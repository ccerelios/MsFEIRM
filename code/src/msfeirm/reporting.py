"""Rebuild Chapter IV tables and figures using portable saved-file inputs only."""

import base64
import csv
import html
import json
from pathlib import Path
import shutil

import yaml

from msfeirm.config import PROJECT_ROOT
from msfeirm.evaluation.metrics import evaluate_slices
from msfeirm.evaluation.protocol import load_protocol
from msfeirm.runtime import environment_info
from msfeirm.training.checkpoint import execution_fingerprint, file_fingerprint

EXPERIMENTS = ("E01", "E02", "E03", "E04", "E05")
VARIABLE = ("E03", "E04", "E05")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def inside(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Report input paths must stay inside their root")
    return result


def collect_inputs(manifest_path, destination, project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()
    manifest_path = inside(root, manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("version") != 1 or manifest.get("model") != "m3":
        raise ValueError("Chapter IV report requires a complete M3 manifest")
    registry_path = inside(root, manifest["registry"])
    registry = read(registry_path)
    if registry["manifest_sha256"] != file_fingerprint(manifest_path):
        raise ValueError("Report manifest does not match the experiment registry")
    unfinished = [eid for eid in EXPERIMENTS if registry["experiments"].get(eid, {}).get("status") != "DONE"]
    if unfinished:
        raise ValueError("Unfinished experiments: " + ", ".join(unfinished))
    run_ids = sorted({registry["experiments"][eid]["configuration"]["run_id"] for eid in EXPERIMENTS})
    if any(registry["runs"][name]["status"] != "DONE" for name in run_ids):
        raise ValueError("All report training runs must be complete")
    protocol = inside(root, manifest["protocol"])
    plan = [(manifest_path, "manifest.yaml"), (registry_path, "registry.json"), (protocol, "protocol.yaml")]

    def registered(path, hashes):
        path = inside(root, path)
        key = str(path.relative_to(root))
        if key not in hashes or file_fingerprint(path) != hashes[key]:
            raise ValueError(f"Registered report artifact changed: {key}")
        return path

    for eid in EXPERIMENTS:
        entry = registry["experiments"][eid]
        result_path = registered(entry["result"], entry["artifacts"])
        result = read(result_path)
        run = registry["runs"][result["run_id"]]
        if (result["status"] != "DONE" or result["experiment_id"] != eid or result["epochs"] != 20
                or result["checkpoint_sha256"] != run["checkpoint_sha256"]):
            raise ValueError("Report result must reference its complete final training run")
        predictions = registered(str(Path(entry["bundle"]) / "predictions/predictions.jsonl"), entry["artifacts"])
        plan.extend([(result_path, f"experiments/{eid}/result.json"), (predictions, f"experiments/{eid}/predictions.jsonl")])
    for name in run_ids:
        run = registry["runs"][name]
        if file_fingerprint(protocol) != run["fingerprints"]["protocol_sha256"]:
            raise ValueError("Report protocol changed from its frozen training run")
        for filename in ["training_curve.csv", "validation_curve.csv", "summary.json", "config.resolved.yaml", "environment.json", "requirements.txt"]:
            source = registered(str(Path(run["configuration"]["output"]) / filename), run["artifacts"])
            plan.append((source, f"training/{name}/{filename}"))
    destination = Path(destination)
    destination.mkdir(parents=True)
    files, sources = {}, []
    for source, target in plan:
        output = inside(destination, target)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        digest = file_fingerprint(output)
        files[target] = digest
        sources.append({"source": str(source.relative_to(root)), "snapshot": target, "sha256": digest})
    write(destination / "inputs.manifest.json", {"version": 1, "status": "DONE", "files": files, "sources": sources})


def verify_inputs(root):
    root = Path(root)
    manifest = read(root / "inputs.manifest.json")
    if manifest["version"] != 1 or manifest["status"] != "DONE":
        raise ValueError("Report input bundle is incomplete")
    for filename, expected in manifest["files"].items():
        if file_fingerprint(inside(root, filename)) != expected:
            raise ValueError(f"Report input snapshot changed: {filename}")
    return manifest


def metric_row(eid, result, members, metrics):
    segmentation, activity = metrics["segmentation"], metrics["activity_matched"]
    return {"experiment_id": eid, "dataset": result["dataset"], "model": "m3", "seed": result["seed"],
            "epochs": result["epochs"], "members": members, "sample_count": metrics["sample_count"],
            **{f"segmentation_{k}": segmentation[k] for k in ("precision", "recall", "f1", "map")},
            "iou_recall_auc": segmentation["iou_recall_auc"],
            **{f"activity_{k}": activity[k] for k in ("precision_macro", "recall_macro", "f1_macro", "accuracy", "coverage")},
            **{f"pairwise_{k}": metrics["membership_pairs"][k] for k in ("precision", "recall", "f1")},
            "cardinality_mae": metrics["cardinality"]["mae"], "activity_end_to_end_f1": metrics["activity_end_to_end"]["f1"],
            "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"]}


def analyze_inputs(root):
    root = Path(root)
    verify_inputs(root)
    registry = read(root / "registry.json")
    protocol = load_protocol(root / "protocol.yaml")
    results, predictions, metrics = {}, {}, {}
    for eid in EXPERIMENTS:
        result = read(root / f"experiments/{eid}/result.json")
        records = [json.loads(line) for line in (root / f"experiments/{eid}/predictions.jsonl").read_text().splitlines()]
        recomputed = evaluate_slices(records, protocol)
        recomputed["overall"].pop("samples")
        if recomputed != result["metrics"]:
            raise ValueError(f"Saved prediction recomputation changed for {eid}")
        results[eid], predictions[eid], metrics[eid] = result, records, recomputed
    wb_checkpoint = results["E02"]["checkpoint_sha256"]
    if any(results[eid]["checkpoint_sha256"] != wb_checkpoint for eid in VARIABLE):
        raise ValueError("All WB report slices must use the same final checkpoint")
    pooled_records = [record for eid in VARIABLE for record in predictions[eid]]
    if len({(r["members"], r["sample_id"]) for r in pooled_records}) != len(pooled_records):
        raise ValueError("WB interval prediction inputs overlap")
    if [r for r in pooled_records if r["members"] == 15] != predictions["E02"]:
        raise ValueError("The fixed N=15 records must match their reused interval records")
    pooled = evaluate_slices(pooled_records, protocol)
    details = pooled["overall"].pop("samples")
    fixed = [metric_row(eid, results[eid], str(predictions[eid][0]["members"]), metrics[eid]["overall"]) for eid in ("E01", "E02")]
    members, intervals = [], []
    for eid in VARIABLE:
        members.extend(metric_row(eid, results[eid], n, m) for n, m in metrics[eid]["by_members"].items())
        sizes = registry["experiments"][eid]["configuration"]["members"]
        key = f"{min(sizes)}-{max(sizes)}"
        intervals.append(metric_row(eid, results[eid], key, metrics[eid]["intervals"][key]))
    members.sort(key=lambda row: int(row["members"]))
    pooled_row = metric_row("WB_ALL", results["E02"], "13-21", pooled["overall"])
    return {"experiments": metrics, "WB_pooled": pooled,
            "tables": {"fixed": fixed, "members": members, "intervals": intervals, "pooled": [pooled_row]},
            "activity_names": {"ut_group": predictions["E01"][0]["activity_names"], "wbsensor": predictions["E02"][0]["activity_names"]}}, details


def csv_table(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_curves(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def markdown_table(rows, columns):
    def cell(value):
        return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(title for _, title in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row[key]) for key, _ in columns) + " |" for row in rows)
    return "\n".join(lines)


def write_documents(directory, analysis, training, figures):
    tables = analysis["tables"]
    segmentation = [("dataset", "数据集"), ("members", "N"), ("sample_count", "样本"),
                    *[(f"segmentation_{k}", label) for k, label in [("precision", "P"), ("recall", "R"), ("f1", "F1"), ("map", "mAP")]], ("iou_recall_auc", "AUC")]
    activity = [("dataset", "数据集"), ("members", "N"), ("sample_count", "样本"),
                *[(f"activity_{k}", label) for k, label in [("precision_macro", "Macro P"), ("recall_macro", "Macro R"), ("f1_macro", "Macro F1"), ("accuracy", "Accuracy"), ("coverage", "覆盖率")]]]
    compact = [("members", "N"), ("sample_count", "样本"), ("segmentation_f1", "分割 F1"),
               ("segmentation_map", "mAP"), ("iou_recall_auc", "AUC"), ("activity_f1_macro", "活动 Macro F1"), ("activity_accuracy", "活动 Accuracy")]
    sections = [("固定人数：子群分割", markdown_table(tables["fixed"], segmentation)),
                ("固定人数：匹配子群活动", markdown_table(tables["fixed"], activity)),
                ("WBSensor：每个人数", markdown_table(tables["members"], compact)),
                ("WBSensor：人数区间", markdown_table(tables["intervals"], compact)),
                ("WBSensor：全部人数合并", markdown_table(tables["pooled"], compact)),
                ("训练记录", markdown_table(training, [("dataset", "数据集"), ("seed", "Seed"), ("epochs", "Epoch"), ("steps", "步数"), ("training_seconds", "训练秒数"), ("best_step", "Best step")]))]
    protocol = load_protocol(directory / "protocol.yaml")
    intro = f"完整 MsFEIRM，seed=42，20 epoch；E01–E05 均已完成。指标为本次实际运行结果，数值范围为 0–1。分割采用 IoU≥{protocol.iou_threshold:g}，mAP 使用 0.50–0.95 的十个阈值，AUC 为 IoU 阈值上的召回率积分。活动主指标按 IoU 匹配的有效子群计算；覆盖率与端到端指标另存 CSV/JSON。"
    notes = "WB 全部人数仅合并 E03–E05 的不重叠记录，共 1,053 个样本；E02 的 N=15 已在 E03 中，不重复计入。AP/AUC/F1 从逐样本记录重算，未平均各人数指标。仅运行 seed 42，不生成跨种子误差条。"
    # The text also works for small report fixtures with different sample counts.
    notes = notes.replace("1,053", f"{tables['pooled'][0]['sample_count']:,}")
    md = ["# MsFEIRM 第四章实际实验报告", "", intro, "", notes]
    for title, table in sections:
        md += ["", "## " + title, "", table]
    for title, stem in figures:
        md += ["", "## " + title, "", f"![{title}](figures/{stem}.png)", "", f"[矢量 SVG](figures/{stem}.svg)"]
    md += ["", "## 文件与重建", "", "完整精度表格见 tables/，完整指标见 metrics.json，输入快照见 inputs/。重建步骤见 RUNNING.md，冻结指标定义见 protocol.yaml；model/protocol/source hashes 见 provenance.json。"]
    (directory / "report.md").write_text("\n".join(md) + "\n")
    def html_table(rows, columns):
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape('—' if row[key] is None else format(row[key], '.6f') if isinstance(row[key], float) else str(row[key]))}</td>" for key, _ in columns) + "</tr>" for row in rows)
        return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    html_sections = [("固定人数：子群分割", tables["fixed"], segmentation), ("固定人数：匹配子群活动", tables["fixed"], activity),
                     ("WBSensor：每个人数", tables["members"], compact), ("WBSensor：人数区间", tables["intervals"], compact), ("WBSensor：全部人数合并", tables["pooled"], compact),
                     ("训练记录", training, [("dataset", "数据集"), ("seed", "Seed"), ("epochs", "Epoch"), ("steps", "步数"), ("training_seconds", "训练秒数"), ("best_step", "Best step")])]
    body = [f"<h1>MsFEIRM 第四章实际实验报告</h1><p>{intro}</p><p>{notes}</p>"]
    for title, rows, columns in html_sections:
        body += [f"<h2>{title}</h2>", html_table(rows, columns)]
    for title, stem in figures:
        data = base64.b64encode((directory / f"figures/{stem}.png").read_bytes()).decode()
        body.append(f"<h2>{title}</h2><img alt='{html.escape(title)}' src='data:image/png;base64,{data}'>")
    body.append("<p>完整精度 CSV/JSON、来源与重建步骤随报告保存。该 HTML 内嵌图像，可单独离线浏览。</p>")
    style = "body{font:15px/1.65 system-ui,sans-serif;color:#172b3a;max-width:1120px;margin:40px auto;padding:0 24px}h1,h2{line-height:1.3}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #dce3e9;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#edf3f8}.table{overflow:auto}img{width:100%;height:auto;border:1px solid #e1e7eb}"
    (directory / "report.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>MsFEIRM 实验报告</title><style>" + style + "</style><body>" + "\n".join(body) + "</body></html>")


def build_report(output, *, manifest_path="configs/experiments/chapter4.yaml", from_inputs=None, project_root=PROJECT_ROOT):
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Report output is not empty; choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "manifest.json", {"stage": "III-3", "status": "IN_PROGRESS"})
    inputs = directory / "inputs"
    if from_inputs is None:
        collect_inputs(manifest_path, inputs, project_root)
    else:
        source = Path(from_inputs)
        original = verify_inputs(source)
        inputs.mkdir()
        for filename in [*original["files"], "inputs.manifest.json"]:
            target = inside(inputs, filename); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(source, filename), target)
    analysis, details = analyze_inputs(inputs)
    write(directory / "metrics.json", analysis)
    tables = directory / "tables"; tables.mkdir()
    figures_dir = directory / "figures"; figures_dir.mkdir()
    for name, values in analysis["tables"].items():
        csv_table(tables / f"{name}.csv", values)
    registry = read(inputs / "registry.json")
    training, figures = [], []
    from msfeirm.plotting import training_figure, confusion_figure, member_figure
    for run_id, row in registry["runs"].items():
        root = inputs / "training" / run_id
        summary = read(root / "summary.json")
        config = yaml.safe_load((root / "config.resolved.yaml").read_text())
        curves, validation = load_curves(root / "training_curve.csv"), load_curves(root / "validation_curve.csv")
        if int(curves[-1]["global_step"]) != summary["global_step"] or summary["completed_epochs"] != 20:
            raise ValueError("Training curves must include the complete final run")
        dataset = row["configuration"]["dataset"]
        name = "ut" if dataset == "ut_group" else "wb"
        training.append({"run_id": run_id, "dataset": dataset, "seed": config["training"]["seed"], "epochs": 20,
                         "steps": summary["global_step"], "training_seconds": summary["elapsed_seconds"],
                         "best_step": summary["best_validation"]["step"], "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"]})
        stem = name + "_training"
        training_figure(curves, validation, ("UT-Group" if name == "ut" else "WBSensor") + " · seed 42 · 20 epochs", figures_dir / stem)
        figures.append((dataset + "：训练与验证曲线", stem))
    csv_table(tables / "training.csv", training)
    for eid, key, title in [("E01", "ut_group", "UT-Group · N=10"), ("E02", "wbsensor", "WBSensor · N=15"), ("WB_ALL", "wbsensor", "WBSensor · N=13–21 pooled")]:
        metrics = analysis["WB_pooled"]["overall"] if eid == "WB_ALL" else analysis["experiments"][eid]["overall"]
        matrix = metrics["activity_matched"]["confusion_matrix"]
        names = analysis["activity_names"][key]
        csv_table(tables / f"confusion_{eid}.csv", [{"target_id": i, "target_activity": names[i], **{str(j): value for j, value in enumerate(row)}} for i, row in enumerate(matrix)])
        stem = "confusion_" + eid
        confusion_figure(matrix, names, title, figures_dir / stem)
        figures.append((title + "：活动混淆矩阵", stem))
    member_figure(analysis["tables"]["members"], figures_dir / "wb_member_metrics")
    figures.append(("WB：变人数指标曲线", "wb_member_metrics"))
    with (directory / "wb_pooled_matches.jsonl").open("w") as stream:
        for detail in details:
            stream.write(json.dumps(detail, allow_nan=False) + "\n")
    shutil.copyfile(inputs / "protocol.yaml", directory / "protocol.yaml")
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", directory / "requirements.txt")
    write(directory / "environment.json", environment_info())
    provenance = {"inputs_manifest_sha256": file_fingerprint(inputs / "inputs.manifest.json"),
                  "registry_sha256": file_fingerprint(inputs / "registry.json"), "protocol_sha256": file_fingerprint(inputs / "protocol.yaml"),
                  "execution_fingerprint": execution_fingerprint(), "completed_experiments": list(EXPERIMENTS),
                  "WB_pooled_sources": list(VARIABLE), "models_loaded": False, "original_sensor_files_required": False,
                  "training": training, "figures": [{"title": title, "stem": stem} for title, stem in figures]}
    write(directory / "provenance.json", provenance)
    write_documents(directory, analysis, training, figures)
    (directory / "RUNNING.md").write_text("# 报告重建\n\n在项目 code 目录和虚拟环境执行，输出必须使用新的空目录：\n\n```bash\n.venv/bin/python -m msfeirm.cli.report --scope chapter4 --from-inputs " + str(inputs.resolve()) + " --output reports/chapter4_rebuilt\n```\n\ninputs/ 已保存重建表图所需的预测、指标、协议与训练曲线；无需模型或原始传感器文件。项目级环境创建、数据准备和实验命令见 code/README.md 与 docs/reproduce.md。所有值来自单一 seed 42；图中滑动均值只用于显示，原始训练点和完整 CSV 均保留。\n")
    artifacts = {str(p.relative_to(directory)): file_fingerprint(p) for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "manifest.json"}
    write(directory / "manifest.json", {"stage": "III-3", "status": "DONE", "completed_experiments": list(EXPERIMENTS),
                                      "WB_unique_sample_count": analysis["WB_pooled"]["overall"]["sample_count"], "figure_count": len(figures), "artifacts": artifacts})
    return {"status": "DONE", "output": str(directory), "WB_unique_sample_count": analysis["WB_pooled"]["overall"]["sample_count"], "figure_count": len(figures)}
