"""Prepare UT-Group or WBSensor window caches and frozen group manifests."""

import argparse
import json
from pathlib import Path
import shutil

import yaml

from msfeirm.config import PROJECT_ROOT, load_yaml
from msfeirm.data.ut_group import prepare_ut_data
from msfeirm.data.wbsensor import prepare_wb_data
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/datasets/ut_group.yaml")
    parser.add_argument("--output")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    name = load_yaml(args.config)["dataset"]["name"]
    adapters = {"ut_group": prepare_ut_data, "wbsensor": prepare_wb_data}
    if name not in adapters:
        parser.error(f"Unsupported dataset: {name}")
    root = project_output_path(args.output or f"data/processed/{name}")
    manifest = adapters[name](args.config, root, reuse=args.reuse)
    (root / "config.resolved.yaml").write_text(yaml.safe_dump({"dataset": manifest["configuration"]}, sort_keys=False), encoding="utf-8")
    (root / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", root / "requirements.txt")
    summary = {key: manifest[key] for key in ("dataset", "window_shape", "split_window_counts", "group_sets", "fingerprint")}
    summary["prepared_root"] = str(root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
