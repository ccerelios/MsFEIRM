# MsFEIRM

本文的完整 M3 方法实现，包含多尺度特征提取、ECA、局部/全局图 Transformer、成员与数量分支、谱聚类、子群分类，以及训练、预测、评价和第四章实验入口。

2026-10-16 按用户选择整理为**仅源码项目**。保留代码、配置、测试、执行计划和技术说明；原始数据、已处理数据、训练权重、实验结果、历史验收快照、虚拟环境和缓存均已删除。运行前需重新建立环境并提供数据。

## 项目结构

- `src/msfeirm/`：模型、数据处理、训练与恢复、评价、实验调度和报告生成。
- `configs/`：两套数据配置、完整 M3、联合损失、训练和 E01–E05 实验清单。
- `tests/`：模块、完整流程、恢复、评价和报告测试；测试自行构造输入。
- `scripts/accept_pipeline.py`：单数据集完整流程验收入口。
- `Plan.md`：方法规格与分阶段执行计划。
- `docs/`：结构、数据格式、指标、运行步骤和当前交接状态。
- `requirements.in` / `requirements.txt`：直接依赖与已验证环境的固定版本。
- `pyproject.toml`：包安装及命令入口。

只实现完整 M3 和第四章 E01–E05，不包含对比方法、消融或第五章扩展实验；无需对齐论文中的结果数值。

## 建立虚拟环境

在本目录执行，使用 Python 3.12：

```bash
cd /Users/klaus/Desktop/GAR/code
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

所有数据处理、测试和实验均使用 `.venv/bin/python`。增加或修改依赖时维护 `requirements.in` 与 `requirements.txt`，并运行 `pip check` 和相关测试。此前已验证 Python 3.12.14 环境；当前目录不再包含环境或离线 wheel 缓存。

## 数据与实验

数据文件不随源码保留。默认配置的原始输入为：

- UT-Group：`../Datasets/smartphoneatwrist.csv`。
- WBSensor：`../Datasets/WBSensor-public/data/*/*.npy`，数组形状为 `[W,6,128]`。

重新提供数据后，可沿用这些位置，也可在 `configs/datasets/*.yaml` 中修改 `dataset.input`。路径相对于本项目目录解析。准备命令会在 `code/data/processed/` 生成新缓存：

```bash
.venv/bin/python -m msfeirm.cli.prepare --config configs/datasets/ut_group.yaml --output data/processed/ut_group
.venv/bin/python -m msfeirm.cli.prepare --config configs/datasets/wbsensor.yaml --output data/processed/wbsensor
```

随后依照 [docs/reproduce.md](docs/reproduce.md) 逐个运行 UT/WB 训练、E01–E05 和最终报告。每次训练只执行一个时间片，预算用尽后保存检查点，再次执行同一命令即可继续。旧数据和检查点已删除，本次需从头运行。

没有真实数据时，可先在虚拟环境运行测试，或运行完整 M3 的合成数据验证：

```bash
.venv/bin/python -m msfeirm.cli.smoke --synthetic --config configs/synthetic/i4_wb.yaml --steps 20 --output runs/synthetic_smoke
```

技术说明：[结构](docs/architecture.md)、[数据格式](docs/data_format.md)、[评价指标](docs/metrics.md)、[交接状态](docs/handoff.md)。`Plan.md` 中的历史完成记录用于说明实现进度，不能视为本目录仍保存对应运行产物。
