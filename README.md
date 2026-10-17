# MsFEIRM

Official implementation of **MsFEIRM**, a unified framework for IMU-based group segmentation and activity recognition using multi-scale feature extraction and interaction relationship modeling. This repository provides the source code, configuration files, and resources for the **UT-Group** and **WBSensor** datasets.

## Repository Structure

```text
MsFEIRM/
├── code/
│   ├── src/msfeirm/       # Models, data processing, training, and evaluation
│   ├── configs/           # Dataset, model, training, and evaluation settings
│   ├── scripts/           # Pipeline verification utilities
│   ├── requirements.in   # Direct dependencies
│   └── requirements.txt  # Pinned dependencies
├── UT-Group/              # Dataset description and download information
└── WBSensor/
    ├── README.md          # Dataset description
    └── data/              # Sensor windows organized by participant
```

## Datasets

Both datasets use six-channel motion signals: three-axis acceleration and three-axis angular velocity.

| Dataset | Data provided | Access |
| --- | --- | --- |
| **UT-Group** | Wrist sensor recordings used to construct group samples | [Dataset Introduction](https://github.com/ccerelios/MsFEIRM/blob/main/UT-Group/README.md) · [Download Dataset](https://github.com/ccerelios/UT-Group/releases/tag/v1.0.0) |
| **WBSensor** | 87 NumPy files in 21 participant folders (`S001`–`S021`); each array has shape `[windows, 6, 128]` | [Dataset Introduction](https://github.com/ccerelios/MsFEIRM/blob/main/WBSensor/README.md) · [Download Dateset](https://github.com/ccerelios/MsFEIRM/tree/main/WBSensor/data) |

For UT-Group, download `smartphoneatwrist.csv` from the release and place it in `UT-Group/`. The default implementation uses the wrist recordings. WBSensor data is included in this repository.

## Environment Setup

Use **Python 3.12** and create the virtual environment at `code/.venv`. The commands below use Bash/Zsh on Linux or macOS.

```bash
git clone https://github.com/ccerelios/MsFEIRM.git
cd MsFEIRM/code

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Run all commands below from `code/`, using `.venv/bin/python`. Repeat the `PYTHONPATH` export when opening a new shell in that directory. The current repository loads the package directly from `src/`.

`requirements.in` lists direct dependencies; `requirements.txt` pins the complete environment. Training checks the installed versions against `requirements.txt`, so keep both files consistent when changing dependencies.

## Quick Start

### 1. Prepare the data

Before running preparation, update **`dataset.input`** in the following configuration files to match this repository layout. These paths are resolved relative to `code/`.

| Configuration | `dataset.input` |
| --- | --- |
| `configs/datasets/ut_group.yaml` | `../UT-Group/smartphoneatwrist.csv` |
| `configs/datasets/wbsensor.yaml` | `../WBSensor/data` |

```bash
.venv/bin/python -m msfeirm.cli.prepare \
  --config configs/datasets/ut_group.yaml \
  --output data/processed/ut_group

.venv/bin/python -m msfeirm.cli.prepare \
  --config configs/datasets/wbsensor.yaml \
  --output data/processed/wbsensor
```

Preparation creates the sensor-window caches and group samples used by training and evaluation. Prepare only the dataset you intend to use. Add `--reuse` to reuse an existing cache with matching data and configuration.

### 2. Train

```bash
# UT-Group
.venv/bin/python -m msfeirm.cli.train \
  --config configs/training/ut_m3.yaml \
  --output runs/ut_m3 --max-wall-time-min 210

# WBSensor
.venv/bin/python -m msfeirm.cli.train \
  --config configs/training/wb_m3.yaml \
  --output runs/wb_m3 --max-wall-time-min 210
```

The supplied configurations train the complete M3 model for 20 epochs on CPU. Training accepts `--device cuda` or `--device mps` when supported by the installed PyTorch build and hardware.

Training saves checkpoints periodically and when the time budget is reached. Continue an unfinished WBSensor run with:

```bash
.venv/bin/python -m msfeirm.cli.train \
  --resume runs/wb_m3/checkpoints/last.pt --max-wall-time-min 210
```

For UT-Group, use `runs/ut_m3/checkpoints/last.pt`. Resume uses the saved configuration; do not pass `--config` or device overrides alongside `--resume`.

### 3. Predict and evaluate

After training, run prediction and evaluation separately. For WBSensor:

```bash
.venv/bin/python -m msfeirm.cli.predict \
  --checkpoint runs/wb_m3/checkpoints/last.pt \
  --prepared data/processed/wbsensor \
  --output runs/wb_m3_predictions

.venv/bin/python -m msfeirm.cli.evaluate \
  --predictions runs/wb_m3_predictions \
  --output results/wb_m3
```

For UT-Group, use the UT checkpoint, `data/processed/ut_group`, and separate prediction and result directories. Prediction uses the test split by default and covers all configured test group sizes. Evaluation reports group segmentation, activity recognition, and subgroup-count metrics using `configs/protocols/default.yaml`.

Choose a new output directory for each fresh run. All generated outputs are stored under `code/`.

For command options, use `.venv/bin/python -m msfeirm.cli.<command> --help`, replacing `<command>` with `prepare`, `train`, `predict`, or `evaluate`.
