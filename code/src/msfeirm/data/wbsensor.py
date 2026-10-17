"""WBSensor NPY windows -> frozen fixed/variable-size group manifests."""

import json
from pathlib import Path
import re

import numpy as np

from msfeirm.config import load_yaml
from msfeirm.data.grouping import GroupConstructionConfig
from msfeirm.data.prepared import (
    FORMAT_VERSION, PreparedWindowGroupDataset, _array, _fingerprint, _json, _resolve, _sha,
    freeze_group_sets, load_window_splits, verify_prepared_files,
)
from msfeirm.data.preprocessing import ChannelStandardizer


def load_wb_configuration(path):
    config = load_yaml(path)["dataset"]
    if config["name"] != "wbsensor":
        raise ValueError("This adapter requires dataset.name=wbsensor")
    if min(config["channels"], config["window_size"]) < 1:
        raise ValueError("channels and window_size must be positive")
    if [recipe["name"] for recipe in config["recipes"]] != config["activity_names"]:
        raise ValueError("recipe names must follow activity_names order")
    bounds = config["model_k_bounds"]
    if len(bounds) != 2 or any(type(value) is not int for value in bounds) or not 1 <= bounds[0] <= bounds[1]:
        raise ValueError("model_k_bounds must contain two ordered positive integers")
    for members in {config["train_members"], *config["test_members"]}:
        construction = GroupConstructionConfig(members=members, **config.get("grouping", {}))
        if construction.k_bounds[0] < bounds[0] or construction.k_bounds[1] > bounds[1] or bounds[1] > members:
            raise ValueError("model_k_bounds must cover every configured group size")
    return config


def inspect_wb_sources(config):
    """Read array headers and hashes needed for executable loading/cache reuse."""
    root = _resolve(config["input"])
    paths = sorted(root.glob(config.get("input_glob", "*/*.npy")))
    if not paths:
        raise ValueError("No WBSensor NPY input files matched input_glob")
    mapping = config["reader"]["file_actions"]
    sources = []
    for path in paths:
        key = re.sub(r"_\d+$", "", path.stem)
        if key not in mapping:
            raise ValueError(f"No configured action for filename: {path.name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 3 or values.shape[1:] != (config["channels"], config["window_size"]):
            raise ValueError(f"Expected [windows,{config['channels']},{config['window_size']}]: {path}")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError(f"Expected real numeric sensor values: {path}")
        sources.append({"file": path.relative_to(root).as_posix(), "action": mapping[key],
                        "shape": list(values.shape), "dtype": str(values.dtype), "sha256": _sha(path)})
    return sources


def read_wb_windows(config, sources=None):
    """Preserve pre-windowed [W,C,T] values and file/window references."""
    sources = inspect_wb_sources(config) if sources is None else sources
    action_names = list(dict.fromkeys(config["reader"]["file_actions"].values()))
    name_to_id = {name: index for index, name in enumerate(action_names)}
    count = sum(source["shape"][0] for source in sources)
    if count == 0:
        raise ValueError("No individual WBSensor windows were available")
    windows = np.empty((count, config["channels"], config["window_size"]), dtype=np.float32)
    actions, file_ids, window_ids = (np.empty(count, dtype=np.int64) for _ in range(3))
    root, cursor = _resolve(config["input"]), 0
    for file_id, source in enumerate(sources):
        values = np.load(root / source["file"], mmap_mode="r", allow_pickle=False)
        end = cursor + len(values)
        windows[cursor:end] = values
        actions[cursor:end] = name_to_id[source["action"]]
        file_ids[cursor:end] = file_id
        window_ids[cursor:end] = np.arange(len(values))
        cursor = end
    return windows, actions, file_ids, window_ids, action_names, sources


def prepare_wb_data(config_path, output_dir, *, reuse=False):
    config = load_wb_configuration(config_path)
    sources = inspect_wb_sources(config)
    source_sha, config_sha = _fingerprint(sources), _fingerprint(config)
    split_path = config.get("split", {}).get("supplied_manifest")
    split_sha = _sha(_resolve(split_path)) if split_path else None
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        if not reuse or not (root / "manifest.json").is_file():
            raise ValueError("Prepared output is not empty; use reuse for a complete matching cache or a new directory")
        manifest = json.loads((root / "manifest.json").read_text())
        if (manifest["source_sha256"] != source_sha or manifest["config_sha256"] != config_sha
                or manifest["split_input_sha256"] != split_sha):
            raise ValueError("Prepared source/configuration does not match")
        verify_prepared_files(root)
        return manifest
    windows, actions, file_ids, window_ids, action_names, _ = read_wb_windows(config, sources)
    splits = load_window_splits(config, len(windows))
    standardizer = ChannelStandardizer.fit(windows, splits["train"], **config.get("normalization", {}))
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "preparation.json", {"status": "IN_PROGRESS", "source_sha256": source_sha, "config_sha256": config_sha})
    for name, values in (("windows", windows), ("actions", actions), ("source_file_ids", file_ids), ("source_window_indices", window_ids)):
        _array(root / f"{name}.npy", values)
    _array(root / "split_indices.npz", splits, archive=True)
    _array(root / "normalization.npz", {"mean": standardizer.mean, "scale": standardizer.scale}, archive=True)
    group_sets = freeze_group_sets(root, config, actions, action_names, splits)
    files = {path.name: _sha(path) for path in sorted(root.iterdir()) if path.suffix in (".npy", ".npz")}
    manifest = {"format_version": FORMAT_VERSION, "dataset": "wbsensor", "source": str(_resolve(config["input"])),
                "source_files": sources, "source_sha256": source_sha, "config_sha256": config_sha,
                "split_input_sha256": split_sha, "configuration": config, "model_k_bounds": config["model_k_bounds"],
                "window_shape": list(windows.shape), "action_names": action_names,
                "split_window_counts": {name: len(indices) for name, indices in splits.items()},
                "group_sets": group_sets, "files": files}
    manifest["fingerprint"] = _fingerprint(manifest)
    _json(root / "manifest.json", manifest)
    _json(root / "preparation.json", {"status": "DONE", "fingerprint": manifest["fingerprint"]})
    return manifest


class PreparedWBSensorDataset(PreparedWindowGroupDataset):
    """Load N-specific groups while retaining a common model K protocol."""

    def __init__(self, root, split="train", members=15):
        super().__init__(root, split, members)
        self.file_ids = np.load(self.root / "source_file_ids.npy", mmap_mode="r", allow_pickle=False)
        self.window_ids = np.load(self.root / "source_window_indices.npy", mmap_mode="r", allow_pickle=False)

    def _source_refs(self, windows):
        return {"window_indices": windows.tolist(),
                "files": [self.manifest["source_files"][file_id]["file"] for file_id in self.file_ids[windows]],
                "file_window_indices": self.window_ids[windows].tolist()}
