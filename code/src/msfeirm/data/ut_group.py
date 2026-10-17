"""UT CSV -> individual windows -> frozen subgroup manifests -> GroupSample."""

import csv
import json
from pathlib import Path

import numpy as np

from msfeirm.config import load_yaml
from msfeirm.data.prepared import (
    FORMAT_VERSION, PreparedWindowGroupDataset, _array, _fingerprint, _json, _resolve, _sha,
    freeze_group_sets, load_window_splits, verify_prepared_files,
)
from msfeirm.data.preprocessing import ChannelStandardizer


def load_ut_configuration(path):
    config = load_yaml(path)["dataset"]
    if config["name"] != "ut_group":
        raise ValueError("This adapter requires dataset.name=ut_group")
    reader = config["reader"]
    if len(reader["sensor_columns"]) != config["channels"] or min(reader["sensor_columns"]) < 0:
        raise ValueError("sensor_columns must describe the configured channel dimension")
    if min(config["window_size"], config["stride"]) < 1:
        raise ValueError("window_size and stride must be positive")
    if len(config["activity_names"]) != len(config["recipes"]):
        raise ValueError("activity_names and recipes must have equal lengths")
    return config


def read_ut_windows(path, reader, *, window_size=200, stride=100):
    """Read configured channels, windowing each contiguous action-code segment.

    Unused CSV fields are never parsed. Unknown/empty action codes end a segment
    and are omitted from its action pool. Non-finite sensor fields are retained
    for the standardizer to fill. Row references are zero-based half-open.
    """
    mapping = {str(key): value for key, value in reader["action_codes"].items()}
    action_names = list(dict.fromkeys(mapping.values()))
    name_to_id = {name: index for index, name in enumerate(action_names)}
    columns, action_column = reader["sensor_columns"], reader["action_column"]
    windows, actions, ranges = [], [], []
    segment, segment_start, current = [], 0, None

    def flush():
        if len(segment) < window_size:
            return
        values = np.asarray(segment, dtype=np.float32)
        for start in range(0, len(values) - window_size + 1, stride):
            windows.append(values[start:start+window_size].T)
            actions.append(name_to_id[mapping[current]])
            ranges.append((segment_start + start, segment_start + start + window_size))

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.reader(stream)):
            if row_number < reader.get("skip_rows", 0):
                continue
            if len(row) <= max(*columns, action_column):
                raise ValueError(f"CSV row {row_number} has too few columns")
            code = row[action_column].strip()
            try:
                code = str(int(float(code)))
            except (ValueError, OverflowError):
                pass
            if code != current:
                flush()
                segment, segment_start, current = [], row_number, code
            if code not in mapping:
                continue
            segment.append([float(row[column]) if row[column].strip() else float("nan") for column in columns])
    flush()
    if not windows:
        raise ValueError("No complete configured windows were available")
    return np.stack(windows), np.asarray(actions, dtype=np.int64), np.asarray(ranges, dtype=np.int64), action_names


def prepare_ut_data(config_path, output_dir, *, reuse=False):
    config = load_ut_configuration(config_path)
    source = _resolve(config["input"])
    source_sha, config_sha = _sha(source), _fingerprint(config)
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
    windows, action_ids, row_ranges, action_names = read_ut_windows(
        source, config["reader"], window_size=config["window_size"], stride=config["stride"],
    )
    splits = load_window_splits(config, len(windows))
    standardizer = ChannelStandardizer.fit(windows, splits["train"], **config.get("normalization", {}))
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "preparation.json", {"status": "IN_PROGRESS", "source_sha256": source_sha, "config_sha256": config_sha})
    _array(root / "windows.npy", windows)
    _array(root / "actions.npy", action_ids)
    _array(root / "row_ranges.npy", row_ranges)
    _array(root / "split_indices.npz", splits, archive=True)
    _array(root / "normalization.npz", {"mean": standardizer.mean, "scale": standardizer.scale}, archive=True)
    group_sets = freeze_group_sets(root, config, action_ids, action_names, splits)
    files = {path.name: _sha(path) for path in sorted(root.iterdir()) if path.suffix in (".npy", ".npz")}
    manifest = {"format_version": FORMAT_VERSION, "dataset": "ut_group", "source": str(source),
                "source_sha256": source_sha, "config_sha256": config_sha, "split_input_sha256": split_sha, "configuration": config,
                "window_shape": list(windows.shape), "action_names": action_names,
                "split_window_counts": {name: len(indices) for name, indices in splits.items()},
                "group_sets": group_sets, "files": files}
    manifest["fingerprint"] = _fingerprint(manifest)
    _json(root / "manifest.json", manifest)
    _json(root / "preparation.json", {"status": "DONE", "fingerprint": manifest["fingerprint"]})
    return manifest


class PreparedUTGroupDataset(PreparedWindowGroupDataset):
    """Read a frozen UT split/N manifest, retaining its original cache format."""

    def __init__(self, root, split="train", members=10):
        super().__init__(root, split, members)
        self.row_ranges = np.load(self.root / "row_ranges.npy", mmap_mode="r", allow_pickle=False)

    def _source_refs(self, windows):
        return {"window_indices": windows.tolist(), "row_ranges": self.row_ranges[windows].tolist()}
