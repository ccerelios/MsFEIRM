"""Shared frozen-window cache utilities and group dataset interface."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from msfeirm.config import PROJECT_ROOT
from msfeirm.data.grouping import ActivityRecipe, GroupConstructionConfig, compose_groups
from msfeirm.data.preprocessing import ChannelStandardizer, split_window_indices
from msfeirm.data.schema import GroupSample


FORMAT_VERSION = 1


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _resolve(path):
    return (PROJECT_ROOT / path).resolve()


def _json(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _array(path, values, *, archive=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        if archive:
            np.savez(stream, **values)
        else:
            np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def load_window_splits(config, count):
    split_config = config.get("split", {})
    supplied = None
    if split_config.get("supplied_manifest"):
        path = _resolve(split_config["supplied_manifest"])
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                supplied = {key: archive[key] for key in ("train", "validation", "test")}
        else:
            supplied = json.loads(path.read_text())
    return split_window_indices(count, ratios=split_config.get("ratios", (.7, .1, .2)),
                                seed=split_config.get("seed", 42), supplied=supplied)


def freeze_group_sets(root, config, action_ids, action_names, splits):
    recipes = tuple(ActivityRecipe.from_mapping(value) for value in config["recipes"])
    group_sets = {}
    for split_index, (split, indices) in enumerate(splits.items()):
        member_counts = [config["train_members"]] if split != "test" else config["test_members"]
        for members in member_counts:
            construction = GroupConstructionConfig(members=members, **config.get("grouping", {}))
            requested = config.get("group_samples", {}).get(split)
            sample_count = len(indices) // members if requested is None else int(requested)
            arrays = compose_groups(action_ids, indices, action_names, recipes, construction, sample_count,
                                    seed=construction.seed + 1000 * split_index + members)
            key = f"{split}_n{members}"
            filename = f"groups_{key}.npz"
            _array(root / filename, arrays, archive=True)
            group_sets[key] = {"file": filename, "count": sample_count, "members": members,
                               "k_min": construction.k_bounds[0], "k_max": construction.k_bounds[1]}
    return group_sets


def verify_prepared_files(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported prepared data format")
    if manifest["fingerprint"] != _fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}):
        raise ValueError("Prepared manifest fingerprint mismatch")
    for filename, expected in manifest["files"].items():
        if _sha(root / filename) != expected:
            raise ValueError(f"Prepared file fingerprint mismatch: {filename}")
    return manifest


class PreparedWindowGroupDataset(Dataset):
    """Load one frozen split/N set; subclasses provide individual references."""

    def __init__(self, root, split, members):
        self.root, self.split, self.members = Path(root).resolve(), split, members
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported prepared data format")
        self.key = f"{split}_n{members}"
        info = self.manifest["group_sets"][self.key]
        self.composition_k_bounds = (info["k_min"], info["k_max"])
        self.k_bounds = tuple(self.manifest.get("model_k_bounds", self.composition_k_bounds))
        self.fingerprint = _fingerprint({"prepared": self.manifest["fingerprint"], "group_set": self.key})
        self.windows = np.load(self.root / "windows.npy", mmap_mode="r", allow_pickle=False)
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r", allow_pickle=False)
        with np.load(self.root / "normalization.npz", allow_pickle=False) as archive:
            self.standardizer = ChannelStandardizer(archive["mean"], archive["scale"])
        with np.load(self.root / info["file"], allow_pickle=False) as archive:
            self.groups = {key: archive[key] for key in archive.files}

    def __len__(self):
        return len(self.groups["window_indices"])

    def _source_refs(self, windows):
        return {"window_indices": windows.tolist()}

    def __getitem__(self, index):
        windows = self.groups["window_indices"][index]
        values = self.standardizer.transform(self.windows[windows])
        ids = torch.from_numpy(self.groups["group_ids"][index].copy())
        k = int(self.groups["k_clusters"][index])
        return GroupSample(
            torch.from_numpy(values), sample_id=f"{self.manifest['dataset']}_{self.key}_{index:06d}", group_ids=ids,
            group_labels=torch.from_numpy(self.groups["group_labels"][index, :k].copy()),
            pair_targets=(ids[:, None] == ids[None, :]).float(), k_clusters=k,
            noise_mask=torch.from_numpy(self.groups["noise_mask"][index].copy()),
            source_refs=self._source_refs(windows),
        )


def open_prepared_dataset(root, split="train", members=None):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    members = manifest["configuration"]["train_members"] if members is None else members
    if manifest["dataset"] == "ut_group":
        from msfeirm.data.ut_group import PreparedUTGroupDataset
        return PreparedUTGroupDataset(root, split, members)
    if manifest["dataset"] == "wbsensor":
        from msfeirm.data.wbsensor import PreparedWBSensorDataset
        return PreparedWBSensorDataset(root, split, members)
    raise ValueError(f"Unsupported prepared dataset: {manifest['dataset']}")
