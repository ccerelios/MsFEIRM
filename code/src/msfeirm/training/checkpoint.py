"""Atomic training snapshots and compatibility fingerprints."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil

import torch

from msfeirm.config import PROJECT_ROOT
from msfeirm.runtime import environment_info


def json_fingerprint(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_fingerprint():
    pins = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            if metadata.version(name) != version:
                raise RuntimeError(f"Installed {name} does not match requirements.txt")
            pins[name] = version
    source = {str(path.relative_to(PROJECT_ROOT)): file_fingerprint(path)
              for path in sorted((PROJECT_ROOT / "src/msfeirm").rglob("*.py"))}
    environment = environment_info()
    return {
        "requirements_sha256": file_fingerprint(PROJECT_ROOT / "requirements.txt"),
        "code_sha256": json_fingerprint(source),
        "runtime_sha256": json_fingerprint({"python": environment["python"], "machine": environment["machine"], "packages": pins}),
    }


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    return value


def atomic_save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    if path.exists():
        previous = path.with_name("previous.pt") if path.name == "last.pt" else path.with_name(path.stem + ".previous.pt")
        previous_temporary = previous.with_suffix(previous.suffix + ".tmp")
        shutil.copyfile(path, previous_temporary)
        previous_temporary.replace(previous)
    temporary.replace(path)
