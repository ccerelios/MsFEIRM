"""Project-local execution and output-path helpers."""

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from msfeirm.config import PROJECT_ROOT


def require_project_virtualenv(expected: Path | None = None) -> Path:
    expected = (expected or PROJECT_ROOT / ".venv").resolve()
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != expected:
        raise RuntimeError(
            f"Use the project virtual environment: {expected / 'bin/python'} "
            f"(current prefix: {sys.prefix})"
        )
    return expected


def project_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Output must remain inside {PROJECT_ROOT}: {candidate}")
    return candidate


def environment_info() -> dict[str, Any]:
    import torch

    dependencies = {}
    for name in (
        "msfeirm", "torch", "numpy", "PyYAML", "scikit-learn", "scipy", "pytest", "setuptools", "wheel",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "virtualenv": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dependencies,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
