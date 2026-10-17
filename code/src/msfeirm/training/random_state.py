"""Weights-only-loadable snapshots of Python, NumPy and PyTorch RNG state."""

import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def capture_random_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(), "position": numpy_state[2],
                  "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA random state requires the same visible device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    if state["torch_mps"] is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS random state requires an available MPS device")
        torch.mps.set_rng_state(state["torch_mps"].cpu())
