"""A single-process shuffled batch cursor committed only after optimizer.step."""

import torch


class StatefulBatchSampler:
    def __init__(self, sample_count, batch_size=1, *, seed=43, shuffle=True):
        if min(sample_count, batch_size) < 1:
            raise ValueError("sample_count and batch_size must be positive")
        self.sample_count, self.batch_size = sample_count, batch_size
        self.seed, self.shuffle = seed, shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.position = 0, 0
        self.order = self._new_order()
        self.pending = None

    def _new_order(self):
        return torch.randperm(self.sample_count, generator=self.generator) if self.shuffle else torch.arange(self.sample_count)

    @property
    def at_epoch_end(self):
        return self.position == self.sample_count

    def peek(self):
        if self.pending is None:
            if self.at_epoch_end:
                self.epoch += 1
                self.position = 0
                self.order = self._new_order()
            self.pending = tuple(self.order[self.position:self.position + self.batch_size].tolist())
        return self.pending

    def commit(self):
        if self.pending is None:
            raise RuntimeError("No pending batch to commit")
        self.position += len(self.pending)
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("Save the sampler at an optimizer-step boundary")
        return {"sample_count": self.sample_count, "batch_size": self.batch_size, "seed": self.seed,
                "shuffle": self.shuffle, "epoch": self.epoch, "position": self.position,
                "order": self.order.clone(), "generator_state": self.generator.get_state()}

    def load_state_dict(self, state):
        for name in ("sample_count", "batch_size", "seed", "shuffle"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Incompatible sampler {name}")
        order = state["order"].cpu()
        if order.shape != (self.sample_count,) or not torch.equal(order.sort().values, torch.arange(self.sample_count)):
            raise ValueError("Invalid saved sampler permutation")
        if not 0 <= state["position"] <= self.sample_count or state["epoch"] < 0:
            raise ValueError("Invalid saved sampler cursor")
        self.order = order.clone()
        self.epoch, self.position = state["epoch"], state["position"]
        self.generator.set_state(state["generator_state"].cpu())
        self.pending = None
