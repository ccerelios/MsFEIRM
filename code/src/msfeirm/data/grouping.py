"""Seeded subgroup composition from individual action windows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActivityRecipe:
    name: str
    member_actions: tuple[str, ...]
    anchor_action: str | None = None

    @classmethod
    def from_mapping(cls, values):
        return cls(values["name"], tuple(values["member_actions"]), values.get("anchor_action"))


@dataclass(frozen=True)
class GroupConstructionConfig:
    members: int = 10
    min_activity_groups: int = 3
    max_activity_groups: int = 5
    min_subgroup_size: int = 2
    max_subgroup_size: int = 5
    noise_members: int = 1
    ignore_index: int = -100
    seed: int = 43

    def __post_init__(self):
        for name in ("members", "min_activity_groups", "max_activity_groups", "min_subgroup_size", "max_subgroup_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.noise_members) is not int or not 0 <= self.noise_members < self.members:
            raise ValueError("noise_members must be an integer in [0,members)")
        if self.min_subgroup_size > self.max_subgroup_size or self.min_activity_groups > self.max_activity_groups:
            raise ValueError("Group range lower bounds must not exceed upper bounds")
        if not self.feasible_activity_counts:
            raise ValueError("No feasible subgroup sizes for the requested member count")

    @property
    def feasible_activity_counts(self):
        active = self.members - self.noise_members
        return tuple(k for k in range(self.min_activity_groups, self.max_activity_groups + 1)
                     if k * self.min_subgroup_size <= active <= k * self.max_subgroup_size)

    @property
    def k_bounds(self):
        return (min(self.feasible_activity_counts) + self.noise_members,
                max(self.feasible_activity_counts) + self.noise_members)


def compose_groups(action_ids, split_indices, action_names, recipes, config, sample_count, *, seed=None):
    """Build compact member-window indices and targets; no sensor data is copied."""
    if sample_count < 0 or max(config.feasible_activity_counts) > len(recipes):
        raise ValueError("Invalid sample count or too few activity recipes")
    generator = np.random.default_rng(config.seed if seed is None else seed)
    pools = {name: split_indices[action_ids[split_indices] == index] for index, name in enumerate(action_names)}
    required = {action for recipe in recipes for action in (*recipe.member_actions, recipe.anchor_action) if action is not None}
    if sample_count and any(name not in pools or len(pools[name]) == 0 for name in required):
        raise ValueError("A required action window pool is empty; supply usable windows or update the recipe configuration")
    max_k = config.k_bounds[1]
    windows = np.zeros((sample_count, config.members), dtype=np.int64)
    group_ids = np.zeros_like(windows)
    group_labels = np.full((sample_count, max_k), config.ignore_index, dtype=np.int64)
    noise_mask = np.zeros_like(windows, dtype=np.bool_)
    counts = np.zeros(sample_count, dtype=np.int64)
    for sample in range(sample_count):
        active_k = int(generator.choice(config.feasible_activity_counts))
        sizes = np.full(active_k, config.min_subgroup_size, dtype=np.int64)
        for _ in range(config.members - config.noise_members - int(sizes.sum())):
            choices = np.flatnonzero(sizes < config.max_subgroup_size)
            sizes[int(generator.choice(choices))] += 1
        activities = generator.choice(len(recipes), active_k, replace=False)
        cursor = 0
        for group, (activity, size) in enumerate(zip(activities, sizes, strict=True)):
            recipe = recipes[activity]
            roles = list(generator.choice(recipe.member_actions, int(size)))
            if recipe.anchor_action is not None:
                roles[0] = recipe.anchor_action
            for role in sorted(set(roles)):
                slots = np.array([index for index, value in enumerate(roles) if value == role])
                selected = generator.choice(pools[role], len(slots), replace=len(pools[role]) < len(slots))
                windows[sample, cursor + slots] = selected
            group_ids[sample, cursor:cursor+size] = group
            group_labels[sample, group] = activity
            cursor += int(size)
        for noise in range(config.noise_members):
            windows[sample, cursor] = generator.choice(split_indices)
            group_ids[sample, cursor] = active_k + noise
            noise_mask[sample, cursor] = True
            cursor += 1
        counts[sample] = active_k + config.noise_members
        permutation = generator.permutation(config.members)
        windows[sample], group_ids[sample], noise_mask[sample] = (value[sample, permutation].copy() for value in (windows, group_ids, noise_mask))
    return {"window_indices": windows, "group_ids": group_ids, "group_labels": group_labels,
            "k_clusters": counts, "noise_mask": noise_mask}
