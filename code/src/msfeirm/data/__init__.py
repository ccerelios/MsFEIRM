"""Group tensor interfaces. Dataset loaders are implemented in stage II."""

from msfeirm.data.collate import collate_groups
from msfeirm.data.schema import GroupBatch, GroupSample

__all__ = ["GroupBatch", "GroupSample", "collate_groups"]
