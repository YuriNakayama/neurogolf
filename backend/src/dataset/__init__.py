"""ARC-AGI task loading and grid <-> tensor encoding for NeuroGolf.

The **data / representation layer**: it loads task JSON (ARC-AGI-1 train /
ARC-GEN) and converts grids to and from the ``[1, NUM_COLORS, GRID_MAX,
GRID_MAX]`` one-hot tensor the ONNX solvers consume.

    types       Example / Task (frozen; as_scorer_dict() feeds evaluate.audit_one)
    encoding    encode_grid / decode_grid ([1,10,30,30], >0 decode, zero-hot border)
    loader      load_task / load_tasks / available_task_ids ("arc-gen" key, taskNNN)
"""

from __future__ import annotations

from dataset.encoding import (
    BATCH,
    CLEAR,
    GRID_MAX,
    NUM_COLORS,
    EncodingError,
    Grid,
    decode_grid,
    encode_grid,
)
from dataset.loader import (
    TaskLoadError,
    available_task_ids,
    load_task,
    load_tasks,
    task_filename,
)
from dataset.types import Example, Task

__all__ = [
    "BATCH",
    "CLEAR",
    "GRID_MAX",
    "NUM_COLORS",
    "EncodingError",
    "Example",
    "Grid",
    "Task",
    "TaskLoadError",
    "available_task_ids",
    "decode_grid",
    "encode_grid",
    "load_task",
    "load_tasks",
    "task_filename",
]
