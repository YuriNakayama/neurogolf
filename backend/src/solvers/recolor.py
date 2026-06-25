"""Recolor solver: apply a fixed per-cell color permutation.

Some ARC tasks just remap colors (e.g. every 1 -> 2) leaving positions intact.
With the one-hot encoding this is a 1x1 conv: a weight of 1 at
``out_channel == mapping[in_channel]``.

:func:`infer_recolor_mapping` derives the mapping from a task's examples; it
returns ``None`` when the task is not a pure recolor (shape changes between
input and output, or one input color maps to two different output colors).
"""

from __future__ import annotations

import onnx

from dataset.encoding import NUM_COLORS, Grid
from dataset.types import Task
from solvers.conv import build_single_layer_conv2d


def _shapes_match(a: Grid, b: Grid) -> bool:
    return len(a) == len(b) and all(
        len(ra) == len(rb) for ra, rb in zip(a, b, strict=True)
    )


def infer_recolor_mapping(task: Task) -> dict[int, int] | None:
    """Infer a consistent color mapping from a task, or ``None`` if not a recolor."""
    mapping: dict[int, int] = {}
    for ex in task.all_examples():
        if not _shapes_match(ex.input, ex.output):
            return None
        for in_row, out_row in zip(ex.input, ex.output, strict=True):
            for src, dst in zip(in_row, out_row, strict=True):
                if mapping.setdefault(src, dst) != dst:
                    return None
    return mapping


def build_recolor_model(mapping: dict[int, int]) -> onnx.ModelProto:
    """Build a recolor ONNX model for ``mapping`` (input color -> output color)."""

    def weight(out_channel: int, in_channel: int, offset: tuple[int, int]) -> float:
        dst = mapping.get(in_channel, in_channel)
        return 1.0 if out_channel == dst and offset == (0, 0) else 0.0

    return build_single_layer_conv2d(weight, kernel_size=1)


def build_recolor_for_task(task: Task) -> onnx.ModelProto | None:
    """Build a recolor model for ``task``, or ``None`` if it is not a recolor."""
    mapping = infer_recolor_mapping(task)
    if mapping is None or all(src == dst for src, dst in mapping.items()):
        return None  # not a recolor, or a no-op (identity handles that)
    if any(dst >= NUM_COLORS for dst in mapping.values()):
        return None
    return build_recolor_model(mapping)
