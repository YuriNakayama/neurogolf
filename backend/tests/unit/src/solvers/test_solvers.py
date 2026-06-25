"""Tests for the per-task ONNX solvers.

These run the built models through onnxruntime to assert the conv weight layout
and spatial transforms are correct end-to-end (a wrong ``itertools.product``
order or perm would fail here).
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime

from dataset.encoding import decode_grid, encode_grid
from dataset.types import Example, Task
from solvers.identity import build_identity_model
from solvers.recolor import (
    build_recolor_for_task,
    build_recolor_model,
    infer_recolor_mapping,
)
from solvers.run import solves_task
from solvers.spatial import (
    build_flip_lr_model,
    build_flip_ud_model,
    build_rot180_model,
    build_transpose_model,
)


def _predict(model: onnx.ModelProto, grid: list[list[int]]) -> list[list[int]]:
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    out = sess.run(["output"], {"input": encode_grid(grid)})[0]
    return decode_grid((out > 0.0).astype(np.float32))


def _task(pairs: list[tuple[list[list[int]], list[list[int]]]]) -> Task:
    train = tuple(Example(input=i, output=o) for i, o in pairs)
    return Task(task_id=1, train=train, test=(), arc_gen=())


def test_identity_round_trips_grid() -> None:
    grid = [[0, 4, 9], [1, 1, 0], [8, 0, 3]]
    assert _predict(build_identity_model(), grid) == grid


def test_identity_solves_identity_task() -> None:
    task = _task([([[1, 2], [3, 4]], [[1, 2], [3, 4]])])
    assert solves_task(build_identity_model(), task)


def test_infer_recolor_mapping_consistent() -> None:
    task = _task([([[1, 2]], [[2, 3]]), ([[1, 1]], [[2, 2]])])
    assert infer_recolor_mapping(task) == {1: 2, 2: 3}


def test_infer_recolor_mapping_conflict_returns_none() -> None:
    task = _task([([[1]], [[2]]), ([[1]], [[3]])])  # 1 -> 2 and 1 -> 3
    assert infer_recolor_mapping(task) is None


def test_infer_recolor_mapping_shape_change_returns_none() -> None:
    task = _task([([[1, 1]], [[1]])])
    assert infer_recolor_mapping(task) is None


def test_recolor_applies_mapping() -> None:
    model = build_recolor_model({1: 2, 0: 0})
    assert _predict(model, [[1, 0], [1, 1]]) == [[2, 0], [2, 2]]


def test_build_recolor_for_task_rejects_pure_identity() -> None:
    task = _task([([[1, 2]], [[1, 2]])])  # no color changes
    assert build_recolor_for_task(task) is None


def test_recolor_solves_recolor_task() -> None:
    task = _task([([[1, 2], [2, 1]], [[2, 3], [3, 2]])])
    model = build_recolor_for_task(task)
    assert model is not None
    assert solves_task(model, task)


def test_transpose_swaps_axes() -> None:
    assert _predict(build_transpose_model(), [[1, 2, 3], [4, 5, 6]]) == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]


def test_flip_models_on_fixed_size_grid() -> None:
    # full 30x30 grid so reversal aligns (no left/top-justify shift)
    grid = [[(r + c) % 9 + 1 for c in range(30)] for r in range(30)]
    assert _predict(build_flip_lr_model(), grid) == [list(reversed(row)) for row in grid]
    assert _predict(build_flip_ud_model(), grid) == list(reversed(grid))
    assert _predict(build_rot180_model(), grid) == [
        list(reversed(row)) for row in reversed(grid)
    ]
