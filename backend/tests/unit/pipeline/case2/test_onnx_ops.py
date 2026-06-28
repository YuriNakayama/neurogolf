"""case2 primitive builders produce valid, scorable, correct ONNX.

Each builder must (1) pass onnx.checker, (2) score via the official-scorer mirror
with a finite cost, and (3) on a synthetic task, reproduce the transform exactly
(anchored top-left like the encoder). These are the invariants the solver relies
on when it emits a primitive as a task's ONNX.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import onnx
import pytest

from evaluate import audit_one
from pipeline.case2 import onnx_ops as ops


def _score(model: onnx.ModelProto, examples: dict[str, Any] | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.onnx")
        onnx.save(model, path)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            result: dict[str, Any] = audit_one(
                path, examples, run_correctness=examples is not None
            )
            return result
        finally:
            os.chdir(cwd)


def _task(inp: list[list[int]], out: list[list[int]]) -> dict[str, Any]:
    return {"train": [{"input": inp, "output": out}], "test": [], "arc-gen": []}


GRID = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (ops.identity(), GRID),
        (ops.transpose(), [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
        (ops.flip_h(3), [[3, 2, 1], [6, 5, 4], [9, 8, 7]]),
        (ops.flip_v(3), [[7, 8, 9], [4, 5, 6], [1, 2, 3]]),
        (ops.rot180(3, 3), [[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
        (ops.rot90(3), [[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
        (ops.rot270(3), [[3, 6, 9], [2, 5, 8], [1, 4, 7]]),
    ],
)
def test_geometry_exact(model: onnx.ModelProto, expected: list[list[int]]) -> None:
    res = _score(model, _task(GRID, expected))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0
    assert res["n_pass"] == 1
    assert res["cost"] is not None


def test_recolor_exact() -> None:
    # swap colors 1<->2: perm[dst]=src, so perm[1]=2, perm[2]=1, else identity.
    perm = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
    inp = [[1, 2, 0], [2, 1, 0]]
    out = [[2, 1, 0], [1, 2, 0]]
    res = _score(ops.recolor(perm), _task(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_tile_exact() -> None:
    inp = [[1, 2], [3, 4]]
    out = (np.tile(np.array(inp), (2, 2))).tolist()
    res = _score(ops.tile(2, 2, 2, 2), _task(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_identity_is_zero_cost() -> None:
    res = _score(ops.identity(), None)
    assert res["cost"] == 0
    assert res["points"] == pytest.approx(25.0)


def test_subgrid_exact() -> None:
    # Crop [1:3, 1:3] from a 3x3 grid → bottom-right 2x2.
    inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = [[5, 6], [8, 9]]
    res = _score(ops.subgrid(1, 3, 1, 3), _task(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0
    assert res["n_pass"] == 1


def test_subgrid_top_right_exact() -> None:
    # Crop [0:2, 1:3] from a 3x3 grid → top-right 2x2.
    inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = [[2, 3], [5, 6]]
    res = _score(ops.subgrid(0, 2, 1, 3), _task(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0
    assert res["n_pass"] == 1
