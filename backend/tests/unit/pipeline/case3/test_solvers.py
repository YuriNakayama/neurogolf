"""case3 ソルバーが正しい ONNX を返し audit_one を通過することを検証する。"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import onnx

from evaluate import audit_one
from pipeline.case3 import solvers
from pipeline.case3.arc import Example, Task


def _task(inp: list[list[int]], out: list[list[int]]) -> Task:
    return Task(
        num=1,
        train=(Example(input=inp, output=out),),
        test=(),
        arc_gen=(),
    )


def _examples(inp: list[list[int]], out: list[list[int]]) -> dict[str, Any]:
    return {"train": [{"input": inp, "output": out}], "test": [], "arc-gen": []}


def _audit(task: Task, examples: dict[str, Any]) -> dict[str, Any]:
    for _name, fn in solvers.SOLVERS:
        model = fn(task)
        if model is not None:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "m.onnx")
                onnx.save(model, path)
                cwd = os.getcwd()
                os.chdir(td)
                try:
                    return audit_one(path, examples, run_correctness=True)
                finally:
                    os.chdir(cwd)
    return {"status": "no_solver"}


def _run_audit(model: onnx.ModelProto, examples: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.onnx")
        onnx.save(model, path)
        cwd = os.getcwd()
        os.chdir(td)
        try:
            return audit_one(path, examples, run_correctness=True)
        finally:
            os.chdir(cwd)


# ─── identity ──────────────────────────────────────────────────────────────


def test_solve_identity() -> None:
    g = [[1, 2], [3, 4]]
    result = _audit(_task(g, g), _examples(g, g))
    assert result["status"] == "ok"
    assert result["n_fail"] == 0


# ─── flip_v ────────────────────────────────────────────────────────────────


def test_solve_flip_v_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flipped = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
    model = solvers.solve_flip_v(_task(g, flipped))
    assert model is not None
    res = _run_audit(model, _examples(g, flipped))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_flip_v_varying_height_returns_none() -> None:
    """異なる高さの example が混在する場合は解けない。"""
    task = Task(
        num=1,
        train=(
            Example(input=[[1, 2], [3, 4]], output=[[3, 4], [1, 2]]),
            Example(
                input=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                output=[[7, 8, 9], [4, 5, 6], [1, 2, 3]],
            ),
        ),
        test=(),
        arc_gen=(),
    )
    assert solvers.solve_flip_v(task) is None


# ─── flip_h ────────────────────────────────────────────────────────────────


def test_solve_flip_h_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flipped = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    model = solvers.solve_flip_h(_task(g, flipped))
    assert model is not None
    res = _run_audit(model, _examples(g, flipped))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


# ─── rot180 ────────────────────────────────────────────────────────────────


def test_solve_rot180_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rot = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    model = solvers.solve_rot180(_task(g, rot))
    assert model is not None
    res = _run_audit(model, _examples(g, rot))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


# ─── rot90 ─────────────────────────────────────────────────────────────────


def test_solve_rot90_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rot = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    model = solvers.solve_rot90(_task(g, rot))
    assert model is not None
    res = _run_audit(model, _examples(g, rot))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_rot90_not_rot90_returns_none() -> None:
    g = [[1, 2], [3, 4]]
    wrong = [[2, 1], [4, 3]]  # flip_h, not rot90
    assert solvers.solve_rot90(_task(g, wrong)) is None


# ─── rot270 ────────────────────────────────────────────────────────────────


def test_solve_rot270_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rot = [[3, 6, 9], [2, 5, 8], [1, 4, 7]]
    model = solvers.solve_rot270(_task(g, rot))
    assert model is not None
    res = _run_audit(model, _examples(g, rot))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0
