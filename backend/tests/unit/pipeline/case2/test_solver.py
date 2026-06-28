"""The solver infers the right primitive from synthetic task JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.case2 import solver


def _write_task(tmp_path: Path, inp: list[list[int]], out: list[list[int]]) -> Path:
    task = {"train": [{"input": inp, "output": out}], "test": [], "arc-gen": []}
    p = tmp_path / "task001.json"
    p.write_text(json.dumps(task))
    return p


def test_solve_identity(tmp_path: Path) -> None:
    g = [[1, 2], [3, 4]]
    sol = solver.solve(_write_task(tmp_path, g, g))
    assert sol is not None
    assert sol.name == "identity"


def test_solve_transpose(tmp_path: Path) -> None:
    sol = solver.solve(
        _write_task(tmp_path, [[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]])
    )
    assert sol is not None
    assert sol.name == "transpose"


def test_solve_flip_h(tmp_path: Path) -> None:
    # Asymmetric placement so no color permutation can mimic the mirror.
    inp = [[1, 1, 2], [3, 0, 0]]
    out = np.array(inp)[:, ::-1].tolist()
    sol = solver.solve(_write_task(tmp_path, inp, out))
    assert sol is not None
    assert sol.name == "flip_h"


def test_solve_recolor(tmp_path: Path) -> None:
    # Position-preserving color change that no geometry reproduces.
    inp = [[1, 2, 1], [2, 1, 2]]
    out = [[3, 4, 3], [4, 3, 4]]
    sol = solver.solve(_write_task(tmp_path, inp, out))
    assert sol is not None
    assert sol.name == "recolor"


def test_solve_unknown_returns_none(tmp_path: Path) -> None:
    # Same input color 1 maps to different outputs by position -> not a recolor,
    # and the shapes/contents match no geometry. No primitive should fit.
    sol = solver.solve(_write_task(tmp_path, [[1, 1], [1, 1]], [[2, 3], [4, 5]]))
    assert sol is None


def test_solve_subgrid_top_right(tmp_path: Path) -> None:
    # Extract top-right 2x2 from a 3x3: crop [0:2, 1:3].
    inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = [[2, 3], [5, 6]]
    sol = solver.solve(_write_task(tmp_path, inp, out))
    assert sol is not None
    assert sol.name == "subgrid"


def test_solve_subgrid_bottom_right(tmp_path: Path) -> None:
    # Extract bottom-right 2x2 from a 3x3: crop [1:3, 1:3].
    inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = [[5, 6], [8, 9]]
    sol = solver.solve(_write_task(tmp_path, inp, out))
    assert sol is not None
    assert sol.name == "subgrid"
