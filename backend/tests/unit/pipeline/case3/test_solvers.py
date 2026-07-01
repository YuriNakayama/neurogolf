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


# ─── floodfill (8-connectivity) ────────────────────────────────────────────


def test_solve_floodfill_simple_enclosed() -> None:
    """5×5: 1 が囲む内側の 0 を 2 で塗りつぶす。"""
    inp = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    out = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    model = solvers.solve_floodfill(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_floodfill_larger_enclosed() -> None:
    """7×7: 線色=3, 塗り色=5 の囲み領域。"""
    inp = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 3, 3, 0],
        [0, 3, 0, 0, 0, 3, 0],
        [0, 3, 0, 0, 0, 3, 0],
        [0, 3, 0, 0, 0, 3, 0],
        [0, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    out = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 3, 3, 0],
        [0, 3, 5, 5, 5, 3, 0],
        [0, 3, 5, 5, 5, 3, 0],
        [0, 3, 5, 5, 5, 3, 0],
        [0, 3, 3, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    model = solvers.solve_floodfill(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_floodfill_not_applicable_returns_none() -> None:
    """非 flood-fill タスクは None を返す。"""
    g = [[1, 2], [3, 4]]
    assert solvers.solve_floodfill(_task(g, g)) is None


def test_solve_floodfill_recolor_not_applicable() -> None:
    """単純 recolor (全体の色変換) は flood-fill ではない。"""
    inp = [[1, 1], [1, 1]]
    out = [[2, 2], [2, 2]]
    assert solvers.solve_floodfill(_task(inp, out)) is None


def test_solve_floodfill_cost_below_threshold() -> None:
    """生成コスト < 10000 であること (基本的な小グリッドの上限チェック)。"""
    inp = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    out = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    model = solvers.solve_floodfill(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["cost"] is not None
    assert res["cost"] < 10000


# ─── panels ────────────────────────────────────────────────────────────────
# Input: 左右(LR)または上下(TB)の 2 パネルを論理演算で合成するタスク。


def _lr_task(
    left: list[list[int]], right: list[list[int]], out: list[list[int]]
) -> Task:
    """左右 2 パネル（LR レイアウト）の入力グリッドを構築。"""
    oh = len(left)
    inp = [left[r] + right[r] for r in range(oh)]
    return Task(
        num=1,
        train=(Example(input=inp, output=out),),
        test=(),
        arc_gen=(),
    )


def _lr_examples(
    left: list[list[int]], right: list[list[int]], out: list[list[int]]
) -> dict[str, Any]:
    oh = len(left)
    inp = [left[r] + right[r] for r in range(oh)]
    return _examples(inp, out)


def test_solve_panels_lr_or() -> None:
    """LR レイアウト OR: 非背景セルの論理和を 1 色で出力。"""
    left = [[1, 0], [0, 0]]
    right = [[0, 0], [0, 2]]
    out = [[3, 0], [0, 3]]
    task = _lr_task(left, right, out)
    model = solvers.solve_panels(task)
    assert model is not None
    res = _run_audit(model, _lr_examples(left, right, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_panels_lr_and() -> None:
    """LR レイアウト AND: 両パネルが非背景な交差セルだけを出力。"""
    left = [[1, 1], [0, 1]]
    right = [[1, 0], [0, 1]]
    out = [[4, 0], [0, 4]]
    task = _lr_task(left, right, out)
    model = solvers.solve_panels(task)
    assert model is not None
    res = _run_audit(model, _lr_examples(left, right, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_panels_lr_xor() -> None:
    """LR レイアウト XOR: 片方のみ非背景なセルを出力。"""
    left = [[1, 0], [1, 0]]
    right = [[1, 1], [0, 0]]
    out = [[0, 5], [5, 0]]
    task = _lr_task(left, right, out)
    model = solvers.solve_panels(task)
    assert model is not None
    res = _run_audit(model, _lr_examples(left, right, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_panels_lr_diff() -> None:
    """LR レイアウト DIFF: 左パネルにあり右パネルにないセルを出力。"""
    left = [[1, 1], [0, 1]]
    right = [[1, 0], [0, 0]]
    out = [[0, 6], [0, 6]]
    task = _lr_task(left, right, out)
    model = solvers.solve_panels(task)
    assert model is not None
    res = _run_audit(model, _lr_examples(left, right, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_panels_tb_or() -> None:
    """TB レイアウト OR: 上下 2 パネルの論理和。"""
    top = [[1, 0], [0, 0]]
    bot = [[0, 0], [0, 1]]
    out = [[7, 0], [0, 7]]
    inp = top + bot
    task = Task(
        num=1,
        train=(Example(input=inp, output=out),),
        test=(),
        arc_gen=(),
    )
    model = solvers.solve_panels(task)
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_panels_not_applicable_returns_none() -> None:
    """パネル合成でないタスクは None を返す。"""
    g = [[1, 2], [3, 4]]
    assert solvers.solve_panels(_task(g, g)) is None


def test_solve_panels_in_solvers_list() -> None:
    """solve_panels が SOLVERS リストに含まれていることを確認。"""
    names = [name for name, _ in solvers.SOLVERS]
    assert "panels" in names


# ─── scale_up_rows ─────────────────────────────────────────────────────────


def test_solve_scale_up_rows_k2() -> None:
    """行方向 2× スケールタスクを正しく解く。"""
    inp = [[1, 2], [3, 4]]
    out = [[1, 2], [1, 2], [3, 4], [3, 4]]
    model = solvers.solve_scale_up_rows(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_scale_up_rows_returns_none_for_same_shape() -> None:
    """同形状（スケールなし）には None を返す。"""
    g = [[1, 2], [3, 4]]
    assert solvers.solve_scale_up_rows(_task(g, g)) is None


def test_solve_scale_up_rows_returns_none_when_cols_also_scale() -> None:
    """行も列も変化（2D スケール）では None を返す。"""
    inp = [[1, 2], [3, 4]]
    out_2d = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    assert solvers.solve_scale_up_rows(_task(inp, out_2d)) is None


# ─── scale_up_cols ─────────────────────────────────────────────────────────


def test_solve_scale_up_cols_k2() -> None:
    """列方向 2× スケールタスクを正しく解く。"""
    inp = [[1, 2], [3, 4]]
    out = [[1, 1, 2, 2], [3, 3, 4, 4]]
    model = solvers.solve_scale_up_cols(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_scale_up_cols_returns_none_for_same_shape() -> None:
    """同形状には None を返す。"""
    g = [[1, 2], [3, 4]]
    assert solvers.solve_scale_up_cols(_task(g, g)) is None


def test_solve_scale_up_cols_returns_none_when_rows_also_scale() -> None:
    """行も列も変化（2D スケール）では None を返す。"""
    inp = [[1, 2], [3, 4]]
    out_2d = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    assert solvers.solve_scale_up_cols(_task(inp, out_2d)) is None


# ─── scale_up_2d ───────────────────────────────────────────────────────────


def test_solve_scale_up_2d_k2() -> None:
    """2D 2× スケールタスクを正しく解く。"""
    inp = [[1, 2], [3, 4]]
    out = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    model = solvers.solve_scale_up_2d(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_scale_up_2d_k3() -> None:
    """2D 3× スケールタスクを正しく解く。"""
    inp = [[1, 2], [3, 4]]
    out = [
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
    ]
    model = solvers.solve_scale_up_2d(_task(inp, out))
    assert model is not None
    res = _run_audit(model, _examples(inp, out))
    assert res["status"] == "ok"
    assert res["n_fail"] == 0


def test_solve_scale_up_2d_returns_none_for_same_shape() -> None:
    """同形状には None を返す。"""
    g = [[1, 2], [3, 4]]
    assert solvers.solve_scale_up_2d(_task(g, g)) is None


def test_solve_scale_up_2d_returns_none_for_rows_only_scale() -> None:
    """行のみスケール（列不変）は 2D ソルバで None（k_h != k_w の検出対象外）。"""
    inp = [[1, 2], [3, 4]]
    out = [[1, 2], [1, 2], [3, 4], [3, 4]]  # 行のみ 2×
    assert solvers.solve_scale_up_2d(_task(inp, out)) is None


def test_solve_scale_up_all_in_solvers_list() -> None:
    """3 つのスケールソルバが SOLVERS リストに含まれる。"""
    names = [name for name, _ in solvers.SOLVERS]
    assert "scale_up_rows" in names
    assert "scale_up_cols" in names
    assert "scale_up_2d" in names
