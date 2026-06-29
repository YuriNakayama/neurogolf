"""LP-based linear separability solver のテスト。

linsep.py: k-local かつ各チャネルが線形分離可能なタスクを
単一 Conv[10,10,k,k] ONNX で解く。
cost = 100*k*k + 10 (k=3→910, k=5→2510, k=7→4910), memory=0。
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from evaluate import audit_one
from pipeline.case3.arc import Example
from pipeline.case3.linsep import build_linsep_conv, try_linsep_conv

# ─── helpers ──────────────────────────────────────────────────────────────────


def _ex(inp: list[list[int]], out: list[list[int]]) -> list[Example]:
    return [Example(input=inp, output=out)]


def _examples_dict(inp: list[list[int]], out: list[list[int]]) -> dict[str, Any]:
    return {"train": [{"input": inp, "output": out}], "test": [], "arc-gen": []}


def _audit(
    model: onnx.ModelProto, inp: list[list[int]], out: list[list[int]]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.onnx")
        onnx.save(model, path)
        cwd = os.getcwd()
        os.chdir(td)
        try:
            result: dict[str, Any] = audit_one(
                path, _examples_dict(inp, out), run_correctness=True
            )
            return result
        finally:
            os.chdir(cwd)


def _run(model: onnx.ModelProto, inp: list[list[int]]) -> np.ndarray:
    """model を onnxruntime で実行し [1,10,30,30] を返す。"""
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    from pipeline.case3.arc import encode_grid

    (out,) = sess.run(None, {"input": encode_grid(inp)})
    return np.asarray(out)


# ─── k=1: recolor (center cell only) ─────────────────────────────────────────


def test_k3_row_shift_returns_model() -> None:
    """row shift (out[r][c] = in[r-1][c]) は k=3 局所で線形分離可能。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is not None


def test_k3_row_shift_onnx_checker() -> None:
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is not None
    onnx.checker.check_model(model, full_check=True)


def test_k3_row_shift_correct_output() -> None:
    """生成モデルが全 example を正答すること。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is not None
    raw = _run(model, inp)
    # out > 0 で二値化してデコード
    for r, row in enumerate(out):
        for c, color in enumerate(row):
            assert raw[0, color, r, c] > 0.0, f"({r},{c}): expected ch {color} > 0"
            for ch in range(10):
                if ch != color:
                    assert raw[0, ch, r, c] <= 0.0, f"({r},{c}): ch {ch} should be <=0"


def test_k3_row_shift_audit_passes() -> None:
    """audit_one (faithful scorer) が n_fail=0 を返すこと。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is not None
    res = _audit(model, inp, out)
    assert res["n_fail"] == 0
    assert res["status"] == "ok"


def test_k3_row_shift_cost() -> None:
    """k=3 の cost = params + memory = 910 + 0 = 910。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is not None
    res = _audit(model, inp, out)
    assert res["cost"] == 910


# ─── ONNX 構造 ─────────────────────────────────────────────────────────────────


def test_linsep_onnx_structure_k3() -> None:
    """ONNX グラフが単一 Conv ノード + W[10,10,3,3] + B[10] であること。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is not None
    graph = model.graph
    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    assert len(conv_nodes) == 1
    assert len(graph.node) == 1
    # initializer shapes
    inits = {i.name: list(i.dims) for i in graph.initializer}
    assert "W" in inits and inits["W"] == [10, 10, 3, 3]
    assert "B" in inits and inits["B"] == [10]


def test_linsep_onnx_structure_k5() -> None:
    """k=5 用: W[10,10,5,5]。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    # k=5 も row-shift なので線形分離可能
    model = build_linsep_conv(_ex(inp, out), 5)
    assert model is not None
    inits = {i.name: list(i.dims) for i in model.graph.initializer}
    assert inits["W"] == [10, 10, 5, 5]


# ─── try_linsep_conv ──────────────────────────────────────────────────────────


def test_try_linsep_conv_returns_arrays() -> None:
    """try_linsep_conv が (W, B) を返し形状が正しいこと。"""
    inp = [[1, 2], [3, 4], [5, 6]]
    out = [[0, 0], [1, 2], [3, 4]]
    result = try_linsep_conv(_ex(inp, out), 3)
    assert result is not None
    W, B = result
    assert W.shape == (10, 10, 3, 3)
    assert B.shape == (10,)


# ─── 非線形分離タスク → None ──────────────────────────────────────────────────


def test_non_klocal_returns_none() -> None:
    """全零入力で異なる位置が異なる出力色 → 同一 window, 異ラベル → LP 不実行 → None。"""
    inp = [[0] * 5 for _ in range(5)]
    out = [[0] * 5 for _ in range(5)]
    out[1][1] = 1
    out[2][2] = 2
    # positions (1,1) and (2,2) share the same k=3 zero window but have different labels
    model = build_linsep_conv(_ex(inp, out), 3)
    assert model is None


# ─── 複数 example での検証 ────────────────────────────────────────────────────


def test_two_examples_consistent() -> None:
    """複数 example が全て k=3 局所的に一致すれば正常に解けること。"""
    inp1 = [[1, 2], [3, 4], [5, 6]]
    out1 = [[0, 0], [1, 2], [3, 4]]
    inp2 = [[2, 3], [4, 5], [6, 7]]
    out2 = [[0, 0], [2, 3], [4, 5]]
    examples = [Example(input=inp1, output=out1), Example(input=inp2, output=out2)]
    model = build_linsep_conv(examples, 3)
    assert model is not None
    onnx.checker.check_model(model, full_check=True)


# ─── recolor タスク: k=3 でも解けること ──────────────────────────────────────


@pytest.mark.parametrize("k", [3, 5])
def test_recolor_solvable(k: int) -> None:
    """色置換は k ≥ 1 で線形分離可能。linsep_conv でも正答を返す。"""
    inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = [[2, 3, 4], [5, 6, 7], [8, 9, 0]]  # shift by 1 (mod 10)
    model = build_linsep_conv(_ex(inp, out), k)
    assert model is not None
    res = _audit(model, inp, out)
    assert res["n_fail"] == 0
