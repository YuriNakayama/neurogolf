"""floodfill ユニットテスト（TDD: RED → GREEN）。"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort

from pipeline.case3.arc import Example, encode_grid
from pipeline.case3.floodfill import (
    build_4conn_oob_safe_flood,
    build_border_bicolor_flood,
)


def _run_bicolor(model: onnx.ModelProto, grid: list[list[int]]) -> list[list[int]]:
    """ONNX モデルを実行し、channel argmax でカラーグリッドを返す。"""
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inp = encode_grid(grid)
    (out,) = sess.run(None, {"input": inp})
    arr = np.asarray(out, dtype=np.float32)[0]  # [10, 30, 30]
    h, w = len(grid), len(grid[0])
    return [[int(np.argmax(arr[:, r, c])) for c in range(w)] for r in range(h)]


def _ex(inp: list[list[int]], out: list[list[int]]) -> Example:
    return Example(input=inp, output=out)


# ─── 検出ロジック ─────────────────────────────────────────────────────────────


def test_detect_rejects_wrong_fill() -> None:
    """0 → 4 など規定外の色変化があれば None を返す。"""
    inp = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    out = [[1, 1, 1], [1, 4, 1], [1, 1, 1]]
    assert build_border_bicolor_flood((_ex(inp, out),)) is None


def test_detect_rejects_wall_change() -> None:
    """壁色が変化する例は None。"""
    inp = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    out = [[2, 1, 1], [1, 2, 1], [1, 1, 1]]
    assert build_border_bicolor_flood((_ex(inp, out),)) is None


# ─── 出力正確性 ──────────────────────────────────────────────────────────────


def test_enclosed_3x3() -> None:
    """中央セルが壁で囲まれ → 2 に。"""
    inp = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    out_expected = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
    model = build_border_bicolor_flood((_ex(inp, out_expected),))
    assert model is not None
    assert _run_bicolor(model, inp) == out_expected


def test_border_reach_3x3() -> None:
    """境界に接する 0 セルが → 3 に。"""
    inp = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    out_expected = [[1, 1, 1], [3, 3, 3], [1, 1, 1]]
    model = build_border_bicolor_flood((_ex(inp, out_expected),))
    assert model is not None
    assert _run_bicolor(model, inp) == out_expected


def test_both_colors_5x5() -> None:
    """外側 ring=3 (border-reach)、内側 center=2 (enclosed)。"""
    inp = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    out_expected = [
        [3, 3, 3, 3, 3],
        [3, 1, 1, 1, 3],
        [3, 1, 2, 1, 3],
        [3, 1, 1, 1, 3],
        [3, 3, 3, 3, 3],
    ]
    model = build_border_bicolor_flood((_ex(inp, out_expected),))
    assert model is not None
    assert _run_bicolor(model, inp) == out_expected


def test_onnx_checker_passes() -> None:
    """生成モデルが onnx.checker を通過する。"""
    inp = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    out_expected = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
    model = build_border_bicolor_flood((_ex(inp, out_expected),))
    assert model is not None
    onnx.checker.check_model(model, full_check=True)


# ─── build_4conn_oob_safe_flood ───────────────────────────────────────────────


def _run_4conn(model: onnx.ModelProto, grid: list[list[int]]) -> list[list[int]]:
    """ONNX モデルを実行し channel argmax でカラーグリッドを返す。"""
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inp = encode_grid(grid)
    (out,) = sess.run(None, {"input": inp})
    arr = np.asarray(out, dtype=np.float32)[0]  # [10, 30, 30]
    h, w = len(grid), len(grid[0])
    return [[int(np.argmax(arr[:, r, c])) for c in range(w)] for r in range(h)]


def test_4conn_rejects_non_floodfill() -> None:
    """flood-fill 規則に合わない task は None を返す。"""
    inp = [[3, 0, 3], [0, 3, 0], [3, 0, 3]]
    out = [[3, 4, 3], [4, 3, 4], [3, 4, 3]]  # 0→4 だが enclosed でない
    assert build_4conn_oob_safe_flood((_ex(inp, out),)) is None


def test_4conn_enclosed_3x3() -> None:
    """中央セルが壁で完全に囲まれて fill_color になる。"""
    inp = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
    out = [[3, 3, 3], [3, 4, 3], [3, 3, 3]]
    model = build_4conn_oob_safe_flood((_ex(inp, out),))
    assert model is not None
    assert _run_4conn(model, inp) == out


def test_4conn_border_bg_unchanged() -> None:
    """境界に接する bg セルは fill されない（border-reachable）。"""
    # 内部の 3×3 だけ壁に囲まれている（外枠は bg）
    inp = [
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0],
    ]
    out = [
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 4, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0],
    ]
    model = build_4conn_oob_safe_flood((_ex(inp, out),))
    assert model is not None
    assert _run_4conn(model, inp) == out


def test_4conn_oob_no_leak() -> None:
    """OOB セル経由の誤 reach が起きないことを確認。

    max_dim=5 の working space で 3×3 と 5×5 の 2 例セット。
    3×3 例では rows 3-4, cols 3-4 が OOB (全チャネル 0) になる。
    8-conn 版はこれを free 扱いして侵食するが、
    4-conn OOB-safe 版は free_u8=ch0 により OOB をブロックし正答を返す。
    """
    # 5×5 example: outer ring is wall, inner 3×3 is all bg (→ all enclosed, fill=4)
    big_inp = [
        [3, 3, 3, 3, 3],
        [3, 0, 0, 0, 3],
        [3, 0, 0, 0, 3],
        [3, 0, 0, 0, 3],
        [3, 3, 3, 3, 3],
    ]
    big_out = [
        [3, 3, 3, 3, 3],
        [3, 4, 4, 4, 3],
        [3, 4, 4, 4, 3],
        [3, 4, 4, 4, 3],
        [3, 3, 3, 3, 3],
    ]
    # 3×3 example: center enclosed (OOB at rows 3-4, cols 3-4 in 5×5 working space)
    small_inp = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
    small_out = [[3, 3, 3], [3, 4, 3], [3, 3, 3]]

    examples = (_ex(big_inp, big_out), _ex(small_inp, small_out))
    model = build_4conn_oob_safe_flood(examples)
    assert model is not None
    assert _run_4conn(model, small_inp) == small_out
    assert _run_4conn(model, big_inp) == big_out


def test_4conn_onnx_checker() -> None:
    """生成モデルが onnx.checker を通過する。"""
    inp = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
    out = [[3, 3, 3], [3, 4, 3], [3, 3, 3]]
    model = build_4conn_oob_safe_flood((_ex(inp, out),))
    assert model is not None
    onnx.checker.check_model(model, full_check=True)
