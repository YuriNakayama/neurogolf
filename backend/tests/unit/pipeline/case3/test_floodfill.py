"""build_border_bicolor_flood のユニットテスト（TDD: RED → GREEN）。"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort

from pipeline.case3.arc import Example, encode_grid
from pipeline.case3.floodfill import build_border_bicolor_flood


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
