"""case3 ビルダーが任意サイズの ARC グリッドで正しく動作することを検証する。

既知のバグ: Slice ベースの build_flip は全 30 要素を反転するため、
30×30 未満グリッドでは content が下端にずれる。
修正後は Gather ベースの次元対応反転を使う。
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from pipeline.case3 import builders as B
from pipeline.case3.arc import encode_grid


def _run(model: onnx.ModelProto, grid: list[list[int]]) -> np.ndarray:
    """model を onnxruntime で実行し [1,10,30,30] 出力を返す。"""
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inp = encode_grid(grid)
    (out,) = sess.run(None, {"input": inp})
    return np.asarray(out)


def _argmax_grid(out: np.ndarray, h: int, w: int) -> list[list[int]]:
    """[1,10,30,30] → h×w の色グリッド（argmax on channel axis）。"""
    arr = out[0]  # [10,30,30]
    grid: list[list[int]] = []
    for r in range(h):
        grid.append([int(np.argmax(arr[:, r, c])) for c in range(w)])
    return grid


# ─── identity ──────────────────────────────────────────────────────────────


def test_identity_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = _run(B.build_identity(), g)
    assert _argmax_grid(out, 3, 3) == g


# ─── flip_v (axis=2): 上下反転 ─────────────────────────────────────────────


def test_flip_v_3x3() -> None:
    """3×3 グリッドの縦反転。content が top-left に収まること。"""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
    out = _run(B.build_flip(3, 2), g)
    assert _argmax_grid(out, 3, 3) == expected


def test_flip_v_2x3() -> None:
    """2×3 グリッド（非正方形）の縦反転。"""
    g = [[1, 2, 3], [4, 5, 6]]
    expected = [[4, 5, 6], [1, 2, 3]]
    out = _run(B.build_flip(2, 2), g)
    assert _argmax_grid(out, 2, 3) == expected


def test_flip_v_padding_stays_zero() -> None:
    """30×30 テンソルのうち、使用外領域 (row >= h) は変化しないこと。"""
    g = [[1, 2], [3, 4]]
    out = _run(B.build_flip(2, 2), g)
    # row 2 以降（padding 領域）はすべてゼロ
    assert np.all(out[0, :, 2:, :] == 0)


# ─── flip_h (axis=3): 左右反転 ─────────────────────────────────────────────


def test_flip_h_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    out = _run(B.build_flip(3, 3), g)
    assert _argmax_grid(out, 3, 3) == expected


def test_flip_h_3x2() -> None:
    g = [[1, 2], [3, 4], [5, 6]]
    expected = [[2, 1], [4, 3], [6, 5]]
    out = _run(B.build_flip(2, 3), g)
    assert _argmax_grid(out, 3, 2) == expected


# ─── rot180 ────────────────────────────────────────────────────────────────


def test_rot180_3x3() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    out = _run(B.build_rot180(3, 3), g)
    assert _argmax_grid(out, 3, 3) == expected


def test_rot180_2x3() -> None:
    g = [[1, 2, 3], [4, 5, 6]]
    expected = [[6, 5, 4], [3, 2, 1]]
    out = _run(B.build_rot180(2, 3), g)
    assert _argmax_grid(out, 2, 3) == expected


# ─── rot90 / rot270 ────────────────────────────────────────────────────────


def test_rot90_3x3() -> None:
    """rot90 CW: [[1,2,3],[4,5,6],[7,8,9]] -> [[7,4,1],[8,5,2],[9,6,3]]"""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    out = _run(B.build_rot90(3), g)
    assert _argmax_grid(out, 3, 3) == expected


def test_rot90_2x3() -> None:
    """rot90 CW: 2×3 -> 3×2"""
    g = [[1, 2, 3], [4, 5, 6]]
    expected = [[4, 1], [5, 2], [6, 3]]
    out = _run(B.build_rot90(2), g)
    assert _argmax_grid(out, 3, 2) == expected


def test_rot270_3x3() -> None:
    """rot270 CCW: [[1,2,3],[4,5,6],[7,8,9]] -> [[3,6,9],[2,5,8],[1,4,7]]"""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[3, 6, 9], [2, 5, 8], [1, 4, 7]]
    out = _run(B.build_rot270(3), g)
    assert _argmax_grid(out, 3, 3) == expected


# ─── onnx.checker pass ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        B.build_identity(),
        B.build_flip(3, 2),
        B.build_flip(3, 3),
        B.build_rot180(3, 3),
        B.build_rot90(3),
        B.build_rot270(3),
        B.build_tile(2, 2, 4, 4),
        B.build_tile(3, 3, 7, 7),
        B.build_tile(1, 3, 1, 6),
    ],
)
def test_onnx_checker_passes(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)


# ─── tile ──────────────────────────────────────────────────────────────────


def test_tile_2x2_to_4x4() -> None:
    """2×2 → 4×4 exact tile（縦横両方を 2 回繰り返す）。"""
    g = [[1, 2], [3, 4]]
    expected = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    out = _run(B.build_tile(2, 2, 4, 4), g)
    assert _argmax_grid(out, 4, 4) == expected


def test_tile_1x3_to_1x6() -> None:
    """1×3 → 1×6 横方向のみのタイリング。"""
    g = [[1, 2, 3]]
    expected = [[1, 2, 3, 1, 2, 3]]
    out = _run(B.build_tile(1, 3, 1, 6), g)
    assert _argmax_grid(out, 1, 6) == expected


def test_tile_3x1_to_6x1() -> None:
    """3×1 → 6×1 縦方向のみのタイリング。"""
    g = [[1], [2], [3]]
    expected = [[1], [2], [3], [1], [2], [3]]
    out = _run(B.build_tile(3, 1, 6, 1), g)
    assert _argmax_grid(out, 6, 1) == expected


def test_tile_non_multiple_3x3_to_7x7() -> None:
    """非整数倍タイル: 3×3 → 7×7 (ceil(7/3)=3, tiled=9×9 を 7×7 に切り出す)。"""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[g[r % 3][c % 3] for c in range(7)] for r in range(7)]
    out = _run(B.build_tile(3, 3, 7, 7), g)
    assert _argmax_grid(out, 7, 7) == expected


def test_tile_padding_stays_zero() -> None:
    """アクティブ領域外（row >= oh）はゼロのままであること。"""
    g = [[1, 2], [3, 4]]
    out = _run(B.build_tile(2, 2, 4, 4), g)
    assert np.all(out[0, :, 4:, :] == 0)
