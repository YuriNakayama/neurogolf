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


# ─── anti_transpose ────────────────────────────────────────────────────────


def test_anti_transpose_3x3() -> None:
    """anti_transpose[r][c] = g[H-1-c][W-1-r]。"""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [[9, 6, 3], [8, 5, 2], [7, 4, 1]]
    out = _run(B.build_anti_transpose(3, 3), g)
    assert _argmax_grid(out, 3, 3) == expected


def test_anti_transpose_2x3() -> None:
    """2×3 非正方形の anti_transpose → 3×2 出力。"""
    g = [[1, 2, 3], [4, 5, 6]]
    expected = [[6, 3], [5, 2], [4, 1]]
    out = _run(B.build_anti_transpose(2, 3), g)
    assert _argmax_grid(out, 3, 2) == expected


# ─── scale ──────────────────────────────────────────────────────────────────


def test_scale_2x2_by_2x2() -> None:
    """2×2 グリッドを 2×2 倍に拡大（各セルが 2×2 ブロックになる）。"""
    g = [[1, 2], [3, 4]]
    expected = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    out = _run(B.build_scale(2, 2, 2, 2), g)
    assert _argmax_grid(out, 4, 4) == expected


def test_scale_3x2_by_1x3() -> None:
    """3×2 グリッドを列方向のみ 3 倍（各行を 3 回繰り返す）。"""
    g = [[1, 2], [3, 4], [5, 6]]
    expected = [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4], [5, 5, 5, 6, 6, 6]]
    out = _run(B.build_scale(3, 2, 1, 3), g)
    assert _argmax_grid(out, 3, 6) == expected


def test_scale_padding_stays_zero() -> None:
    """拡大後の 30×30 テンソルで使用外領域はゼロのまま。"""
    g = [[1, 2], [3, 4]]
    out = _run(B.build_scale(2, 2, 2, 2), g)
    assert np.all(out[0, :, 4:, :] == 0)


# ─── tile ───────────────────────────────────────────────────────────────────


def test_tile_2x2_by_2x2() -> None:
    """2×2 グリッドを 2×2 回繰り返す。"""
    g = [[1, 2], [3, 4]]
    expected = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    out = _run(B.build_tile(2, 2, 2, 2), g)
    assert _argmax_grid(out, 4, 4) == expected


def test_tile_2x3_by_1x2() -> None:
    """2×3 グリッドを列方向のみ 2 回繰り返す。"""
    g = [[1, 2, 3], [4, 5, 6]]
    expected = [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]
    out = _run(B.build_tile(2, 3, 1, 2), g)
    assert _argmax_grid(out, 2, 6) == expected


# ─── mosaic ─────────────────────────────────────────────────────────────────


def test_mosaic_2x2_by_2x2() -> None:
    """2×2 を 2×2 mirror-tile: 隣接タイルが反転して辺を共有する。"""
    g = [[1, 2], [3, 4]]
    expected = [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3], [1, 2, 2, 1]]
    out = _run(B.build_mosaic(2, 2, 2, 2), g)
    assert _argmax_grid(out, 4, 4) == expected


def test_mosaic_2x3_by_2x1() -> None:
    """2×3 を行方向のみ 2 回 mirror-tile（上下反転で折り返す）。"""
    g = [[1, 2, 3], [4, 5, 6]]
    expected = [[1, 2, 3], [4, 5, 6], [4, 5, 6], [1, 2, 3]]
    out = _run(B.build_mosaic(2, 3, 2, 1), g)
    assert _argmax_grid(out, 4, 3) == expected


# ─── keep_color ─────────────────────────────────────────────────────────────


def test_keep_color_retains_target() -> None:
    """keep_color(2): 色 2 だけ残り、他はすべて 0 になる。"""
    g = [[1, 2, 3], [2, 2, 2], [3, 1, 0]]
    expected = [[0, 2, 0], [2, 2, 2], [0, 0, 0]]
    out = _run(B.build_keep_color(2), g)
    assert _argmax_grid(out, 3, 3) == expected


def test_keep_color_background_preserved() -> None:
    """背景（色 0）は常に保持される。"""
    g = [[0, 1], [2, 0]]
    expected = [[0, 0], [2, 0]]
    out = _run(B.build_keep_color(2), g)
    assert _argmax_grid(out, 2, 2) == expected


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
        B.build_anti_transpose(3, 3),
        B.build_scale(2, 2, 2, 2),
        B.build_tile(2, 2, 2, 2),
        B.build_mosaic(2, 2, 2, 2),
        B.build_keep_color(3),
    ],
)
def test_onnx_checker_passes(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
