"""Golf-minimal ONNX graph builders for case3 (self-contained).

スコア ``cost = params + memory``。memory は **中間テンソル**（input/output を除く
node 出力）の要素数 × itemsize の総和（公式 ``calculate_memory``）。よって

* 中間テンソルを増やさない / 小さくする
* params（initializer 要素数）を増やさない

のが golf の本質。各 builder は I/O ``[1,10,30,30]`` float32・IR10/opset10 を守る。
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

NUM_COLORS = 10
GRID_MAX = 30
GRID_SHAPE = [1, NUM_COLORS, GRID_MAX, GRID_MAX]
IR_VERSION = 10
OPSET = [helper.make_opsetid("", 10)]
_DTYPE = TensorProto.FLOAT


def _model(
    nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", _DTYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DTYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "g", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)


def build_identity() -> onnx.ModelProto:
    """output = input。中間テンソル 0・params 0 → cost 0（25 点）。"""
    node = helper.make_node("Identity", ["input"], ["output"])
    return _model([node], [])


def build_recolor(mapping: dict[int, int]) -> onnx.ModelProto:
    """色 c -> mapping[c] の置換を 1x1 Conv で表現。

    重み W[o,i,0,0] = 1 if mapping[i]==o else 0。出力は input/output 名のみで
    中間テンソルなし → memory は W のみに依存（params=NUM_COLORS^2=100）。
    """
    w = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
    for src, dst in mapping.items():
        w[dst, src, 0, 0] = 1.0
    w_init = helper.make_tensor(
        "W", _DTYPE, [NUM_COLORS, NUM_COLORS, 1, 1], w.flatten()
    )
    node = helper.make_node(
        "Conv", ["input", "W"], ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
    )
    return _model([node], [w_init])


def build_recolor_gather(mapping: dict[int, int]) -> onnx.ModelProto:
    """色置換を Gather(axis=1) で表現。indices[o] = 元チャネル（params=10）。

    output[:, o] = input[:, indices[o]]。各出力色 o に対し「o に写る元色」を 1 つ選ぶ
    必要があるが、recolor は単射とは限らない（複数色が 1 色に潰れる）。Gather は
    各出力チャネルに 1 つの入力チャネルしか引けないので、**単射な mapping**（逆引き
    可能）のときのみ使える。それ以外は Conv 版を使う。
    """
    inv: dict[int, int] = {}
    for src, dst in mapping.items():
        inv[dst] = src  # 単射前提（呼び出し側が保証）
    indices = [inv.get(o, o) for o in range(NUM_COLORS)]
    idx = helper.make_tensor("idx", TensorProto.INT64, [NUM_COLORS], indices)
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)
    return _model([node], [idx])


def build_permute_axes(perm: tuple[int, int, int, int]) -> onnx.ModelProto:
    """空間軸の Transpose（例: 転置・軸入替）。perm は 4 軸の並び。"""
    node = helper.make_node("Transpose", ["input"], ["output"], perm=list(perm))
    return _model([node], [])


def _rev_idx(size: int) -> list[int]:
    """``0..size-1`` を逆順にし、``size..29`` はそのまま残すインデックス。"""
    return list(range(size - 1, -1, -1)) + list(range(size, GRID_MAX))


def build_flip(size: int, axis: int) -> onnx.ModelProto:
    """Gather で先頭 ``size`` 要素だけ逆順にする（残りは padding でゼロ維持）。

    Slice 版と異なり次元依存の Gather を使うため、任意サイズの ARC グリッドで
    content が top-left に留まる。params = 30（index initializer 1 本）。
    """
    idx = _rev_idx(size)
    init = helper.make_tensor("idx", TensorProto.INT64, [GRID_MAX], idx)
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=axis)
    return _model([node], [init])


def build_rot180(h: int, w: int) -> onnx.ModelProto:
    """h×w グリッドを 180° 回転（縦・横それぞれ Gather で逆順）。"""
    hidx = _rev_idx(h)
    widx = _rev_idx(w)
    h_init = helper.make_tensor("hidx", TensorProto.INT64, [GRID_MAX], hidx)
    w_init = helper.make_tensor("widx", TensorProto.INT64, [GRID_MAX], widx)
    g1 = helper.make_node("Gather", ["input", "hidx"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "widx"], ["output"], axis=3)
    return _model([g1, g2], [h_init, w_init])


def build_rot90(h: int) -> onnx.ModelProto:
    """rot90 CW: Transpose(0,1,3,2) → Gather 新 width 軸を逆順（= 旧 height h）。"""
    idx = _rev_idx(h)
    init = helper.make_tensor("idx", TensorProto.INT64, [GRID_MAX], idx)
    t = helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
    g = helper.make_node("Gather", ["t", "idx"], ["output"], axis=3)
    return _model([t, g], [init])


def build_rot270(w: int) -> onnx.ModelProto:
    """rot270 CCW: Transpose(0,1,3,2) → Gather 新 height 軸を逆順（= 旧 width w）。"""
    idx = _rev_idx(w)
    init = helper.make_tensor("idx", TensorProto.INT64, [GRID_MAX], idx)
    t = helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
    g = helper.make_node("Gather", ["t", "idx"], ["output"], axis=2)
    return _model([t, g], [init])


def _scale_idx(dim_in: int, k: int) -> list[int]:
    """k 倍スケールアップの 30 要素インデックスマップ。

    content 行/列 [0..dim_in) を各 k 回繰り返し、残りはゼロパディング行を維持する。
    k * dim_in <= GRID_MAX を前提とする。
    """
    scaled = [r // k for r in range(k * dim_in)]
    padding = list(range(dim_in, dim_in + (GRID_MAX - k * dim_in)))
    return scaled + padding


def build_scale_up_rows(h: int, k: int) -> onnx.ModelProto:
    """各行を k 回繰り返す行方向スケール（params=30, memory=0）。

    Gather(axis=2) 1 本で input → output へ直結するため中間テンソルが生じない。
    k * h <= GRID_MAX を前提とする。
    """
    row_idx = _scale_idx(h, k)
    init = helper.make_tensor("scale_row_idx", TensorProto.INT64, [GRID_MAX], row_idx)
    node = helper.make_node("Gather", ["input", "scale_row_idx"], ["output"], axis=2)
    return _model([node], [init])


def build_scale_up_cols(w: int, k: int) -> onnx.ModelProto:
    """各列を k 回繰り返す列方向スケール（params=30, memory=0）。

    Gather(axis=3) 1 本で input → output へ直結するため中間テンソルが生じない。
    k * w <= GRID_MAX を前提とする。
    """
    col_idx = _scale_idx(w, k)
    init = helper.make_tensor("scale_col_idx", TensorProto.INT64, [GRID_MAX], col_idx)
    node = helper.make_node("Gather", ["input", "scale_col_idx"], ["output"], axis=3)
    return _model([node], [init])


def build_scale_up_2d(h: int, w: int, k: int) -> onnx.ModelProto:
    """行・列を共に k 回繰り返す 2D スケール（params=60, memory=36000）。

    Gather(axis=2) で行を展開した中間テンソルに続き Gather(axis=3) で列を展開する。
    k * h <= GRID_MAX かつ k * w <= GRID_MAX を前提とする。
    """
    row_idx = _scale_idx(h, k)
    col_idx = _scale_idx(w, k)
    row_init = helper.make_tensor(
        "scale_row_idx", TensorProto.INT64, [GRID_MAX], row_idx
    )
    col_init = helper.make_tensor(
        "scale_col_idx", TensorProto.INT64, [GRID_MAX], col_idx
    )
    g_rows = helper.make_node(
        "Gather", ["input", "scale_row_idx"], ["scale_mid"], axis=2
    )
    g_cols = helper.make_node(
        "Gather", ["scale_mid", "scale_col_idx"], ["output"], axis=3
    )
    return _model([g_rows, g_cols], [row_init, col_init])


def _tile_idx(dim_in: int, reps: int) -> list[int]:
    """np.tile の循環インデックスマップ（30 要素）。

    content 位置 [0..reps*dim_in) は ``i % dim_in`` で循環し、
    残り [reps*dim_in..29] はゼロパディング行/列へ写す。
    reps * dim_in <= GRID_MAX を前提とする。
    """
    tiled = [i % dim_in for i in range(reps * dim_in)]
    padding = list(range(dim_in, dim_in + (GRID_MAX - reps * dim_in)))
    return tiled + padding


def build_tile_rows(h: int, reps: int) -> onnx.ModelProto:
    """行方向 reps 回循環タイル（np.tile axis=0）。params=30, memory=0。

    Gather(axis=2) 1 本で input → output へ直結。reps * h <= GRID_MAX を前提とする。
    """
    row_idx = _tile_idx(h, reps)
    init = helper.make_tensor("tile_row_idx", TensorProto.INT64, [GRID_MAX], row_idx)
    node = helper.make_node("Gather", ["input", "tile_row_idx"], ["output"], axis=2)
    return _model([node], [init])


def build_tile_cols(w: int, reps: int) -> onnx.ModelProto:
    """列方向 reps 回循環タイル（np.tile axis=1）。params=30, memory=0。

    Gather(axis=3) 1 本で input → output へ直結。reps * w <= GRID_MAX を前提とする。
    """
    col_idx = _tile_idx(w, reps)
    init = helper.make_tensor("tile_col_idx", TensorProto.INT64, [GRID_MAX], col_idx)
    node = helper.make_node("Gather", ["input", "tile_col_idx"], ["output"], axis=3)
    return _model([node], [init])


def build_tile(h: int, w: int, reps_h: int, reps_w: int) -> onnx.ModelProto:
    """2D 循環タイル（np.tile）。params=60, memory=36000。

    Gather(axis=2) で行を展開した中間テンソルに続き Gather(axis=3) で列を展開する。
    reps_h * h <= GRID_MAX かつ reps_w * w <= GRID_MAX を前提とする。
    """
    row_idx = _tile_idx(h, reps_h)
    col_idx = _tile_idx(w, reps_w)
    row_init = helper.make_tensor(
        "tile_row_idx", TensorProto.INT64, [GRID_MAX], row_idx
    )
    col_init = helper.make_tensor(
        "tile_col_idx", TensorProto.INT64, [GRID_MAX], col_idx
    )
    g_rows = helper.make_node(
        "Gather", ["input", "tile_row_idx"], ["tile_mid"], axis=2
    )
    g_cols = helper.make_node(
        "Gather", ["tile_mid", "tile_col_idx"], ["output"], axis=3
    )
    return _model([g_rows, g_cols], [row_init, col_init])
