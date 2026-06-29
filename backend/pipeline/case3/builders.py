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


def _tile_idx(n: int, reps: int) -> list[int]:
    """n を reps 回繰り返すインデックス（reps*n > 30 はクリップ）、残りは GRID_MAX-1。"""
    out = (list(range(n)) * reps)[:GRID_MAX]
    return out + [GRID_MAX - 1] * (GRID_MAX - len(out))


def build_tile(h: int, w: int, reps_h: int, reps_w: int) -> onnx.ModelProto:
    """h×w グリッドを reps_h × reps_w 回タイリング（各軸 Gather）。params=60。"""
    rows = _tile_idx(h, reps_h)
    cols = _tile_idx(w, reps_w)
    h_init = helper.make_tensor("rows", TensorProto.INT64, [GRID_MAX], rows)
    w_init = helper.make_tensor("cols", TensorProto.INT64, [GRID_MAX], cols)
    g1 = helper.make_node("Gather", ["input", "rows"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "cols"], ["output"], axis=3)
    return _model([g1, g2], [h_init, w_init])


def _scale_idx(n: int, factor: int) -> list[int]:
    """各セルを factor 回繰り返す（output[i] = input[i // factor]）。"""
    out = [i // factor for i in range(min(n * factor, GRID_MAX))]
    return out + [GRID_MAX - 1] * (GRID_MAX - len(out))


def build_scale(h: int, w: int, sh: int, sw: int) -> onnx.ModelProto:
    """h×w グリッドを sh×sw 倍にズーム（各セルを sh×sw ブロックに）。params=60。"""
    rows = _scale_idx(h, sh)
    cols = _scale_idx(w, sw)
    h_init = helper.make_tensor("rows", TensorProto.INT64, [GRID_MAX], rows)
    w_init = helper.make_tensor("cols", TensorProto.INT64, [GRID_MAX], cols)
    g1 = helper.make_node("Gather", ["input", "rows"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "cols"], ["output"], axis=3)
    return _model([g1, g2], [h_init, w_init])
