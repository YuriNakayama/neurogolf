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


def build_flip(axis: int) -> onnx.ModelProto:
    """空間反転を Slice の負ステップで表現（axis: 2=縦, 3=横）。

    Slice の starts/ends/axes/steps を initializer で与える。中間テンソルなし。
    """
    starts = helper.make_tensor("s", TensorProto.INT64, [1], [-1])
    ends = helper.make_tensor("e", TensorProto.INT64, [1], [-(GRID_MAX + 1)])
    axes = helper.make_tensor("a", TensorProto.INT64, [1], [axis])
    steps = helper.make_tensor("t", TensorProto.INT64, [1], [-1])
    node = helper.make_node("Slice", ["input", "s", "e", "a", "t"], ["output"])
    return _model([node], [starts, ends, axes, steps])
