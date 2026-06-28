"""小空間ルックアップ solver: 固定サイズタスクを h×w 空間で処理し memory を最小化。

公式スコアラの memory は中間テンソルの **静的形状** 要素数 × itemsize（input/output 除外）。
ベースラインは全中間を [1,10,30,30]=9000 要素で持つため高 cost。本 solver は

  Slice([1,10,30,30] -> [1,10,h,w])  →  小空間で lookup/conv  →  Pad(-> [1,10,30,30])

とし、全中間を h×w スケール（数十〜数百要素）に抑える。h,w は「全 example で入出力
サイズが一定」のタスクのみ確定できる。近傍ルックアップ部は ``lookup`` と同じ
2 層 Conv（完全一致検出 → 色写像）だが、小空間なので P チャネルでも安価。
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from .arc import NUM_COLORS, Example
from .lookup import _build_table, detector_weights

_DTYPE = TensorProto.FLOAT
GRID = 30
GRID_SHAPE = [1, NUM_COLORS, GRID, GRID]


def _const_size(examples: tuple[Example, ...]) -> tuple[int, int, int, int] | None:
    sizes = {
        ((len(e.input), len(e.input[0])), (len(e.output), len(e.output[0])))
        for e in examples
    }
    if len(sizes) != 1:
        return None
    (ih, iw), (oh, ow) = next(iter(sizes))
    return ih, iw, oh, ow


def build_small_lookup(examples: tuple[Example, ...], k: int) -> onnx.ModelProto | None:
    """固定サイズ・近傍決定的タスクを小空間 2 層 Conv で厳密構成。"""
    size = _const_size(examples)
    if size is None:
        return None
    ih, iw, oh, ow = size
    if (ih, iw) != (oh, ow):  # 同形状のみ（crop/upscale は別 solver）
        return None
    table = _build_table(examples, k)
    if table is None:
        return None
    patterns = list(table.items())
    p = len(patterns)
    pad = k // 2
    pat_grids = [
        np.frombuffer(key, dtype=np.int64).reshape(k, k) for key, _ in patterns
    ]
    out_colors = [oc for _, oc in patterns]

    w1, b1 = detector_weights(pat_grids, k)
    w2 = np.zeros((NUM_COLORS, p, 1, 1), dtype=np.float32)
    for pi, oc in enumerate(out_colors):
        w2[oc, pi, 0, 0] = 1.0

    # Slice input [1,10,30,30] -> [1,10,ih,iw] (top-left)
    starts = helper.make_tensor("st", TensorProto.INT64, [2], [0, 0])
    ends = helper.make_tensor("en", TensorProto.INT64, [2], [ih, iw])
    axes = helper.make_tensor("ax", TensorProto.INT64, [2], [2, 3])
    slice_node = helper.make_node("Slice", ["input", "st", "en", "ax"], ["xs"])

    w1_t = helper.make_tensor("W1", _DTYPE, [p, NUM_COLORS, k, k], w1.flatten())
    b1_t = helper.make_tensor("B1", _DTYPE, [p], b1.flatten())
    w2_t = helper.make_tensor("W2", _DTYPE, [NUM_COLORS, p, 1, 1], w2.flatten())
    conv1 = helper.make_node(
        "Conv", ["xs", "W1", "B1"], ["h0"], kernel_shape=[k, k], pads=[pad] * 4
    )
    relu = helper.make_node("Relu", ["h0"], ["h1"])
    conv2 = helper.make_node(
        "Conv", ["h1", "W2"], ["ys"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
    )

    # Pad [1,10,oh,ow] -> [1,10,30,30] (bottom/right zeros). opset10: pads は属性。
    pad_node = helper.make_node(
        "Pad",
        ["ys"],
        ["output"],
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, GRID - oh, GRID - ow],
        value=0.0,
    )

    x = helper.make_tensor_value_info("input", _DTYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DTYPE, GRID_SHAPE)
    graph = helper.make_graph(
        [slice_node, conv1, relu, conv2, pad_node],
        "small_lookup",
        [x],
        [y],
        [starts, ends, axes, w1_t, b1_t, w2_t],
    )
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )
