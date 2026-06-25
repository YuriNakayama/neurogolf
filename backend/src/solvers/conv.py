"""Single-layer Conv2d ONNX builder — the shared base for NeuroGolf solvers.

This re-implements the official ``neurogolf_utils.single_layer_conv2d_network``
with bare ``onnx.helper`` (the official module can't be imported — it pulls in
IPython / matplotlib / onnx_tool). The weight layout, kernel offsets, padding,
ir_version and opset are reproduced verbatim so the resulting graph scores
identically under the competition scorer.

A solver supplies ``weight_fn(out_channel, in_channel, (row, col)) -> float``
where ``(row, col)`` ranges over the centered kernel offsets.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import onnx

from dataset.encoding import BATCH, GRID_MAX, NUM_COLORS

INPUT_NAME = "input"
OUTPUT_NAME = "output"
_DATA_TYPE = onnx.TensorProto.FLOAT
_GRID_SHAPE = [BATCH, NUM_COLORS, GRID_MAX, GRID_MAX]
_IR_VERSION = 10
_OPSET_IMPORTS = [onnx.helper.make_opsetid("", 10)]

WeightFn = Callable[[int, int, tuple[int, int]], float]


def build_single_layer_conv2d(weight_fn: WeightFn, kernel_size: int) -> onnx.ModelProto:
    """Build a constraint-compliant single Conv layer ``[1,10,30,30] -> same``."""
    kernel_offsets = range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel_shape = [kernel_size, kernel_size]
    w_shape = [NUM_COLORS, NUM_COLORS, kernel_size, kernel_size]
    pads = [kernel_size // 2] * 4
    weight_cells = itertools.product(
        range(NUM_COLORS), range(NUM_COLORS), kernel_offsets, kernel_offsets
    )
    weights = [weight_fn(o, i, (r, c)) for (o, i, r, c) in weight_cells]

    x = onnx.helper.make_tensor_value_info(INPUT_NAME, _DATA_TYPE, _GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info(OUTPUT_NAME, _DATA_TYPE, _GRID_SHAPE)
    w = onnx.helper.make_tensor("W", _DATA_TYPE, w_shape, weights)
    node = onnx.helper.make_node(
        "Conv", [INPUT_NAME, "W"], [OUTPUT_NAME], kernel_shape=kernel_shape, pads=pads
    )
    graph = onnx.helper.make_graph([node], "graph", [x], [y], [w])
    model = onnx.helper.make_model(
        graph, ir_version=_IR_VERSION, opset_imports=_OPSET_IMPORTS
    )
    onnx.checker.check_model(model)
    return model
