"""Spatial solvers: transforms that move cells without changing color.

These are zero-parameter graphs built from ``Transpose`` / ``Gather`` (all
allowed, statically shaped):

- ``transpose``  out[r][c] = in[c][r]  — safe for any grid (left-justified
  encoding stays consistent because H and W swap together).
- ``flip_lr`` / ``flip_ud`` / ``rot180`` reverse along the 30-wide axes. These
  are only correct when every example shares one fixed H x W (otherwise the
  left/top-justified encoding shifts the active region). The build-and-verify
  orchestration discards a model that fails any example, so emitting them is
  safe — they simply won't be adopted for variable-size tasks.
"""

from __future__ import annotations

import onnx

from dataset.encoding import BATCH, GRID_MAX, NUM_COLORS
from solvers.conv import INPUT_NAME, OUTPUT_NAME

_DATA_TYPE = onnx.TensorProto.FLOAT
_INT64 = onnx.TensorProto.INT64
_GRID_SHAPE = [BATCH, NUM_COLORS, GRID_MAX, GRID_MAX]
_IR_VERSION = 10
_OPSET_IMPORTS = [onnx.helper.make_opsetid("", 10)]
_REVERSED = list(range(GRID_MAX - 1, -1, -1))


def _wrap(
    node: onnx.NodeProto, initializers: list[onnx.TensorProto]
) -> onnx.ModelProto:
    x = onnx.helper.make_tensor_value_info(INPUT_NAME, _DATA_TYPE, _GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info(OUTPUT_NAME, _DATA_TYPE, _GRID_SHAPE)
    graph = onnx.helper.make_graph([node], "graph", [x], [y], initializers)
    model = onnx.helper.make_model(
        graph, ir_version=_IR_VERSION, opset_imports=_OPSET_IMPORTS
    )
    onnx.checker.check_model(model)
    return model


def build_transpose_model() -> onnx.ModelProto:
    """Transpose H and W (axes 2 and 3): out[r][c] = in[c][r]."""
    node = onnx.helper.make_node(
        "Transpose", [INPUT_NAME], [OUTPUT_NAME], perm=[0, 1, 3, 2]
    )
    return _wrap(node, [])


def _gather_model(axis: int) -> onnx.ModelProto:
    indices = onnx.helper.make_tensor("rev", _INT64, [GRID_MAX], _REVERSED)
    node = onnx.helper.make_node(
        "Gather", [INPUT_NAME, "rev"], [OUTPUT_NAME], axis=axis
    )
    return _wrap(node, [indices])


def build_flip_ud_model() -> onnx.ModelProto:
    """Reverse rows (axis 2). Correct only for fixed-size grids."""
    return _gather_model(axis=2)


def build_flip_lr_model() -> onnx.ModelProto:
    """Reverse columns (axis 3). Correct only for fixed-size grids."""
    return _gather_model(axis=3)


def build_rot180_model() -> onnx.ModelProto:
    """Reverse both H and W. Correct only for fixed-size grids."""
    rev_r = onnx.helper.make_tensor("rev_r", _INT64, [GRID_MAX], _REVERSED)
    rev_c = onnx.helper.make_tensor("rev_c", _INT64, [GRID_MAX], _REVERSED)
    n1 = onnx.helper.make_node("Gather", [INPUT_NAME, "rev_r"], ["tmp"], axis=2)
    n2 = onnx.helper.make_node("Gather", ["tmp", "rev_c"], [OUTPUT_NAME], axis=3)
    x = onnx.helper.make_tensor_value_info(INPUT_NAME, _DATA_TYPE, _GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info(OUTPUT_NAME, _DATA_TYPE, _GRID_SHAPE)
    graph = onnx.helper.make_graph([n1, n2], "graph", [x], [y], [rev_r, rev_c])
    model = onnx.helper.make_model(
        graph, ir_version=_IR_VERSION, opset_imports=_OPSET_IMPORTS
    )
    onnx.checker.check_model(model)
    return model
