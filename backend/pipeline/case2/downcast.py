"""Lossless float32 -> float16 downcast pass for baseline ONNX graphs.

94% of a typical baseline task's ``cost = params + memory`` is intermediate
tensor memory (float32, 4 bytes/elem). The grids these graphs carry are tiny
integers (one-hot 0/1, colors 0-9), so float16 represents every value exactly —
halving the memory of every float intermediate with no loss of correctness.

The pass rewrites the graph so all FLOAT tensors/initializers become FLOAT16,
keeping the model's I/O contract (``input``/``output`` stay FLOAT) via a Cast at
each boundary. Ops that are not float16-capable in onnxruntime are left untouched
by reverting the whole task if the recast model fails the scorer's checks — the
build pipeline only overrides a task when the recast is provably exact and
cheaper (verified through ``src/evaluate``), so an unsupported op simply means
that task keeps its baseline ONNX.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_F32 = TensorProto.FLOAT
_F16 = TensorProto.FLOAT16
IO_NAMES = ("input", "output")


def _recast_initializers(graph: onnx.GraphProto) -> None:
    for init in list(graph.initializer):
        if init.data_type != _F32:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        graph.initializer.remove(init)
        graph.initializer.append(numpy_helper.from_array(arr, name=init.name))


def _recast_value_infos(graph: onnx.GraphProto) -> None:
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name in IO_NAMES:
            continue
        tt = vi.type.tensor_type
        if tt.elem_type == _F32:
            tt.elem_type = _F16


def _recast_constant_nodes(graph: onnx.GraphProto) -> None:
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == _F32:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=attr.t.name))


def _recast_cast_nodes(graph: onnx.GraphProto) -> None:
    """Rewrite ``Cast(to=FLOAT)`` ops to ``Cast(to=FLOAT16)`` (internal casts)."""
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == _F32:
                attr.i = _F16


def to_float16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of ``model`` with all float32 internals recast to float16.

    The public ``input``/``output`` remain float32; a Cast bridges each boundary.
    Raises on checker failure so the caller can fall back to the baseline.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    graph = m.graph

    # Bridge input: input(f32) -> Cast -> input_f16, rename first consumers.
    in_cast_out = "input_f16"
    out_cast_in = "output_f16"
    producers_of_output = [n for n in graph.node for o in n.output if o == "output"]

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == "input":
                node.input[i] = in_cast_out
        for i, name in enumerate(node.output):
            if name == "output":
                node.output[i] = out_cast_in

    in_cast = helper.make_node("Cast", ["input"], [in_cast_out], to=_F16)
    out_cast = helper.make_node("Cast", [out_cast_in], ["output"], to=_F32)

    _recast_initializers(graph)
    _recast_constant_nodes(graph)
    _recast_cast_nodes(graph)
    _recast_value_infos(graph)

    new_nodes = [in_cast, *graph.node, out_cast]
    del graph.node[:]
    graph.node.extend(new_nodes)
    # output_f16 is float16; declare it so shape inference is consistent.
    if producers_of_output:
        graph.value_info.append(
            helper.make_tensor_value_info(out_cast_in, _F16, [1, 10, 30, 30])
        )

    onnx.checker.check_model(m, full_check=True)
    return m
