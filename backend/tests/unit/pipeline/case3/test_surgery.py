"""surgery.prune_and_guard のユニットテスト。"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from pipeline.case3.surgery import prune_and_guard


def _make_and_guard_model() -> onnx.ModelProto:
    """And(guard_init, x) → intermediate → Cast(float) → output を構築。

    guard_init は全 True な bool 定数。
    And は中間テンソルを出力し、graph output は Cast 経由。
    """
    guard_arr = np.ones((1, 1, 3, 3), dtype=np.bool_)
    guard_init = numpy_helper.from_array(guard_arr, name="guard")
    and_node = helper.make_node("And", ["guard", "x"], ["intermediate"])
    cast_node = helper.make_node("Cast", ["intermediate"], ["output"], to=TensorProto.FLOAT)
    graph = helper.make_graph(
        [and_node, cast_node],
        "test_guard",
        [helper.make_tensor_value_info("x", TensorProto.BOOL, [1, 1, 3, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])],
        initializer=[guard_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def test_prune_and_guard_removes_and_node() -> None:
    """And(guard, x) の And ノードが除去されること。"""
    model = _make_and_guard_model()
    result = prune_and_guard(model, guard_name="guard")

    ops = [n.op_type for n in result.graph.node]
    assert "And" not in ops, f"And node should be removed, got {ops}"


def test_prune_and_guard_removes_dead_initializer() -> None:
    """guard_init が DCE で除去されること。"""
    model = _make_and_guard_model()
    result = prune_and_guard(model, guard_name="guard")

    init_names = {init.name for init in result.graph.initializer}
    assert "guard" not in init_names, f"guard initializer should be removed, got {init_names}"


def test_prune_and_guard_model_runs_correctly() -> None:
    """除去後のモデルが元と同じ結果を返すこと（All-True guard は恒等）。"""
    model = _make_and_guard_model()
    result = prune_and_guard(model, guard_name="guard")

    onnx.checker.check_model(result)

    inp = np.array(
        [[[[True, False, True], [False, True, False], [True, True, False]]]],
        dtype=np.bool_,
    )
    sess = ort.InferenceSession(
        result.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out = sess.run(None, {"x": inp})[0]
    np.testing.assert_array_equal(out, inp.astype(np.float32))


def test_prune_and_guard_noop_when_no_match() -> None:
    """guard_name が存在しない場合は model をそのまま返すこと。"""
    model = _make_and_guard_model()
    result = prune_and_guard(model, guard_name="nonexistent_guard")

    assert result is model, "Should return original model when no pattern found"
