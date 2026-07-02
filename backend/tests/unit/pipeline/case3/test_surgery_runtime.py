"""equivalence_alias_runtime surgery pass のユニットテスト (TDD: RED → GREEN)。

ランタイム等値テンソルのエイリアス化により、モデルのメモリコストが
下がることを確認する。
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from pipeline.case3.surgery import equivalence_alias_runtime


def _run(model: onnx.ModelProto, grid: list[list[int]]) -> np.ndarray:
    inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            inp[0, color, r, c] = 1.0
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (out,) = sess.run(None, {"input": inp})
    return np.asarray(out)


def _count_nodes(model: onnx.ModelProto) -> int:
    return len(model.graph.node)


def _make_simple_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    return {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2], [3, 4]],
            },
            {
                "input": [[5, 0], [0, 9]],
                "output": [[5, 0], [0, 9]],
            },
        ]
    }


def _build_duplicate_path_model() -> onnx.ModelProto:
    """relu_a と relu_b が同一計算 (Max(input, 0)) を持つモデル。

    input → relu_a = Max(input, 0)
          → relu_b = Max(input, 0)   ← duplicate of relu_a
    output = Add(relu_a, relu_b)
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    zero = numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=np.float32), "zero")
    node_a = helper.make_node("Max", ["input", "zero"], ["relu_a"])
    node_b = helper.make_node("Max", ["input", "zero"], ["relu_b"])
    node_add = helper.make_node("Add", ["relu_a", "relu_b"], ["output"])
    graph = helper.make_graph(
        [node_a, node_b, node_add],
        "dup_test",
        [X],
        [Y],
        [zero],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


def _build_unique_model() -> onnx.ModelProto:
    """中間テンソルが全て異なる値のモデル (Gather で異なる channel を選択)。

    input → a = Gather(input, [0], axis=1)
          → b = Gather(input, [1], axis=1)
    output = Add(a, b)   ← then pad to [1,10,30,30]
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    idx0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "idx0")
    idx1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "idx1")
    pads_v = numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, 9, 0, 0], dtype=np.int64), "pads_v"
    )
    node_a = helper.make_node("Gather", ["input", "idx0"], ["ch0"], axis=1)
    node_b = helper.make_node("Gather", ["input", "idx1"], ["ch1"], axis=1)
    node_add = helper.make_node("Add", ["ch0", "ch1"], ["ab"])
    # pad ch axis from 1 to 10 so output has shape [1,10,30,30]
    node_pad = helper.make_node("Pad", ["ab", "pads_v"], ["output"])
    graph = helper.make_graph(
        [node_a, node_b, node_add, node_pad],
        "unique_test",
        [X],
        [Y],
        [idx0, idx1, pads_v],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


# ─── tests ──────────────────────────────────────────────────────────────────


def test_duplicate_path_node_count_reduced() -> None:
    """重複計算パスがあるモデルで、ノード数が減ること。"""
    model = _build_duplicate_path_model()
    orig_nodes = _count_nodes(model)
    result = equivalence_alias_runtime(model, _make_simple_examples())
    assert _count_nodes(result) < orig_nodes, "dead node should be eliminated"


def test_duplicate_path_output_preserved() -> None:
    """エイリアス後もモデル出力が変わらないこと。"""
    model = _build_duplicate_path_model()
    grid = [[1, 2], [3, 4]]
    expected = _run(model, grid)
    result = equivalence_alias_runtime(model, _make_simple_examples())
    got = _run(result, grid)
    np.testing.assert_array_equal(got, expected)


def test_duplicate_path_onnx_valid() -> None:
    """エイリアス後のモデルが onnx.checker を通ること。"""
    model = _build_duplicate_path_model()
    result = equivalence_alias_runtime(model, _make_simple_examples())
    onnx.checker.check_model(result)


def test_unique_model_unchanged() -> None:
    """全中間テンソルが固有値のモデルでは、ノード数が変わらないこと。"""
    model = _build_unique_model()
    orig_nodes = _count_nodes(model)
    result = equivalence_alias_runtime(model, _make_simple_examples())
    assert _count_nodes(result) == orig_nodes


def test_empty_examples_returns_original() -> None:
    """examples が空の場合はモデルが変わらないこと。"""
    model = _build_duplicate_path_model()
    result = equivalence_alias_runtime(model, {"train": [], "test": []})
    assert _count_nodes(result) == _count_nodes(model)


def test_result_does_not_mutate_input() -> None:
    """元のモデルが変更されないこと（deepcopy が正しく使われていること）。"""
    model = _build_duplicate_path_model()
    orig_nodes = _count_nodes(model)
    equivalence_alias_runtime(model, _make_simple_examples())
    assert _count_nodes(model) == orig_nodes, "input model must not be mutated"
