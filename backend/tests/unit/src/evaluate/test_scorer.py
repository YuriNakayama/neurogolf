"""``evaluate.scorer``（公式スコアラーのミラー）の単体テスト。

手製の小さな ONNX を ``onnx.helper`` で組み、``audit_one`` が
params / memory / cost / points と correctness を期待どおり返すこと、
banned op / filesize 超過 / incorrect で points=0 になることを検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from evaluate import audit_one, convert_to_numpy

_SHAPE = [1, 10, 30, 30]


def _identity_onnx(path: Path) -> Path:
    """input をそのまま output に流す恒等 ONNX を保存する。"""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, _SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, _SHAPE)
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _banned_onnx(path: Path) -> Path:
    """禁止 op（NonZero）を含む ONNX を保存する。"""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, _SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.INT64, [4, 1])
    node = helper.make_node("NonZero", ["input"], ["output"])
    graph = helper.make_graph([node], "banned", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def _grid(values: list[list[int]]) -> list[list[int]]:
    return values


def _identity_task() -> dict[str, object]:
    """input == output の恒等タスク（恒等 ONNX なら全 pass するはず）。"""
    g = _grid([[1, 2], [3, 0]])
    return {"train": [{"input": g, "output": g}], "test": [], "arc-gen": []}


def _swap_task() -> dict[str, object]:
    """input != output のタスク（恒等 ONNX なら fail するはず）。"""
    return {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [],
        "arc-gen": [],
    }


def test_convert_to_numpy_one_hot() -> None:
    b = convert_to_numpy({"input": [[3]], "output": [[3]]})
    assert b is not None
    assert b["input"].shape == (1, 10, 30, 30)
    # color 3 at (0,0) is hot, everything else zero.
    assert b["input"][0, 3, 0, 0] == 1.0
    assert b["input"].sum() == 1.0


def test_convert_to_numpy_rejects_oversize() -> None:
    big = [[0] * 31]
    assert convert_to_numpy({"input": big, "output": big}) is None


def test_audit_one_identity_correct(tmp_path: Path) -> None:
    path = _identity_onnx(tmp_path / "task001.onnx")
    res = audit_one(str(path), _identity_task(), run_correctness=True)
    assert res["status"] == "ok"
    assert res["n_fail"] == 0
    assert res["n_pass"] == 1
    assert res["params"] == 0
    # Identity graph: input/output are excluded from memory, so cost == 0 and
    # points are capped at max(1, 25 - ln(1)) == 25 — this is the mirror's
    # intended behaviour for a pure passthrough.
    assert res["cost"] == 0
    assert res["points"] == 25.0


def test_audit_one_identity_incorrect_scores_zero(tmp_path: Path) -> None:
    path = _identity_onnx(tmp_path / "task002.onnx")
    res = audit_one(str(path), _swap_task(), run_correctness=True)
    assert res["n_fail"] == 1
    assert res["status"] == "INCORRECT"
    assert res["points"] == 0.0


def test_audit_one_no_correctness_still_scores(tmp_path: Path) -> None:
    path = _identity_onnx(tmp_path / "task003.onnx")
    res = audit_one(str(path), None, run_correctness=False)
    assert res["status"] == "ok"
    assert res["points"] is not None and res["points"] > 1.0


def test_audit_one_banned_op(tmp_path: Path) -> None:
    path = _banned_onnx(tmp_path / "task004.onnx")
    res = audit_one(str(path), None, run_correctness=False)
    assert res["status"].startswith("BANNED_OP")
    assert res["points"] == 0.0


def test_audit_one_filesize_over_limit(tmp_path: Path) -> None:
    path = tmp_path / "task005.onnx"
    path.write_bytes(b"\x00" * (int(1.44 * 1024 * 1024) + 1))
    res = audit_one(str(path), None, run_correctness=False)
    assert res["status"] == "FILESIZE_OVER_LIMIT"
    assert res["points"] == 0.0


def test_audit_one_load_error(tmp_path: Path) -> None:
    path = tmp_path / "task006.onnx"
    path.write_bytes(b"not an onnx model")
    res = audit_one(str(path), None, run_correctness=False)
    assert res["status"].startswith("load_error")
    assert res["points"] == 0.0


def test_audit_one_cheaper_has_higher_points(tmp_path: Path) -> None:
    """同一構造なら cost が小さいほど points が高い（log スコアの単調性）。"""
    path = _identity_onnx(tmp_path / "task007.onnx")
    res = audit_one(str(path), None, run_correctness=False)
    assert res["cost"] is not None
    # points = 25 - ln(cost) なので cost と points は逆相関。
    import math

    assert np.isclose(res["points"], max(1.0, 25.0 - math.log(max(1.0, res["cost"]))))
