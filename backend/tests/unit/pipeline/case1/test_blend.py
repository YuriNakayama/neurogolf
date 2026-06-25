"""``pipeline.case1`` の単体テスト（blend 選択 + バンドル解決）。

- ``better()`` の tie-break 真理値表
- ``resolve_bundle()`` の dir / zip 解決
- ダミー A/B バンドルに対する ``blend()`` の選択結果
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from pipeline.case1 import better, blend, resolve_bundle
from pipeline.case1.bundle import BundleError

_SHAPE = [1, 10, 30, 30]


def _r(n_fail: int, cost: int | None, status: str = "ok") -> dict[str, object]:
    """audit_one 互換の最小スコア dict を作る。"""
    points = 0.0 if (n_fail > 0 or cost is None) else 10.0
    return {"n_fail": n_fail, "cost": cost, "status": status, "points": points}


def test_better_only_a_correct() -> None:
    assert better(_r(0, 100), _r(1, 50)) == "A"


def test_better_only_b_correct() -> None:
    assert better(_r(1, 50), _r(0, 100)) == "B"


def test_better_both_correct_picks_cheaper() -> None:
    assert better(_r(0, 100), _r(0, 50)) == "B"
    assert better(_r(0, 50), _r(0, 100)) == "A"


def test_better_equal_cost_prefers_a() -> None:
    assert better(_r(0, 100), _r(0, 100)) == "A"


def test_better_neither_correct_fewer_fails() -> None:
    assert better(_r(3, 100), _r(1, 100)) == "B"
    assert better(_r(1, 100), _r(3, 100)) == "A"


def test_better_none_candidate() -> None:
    assert better(None, _r(0, 100)) == "B"
    assert better(_r(0, 100), None) == "A"


def _identity_onnx(path: Path, n_extra_const: int = 0) -> None:
    """恒等 ONNX を保存する。``n_extra_const`` で cost をわずかに増やせる。"""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, _SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, _SHAPE)
    nodes = [helper.make_node("Identity", ["input"], ["output"])]
    inits = []
    for i in range(n_extra_const):
        inits.append(helper.make_tensor(f"k{i}", TensorProto.FLOAT, [1], [float(i)]))
    graph = helper.make_graph(nodes, "identity", [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))


def _identity_task_json(path: Path) -> None:
    import json

    g = [[1, 2], [3, 0]]
    path.write_text(json.dumps({"train": [{"input": g, "output": g}]}))


def test_resolve_bundle_dir(tmp_path: Path) -> None:
    d = tmp_path / "bundle"
    d.mkdir()
    _identity_onnx(d / "task001.onnx")
    assert resolve_bundle(str(d), None, "A", str(tmp_path / "work")) == str(d)


def test_resolve_bundle_zip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _identity_onnx(src / "task001.onnx")
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(src / "task001.onnx", "task001.onnx")
    resolved = resolve_bundle(None, str(zip_path), "B", str(tmp_path / "work"))
    assert (Path(resolved) / "task001.onnx").is_file()


def test_resolve_bundle_missing_raises(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(BundleError):
        resolve_bundle(None, None, "A", str(tmp_path / "work"))


def test_blend_picks_cheaper_correct(tmp_path: Path) -> None:
    """両方 correct のとき、cost の低い側（A: 余分な const なし）を採用する。"""
    a = tmp_path / "A"
    b = tmp_path / "B"
    tasks = tmp_path / "tasks"
    for d in (a, b, tasks):
        d.mkdir()
    # task001: A は素の恒等（安い）、B は const 入り（高い）。両方 correct。
    _identity_onnx(a / "task001.onnx")
    _identity_onnx(b / "task001.onnx", n_extra_const=5)
    _identity_task_json(tasks / "task001.json")

    stage = tmp_path / "stage"
    summary = blend(str(a), str(b), str(tasks), str(stage))

    assert summary.staged == 1
    assert summary.picks["A"] == 1
    assert summary.picks["B"] == 0
    assert (stage / "task001.onnx").is_file()


def test_blend_uses_b_when_a_missing(tmp_path: Path) -> None:
    a = tmp_path / "A"
    b = tmp_path / "B"
    tasks = tmp_path / "tasks"
    for d in (a, b, tasks):
        d.mkdir()
    # A には task001 が無い → B が採用される。
    _identity_onnx(b / "task001.onnx")
    _identity_task_json(tasks / "task001.json")

    stage = tmp_path / "stage"
    summary = blend(str(a), str(b), str(tasks), str(stage))

    assert summary.picks["B"] == 1
    assert summary.missing_a == 400
    assert (stage / "task001.onnx").is_file()
