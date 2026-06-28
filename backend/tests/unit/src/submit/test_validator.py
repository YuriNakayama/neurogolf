"""validator の hard/soft 失敗の切り分けテスト（audit_one はモック）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from submit import validator


def _audit(status: str, cost: int | None, points: float | None) -> dict[str, Any]:
    return {"status": status, "cost": cost, "points": points}


def _patch_audit(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> None:
    monkeypatch.setattr(validator, "audit_one", lambda *_a, **_k: result)


def _onnx(tmp_path: Path, name: str = "task001.onnx") -> Path:
    p = tmp_path / name
    p.write_bytes(b"x")  # size only; audit_one is mocked
    return p


def test_ok_is_scorable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audit(monkeypatch, _audit("ok", 1234, 12.5))
    r = validator.validate_onnx_file(_onnx(tmp_path))
    assert r.scorable is True
    assert r.cost == 1234
    assert r.score == 12.5


def test_score_error_is_soft_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORT が load できない false-negative は raise せず scorable=False で残す。"""
    _patch_audit(monkeypatch, _audit("score_error:[ShapeInferenceError]", None, 0.0))
    r = validator.validate_onnx_file(_onnx(tmp_path))
    assert r.scorable is False
    assert r.cost is None
    assert r.score is None


def test_session_error_is_soft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audit(monkeypatch, _audit("session_error:boom", None, None))
    r = validator.validate_onnx_file(_onnx(tmp_path))
    assert r.scorable is False


def test_banned_op_is_hard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audit(monkeypatch, _audit("BANNED_OP:Loop", None, None))
    with pytest.raises(validator.ValidationError):
        validator.validate_onnx_file(_onnx(tmp_path))


def test_load_error_is_hard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audit(monkeypatch, _audit("load_error:corrupt", None, None))
    with pytest.raises(validator.ValidationError):
        validator.validate_onnx_file(_onnx(tmp_path))


def test_oversize_is_hard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _onnx(tmp_path)
    p.write_bytes(b"x" * (validator.MAX_ONNX_BYTES + 1))
    # audit_one must not even be consulted for an oversize file.
    monkeypatch.setattr(
        validator,
        "audit_one",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises(validator.ValidationError, match="サイズ超過"):
        validator.validate_onnx_file(p)


def test_files_keep_soft_and_raise_only_on_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """soft 混在は raise せず全件返す。hard が 1 件でもあれば raise。"""
    soft = _onnx(tmp_path, "task001.onnx")
    ok = _onnx(tmp_path, "task002.onnx")

    def fake_audit(path: str, *_a: object, **_k: object) -> dict[str, Any]:
        return (
            _audit("score_error:x", None, 0.0)
            if path.endswith("task001.onnx")
            else _audit("ok", 10, 1.0)
        )

    monkeypatch.setattr(validator, "audit_one", fake_audit)
    results = validator.validate_onnx_files([soft, ok])
    assert len(results) == 2
    assert sum(1 for r in results if not r.scorable) == 1
