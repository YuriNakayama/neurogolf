"""Tests for the pre-submit candidate gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from submit import gate


def _onnx(tmp_path: Path, name: str = "task001.onnx") -> Path:
    path = tmp_path / name
    path.write_bytes(b"onnx")
    return path


def _task_json(tmp_path: Path) -> Path:
    path = tmp_path / "task001.json"
    path.write_text(json.dumps({"train": []}))
    return path


def _patch_static_model(
    monkeypatch: pytest.MonkeyPatch, *, functions: int = 0, forbidden: tuple[str, ...] = ()
) -> None:
    monkeypatch.setattr(gate, "_functions_and_forbidden", lambda _p: (functions, forbidden))


def _patch_audits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_cost: int = 1000,
    cand_cost: int = 900,
    cand_status: str = "ok",
    n_fail: int = 0,
) -> None:
    def fake_audit(path: Path, _examples: dict[str, Any]) -> dict[str, Any]:
        if path.name == "baseline.onnx":
            return {"status": "ok", "cost": base_cost, "n_fail": 0}
        return {"status": cand_status, "cost": cand_cost, "n_fail": n_fail}

    monkeypatch.setattr(gate, "_audit_clean", fake_audit)


def test_submit_candidate_when_exact_and_gain_clears_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=900)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx", _onnx(tmp_path), _task_json(tmp_path)
    )

    assert result.decision == "submit-candidate"
    assert result.gain is not None
    assert result.gain > 0.02


def test_bank_low_gain_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=995)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx", _onnx(tmp_path), _task_json(tmp_path)
    )

    assert result.decision == "bank-low-gain"


def test_mid_gain_requires_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=985)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx", _onnx(tmp_path), _task_json(tmp_path)
    )

    assert result.decision == "review-mid-gain"


def test_local_fail_blocks_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, cand_status="INCORRECT", n_fail=1)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx", _onnx(tmp_path), _task_json(tmp_path)
    )

    assert result.decision == "blocked-local-fail"


def test_forbidden_op_blocks_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch)
    _patch_static_model(monkeypatch, forbidden=("Loop",))

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx", _onnx(tmp_path), _task_json(tmp_path)
    )

    assert result.decision == "blocked-forbidden-op"


def test_task_num_from_path_rejects_non_task_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="taskNNN"):
        gate.task_num_from_path(tmp_path / "candidate.onnx")


def test_task_num_from_path_accepts_descriptive_scratch_name(tmp_path: Path) -> None:
    assert gate.task_num_from_path(tmp_path / "task364_PA7_to_PA6.onnx") == 364
