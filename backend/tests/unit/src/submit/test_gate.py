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


def _task_json_dir(tmp_path: Path, *tasks: int) -> Path:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for task in tasks:
        (task_dir / f"task{task:03d}.json").write_text(json.dumps({"train": []}))
    return task_dir


def _patch_static_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    functions: int = 0,
    forbidden: tuple[str, ...] = (),
    op_types: tuple[str, ...] = (),
) -> None:
    if not op_types:
        op_types = ("TopK",) if "TopK" in forbidden else ()
    monkeypatch.setattr(
        gate,
        "_model_static_risks",
        lambda _p: (functions, forbidden, op_types),
    )


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


def test_bank_low_gain_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=995)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx", _onnx(tmp_path), _task_json(tmp_path)
    )

    assert result.decision == "bank-low-gain"


def test_mid_gain_requires_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_known_hidden_risk_blocks_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=900)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx",
        _onnx(tmp_path, "task017_prune.onnx"),
        _task_json(tmp_path),
    )

    assert result.decision == "blocked-known-risk"


def test_known_runtime_risk_requires_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=900)
    monkeypatch.setattr(gate, "_model_static_risks", lambda _p: (0, (), ("TopK",)))

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx",
        _onnx(tmp_path, "task285_topk.onnx"),
        _task_json(tmp_path),
    )

    assert result.decision == "review-known-risk"


def test_index_runtime_risk_requires_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=900)
    _patch_static_model(monkeypatch, op_types=("ScatterND",))

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx",
        _onnx(tmp_path, "task284_index_i32.onnx"),
        _task_json(tmp_path),
    )

    assert result.decision == "review-known-risk"


def test_known_unchanged_micro_requires_review_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=900)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx",
        _onnx(tmp_path, "task342_larger_repair.onnx"),
        _task_json(tmp_path),
    )

    assert result.decision == "review-known-risk"


def test_case475_regression_member_requires_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_audits(monkeypatch, base_cost=1000, cand_cost=990)
    _patch_static_model(monkeypatch)

    result = gate.evaluate_candidate_gate(
        tmp_path / "baseline.onnx",
        _onnx(tmp_path, "task110_case475_micro.onnx"),
        _task_json(tmp_path),
    )

    assert result.decision == "review-known-risk"


def test_changed_candidate_files_returns_only_byte_differences(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    (baseline_dir / "task001.onnx").write_bytes(b"same")
    (candidate_dir / "task001.onnx").write_bytes(b"same")
    (baseline_dir / "task002.onnx").write_bytes(b"old")
    changed = candidate_dir / "task002.onnx"
    changed.write_bytes(b"new")

    result = gate.changed_candidate_files(
        [candidate_dir / "task001.onnx", changed], baseline_dir
    )

    assert result == [changed]


def test_bundle_gate_blocks_low_gain_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline = baseline_dir / "task001.onnx"
    candidate = candidate_dir / "task001.onnx"
    baseline.write_bytes(b"old")
    candidate.write_bytes(b"new")
    task_dir = _task_json_dir(tmp_path, 1)
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {
            "status": "ok",
            "cost": 1000 if path == baseline else 995,
            "n_fail": 0,
        },
    )
    _patch_static_model(monkeypatch)

    result = gate.evaluate_bundle_gate([candidate], baseline_dir, task_dir)

    assert result.decision == "blocked-bundle-gate"
    assert result.blocked[0].decision == "bank-low-gain"


def test_bundle_gate_accepts_explicit_micro_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline1 = baseline_dir / "task001.onnx"
    candidate1 = candidate_dir / "task001.onnx"
    baseline2 = baseline_dir / "task003.onnx"
    candidate2 = candidate_dir / "task003.onnx"
    baseline1.write_bytes(b"old1")
    candidate1.write_bytes(b"new1")
    baseline2.write_bytes(b"old2")
    candidate2.write_bytes(b"new2")
    task_dir = _task_json_dir(tmp_path, 1, 3)
    costs = {baseline1: 1000, baseline2: 1000, candidate1: 992, candidate2: 992}
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {"status": "ok", "cost": costs[path], "n_fail": 0},
    )
    _patch_static_model(monkeypatch)

    blocked = gate.evaluate_bundle_gate(
        [candidate1, candidate2], baseline_dir, task_dir
    )
    allowed = gate.evaluate_bundle_gate(
        [candidate1, candidate2],
        baseline_dir,
        task_dir,
        allow_micro_bundle=True,
        micro_bundle_gain=0.015,
    )

    assert blocked.decision == "blocked-bundle-gate"
    assert allowed.decision == "submit-micro-bundle"
    assert allowed.blocked == ()
    assert allowed.total_gain > 0.015


def test_micro_bundle_gate_keeps_review_risk_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline1 = baseline_dir / "task001.onnx"
    candidate1 = candidate_dir / "task001.onnx"
    baseline2 = baseline_dir / "task090.onnx"
    candidate2 = candidate_dir / "task090.onnx"
    baseline1.write_bytes(b"old1")
    candidate1.write_bytes(b"new1")
    baseline2.write_bytes(b"old2")
    candidate2.write_bytes(b"new2")
    task_dir = _task_json_dir(tmp_path, 1, 90)
    costs = {baseline1: 1000, baseline2: 1000, candidate1: 992, candidate2: 900}
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {"status": "ok", "cost": costs[path], "n_fail": 0},
    )
    _patch_static_model(monkeypatch)

    result = gate.evaluate_bundle_gate(
        [candidate1, candidate2],
        baseline_dir,
        task_dir,
        allow_micro_bundle=True,
        micro_bundle_gain=0.015,
    )

    assert result.decision == "blocked-bundle-gate"
    assert [blocked.decision for blocked in result.blocked] == [
        "bank-low-gain",
        "review-known-risk",
    ]


def test_micro_bundle_gate_respects_task_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    task_dir = _task_json_dir(tmp_path, 1, 3, 4)
    candidates = []
    costs = {}
    for task in (1, 3, 4):
        baseline = baseline_dir / f"task{task:03d}.onnx"
        candidate = candidate_dir / f"task{task:03d}.onnx"
        baseline.write_bytes(f"old{task}".encode())
        candidate.write_bytes(f"new{task}".encode())
        costs[baseline] = 1000
        costs[candidate] = 992
        candidates.append(candidate)
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {"status": "ok", "cost": costs[path], "n_fail": 0},
    )
    _patch_static_model(monkeypatch)

    result = gate.evaluate_bundle_gate(
        candidates,
        baseline_dir,
        task_dir,
        allow_micro_bundle=True,
        micro_bundle_gain=0.015,
        micro_bundle_max_tasks=2,
    )

    assert result.decision == "blocked-bundle-gate"
    assert [blocked.decision for blocked in result.blocked] == [
        "bank-low-gain",
        "bank-low-gain",
        "bank-low-gain",
    ]


def test_bundle_gate_accepts_submit_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline = baseline_dir / "task001.onnx"
    candidate = candidate_dir / "task001.onnx"
    baseline.write_bytes(b"old")
    candidate.write_bytes(b"new")
    task_dir = _task_json_dir(tmp_path, 1)
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {
            "status": "ok",
            "cost": 1000 if path == baseline else 900,
            "n_fail": 0,
        },
    )
    _patch_static_model(monkeypatch)

    result = gate.evaluate_bundle_gate([candidate], baseline_dir, task_dir)

    assert result.decision == "submit-bundle"
    assert result.blocked == ()
    assert result.total_gain > 0.02


def test_bundle_gate_review_requires_explicit_allow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline = baseline_dir / "task001.onnx"
    candidate = candidate_dir / "task001.onnx"
    baseline.write_bytes(b"old")
    candidate.write_bytes(b"new")
    task_dir = _task_json_dir(tmp_path, 1)
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {
            "status": "ok",
            "cost": 1000 if path == baseline else 985,
            "n_fail": 0,
        },
    )
    _patch_static_model(monkeypatch)

    blocked = gate.evaluate_bundle_gate([candidate], baseline_dir, task_dir)
    allowed = gate.evaluate_bundle_gate(
        [candidate], baseline_dir, task_dir, allow_review=True
    )

    assert blocked.decision == "blocked-bundle-gate"
    assert blocked.blocked[0].decision == "review-mid-gain"
    assert allowed.decision == "submit-bundle"


def test_bundle_gate_allows_only_one_review_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline1 = baseline_dir / "task001.onnx"
    candidate1 = candidate_dir / "task001.onnx"
    baseline2 = baseline_dir / "task003.onnx"
    candidate2 = candidate_dir / "task003.onnx"
    baseline1.write_bytes(b"old1")
    candidate1.write_bytes(b"new1")
    baseline2.write_bytes(b"old2")
    candidate2.write_bytes(b"new2")
    task_dir = _task_json_dir(tmp_path, 1, 3)
    monkeypatch.setattr(
        gate,
        "_audit_clean",
        lambda path, _examples: {
            "status": "ok",
            "cost": 1000 if path in {baseline1, baseline2} else 985,
            "n_fail": 0,
        },
    )
    _patch_static_model(monkeypatch)

    result = gate.evaluate_bundle_gate(
        [candidate1, candidate2], baseline_dir, task_dir, allow_review=True
    )

    assert result.decision == "blocked-bundle-gate"
    assert [blocked.decision for blocked in result.blocked] == [
        "review-mid-gain",
        "review-mid-gain",
    ]


def test_bundle_gate_blocks_missing_baseline(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    candidate = candidate_dir / "task001.onnx"
    candidate.write_bytes(b"new")
    task_dir = _task_json_dir(tmp_path, 1)

    result = gate.evaluate_bundle_gate([candidate], baseline_dir, task_dir)

    assert result.decision == "blocked-bundle-gate"
    assert result.blocked[0].decision == "blocked-missing-baseline"


def test_task_num_from_path_rejects_non_task_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="taskNNN"):
        gate.task_num_from_path(tmp_path / "candidate.onnx")


def test_task_num_from_path_accepts_descriptive_scratch_name(tmp_path: Path) -> None:
    assert gate.task_num_from_path(tmp_path / "task364_PA7_to_PA6.onnx") == 364
