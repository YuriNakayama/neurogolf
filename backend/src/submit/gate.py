"""Pre-submit gate for one-task NeuroGolf candidate ONNX files."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import onnx

from evaluate import audit_one

TASK_RE = re.compile(r"task(\d{3})")

HARD_RISK_TASKS = {
    17: "task017 candidate-row pruning regressed hidden/Public validation",
    44: "case532 isolated task044 bottom-match alias regressed Public validation",
    101: "task101 local-equivalence aliases have regressed Public validation",
    135: "case541 unscorable-baseline crop replacement regressed Public validation",
    192: (
        "case544 task192 score-weight repair passed local but regressed "
        "Public validation"
    ),
    286: "broad task286 bitwise-chain equivalence bundle regressed Public validation",
    305: "task305 suffix-row prune was contaminated by a regressing filler bundle",
    366: "task366 broad alias bundle was contaminated by a regressing filler bundle",
}

REVIEW_RISK_TASKS = {
    2: "task002 bit-topology aliases have prior Public-regression risk",
    18: "case514 clean micro-bundle member produced Kaggle ERROR",
    35: "case395 batch-ERROR family member",
    37: "case395 batch-ERROR family member",
    46: "case475 regressing micro bundle member",
    64: "case475 regressing micro bundle member",
    90: "case414 unisolated micro bundle member",
    92: "case514 clean micro-bundle member produced Kaggle ERROR",
    96: "case514 clean micro-bundle member produced Kaggle ERROR",
    110: "case475 regressing micro bundle member",
    173: "case510 task173 grid-score dtype shrink produced Kaggle ERROR",
    175: "case509 guarded task175 index repair regressed Public validation",
    209: "case414 unisolated micro bundle member",
    263: "case514 clean micro-bundle member produced Kaggle ERROR",
    293: "case414 unisolated micro bundle member",
    308: "case414 unisolated micro bundle member",
    338: "standalone micro did not move Public LB",
    342: "standalone micro did not move Public LB",
    370: "case414 unisolated micro bundle member",
    378: "case475 regressing micro bundle member",
    392: "standalone micro did not move Public LB",
}

TOPK_RUNTIME_RISK_TASKS = {
    233: "task233 uint8 TopK family produced Kaggle ERROR",
    285: "task285 uint8 TopK family produced Kaggle ERROR",
}

INDEX_RUNTIME_RISK_TASKS = {
    131: "task131 ScatterND/Gather index dtype surgery failed local/runtime gates",
    284: "task284 ScatterND index narrowing failed local/runtime gates",
    371: "task371 ScatterND index narrowing failed local/runtime gates",
}

INDEX_RUNTIME_OPS = {
    "Gather",
    "GatherElements",
    "GatherND",
    "ScatterElements",
    "ScatterND",
}


@dataclass(frozen=True)
class CandidateGate:
    """Decision for one candidate compared with the accepted baseline."""

    task: int
    baseline_cost: int | None
    candidate_cost: int | None
    gain: float | None
    n_fail: int
    status: str
    functions: int
    forbidden_ops: tuple[str, ...]
    decision: str
    reason: str


@dataclass(frozen=True)
class BundleGate:
    """Decision for changed candidates in a submission bundle."""

    changed: tuple[CandidateGate, ...]
    allowed_review: bool
    allowed_micro_bundle: bool = False
    micro_bundle_gain: float = 0.015
    micro_bundle_max_tasks: int = 5

    @property
    def accepted_decisions(self) -> tuple[str, ...]:
        decisions = ["submit-candidate"]
        if self.allowed_review:
            decisions.extend(["review-mid-gain", "review-known-risk"])
        return tuple(decisions)

    @property
    def blocked(self) -> tuple[CandidateGate, ...]:
        if self.is_micro_bundle:
            return ()
        accepted = set(self.accepted_decisions)
        blocked = [gate for gate in self.changed if gate.decision not in accepted]
        review = [gate for gate in self.changed if gate.decision.startswith("review-")]
        if self.allowed_review and len(review) > 1:
            blocked.extend(review)
        return tuple(blocked)

    @property
    def decision(self) -> str:
        if not self.changed:
            return "blocked-no-changes"
        if self.is_micro_bundle:
            return "submit-micro-bundle"
        if self.blocked:
            return "blocked-bundle-gate"
        return "submit-bundle"

    @property
    def total_gain(self) -> float:
        return sum(gate.gain or 0.0 for gate in self.changed)

    @property
    def is_micro_bundle(self) -> bool:
        if not self.allowed_micro_bundle:
            return False
        if not self.changed or len(self.changed) > self.micro_bundle_max_tasks:
            return False
        decisions = {gate.decision for gate in self.changed}
        if not decisions <= {"bank-low-gain", "submit-candidate"}:
            return False
        return self.total_gain >= self.micro_bundle_gain


@dataclass(frozen=True)
class CandidateScan:
    """Result from screening one scratch candidate file."""

    path: Path
    gate: CandidateGate


def task_num_from_path(path: Path) -> int:
    """Extract the task number from a `taskNNN.onnx` path."""

    match = TASK_RE.search(path.name)
    if match is None:
        raise ValueError(f"ファイル名に taskNNN が含まれていません: {path.name}")
    return int(match.group(1))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def changed_candidate_files(files: list[Path], baseline_dir: Path) -> list[Path]:
    """Return candidate files whose bytes differ from the baseline task file."""

    changed: list[Path] = []
    for candidate in files:
        task = task_num_from_path(candidate)
        baseline = baseline_dir / f"task{task:03d}.onnx"
        if not baseline.is_file() or _digest(candidate) != _digest(baseline):
            changed.append(candidate)
    return changed


def _audit_clean(path: Path, examples: dict[str, Any]) -> dict[str, Any]:
    """Run audit_one while containing profiler output in a temporary directory."""

    resolved = path.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path.cwd()
        try:
            os.chdir(tmp)
            return audit_one(str(resolved), examples, run_correctness=True)
        finally:
            os.chdir(cwd)


def _model_static_risks(path: Path) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    model = onnx.load(str(path))
    forbidden = {
        "Loop",
        "Scan",
        "NonZero",
        "Unique",
        "Script",
        "Function",
        "Compress",
    }
    op_types = sorted({node.op_type for node in model.graph.node})
    found = sorted(set(op_types) & forbidden)
    return len(model.functions), tuple(found), tuple(op_types)


def _known_risk_reason(task: int, op_types: tuple[str, ...]) -> tuple[str, str] | None:
    if task in HARD_RISK_TASKS:
        return "blocked-known-risk", HARD_RISK_TASKS[task]
    if task in TOPK_RUNTIME_RISK_TASKS and "TopK" in op_types:
        return "review-known-risk", TOPK_RUNTIME_RISK_TASKS[task]
    if task in INDEX_RUNTIME_RISK_TASKS and (set(op_types) & INDEX_RUNTIME_OPS):
        return "review-known-risk", INDEX_RUNTIME_RISK_TASKS[task]
    if task in REVIEW_RISK_TASKS:
        return "review-known-risk", REVIEW_RISK_TASKS[task]
    return None


def evaluate_candidate_gate(
    baseline: Path,
    candidate: Path,
    task_json: Path,
    *,
    submit_gain: float = 0.020,
    mid_gain: float = 0.010,
) -> CandidateGate:
    """Compare a candidate against baseline and classify submit readiness."""

    task = task_num_from_path(candidate)
    examples = json.loads(task_json.read_text())
    base = _audit_clean(baseline, examples)
    cand = _audit_clean(candidate, examples)
    functions, forbidden_ops, op_types = _model_static_risks(candidate)

    baseline_cost = base["cost"]
    candidate_cost = cand["cost"]
    status = str(cand["status"])
    n_fail = int(cand["n_fail"])
    gain = None
    if baseline_cost is not None and candidate_cost is not None and candidate_cost > 0:
        gain = math.log(int(baseline_cost) / int(candidate_cost))

    if base["status"] != "ok" or base["cost"] is None:
        decision = "blocked-baseline"
        reason = f"baseline audit failed: {base['status']}"
    elif status != "ok" or n_fail != 0:
        decision = "blocked-local-fail"
        reason = f"candidate local audit failed: status={status} n_fail={n_fail}"
    elif functions:
        decision = "blocked-functions"
        reason = f"candidate contains {functions} ONNX functions"
    elif forbidden_ops:
        decision = "blocked-forbidden-op"
        reason = "candidate contains forbidden ops: " + ", ".join(forbidden_ops)
    elif candidate_cost is None or gain is None:
        decision = "blocked-unscorable"
        reason = "candidate cost could not be estimated"
    elif int(candidate_cost) >= int(baseline_cost):
        decision = "blocked-no-gain"
        reason = f"candidate does not reduce cost: {baseline_cost}->{candidate_cost}"
    elif known_risk := _known_risk_reason(task, op_types):
        decision, known_reason = known_risk
        reason = f"{known_reason}; requires a new isolated repair plan before submit"
    elif gain >= submit_gain:
        decision = "submit-candidate"
        reason = f"gain {gain:.6f} >= submit threshold {submit_gain:.6f}"
    elif gain >= mid_gain:
        decision = "review-mid-gain"
        reason = f"gain {gain:.6f} requires prior Kaggle-valid evidence"
    else:
        decision = "bank-low-gain"
        reason = f"gain {gain:.6f} is below bank threshold {mid_gain:.6f}"

    return CandidateGate(
        task=task,
        baseline_cost=int(baseline_cost) if baseline_cost is not None else None,
        candidate_cost=int(candidate_cost) if candidate_cost is not None else None,
        gain=gain,
        n_fail=n_fail,
        status=status,
        functions=functions,
        forbidden_ops=forbidden_ops,
        decision=decision,
        reason=reason,
    )


def evaluate_bundle_gate(
    files: list[Path],
    baseline_dir: Path,
    task_dir: Path,
    *,
    submit_gain: float = 0.020,
    mid_gain: float = 0.010,
    allow_review: bool = False,
    allow_micro_bundle: bool = False,
    micro_bundle_gain: float = 0.015,
    micro_bundle_max_tasks: int = 5,
) -> BundleGate:
    """Gate every changed task in a candidate submission bundle."""

    changed = changed_candidate_files(files, baseline_dir)
    results: list[CandidateGate] = []
    for candidate in changed:
        task = task_num_from_path(candidate)
        baseline = baseline_dir / f"task{task:03d}.onnx"
        task_json = task_dir / f"task{task:03d}.json"
        if not baseline.is_file():
            results.append(
                CandidateGate(
                    task=task,
                    baseline_cost=None,
                    candidate_cost=None,
                    gain=None,
                    n_fail=0,
                    status="missing-baseline",
                    functions=0,
                    forbidden_ops=(),
                    decision="blocked-missing-baseline",
                    reason=f"baseline is missing: {baseline}",
                )
            )
            continue
        if not task_json.is_file():
            results.append(
                CandidateGate(
                    task=task,
                    baseline_cost=None,
                    candidate_cost=None,
                    gain=None,
                    n_fail=0,
                    status="missing-task-json",
                    functions=0,
                    forbidden_ops=(),
                    decision="blocked-missing-task-json",
                    reason=f"task json is missing: {task_json}",
                )
            )
            continue
        results.append(
            evaluate_candidate_gate(
                baseline,
                candidate,
                task_json,
                submit_gain=submit_gain,
                mid_gain=mid_gain,
            )
        )
    return BundleGate(
        changed=tuple(results),
        allowed_review=allow_review,
        allowed_micro_bundle=allow_micro_bundle,
        micro_bundle_gain=micro_bundle_gain,
        micro_bundle_max_tasks=micro_bundle_max_tasks,
    )


def _blocked_scan_result(
    candidate: Path,
    *,
    decision: str,
    status: str,
    reason: str,
) -> CandidateScan:
    try:
        task = task_num_from_path(candidate)
    except ValueError:
        task = 0
    return CandidateScan(
        path=candidate,
        gate=CandidateGate(
            task=task,
            baseline_cost=None,
            candidate_cost=None,
            gain=None,
            n_fail=0,
            status=status,
            functions=0,
            forbidden_ops=(),
            decision=decision,
            reason=reason,
        ),
    )


def scan_candidate_files(
    files: list[Path],
    baseline_dir: Path,
    task_dir: Path,
    *,
    submit_gain: float = 0.020,
    mid_gain: float = 0.010,
) -> tuple[CandidateScan, ...]:
    """Gate scratch candidates independently, converting failures into rows."""

    results: list[CandidateScan] = []
    for candidate in files:
        try:
            task = task_num_from_path(candidate)
        except ValueError as exc:
            results.append(
                _blocked_scan_result(
                    candidate,
                    decision="blocked-bad-filename",
                    status="bad-filename",
                    reason=str(exc),
                )
            )
            continue
        baseline = baseline_dir / f"task{task:03d}.onnx"
        task_json = task_dir / f"task{task:03d}.json"
        if not baseline.is_file():
            results.append(
                _blocked_scan_result(
                    candidate,
                    decision="blocked-missing-baseline",
                    status="missing-baseline",
                    reason=f"baseline is missing: {baseline}",
                )
            )
            continue
        if not task_json.is_file():
            results.append(
                _blocked_scan_result(
                    candidate,
                    decision="blocked-missing-task-json",
                    status="missing-task-json",
                    reason=f"task json is missing: {task_json}",
                )
            )
            continue
        try:
            gate = evaluate_candidate_gate(
                baseline,
                candidate,
                task_json,
                submit_gain=submit_gain,
                mid_gain=mid_gain,
            )
        except Exception as exc:
            results.append(
                _blocked_scan_result(
                    candidate,
                    decision="blocked-exception",
                    status=type(exc).__name__,
                    reason=str(exc),
                )
            )
            continue
        results.append(CandidateScan(path=candidate, gate=gate))
    return tuple(results)
