"""Build a case2 override bundle on top of the case1 baseline.

For every task we try the case2 DSL solver. A candidate ONNX *overrides* the
baseline's ``taskNNN.onnx`` only when, judged by the official-scorer mirror
(``src/evaluate.audit_one`` with correctness ON over all train+test+arc-gen
pairs), it is **exactly correct** (``n_fail == 0``) **and strictly cheaper**
than the baseline. Otherwise the baseline file is kept verbatim.

This guarantees the bundle never regresses below the baseline locally, and —
because correctness is verified on hundreds of arc-gen pairs per task — an
override that wins locally is strong evidence it captures the true rule and so
holds on the hidden private split too.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import onnx

from evaluate import audit_one

from . import solver

logger = logging.getLogger(__name__)

MAX_TASKS = 400


@dataclass(frozen=True)
class TaskOutcome:
    """Per-task build result."""

    task: int
    primitive: str | None
    baseline_cost: int | None
    baseline_points: float
    case2_cost: int | None
    case2_points: float | None
    overridden: bool


@dataclass(frozen=True)
class AuditResult:
    """The subset of ``audit_one`` fields the build pipeline needs."""

    cost: int | None
    points: float
    n_fail: int
    status: str


def _audit(onnx_path: Path, examples: dict[str, object] | None) -> AuditResult:
    """audit_one inside a temp cwd so the ORT profile file is contained."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            res = audit_one(
                str(onnx_path.resolve()),
                examples,
                run_correctness=examples is not None,
            )
        finally:
            os.chdir(cwd)
    cost = res["cost"]
    pts = res["points"]
    return AuditResult(
        cost=int(cost) if cost is not None else None,
        points=float(pts) if pts is not None else 0.0,
        n_fail=int(res.get("n_fail", 0)),
        status=str(res.get("status", "")),
    )


def build_override_bundle(
    baseline_dir: Path,
    task_dir: Path,
    out_dir: Path,
) -> list[TaskOutcome]:
    """Copy baseline ONNX into ``out_dir``, overriding where case2 wins.

    Returns one :class:`TaskOutcome` per task that had a baseline file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[TaskOutcome] = []

    for n in range(1, MAX_TASKS + 1):
        base_onnx = baseline_dir / f"task{n:03d}.onnx"
        if not base_onnx.is_file():
            continue
        task_json = task_dir / f"task{n:03d}.json"
        examples = json.loads(task_json.read_text()) if task_json.is_file() else None

        base_res = _audit(base_onnx, examples)
        base_cost = base_res.cost
        base_pts = base_res.points

        dst = out_dir / f"task{n:03d}.onnx"
        shutil.copyfile(base_onnx, dst)

        sol = solver.solve(task_json) if task_json.is_file() else None
        case2_cost: int | None = None
        case2_pts: float | None = None
        overridden = False

        if sol is not None:
            cand = out_dir / f"task{n:03d}.cand.onnx"
            onnx.save(sol.model, cand)
            cres = _audit(cand, examples)
            case2_cost = cres.cost
            case2_pts = cres.points
            exact = cres.status == "ok" and cres.n_fail == 0
            # Only override when BOTH the base and the candidate are locally
            # scorable AND the candidate is strictly cheaper. A base that scores
            # 0 / errors locally is almost always a FALSE NEGATIVE: the local
            # fallback scorer can't run some ops (MaxUnpool, neg-pad ConvTranspose)
            # that Kaggle's official scorer handles, so the base actually works on
            # the hidden set. Overriding those regresses the LB (empirically -5
            # on task347). Never trust a local base failure as a win opportunity.
            cheaper = (
                case2_cost is not None
                and base_cost is not None
                and base_res.status == "ok"
                and base_res.n_fail == 0
                and case2_cost < base_cost
            )
            if exact and cheaper:
                shutil.copyfile(cand, dst)
                overridden = True
            cand.unlink()

        outcomes.append(
            TaskOutcome(
                task=n,
                primitive=sol.name if sol else None,
                baseline_cost=base_cost,
                baseline_points=base_pts,
                case2_cost=case2_cost,
                case2_points=case2_pts,
                overridden=overridden,
            )
        )
        if overridden:
            logger.info(
                "task%03d overridden by %s: cost %s -> %s",
                n,
                sol.name if sol else "?",
                base_cost,
                case2_cost,
            )

    return outcomes


def summarize(outcomes: list[TaskOutcome]) -> dict[str, float]:
    """Return aggregate stats for a build run."""
    n_over = sum(1 for o in outcomes if o.overridden)
    base_total = sum(o.baseline_points for o in outcomes)
    new_total = sum(
        (o.case2_points if o.overridden and o.case2_points else o.baseline_points)
        for o in outcomes
    )
    return {
        "tasks": len(outcomes),
        "overridden": n_over,
        "baseline_total": round(base_total, 2),
        "new_total": round(new_total, 2),
        "gain": round(new_total - base_total, 2),
    }
