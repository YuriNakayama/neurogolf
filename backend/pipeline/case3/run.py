"""case3 オーケストレーション: 各タスクで候補 solver を試し、厳密検証して最小 cost を採る。

正答性・cost は ``src/evaluate`` の公式スコアラミラー（``audit_one``）で判定する
（case 独立だが scoring は競技固定の共有不変なので src/evaluate のみ依存）。
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import onnx

from .arc import Task, load_task
from .solvers import SOLVERS

logger = logging.getLogger(__name__)

# src/evaluate の公式スコアラミラーを使う（競技固定 scoring の共有不変）。
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evaluate.scorer import audit_one  # noqa: E402


@dataclass(frozen=True)
class Solved:
    task: int
    solver: str
    cost: int
    points: float
    onnx_path: Path


def _examples_dict(task: Task) -> dict[str, list[dict[str, list[list[int]]]]]:
    def conv(exs: tuple) -> list[dict[str, list[list[int]]]]:  # type: ignore[type-arg]
        return [{"input": e.input, "output": e.output} for e in exs]

    return {
        "train": conv(task.train),
        "test": conv(task.test),
        "arc-gen": conv(task.arc_gen),
    }


def solve_task(task: Task, work_dir: Path) -> Solved | None:
    """全 solver を試し、厳密正答かつ最小 cost の ONNX を保存して返す。"""
    examples = _examples_dict(task)
    best: Solved | None = None
    for name, solver in SOLVERS:
        try:
            model = solver(task)
        except Exception:
            logger.debug("solver %s raised on task %d", name, task.num, exc_info=True)
            continue
        if model is None:
            continue
        tmp = work_dir / f"_cand_{task.num:03d}_{name}.onnx"
        try:
            onnx.save(model, str(tmp))
        except Exception:
            continue
        res = audit_one(str(tmp), examples, run_correctness=True)
        if res["status"] == "ok" and res["n_fail"] == 0 and res["points"] is not None:
            cost = int(res["cost"])
            if best is None or cost < best.cost:
                final = work_dir / f"task{task.num:03d}.onnx"
                onnx.save(model, str(final))
                best = Solved(task.num, name, cost, float(res["points"]), final)
        tmp.unlink(missing_ok=True)
    return best


def run(task_dir: Path, out_dir: Path, tasks: list[int] | None = None) -> list[Solved]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nums = tasks or list(range(1, 401))
    solved: list[Solved] = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for n in nums:
            task = load_task(task_dir, n)
            s = solve_task(task, work)
            if s is not None:
                # 確定 ONNX を out_dir へ
                final = out_dir / f"task{n:03d}.onnx"
                onnx.save(onnx.load(str(s.onnx_path)), str(final))
                solved.append(Solved(s.task, s.solver, s.cost, s.points, final))
                logger.info(
                    "task %d solved by %s cost=%d pts=%.3f",
                    n,
                    s.solver,
                    s.cost,
                    s.points,
                )
    return solved


def write_manifest(solved: list[Solved], path: Path) -> None:
    rows = [
        {
            "task": s.task,
            "solver": s.solver,
            "cost": s.cost,
            "points": round(s.points, 4),
        }
        for s in sorted(solved, key=lambda x: x.task)
    ]
    path.write_text(json.dumps(rows, indent=2))
