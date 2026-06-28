"""per-task MAX-blend: 複数バンドルから各タスク最小 cost の正答 ONNX を選ぶ。

公式スコアラミラー（``src/evaluate.audit_one``）で全 example 厳密検証し、正答かつ
cost 最小の ONNX をタスクごとに採用する。正答しない ONNX は採用しない（不正解は
0 点、欠損も 0 点だが malformed zip を避けるため不正解は出さない）。
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evaluate.scorer import audit_one  # noqa: E402


@dataclass(frozen=True)
class Pick:
    task: int
    source: str
    cost: int
    points: float
    onnx_path: Path
    status: str


def _examples_for(task_dir: Path, num: int) -> dict[str, object] | None:
    p = task_dir / f"task{num:03d}.json"
    if not p.is_file():
        return None
    data: dict[str, object] = json.loads(p.read_text())
    return data


def _candidate_dirs(bundle_dirs: list[Path]) -> dict[int, list[tuple[str, Path]]]:
    """各タスク番号 -> [(source_label, onnx_path)] を収集（再帰探索）。"""
    cands: dict[int, list[tuple[str, Path]]] = {}
    for bd in bundle_dirs:
        label = bd.name
        for op in sorted(bd.rglob("task*.onnx")):
            digits = "".join(ch for ch in op.stem if ch.isdigit())[:3]
            if not digits:
                continue
            num = int(digits)
            cands.setdefault(num, []).append((f"{label}/{op.parent.name}", op))
    return cands


def blend(
    bundle_dirs: list[Path],
    task_dir: Path,
    out_dir: Path,
    run_correctness: bool = True,
) -> list[Pick]:
    """全バンドルを走査し、タスクごとに最小 cost の正答 ONNX を out_dir に確定。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = _candidate_dirs(bundle_dirs)
    picks: list[Pick] = []
    for num in sorted(cands):
        examples = _examples_for(task_dir, num)
        best: Pick | None = None
        for source, op in cands[num]:
            res = audit_one(
                str(op),
                examples,
                run_correctness=run_correctness and examples is not None,
            )
            ok = (
                res["status"] == "ok"
                and res["n_fail"] == 0
                and res["points"] is not None
            )
            if not ok:
                continue
            cost = int(res["cost"])
            if best is None or cost < best.cost:
                best = Pick(num, source, cost, float(res["points"]), op, res["status"])
        if best is not None:
            shutil.copyfile(best.onnx_path, out_dir / f"task{num:03d}.onnx")
            picks.append(best)
            logger.info(
                "task %d <- %s cost=%d pts=%.3f",
                num,
                best.source,
                best.cost,
                best.points,
            )
    return picks


def write_manifest(picks: list[Pick], path: Path) -> None:
    rows = [
        {
            "task": p.task,
            "source": p.source,
            "cost": p.cost,
            "points": round(p.points, 4),
        }
        for p in sorted(picks, key=lambda x: x.task)
    ]
    path.write_text(json.dumps(rows, indent=2))
