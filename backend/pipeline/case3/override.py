"""override-blend: 基準バンドル（frank7166, 実LB 7166）を土台に、各タスクで
**ローカル厳密検証で正答かつ cost 厳密減** の ONNX のみを差し替える。

重要: 基準バンドルにはローカル strict shape-inference で 0 点になるが Kaggle 実採点では
得点するタスクがある（frank7166 で 8 個）。これらを誤って捨てると実 LB が落ちるため、
**override は「候補がローカル正答 かつ 基準より cost が小さい」場合のみ**行い、それ以外は
基準 ONNX をそのまま残す。基準の onnx は無条件でコピーする（タスクを落とさない）。
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
class Override:
    task: int
    source: str
    base_cost: int | None
    new_cost: int


def _examples_for(task_dir: Path, num: int) -> dict[str, object] | None:
    p = task_dir / f"task{num:03d}.json"
    if not p.is_file():
        return None
    data: dict[str, object] = json.loads(p.read_text())
    return data


def _audit_cost(onnx_path: Path, examples: dict[str, object] | None) -> int | None:
    """ローカル正答なら cost、そうでなければ None。"""
    res = audit_one(str(onnx_path), examples, run_correctness=examples is not None)
    if res["status"] == "ok" and res["n_fail"] == 0 and res["cost"] is not None:
        return int(res["cost"])
    return None


def override_blend(
    base_dir: Path,
    candidate_dirs: list[Path],
    task_dir: Path,
    out_dir: Path,
) -> list[Override]:
    """base_dir を土台に、候補が正答かつ cost 減のタスクのみ差し替えて out_dir を作る。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides: list[Override] = []
    base_files = sorted(base_dir.glob("task*.onnx"))
    for bf in base_files:
        num = int("".join(ch for ch in bf.stem if ch.isdigit())[:3])
        examples = _examples_for(task_dir, num)
        # base を無条件採用（タスクを落とさない）
        shutil.copyfile(bf, out_dir / bf.name)
        base_cost = _audit_cost(
            bf, examples
        )  # None = ローカル不可（だが Kaggle で得点しうる）
        best_cost = base_cost
        best_src = bf
        best_name = "base"
        for cd in candidate_dirs:
            cand = cd / bf.name
            if not cand.is_file():
                matches = list(cd.rglob(bf.name))
                if not matches:
                    continue
                cand = matches[0]
            c = _audit_cost(cand, examples)
            # 差し替えは「候補がローカル正答 かつ base よりローカル cost が小さい」場合のみ。
            # base_cost が None（ローカル不可）のタスクは触らない（Kaggle 得点を守る）。
            if c is None or base_cost is None or c >= base_cost:
                continue
            if best_cost is None or c < best_cost:
                best_cost = c
                best_src = cand
                best_name = cd.name
        if best_name != "base":
            shutil.copyfile(best_src, out_dir / bf.name)
            overrides.append(Override(num, best_name, base_cost, best_cost or 0))
            logger.info(
                "override t%d <- %s cost %s->%d", num, best_name, base_cost, best_cost
            )
    return overrides
