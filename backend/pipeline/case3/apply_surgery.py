"""グラフサージェリ適用ドライバ: バンドルへ surgery パスを安全に適用する。

ベースバンドル（frank7166 等）の各 ``taskNNN.onnx`` に対し、``surgery.PASSES`` を
安全側→危険側の順に**チェイン適用**する。各パスの出力は ``src/evaluate.audit_one``
で採点し、

  - ``onnx.checker.check_model`` を通る、かつ
  - 全 example 正答（``n_fail == 0`` かつ ``points > 0``）、かつ
  - cost が**厳密に**減少（``cost_after < cost_before``）

の 3 条件を満たした場合のみ採用して次パスへ繋ぐ。1 つでも崩れたパスはスキップし、
直前の（採用済み）モデルを保持する。最終的に少しでも cost が減ったタスクのみ
出力先へ surgered ONNX を書き、それ以外は**元ファイルをそのままコピー**する。

ローカル採点が Kaggle 実 LB と一致しない実証があるため、このドライバは
**タスクを 1 つも落とさない**（元バンドルの全タスクが出力にも必ず存在する）。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import onnx

from evaluate.scorer import audit_one

from .surgery import PASSES, _load_task_json

logger = logging.getLogger(__name__)


def _cost(
    onnx_path: Path, examples: dict[str, object] | None
) -> tuple[float, int] | None:
    """(points, cost) を返す。不正答/unscorable は None。"""
    res = audit_one(str(onnx_path), examples, run_correctness=examples is not None)
    if res["points"] is None or res["points"] <= 0.0 or res["cost"] is None:
        return None
    if examples is not None and res["n_fail"] > 0:
        return None
    return float(res["points"]), int(res["cost"])


def surgeon_task(
    src_onnx: Path,
    examples: dict[str, object] | None,
    work_dir: Path,
) -> tuple[onnx.ModelProto, float, int] | None:
    """1 タスクへ全パスをチェイン適用。改善できれば (model, points, cost) を返す。

    改善できなければ None（呼び出し側は元ファイルをコピーする）。
    """
    base = _cost(src_onnx, examples)
    if base is None:
        return None
    cur_points, cur_cost = base
    cur_model = onnx.load(str(src_onnx))
    improved = False
    tmp = work_dir / src_onnx.name

    for name, fn in PASSES:
        try:
            cand = fn(cur_model)
            onnx.checker.check_model(cand)
        except Exception as e:
            logger.debug("%s: pass %s failed: %s", src_onnx.name, name, str(e)[:80])
            continue
        onnx.save(cand, str(tmp))
        scored = _cost(tmp, examples)
        if scored is None:
            continue
        pts, cost = scored
        if cost < cur_cost:
            cur_model, cur_points, cur_cost = cand, pts, cost
            improved = True
            logger.debug("%s: %s accepted -> cost %d", src_onnx.name, name, cost)

    if not improved:
        return None
    return cur_model, cur_points, cur_cost


def apply_surgery(
    base_dir: Path,
    task_dir: Path,
    out_dir: Path,
) -> dict[str, tuple[float, int, float, int]]:
    """base_dir の全 taskNNN.onnx へ surgery を適用し out_dir へ書く。

    返り値: {task_name: (points_before, cost_before, points_after, cost_after)}（改善分のみ）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    improvements: dict[str, tuple[float, int, float, int]] = {}
    onnx_files = sorted(base_dir.glob("task*.onnx"))
    logger.info("applying surgery to %d onnx files", len(onnx_files))

    for i, src in enumerate(onnx_files):
        tnum_digits = "".join(ch for ch in src.stem if ch.isdigit())[:3]
        examples = _load_task_json(task_dir, int(tnum_digits)) if tnum_digits else None
        before = _cost(src, examples)
        result = surgeon_task(src, examples, work_dir)
        dst = out_dir / src.name
        if result is None or before is None:
            shutil.copy2(src, dst)
        else:
            model, pts_after, cost_after = result
            onnx.save(model, str(dst))
            improvements[src.name] = (before[0], before[1], pts_after, cost_after)
        if (i + 1) % 25 == 0:
            logger.info(
                "  .. %d/%d done (%d improved)",
                i + 1,
                len(onnx_files),
                len(improvements),
            )

    shutil.rmtree(work_dir, ignore_errors=True)
    gain = sum(a[2] - a[0] for a in improvements.values())
    logger.info(
        "surgery done: %d/%d tasks improved, local points delta +%.2f",
        len(improvements),
        len(onnx_files),
        gain,
    )
    return improvements
