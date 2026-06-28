"""CSV 駆動の高速 MAX-blend。

各バンドルを一度だけ ``audit_dir`` で採点した CSV（task, cost, points, n_fail,
status を含む）を読み、タスクごとに **正答（status==ok かつ n_fail==0）かつ cost 最小**
のバンドルを選んで ONNX をコピーする。再採点しないので blend.py より桁違いに速い。

CSV は ``src/evaluate.audit_dir(out_csv=...)`` が出力する形式を前提（列:
task,onnx,points,cost,params,memory,filesize,n_pass,n_fail,status）。
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    name: str
    onnx_dir: Path
    csv_path: Path


@dataclass(frozen=True)
class Pick:
    task: int
    source: str
    cost: int
    points: float


def _load_csv(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if not r.get("task"):
                continue
            rows[int(r["task"])] = r
    return rows


def _ok(row: dict[str, str]) -> bool:
    return (
        row.get("status") == "ok"
        and (row.get("n_fail") in (None, "", "0"))
        and row.get("cost") not in (None, "")
    )


def csv_blend(sources: list[Source], out_dir: Path) -> list[Pick]:
    """CSV から各タスク最小 cost の正答バンドルを選び ONNX を out_dir に確定。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {s.name: _load_csv(s.csv_path) for s in sources}
    by_name = {s.name: s for s in sources}
    all_tasks = sorted({t for tbl in tables.values() for t in tbl})
    picks: list[Pick] = []
    for num in all_tasks:
        best: tuple[str, int, float] | None = None
        for s in sources:
            row = tables[s.name].get(num)
            if row is None or not _ok(row):
                continue
            cost = int(row["cost"])
            pts = float(row["points"]) if row["points"] else 0.0
            if best is None or cost < best[1]:
                best = (s.name, cost, pts)
        if best is None:
            continue
        name, cost, pts = best
        src_onnx = by_name[name].onnx_dir / f"task{num:03d}.onnx"
        if not src_onnx.is_file():
            # 一部バンドルは onnx をサブディレクトリに持つ
            matches = list(by_name[name].onnx_dir.rglob(f"task{num:03d}.onnx"))
            if not matches:
                continue
            src_onnx = matches[0]
        shutil.copyfile(src_onnx, out_dir / f"task{num:03d}.onnx")
        picks.append(Pick(num, name, cost, pts))
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
