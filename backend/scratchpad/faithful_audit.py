"""Faithful audit harness (onnx 1.20.0 / ort 1.24.1, offset +0.11).

Usage:
  rank   <onnx_dir> [csv]            cost-only worst-first ranking (no data)
  one    <onnx_path> <task_num>      correctness+cost for a single task
  total  <onnx_dir>                  full audit total (correctness+cost)
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, "src")
from evaluate.scorer import audit_one  # noqa: E402

LAKE = "../data/lake/neurogolf-2026"


def _load(tnum: int):
    p = os.path.join(LAKE, f"task{tnum:03d}.json")
    with open(p) as f:
        return json.load(f)


def cmd_rank(onnx_dir: str, out_csv: str | None = None) -> None:
    import glob

    rows = []
    for f in sorted(glob.glob(os.path.join(onnx_dir, "task*.onnx"))):
        r = audit_one(f, None, run_correctness=False)
        tnum = int(os.path.basename(f)[4:7])
        rows.append((tnum, r["cost"], r["params"], r["memory"], r["status"]))
    rows.sort(key=lambda x: -(x[1] or 0))
    tot = sum(max(1.0, 25.0 - math.log(max(1, c))) for _, c, *_ in rows if c)
    print(f"tasks={len(rows)} cost_only_total={tot:.4f}")
    for t, c, p, m, s in rows[:40]:
        pts = max(1.0, 25.0 - math.log(max(1, c))) if c else 0
        print(f"t{t:03d} cost={c} params={p} mem={m} pts={pts:.3f} {s}")
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "cost", "params", "memory", "status"])
            w.writerows(rows)


def cmd_one(onnx_path: str, tnum: int) -> None:
    r = audit_one(onnx_path, _load(tnum), run_correctness=True)
    print(
        json.dumps(
            {
                k: r[k]
                for k in [
                    "params",
                    "memory",
                    "cost",
                    "points",
                    "n_pass",
                    "n_fail",
                    "status",
                ]
            }
        )
    )


def cmd_total(onnx_dir: str) -> None:
    import glob

    tot = 0.0
    zeros = []
    for f in sorted(glob.glob(os.path.join(onnx_dir, "task*.onnx"))):
        tnum = int(os.path.basename(f)[4:7])
        r = audit_one(f, _load(tnum), run_correctness=True)
        tot += r["points"] or 0.0
        if not r["points"]:
            zeros.append((tnum, r["status"]))
    print(f"total={tot:.4f} zero_tasks={len(zeros)}")
    for t, s in zeros[:30]:
        print(f"  t{t:03d} {s}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rank":
        cmd_rank(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "one":
        cmd_one(sys.argv[2], int(sys.argv[3]))
    elif cmd == "total":
        cmd_total(sys.argv[2])
