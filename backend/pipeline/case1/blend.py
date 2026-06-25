"""per-task MAX BLEND の選択ロジック（case1）。

Kaggle ノートブック ``biohack44/neurogolf-2026-blend-max`` の Cell 8 / 10 / 12
を移植。2 つの候補バンドル A / B を受け取り、タスクごとに
**correct かつ cheaper** な ONNX を選んでステージし、``submission.zip`` を
再パッケージする。

  - only one correct -> それを採用
  - both correct      -> 低 cost（高 points）を採用
  - both correct で同 cost -> A（任意・安定）
  - neither correct   -> fail 少を採用、なお同じなら A

新しいモデルは作らない。各タスクは A か B の実提出が既に含む ONNX のいずれか
なので private-set の挙動は保たれ、blend 後 LB >= max(A_LB, B_LB)。

この選択ロジックは手法変更で変わりうるため ``src`` ではなく case ディレクトリ
に置く。公式スコアラーのミラーだけ ``evaluate`` に共通化している。
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from evaluate import audit_dir, audit_one

MAX_TASKS = 400


@dataclass
class BlendSummary:
    """blend 実行の集計結果。"""

    picks: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0})
    missing_a: int = 0
    missing_b: int = 0
    staged: int = 0
    gain_vs_a: float = 0.0
    gain_vs_b: float = 0.0
    # (task, winner, a_pts, b_pts, win_pts)
    rows: list[tuple[int, str, float, float, float]] = field(default_factory=list)


def load_task(task_dir: str, t: int) -> dict[str, Any] | None:
    """``task_dir`` から ``task{t:03d}.json`` を読み込む（無ければ None）。"""
    p = f"{task_dir}/task{t:03d}.json"
    return json.load(open(p)) if os.path.isfile(p) else None


def _pa(a_dir: str, t: int) -> str:
    return os.path.join(a_dir, f"task{t:03d}.onnx")


def _pb(b_dir: str, t: int) -> str:
    return os.path.join(b_dir, f"task{t:03d}.onnx")


def score(path: str, ex: dict[str, Any] | None) -> dict[str, Any] | None:
    """ONNX を公式スコアラーのミラーで採点（存在しなければ None）。"""
    if not os.path.isfile(path):
        return None
    return audit_one(path, ex, run_correctness=(ex is not None))


def better(ra: dict[str, Any] | None, rb: dict[str, Any] | None) -> str:
    """tie-break ルールに従って勝者 ``'A'`` / ``'B'`` を返す。"""
    aok = (
        ra is not None
        and ra["n_fail"] == 0
        and ra["status"] == "ok"
        and ra["cost"] is not None
    )
    bok = (
        rb is not None
        and rb["n_fail"] == 0
        and rb["status"] == "ok"
        and rb["cost"] is not None
    )
    if aok and not bok:
        return "A"
    if bok and not aok:
        return "B"
    if aok and bok:
        assert ra is not None and rb is not None
        if ra["cost"] < rb["cost"]:
            return "A"
        if rb["cost"] < ra["cost"]:
            return "B"
        return "A"  # equal cost -> A
    # neither fully ok: choose fewer fails, then A
    af = ra["n_fail"] if ra else 10**9
    bf = rb["n_fail"] if rb else 10**9
    if af <= bf:
        return "A"
    return "B"


def blend(a_dir: str, b_dir: str, task_dir: str, stage: str) -> BlendSummary:
    """A / B をタスクごとに採点し、勝者を ``stage`` にコピーする。"""
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    summary = BlendSummary()
    for t in range(1, MAX_TASKS + 1):
        ex = load_task(task_dir, t)
        ra = score(_pa(a_dir, t), ex)
        rb = score(_pb(b_dir, t), ex)
        if ra is None:
            summary.missing_a += 1
        if rb is None:
            summary.missing_b += 1
        if ra is None and rb is None:
            print(f"task{t:03d}: MISSING in BOTH -> task will score 0")
            continue
        w = better(ra, rb)
        summary.picks[w] += 1
        src = _pa(a_dir, t) if w == "A" else _pb(b_dir, t)
        shutil.copy(src, os.path.join(stage, f"task{t:03d}.onnx"))
        pa_pts = ra["points"] if (ra and ra["points"] is not None) else 0.0
        pb_pts = rb["points"] if (rb and rb["points"] is not None) else 0.0
        win_pts = pa_pts if w == "A" else pb_pts
        summary.gain_vs_a += win_pts - pa_pts
        summary.gain_vs_b += win_pts - pb_pts
        summary.rows.append((t, w, pa_pts, pb_pts, win_pts))

    summary.staged = len(glob.glob(stage + "/task*.onnx"))
    print(
        f"\npicks: A={summary.picks['A']}  B={summary.picks['B']}  "
        f"(missingA={summary.missing_a}, missingB={summary.missing_b})"
    )
    print(f"staged tasks: {summary.staged}")
    print(
        f"local gain vs A-only: +{summary.gain_vs_a:.2f}   "
        f"vs B-only: +{summary.gain_vs_b:.2f}"
    )
    return summary


def print_diff(summary: BlendSummary, top: int = 30) -> None:
    """A と B で points が異なるタスクを差の大きい順に表示（Cell 10）。"""
    diff = [r for r in summary.rows if abs(r[2] - r[3]) > 1e-9]
    diff.sort(key=lambda r: -abs(r[2] - r[3]))
    print(f"{len(diff)} tasks where A and B differ in points. Top {top} by gap:")
    print(f"{'task':>4} {'win':>3} {'A_pts':>7} {'B_pts':>7} {'used':>7}")
    for t, w, ap, bp, wp in diff[:top]:
        print(f"{t:>4} {w:>3} {ap:>7.3f} {bp:>7.3f} {wp:>7.3f}")


def package(stage: str, out: str, task_dir: str | None = None) -> str:
    """``stage`` を ``out.zip`` に固め、最終 audit を実行する（Cell 12）。

    ``out`` は拡張子なしのパス（``shutil.make_archive`` の規約）。
    生成した zip パスを返す。
    """
    zip_path = out + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(out, "zip", stage)
    print("wrote", zip_path, " (", len(glob.glob(stage + "/task*.onnx")), "tasks)")
    if task_dir is not None:
        final = audit_dir(
            stage,
            task_dir=task_dir,
            out_csv=f"{os.path.dirname(out) or '.'}/blend_audit.csv",
            run_correctness=True,
        )
        tot = sum(r["points"] for r in final if r["points"] is not None)
        print(f"\nblended local total (public-only): {tot:.2f}")
        print(
            "Selection blend: every task == one of your real submissions, "
            "so private-set"
        )
        print("behavior is preserved. Blended LB >= max(A_LB, B_LB).")
    return zip_path
