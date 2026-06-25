"""``python -m pipeline.case0`` — generate an ONNX bundle from the solvers.

Tries each solver against every task and saves ``taskNNN.onnx`` for the ones it
solves exactly. The resulting ``--out`` directory is a bundle ready for
``python -m submit submit --onnx-dir <out>`` or ``pipeline.case1`` blending.

例::

    uv run python -m pipeline.case0 build \\
        --task-dir ../data/lake/neurogolf-2026 \\
        --out ../data/output/onnx
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evaluate import SOURCE
from pipeline.case0.build import BuildSummary, build_bundle

app = typer.Typer(
    help="NeuroGolf 2026 case0 — generate a per-task ONNX bundle",
    no_args_is_help=True,
)
console = Console()

DEFAULT_TASK_DIR = Path("../data/lake/neurogolf-2026")
DEFAULT_OUT = Path("../data/output/onnx")


@app.callback()
def _main() -> None:
    """case0 のサブコマンド群（``build`` など）。"""


def _summary_table(summary: BuildSummary) -> Table:
    table = Table(title="case0 build — solver ごとの採択数")
    table.add_column("solver")
    table.add_column("count", justify="right")
    counts: dict[str, int] = {}
    for b in summary.solved:
        counts[b.solver or "?"] = counts.get(b.solver or "?", 0) + 1
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        table.add_row(name, str(n))
    return table


@app.command("build")
def build_cmd(
    task_dir: Path = typer.Option(
        DEFAULT_TASK_DIR, "--task-dir", help="タスク JSON ディレクトリ"
    ),
    out: Path = typer.Option(
        DEFAULT_OUT, "--out", help="taskNNN.onnx 出力ディレクトリ"
    ),
) -> None:
    """全タスクを総当りでソルバ検証し、解けたものを ``out`` に保存する。"""
    console.print(f"[dim]scorer source: {SOURCE}[/dim]")
    summary = build_bundle(task_dir, out)
    console.print(_summary_table(summary))
    console.print(
        f"solved: [bold]{len(summary.solved)}[/bold] / {len(summary.builds)}  "
        f"推定合計スコア: [bold]{summary.total_points:.3f}[/bold]  "
        f"-> {out}"
    )


if __name__ == "__main__":
    app()
