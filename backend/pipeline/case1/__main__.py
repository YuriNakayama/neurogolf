"""``python -m pipeline.case1`` エントリポイント — per-task MAX BLEND。

2 つの提出バンドル A / B（dir または submission.zip）をタスクごとに採点し、
correct かつ cheaper な ONNX を選んで ``submission.zip`` を再パッケージする。

例::

    uv run python -m pipeline.case1 blend \\
        --a-dir /path/to/A_bundle \\
        --b-zip /path/to/B/submission.zip \\
        --task-dir /kaggle/input/competitions/neurogolf-2026 \\
        --out data/output/case1/submission

元ノートブック ``biohack44/neurogolf-2026-blend-max`` の Cell 6 / 8 / 10 / 12 に対応。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from evaluate import SOURCE
from pipeline.case1.blend import blend as run_blend
from pipeline.case1.blend import package, print_diff
from pipeline.case1.bundle import BundleError, resolve_bundle

app = typer.Typer(
    help="NeuroGolf 2026 case1 — per-task MAX BLEND (selection blend)",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _main() -> None:
    """case1 のサブコマンド群（``blend`` など）。

    コールバックを置くことで、コマンドが 1 つでも typer がサブコマンド
    ルーティング（``... blend ...``）を維持する。
    """


# 元ノートブックの Kaggle 既定パス（ローカルでは --task-dir 等で上書きする）。
DEFAULT_TASK_DIR = Path("/kaggle/input/competitions/neurogolf-2026")
DEFAULT_WORK = Path("/kaggle/working")
DEFAULT_OUT = DEFAULT_WORK / "submission"


@app.command("blend")
def blend_cmd(
    a_dir: Path | None = typer.Option(
        None, "--a-dir", help="バンドル A: taskNNN.onnx を含むディレクトリ"
    ),
    a_zip: Path | None = typer.Option(
        None, "--a-zip", help="バンドル A: submission.zip"
    ),
    b_dir: Path | None = typer.Option(
        None, "--b-dir", help="バンドル B: taskNNN.onnx を含むディレクトリ"
    ),
    b_zip: Path | None = typer.Option(
        None, "--b-zip", help="バンドル B: submission.zip"
    ),
    task_dir: Path = typer.Option(
        DEFAULT_TASK_DIR, "--task-dir", help="競技タスク JSON のディレクトリ"
    ),
    out: Path = typer.Option(
        DEFAULT_OUT, "--out", help="出力 submission.zip のパス（拡張子なし）"
    ),
    work: Path = typer.Option(
        DEFAULT_WORK, "--work", help="zip 展開や stage の作業ディレクトリ"
    ),
    show_diff: bool = typer.Option(
        True, "--diff/--no-diff", help="A/B で勝敗が分かれたタスクを表示する"
    ),
) -> None:
    """A / B をタスクごとに blend して submission.zip を生成する。"""
    console.print("[bold cyan]== NeuroGolf case1: per-task MAX BLEND ==[/]")
    console.print(f"scoring source: {SOURCE}")

    work.mkdir(parents=True, exist_ok=True)
    try:
        a = resolve_bundle(
            str(a_dir) if a_dir else None,
            str(a_zip) if a_zip else None,
            "A",
            str(work),
        )
        b = resolve_bundle(
            str(b_dir) if b_dir else None,
            str(b_zip) if b_zip else None,
            "B",
            str(work),
        )
    except BundleError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    stage = str(work / "_stage")
    summary = run_blend(a, b, str(task_dir), stage)

    if show_diff:
        console.print("\n[bold]A/B 差分（勝敗が分かれたタスク）[/]")
        print_diff(summary)

    out.parent.mkdir(parents=True, exist_ok=True)
    console.print("\n[bold]submission.zip 生成 + 最終 audit[/]")
    zip_path = package(stage, str(out), task_dir=str(task_dir))
    console.print(f"[green]完了: {zip_path}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
