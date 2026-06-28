"""case3 CLI — solver bank で ONNX を生成 / 公開バンドルを override-blend する。

``src/submit/__main__.py`` と同じ typer + rich の薄い CLI。業務ロジックは
``run`` / ``override`` モジュールに委譲する。
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from .apply_surgery import apply_surgery
from .override import override_blend
from .run import run, write_manifest

console = Console()

app = typer.Typer(
    add_completion=False, help="case3: minimal-ONNX solver bank + override-blend"
)

DEFAULT_TASK_DIR = Path("../data/lake/neurogolf-2026")


@app.callback()
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command("solve")
def solve_cmd(
    task_dir: Path = typer.Option(DEFAULT_TASK_DIR, "--task-dir"),
    out: Path = typer.Option(Path("../data/output/onnx/case3"), "--out"),
) -> None:
    """solver bank で各タスク最小 cost の ONNX を生成（厳密検証済みのみ）。"""
    solved = run(task_dir, out)
    write_manifest(solved, out.parent / "case3_manifest.json")
    typer.echo(f"solved {len(solved)} tasks -> {out}")


@app.command("blend")
def blend_cmd(
    base: Path = typer.Option(..., "--base", help="基準バンドル onnx ディレクトリ"),
    candidate: list[Path] = typer.Option(
        [], "--candidate", help="差し替え候補ディレクトリ"
    ),
    task_dir: Path = typer.Option(DEFAULT_TASK_DIR, "--task-dir"),
    out: Path = typer.Option(Path("../data/output/onnx/blend"), "--out"),
) -> None:
    """基準バンドルを土台に、正答かつ cost 減の候補のみ差し替える（タスクは落とさない）。"""
    overrides = override_blend(base, list(candidate), task_dir, out)
    typer.echo(f"overrode {len(overrides)} tasks -> {out}")


@app.command("surgery")
def surgery_cmd(
    base: Path = typer.Option(..., "--base", help="基準バンドル onnx ディレクトリ"),
    task_dir: Path = typer.Option(DEFAULT_TASK_DIR, "--task-dir"),
    out: Path = typer.Option(Path("../data/output/onnx/surgery"), "--out"),
) -> None:
    """基準バンドルへ意味保存のグラフサージェリを適用（正答かつ cost 減のみ採用）。"""
    improvements = apply_surgery(base, task_dir, out)
    typer.echo(f"improved {len(improvements)} tasks -> {out}")


if __name__ == "__main__":
    app()
