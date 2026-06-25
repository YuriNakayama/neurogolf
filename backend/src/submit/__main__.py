"""`uv run python -m submit` entry point — build & submit a NeuroGolf zip."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from submit import (
    AuthError,
    PackagingError,
    ValidationError,
    build_submission_zip,
    collect_onnx_files,
    confirm_submission,
    ensure_credentials,
    kaggle_submit,
    poll,
    record,
    validate_onnx_files,
)
from submit.kaggle_api import KaggleCLIError
from submit.validator import TaskValidation

app = typer.Typer(
    help="Kaggle NeuroGolf 2026 への提出自動化CLI",
    no_args_is_help=True,
)
console = Console()

DEFAULT_ONNX_DIR = Path("data/output/onnx")
DEFAULT_OUTPUT_DIR = Path("data/output/submit")


def _validations_table(results: list[TaskValidation]) -> Table:
    table = Table(title="ONNX 検証結果")
    table.add_column("task")
    table.add_column("bytes", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("score", justify="right")
    for r in results:
        table.add_row(
            r.path.name,
            f"{r.size_bytes:,}",
            f"{r.cost:,}",
            f"{r.score:.3f}",
        )
    return table


@app.command("submit")
def submit_cmd(
    message: str = typer.Option(..., "-m", "--message", help="提出メッセージ"),
    onnx_dir: Path = typer.Option(
        DEFAULT_ONNX_DIR,
        "--onnx-dir",
        help="taskNNN.onnx を収めたディレクトリ",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        help="submission.zip 生成先",
    ),
    dry_run_only: bool = typer.Option(
        False,
        "--dry-run",
        help="提出は行わず、検証とパッケージングのみ実行",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="提出後に validation 結果をポーリング",
    ),
) -> None:
    """ONNX 群を検証して submission.zip を作り Kaggle に提出する。"""

    console.print("[bold cyan]== NeuroGolf 提出フロー ==[/]")
    console.print(f"onnx dir : {onnx_dir}")
    console.print(f"dry-run  : {dry_run_only}")

    console.print("\n[bold]1) ONNX 収集 & 検証[/]")
    try:
        files = collect_onnx_files(onnx_dir)
        results = validate_onnx_files(files)
    except (PackagingError, ValidationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc
    console.print(_validations_table(results))
    total_score = sum(r.score for r in results)
    console.print(f"  tasks: {len(results)}  推定合計スコア: {total_score:.3f}")

    console.print("\n[bold]2) submission.zip 生成[/]")
    archive = build_submission_zip(onnx_dir, output_dir, files=files)
    console.print(f"  生成: {archive}")

    if dry_run_only:
        record(
            base_dir=output_dir,
            case=onnx_dir.name,
            message=message,
            archive=archive,
            dry_run=True,
            result={"mode": "dry-run", "tasks": len(results)},
        )
        console.print("[green]dry-run 完了。提出は行いませんでした。[/]")
        raise typer.Exit(code=0)

    console.print("\n[bold]3) 認証確認[/]")
    try:
        method = ensure_credentials()
    except AuthError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=4) from exc
    console.print(f"  認証方式: {method}")

    console.print("\n[bold]4) Kaggle へ提出[/]")
    try:
        stdout = kaggle_submit(archive, message)
    except KaggleCLIError as exc:
        console.print(f"[red]提出失敗:[/]\n{exc}")
        record(
            base_dir=output_dir,
            case=onnx_dir.name,
            message=message,
            archive=archive,
            dry_run=False,
            result={"error": str(exc)},
        )
        raise typer.Exit(code=7) from exc
    console.print(stdout)

    result: dict[str, object] = {"stdout": stdout, "tasks": len(results)}

    console.print("\n[bold]4b) 履歴APIで提出確認[/]")
    confirmed = confirm_submission(message)
    if confirmed is None:
        console.print(
            "[yellow]履歴に提出行が見つかりません。Kaggle UI を確認してください。[/]"
        )
        result["confirmed"] = False
    else:
        console.print(f"  確認: status={confirmed.get('status')}")
        result["confirmed"] = True
        result["confirmed_row"] = confirmed

    if wait:
        console.print("\n[bold]5) Validation ポーリング[/]")
        outcome = poll(message)
        result["poll"] = outcome
        console.print(f"  結果: {outcome.get('status')}")

    record(
        base_dir=output_dir,
        case=onnx_dir.name,
        message=message,
        archive=archive,
        dry_run=False,
        result=result,
    )
    console.print("\n[green]提出完了。[/]")


@app.command("validate")
def validate_cmd(
    onnx_dir: Path = typer.Option(
        DEFAULT_ONNX_DIR,
        "--onnx-dir",
        help="taskNNN.onnx を収めたディレクトリ",
    ),
) -> None:
    """ONNX 群を検証してコスト/スコアを表示する（提出はしない）。"""

    try:
        files = collect_onnx_files(onnx_dir)
        results = validate_onnx_files(files)
    except (PackagingError, ValidationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc
    console.print(_validations_table(results))
    console.print(
        f"  tasks: {len(results)}  推定合計スコア: {sum(r.score for r in results):.3f}"
    )


@app.command("submissions")
def submissions_cmd() -> None:
    """現在の提出一覧を取得する。"""

    from submit.kaggle_api import list_submissions

    try:
        rows = list_submissions()
    except KaggleCLIError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    for row in rows:
        console.print(row)


if __name__ == "__main__":  # pragma: no cover
    app()
