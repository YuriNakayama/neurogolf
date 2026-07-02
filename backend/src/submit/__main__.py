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
from submit.gate import evaluate_candidate_gate, task_num_from_path
from submit.kaggle_api import KaggleCLIError
from submit.validator import TaskValidation

app = typer.Typer(
    help="Kaggle NeuroGolf 2026 への提出自動化CLI",
    no_args_is_help=True,
)
console = Console()

# DVC は repo-root の data/output/onnx に pull するため、backend cwd
# （python -m submit の実行位置）からは ../data/output/onnx を指す。
DEFAULT_ONNX_DIR = Path("../data/output/onnx")
DEFAULT_OUTPUT_DIR = Path("../data/output/submit")


def _validations_table(results: list[TaskValidation]) -> Table:
    table = Table(title="ONNX 検証結果")
    table.add_column("task")
    table.add_column("bytes", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("score", justify="right")
    for r in results:
        cost = f"{r.cost:,}" if r.cost is not None else "—"
        score = f"{r.score:.3f}" if r.score is not None else "(local未推定)"
        table.add_row(r.path.name, f"{r.size_bytes:,}", cost, score)
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
    total_score = sum(r.score for r in results if r.score is not None)
    unscorable = [r for r in results if not r.scorable]
    summary = f"  tasks: {len(results)}  推定合計スコア: {total_score:.3f}"
    if unscorable:
        summary += (
            f"  (うち {len(unscorable)} 件は local 未推定だが提出に含める"
            " — Kaggle 採点に委ねる)"
        )
    console.print(summary)

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
    total = sum(r.score for r in results if r.score is not None)
    unscorable = sum(1 for r in results if not r.scorable)
    line = f"  tasks: {len(results)}  推定合計スコア: {total:.3f}"
    if unscorable:
        line += f"  (うち {unscorable} 件は local 未推定)"
    console.print(line)


@app.command("gate")
def gate_cmd(
    candidate: Path = typer.Argument(..., help="評価する taskNNN.onnx 候補"),
    baseline_dir: Path = typer.Option(
        DEFAULT_ONNX_DIR,
        "--baseline-dir",
        help="採用済み baseline の taskNNN.onnx ディレクトリ",
    ),
    task_dir: Path = typer.Option(
        Path("../data/lake/neurogolf-2026"),
        "--task-dir",
        help="taskNNN.json を収めたディレクトリ",
    ),
    submit_gain: float = typer.Option(
        0.020,
        "--submit-gain",
        help="通常 submit 可能とみなす最小 relative gain",
    ),
    mid_gain: float = typer.Option(
        0.010,
        "--mid-gain",
        help="要レビュー候補と bank-only 候補の境界",
    ),
) -> None:
    """候補 ONNX 1 件を baseline と比較し、submit 前ゲートを表示する。"""

    try:
        task = task_num_from_path(candidate)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc
    task_name = f"task{task:03d}"
    baseline = baseline_dir / f"{task_name}.onnx"
    task_json = task_dir / f"{task_name}.json"
    if not baseline.is_file():
        console.print(f"[red]baseline が見つかりません:[/] {baseline}")
        raise typer.Exit(code=2)
    if not candidate.is_file():
        console.print(f"[red]candidate が見つかりません:[/] {candidate}")
        raise typer.Exit(code=2)
    if not task_json.is_file():
        console.print(f"[red]task json が見つかりません:[/] {task_json}")
        raise typer.Exit(code=2)

    result = evaluate_candidate_gate(
        baseline,
        candidate,
        task_json,
        submit_gain=submit_gain,
        mid_gain=mid_gain,
    )
    gain = "—" if result.gain is None else f"{result.gain:.6f}"
    forbidden = ", ".join(result.forbidden_ops) if result.forbidden_ops else "[]"
    console.print(f"task          : {result.task:03d}")
    console.print(f"baseline cost : {result.baseline_cost}")
    console.print(f"candidate cost: {result.candidate_cost}")
    console.print(f"gain          : {gain}")
    console.print(f"status        : {result.status}")
    console.print(f"n_fail        : {result.n_fail}")
    console.print(f"functions     : {result.functions}")
    console.print(f"forbidden ops : {forbidden}")
    console.print(f"decision      : [bold]{result.decision}[/]")
    console.print(f"reason        : {result.reason}")


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
