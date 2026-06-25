"""``python -m pipeline.case1`` — 既知良好バンドルを再現して提出するベースライン。

高スコアの公開ノートブック出力（``submission.zip``）を取得・検証し、目標 Public
Score に到達するまで Kaggle に提出する。新しいモデルは作らない（後処理ケース）。

例::

    # ローカルにある検証済み zip をそのまま提出
    uv run python -m pipeline.case1 submit \\
        --local-zip ../data/lake/case1-baseline/submission.zip

    # Kaggle から取得 → 検証 → 提出（目標スコアまで）
    uv run python -m pipeline.case1 submit
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pipeline.case1 import (
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    TARGET_KERNEL,
    TARGET_PUBLIC_SCORE,
    ReproduceError,
    reached,
    resolve_target,
    run_until_target,
)
from submit import AuthError, ensure_credentials, record
from submit.kaggle_api import KaggleCLIError

app = typer.Typer(
    help="NeuroGolf 2026 case1 — 既知良好バンドルを再現・提出するベースライン",
    no_args_is_help=True,
)
console = Console()

DEFAULT_WORK = Path("../data/lake/case1-baseline")
DEFAULT_OUTPUT_DIR = Path("data/output/submit")
CASE = "case1"


@app.callback()
def _main() -> None:
    """case1 のサブコマンド群（``verify`` / ``submit``）。"""


@app.command("verify")
def verify_cmd(
    local_zip: Path = typer.Option(
        None,
        "--local-zip",
        help="検証する submission.zip（省略時は Kaggle から取得）",
    ),
    work: Path = typer.Option(DEFAULT_WORK, "--work", help="取得物の作業ディレクトリ"),
) -> None:
    """ベースラインバンドルを取得（必要時）し、SHA256 / バイト数を検証する。"""

    console.print(f"[dim]target kernel: {TARGET_KERNEL}[/dim]")
    console.print(
        f"[dim]expected: {EXPECTED_BYTES:,} bytes  sha256={EXPECTED_SHA256}[/dim]"
    )
    try:
        bundle = resolve_target(work, local_zip=local_zip)
    except ReproduceError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc
    console.print(
        f"[green]検証 OK[/]: {bundle.zip_path}  "
        f"{bundle.size_bytes:,} bytes  sha256={bundle.sha256}"
    )


@app.command("submit")
def submit_cmd(
    message: str = typer.Option(
        "case1 reproduce baseline", "-m", "--message", help="提出メッセージ"
    ),
    local_zip: Path = typer.Option(
        None,
        "--local-zip",
        help="提出する submission.zip（省略時は Kaggle から取得）",
    ),
    work: Path = typer.Option(DEFAULT_WORK, "--work", help="取得物の作業ディレクトリ"),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", help="提出履歴の保存先"
    ),
    target: float = typer.Option(
        TARGET_PUBLIC_SCORE, "--target", help="到達目標 Public Score"
    ),
    max_attempts: int = typer.Option(
        3, "--max-attempts", help="目標未達時の最大提出回数"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="検証のみで提出はしない"),
) -> None:
    """ベースラインバンドルを検証し、目標 Public Score まで提出する。"""

    console.print("[bold cyan]== case1 reproduce baseline 提出フロー ==[/]")
    console.print(f"target kernel : {TARGET_KERNEL}")
    console.print(f"target score  : {target}")

    console.print("\n[bold]1) バンドル取得 & 検証[/]")
    try:
        bundle = resolve_target(work, local_zip=local_zip)
    except ReproduceError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc
    console.print(
        f"  [green]検証 OK[/]: {bundle.zip_path}  {bundle.size_bytes:,} bytes"
    )

    if dry_run:
        record(
            base_dir=output_dir,
            case=CASE,
            message=message,
            archive=bundle.zip_path,
            dry_run=True,
            result={"mode": "dry-run", "sha256": bundle.sha256},
        )
        console.print("[green]dry-run 完了。提出は行いませんでした。[/]")
        raise typer.Exit(code=0)

    console.print("\n[bold]2) 認証確認[/]")
    try:
        method = ensure_credentials()
    except AuthError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=4) from exc
    console.print(f"  認証方式: {method}")

    console.print("\n[bold]3) 目標スコアまで提出[/]")
    try:
        outcome = run_until_target(
            bundle.zip_path, message, target, max_attempts=max_attempts
        )
    except KaggleCLIError as exc:
        console.print(f"[red]提出失敗:[/]\n{exc}")
        record(
            base_dir=output_dir,
            case=CASE,
            message=message,
            archive=bundle.zip_path,
            dry_run=False,
            result={"error": str(exc)},
        )
        raise typer.Exit(code=7) from exc

    console.print(f"  status={outcome.status}  publicScore={outcome.public_score}")
    ok = reached(outcome.public_score, target)
    record(
        base_dir=output_dir,
        case=CASE,
        message=message,
        archive=bundle.zip_path,
        dry_run=False,
        result={
            "status": outcome.status,
            "public_score": outcome.public_score,
            "target": target,
            "reached": ok,
            "sha256": bundle.sha256,
        },
    )
    if ok:
        console.print("\n[green]目標スコア到達。提出完了。[/]")
    else:
        console.print(
            "\n[yellow]目標未達（pending の可能性）。Kaggle UI を確認してください。[/]"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
