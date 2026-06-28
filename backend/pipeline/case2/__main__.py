"""case2 CLI — build a DSL-override bundle and submit it.

Thin typer wrapper: ``build`` re-solves tasks with case2 DSL primitives and
overrides the case1 baseline where a case2 net is exactly correct and strictly
cheaper; ``submit`` packages and submits the bundle until a target Public Score.
Business logic lives in :mod:`build` / :mod:`solver`; this only parses args.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from submit.packager import build_submission_zip

from . import TARGET_PUBLIC_SCORE, build_override_bundle, summarize

app = typer.Typer(add_completion=False, help="case2 DSL-primitive override pipeline")
console = Console()
logger = logging.getLogger(__name__)

_DEFAULT_BASELINE = Path("../data/lake/case2-base7166/onnx")
_DEFAULT_TASKS = Path("../data/lake/neurogolf-2026")
_DEFAULT_OUT = Path("../data/output/onnx/case2")


@app.callback()
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def build(
    baseline_dir: Path = typer.Option(
        _DEFAULT_BASELINE, help="baseline taskNNN.onnx dir"
    ),
    task_dir: Path = typer.Option(_DEFAULT_TASKS, help="task JSON dir"),
    out_dir: Path = typer.Option(_DEFAULT_OUT, help="output bundle dir"),
) -> None:
    """Re-solve all tasks; override baseline where case2 is exact and cheaper."""
    outcomes = build_override_bundle(baseline_dir, task_dir, out_dir)
    stats = summarize(outcomes)
    console.print_json(json.dumps(stats))


@app.command()
def package(
    onnx_dir: Path = typer.Option(_DEFAULT_OUT, help="bundle dir to zip"),
    out_dir: Path = typer.Option(
        Path("../data/output/onnx/case2"), help="zip output dir"
    ),
) -> None:
    """Zip the bundle into submission.zip."""
    zip_path = build_submission_zip(onnx_dir, out_dir)
    console.print(f"built {zip_path}")


@app.command()
def submit(
    zip_path: Path = typer.Option(..., help="submission.zip to submit"),
    target: float = typer.Option(TARGET_PUBLIC_SCORE, help="target Public Score"),
    max_attempts: int = typer.Option(3, help="max submit retries"),
    dry_run: bool = typer.Option(False, help="validate only, do not submit"),
) -> None:
    """Submit the bundle, polling Public Score until the target is reached."""
    from .submit_loop import run_until_target

    if dry_run:
        console.print(f"[dry-run] would submit {zip_path} targeting {target}")
        return
    outcome = run_until_target(
        zip_path, "case2 dsl override", target, max_attempts=max_attempts
    )
    console.print(
        f"status={outcome.status} public_score={outcome.public_score} "
        f"reached={outcome.public_score is not None and outcome.public_score >= target}"
    )


if __name__ == "__main__":
    app()
