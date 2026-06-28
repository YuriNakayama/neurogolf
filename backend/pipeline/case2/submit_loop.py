"""case1 baseline — 既知良好バンドルを Public Score 目標まで提出するループ。

NeuroGolf は 1 日 100 回まで無検証で提出できる。本モジュールは検証済みの
``submission.zip`` を提出し、履歴 API で status / publicScore をポーリングして、
目標スコアに到達するか試行回数を使い切るまで繰り返す。

提出そのものは ``src/submit`` の Kaggle CLI ラッパーに委譲する（case 横断の
不変処理）。ここに置くのは「目標到達まで回す」case 固有のループ制御だけ。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from submit.kaggle_api import (
    KaggleCLIError,
    confirm_submission,
    list_submissions,
    poll,
)
from submit.kaggle_api import submit as kaggle_submit

# 同点扱いの許容誤差（公式 publicScore の表示桁ゆらぎ吸収）。
SCORE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SubmitOutcome:
    """1 回の提出結果。"""

    submitted: bool
    status: str
    public_score: float | None
    raw: str


def reached(score: float | None, target: float) -> bool:
    """``score`` が ``target`` 以上（許容誤差込み）なら True。None は未到達。"""

    if score is None:
        return False
    return score >= target - SCORE_TOLERANCE


def _public_score(row: dict[str, str]) -> float | None:
    raw = row.get("publicScore") or row.get("public_score") or ""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def submit_once(
    zip_path: str | Path,
    message: str,
    *,
    poll_seconds: int = 600,
    poll_interval: int = 30,
) -> SubmitOutcome:
    """バンドルを 1 回提出し、status / publicScore をポーリングして返す。"""

    raw = kaggle_submit(Path(zip_path), message)
    confirm_submission(message, timeout_s=poll_interval * 2, interval_s=poll_interval)
    outcome = poll(message, timeout_s=poll_seconds, interval_s=poll_interval)
    row = outcome.get("row") or {}
    status = str(outcome.get("status") or row.get("status") or "unknown")
    return SubmitOutcome(
        submitted=True,
        status=status,
        public_score=_public_score(row),
        raw=raw,
    )


def latest_public_score(message: str) -> float | None:
    """履歴 API から ``message`` を含む最新提出の publicScore を取得する。"""

    for row in list_submissions():
        description = row.get("description") or row.get("message") or ""
        if message in description:
            return _public_score(row)
    return None


def run_until_target(
    zip_path: str | Path,
    message: str,
    target: float,
    *,
    max_attempts: int = 3,
    poll_seconds: int = 600,
    poll_interval: int = 30,
    sleep_between: int = 0,
) -> SubmitOutcome:
    """目標スコア到達まで提出を繰り返す（最大 ``max_attempts`` 回）。

    成功した時点で即座に止める。全試行で未到達なら最後の結果を返す。
    同一バンドルの再提出は冪等（private-set の挙動は変わらない）なので、
    pending / 一時失敗のリトライ手段として使う。
    """

    last = SubmitOutcome(
        submitted=False, status="not_attempted", public_score=None, raw=""
    )
    for attempt in range(1, max_attempts + 1):
        attempt_message = message if attempt == 1 else f"{message} (retry {attempt})"
        try:
            last = submit_once(
                zip_path,
                attempt_message,
                poll_seconds=poll_seconds,
                poll_interval=poll_interval,
            )
        except KaggleCLIError as exc:
            last = SubmitOutcome(
                submitted=False, status="error", public_score=None, raw=str(exc)
            )
        if reached(last.public_score, target):
            return last
        if attempt < max_attempts and sleep_between > 0:
            time.sleep(sleep_between)
    return last
