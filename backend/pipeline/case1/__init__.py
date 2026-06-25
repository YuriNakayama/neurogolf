"""case1 — 既知良好バンドルを再現して Public Score を担保するベースライン。

高スコアの公開ノートブック出力（400 タスク完成済み ONNX バンドル）を取得し、
バイト数 / SHA256 で固定照合してから Kaggle に提出する。新しいモデルは作らず、
**ある程度の Public Score を保証する土台**として使う後処理ケース。

- 取得 + 検証: ``reproduce`` (``resolve_target`` / ``verify_bundle`` / ``fetch_target``)
- 目標まで提出: ``submit_loop`` (``run_until_target`` / ``submit_once`` / ``reached``)

実行: ``uv run python -m pipeline.case1 submit --help``
"""

from __future__ import annotations

from pipeline.case1.reproduce import (
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    TARGET_KERNEL,
    Bundle,
    ReproduceError,
    fetch_target,
    resolve_target,
    verify_bundle,
)
from pipeline.case1.submit_loop import (
    SubmitOutcome,
    reached,
    run_until_target,
    submit_once,
)

# 固定ベースラインの公開スコア（boristown/agi-neural-golf-visualization-baseline）。
TARGET_PUBLIC_SCORE = 7159.44

__all__ = [
    "EXPECTED_BYTES",
    "EXPECTED_SHA256",
    "TARGET_KERNEL",
    "TARGET_PUBLIC_SCORE",
    "Bundle",
    "ReproduceError",
    "SubmitOutcome",
    "fetch_target",
    "reached",
    "resolve_target",
    "run_until_target",
    "submit_once",
    "verify_bundle",
]
