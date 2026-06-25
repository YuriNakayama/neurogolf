"""case1 — per-task MAX BLEND（biohack44/neurogolf-2026-blend-max の移植）。

2 つの提出バンドル A / B から、タスクごとに correct かつ cheaper な ONNX を
選んで ``submission.zip`` を再パッケージする選択ブレンド。公式スコアラーの
ミラーは ``evaluate`` に共通化し、ここには case 固有の選択ロジックだけを置く。

実行: ``uv run python -m pipeline.case1 blend --help``
"""

from __future__ import annotations

from pipeline.case1.blend import (
    BlendSummary,
    better,
    blend,
    package,
    print_diff,
    score,
)
from pipeline.case1.bundle import BundleError, resolve_bundle

__all__ = [
    "BlendSummary",
    "BundleError",
    "better",
    "blend",
    "package",
    "print_diff",
    "resolve_bundle",
    "score",
]
