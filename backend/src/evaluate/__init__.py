"""NeuroGolf 2026 公式スコアラーのミラー（全ケース共通）。

Kaggle ノートブック ``biohack44/neurogolf-2026-blend-max`` の audit harness を
移植したもの。任意の ``taskNNN.onnx`` を競技の公式スコアリングと同じ手順で
採点する（``cost = params + memory_bytes``、correctness、``points``）。

case 固有のロジック（blend 選択など）はここには置かず ``pipeline/<case>`` に置く。
"""

from __future__ import annotations

from evaluate.scorer import (
    SOURCE,
    audit_dir,
    audit_one,
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    sanitize_model,
)

__all__ = [
    "SOURCE",
    "audit_dir",
    "audit_one",
    "calculate_memory",
    "calculate_params",
    "convert_to_numpy",
    "sanitize_model",
]
