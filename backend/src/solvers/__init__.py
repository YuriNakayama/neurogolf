"""NeuroGolf ARC solvers — small per-task ONNX models (the **generation layer**).

Each builder returns a constraint-compliant ``onnx.ModelProto``; :mod:`run`
audits a model against a task via the official-scorer mirror.

    conv        build_single_layer_conv2d (verbatim official single-layer Conv)
    identity    build_identity_model (output == input)
    recolor     build_recolor_for_task / infer_recolor_mapping (color permutation)
    spatial     transpose / flip_lr / flip_ud / rot180 (zero-param Transpose/Gather)
    run         solves_task / audit_model (correctness + cost via evaluate)
"""

from __future__ import annotations

from solvers.conv import INPUT_NAME, OUTPUT_NAME, build_single_layer_conv2d
from solvers.identity import build_identity_model
from solvers.recolor import (
    build_recolor_for_task,
    build_recolor_model,
    infer_recolor_mapping,
)
from solvers.run import SolveResult, audit_model, solves_task
from solvers.spatial import (
    build_flip_lr_model,
    build_flip_ud_model,
    build_rot180_model,
    build_transpose_model,
)

__all__ = [
    "INPUT_NAME",
    "OUTPUT_NAME",
    "SolveResult",
    "audit_model",
    "build_flip_lr_model",
    "build_flip_ud_model",
    "build_identity_model",
    "build_recolor_for_task",
    "build_recolor_model",
    "build_rot180_model",
    "build_single_layer_conv2d",
    "build_transpose_model",
    "infer_recolor_mapping",
    "solves_task",
]
