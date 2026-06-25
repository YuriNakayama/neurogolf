"""Identity solver: output == input.

A 1x1 conv whose weight is 1 exactly when the output channel equals the input
channel (and at the single kernel position). The canonical baseline; solves
tasks whose transform is the identity map.
"""

from __future__ import annotations

import onnx

from solvers.conv import build_single_layer_conv2d


def _identity_weight(
    out_channel: int, in_channel: int, offset: tuple[int, int]
) -> float:
    return 1.0 if out_channel == in_channel and offset == (0, 0) else 0.0


def build_identity_model() -> onnx.ModelProto:
    """Build the identity ONNX model."""
    return build_single_layer_conv2d(_identity_weight, kernel_size=1)
