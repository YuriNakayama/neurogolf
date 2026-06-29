"""Build a single-Conv ONNX from a saved sep_tNNN_kK.npz (W[10,D],b[10]) and
faithful-audit it. Conv weight = W.reshape(10,10,k,k), pads=k//2, output named
'output' so memory=0; cost=params=10*10*k*k+10 if correct."""

import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, "src")
from evaluate.scorer import audit_one  # noqa: E402

LAKE = "../data/lake/neurogolf-2026"


def build(t, k, scale=1.0):
    d = np.load(f"scratchpad/sep_t{t:03d}_k{k}.npz")
    W = (d["W"] * scale).astype(np.float32).reshape(10, 10, k, k)
    b = (d["b"] * scale).astype(np.float32)
    pad = k // 2
    node = helper.make_node(
        "Conv",
        ["input", "W", "B"],
        ["output"],
        kernel_shape=[k, k],
        pads=[pad, pad, pad, pad],
        strides=[1, 1],
    )
    graph = helper.make_graph(
        [node],
        f"t{t:03d}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "B")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    path = f"scratchpad/cand_t{t:03d}.onnx"
    onnx.save(m, path)
    return path


def main():
    t = int(sys.argv[1])
    k = int(sys.argv[2])
    path = build(t, k)
    with open(f"{LAKE}/task{t:03d}.json") as f:
        task = json.load(f)
    r = audit_one(path, task, run_correctness=True)
    print(
        f"t{t:03d} k={k}: "
        + json.dumps(
            {
                kk: r[kk]
                for kk in [
                    "params",
                    "memory",
                    "cost",
                    "points",
                    "n_pass",
                    "n_fail",
                    "status",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
