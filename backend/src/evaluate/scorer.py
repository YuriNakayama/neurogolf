"""NeuroGolf 2026 公式スコアラーのミラー（audit harness）。

Kaggle ノートブック ``biohack44/neurogolf-2026-blend-max`` の Cell 4
（Local Audit Harness）を**ロジックそのまま**に移植したもの。競技の公式
スコアリング（``neurogolf_utils.py``）を再現し、任意の ``taskNNN.onnx`` を
ローカルで Kaggle validator と同じ手順で採点する:

  1. 全ての有効な (<=30x30) 例（train + test + arc-gen）で functional
     correctness を検証し、
  2. ``cost = params + memory_bytes`` と
     ``points = max(1, 25 - ln(max(1, cost)))`` を計算する。

実装方針（ミラー忠実性）:
- ``neurogolf_utils`` が import 可能ならその関数を**そのまま**使う（最も安全）。
- import できない場合は、2026-05-14 版 ``neurogolf_utils.py`` と byte-for-byte
  整合な fallback 実装を使う。
- リポジトリ内の他モジュール（``arc`` / ``onnxgolf`` 等）には依存せず、
  ``onnx`` + ``onnxruntime`` + ``numpy`` のみで自己完結する。

この harness は全ケース共通で変わらない（競技固定のスコアリング）ため
``src/evaluate`` に置く。case 固有の blend 選択ロジックは ``pipeline/case1``。
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from typing import Any

import numpy as np
import onnx
import onnxruntime

# ----------------------------------------------------------------------------
# Try to use the OFFICIAL utils verbatim. This is the safest source of truth.
# IMPORTANT: a module named `neurogolf_utils` may exist on the path that is NOT
# the real competition utils (a stub, an empty namespace pkg, or a partial copy).
# So we do NOT trust a successful import alone -- we verify it actually exposes
# the functions we need. If it doesn't, we fall back to the local re-impl.
# ----------------------------------------------------------------------------
_REQUIRED = (
    "convert_to_numpy",
    "sanitize_model",
    "calculate_params",
    "calculate_memory",
    "score_network",
)
_OFFICIAL: Any | None = None
for cand in [
    "/kaggle/input/competitions/neurogolf-2026",
    "/kaggle/usr/lib/neurogolf_utils",
    "./neurogolf_utils",
    ".",
]:
    if cand not in sys.path:
        sys.path.insert(0, cand)
try:
    import neurogolf_utils as _cand_mod

    if all(hasattr(_cand_mod, fn) for fn in _REQUIRED):
        _OFFICIAL = _cand_mod
    else:
        _missing = [fn for fn in _REQUIRED if not hasattr(_cand_mod, fn)]
        print(
            f"[audit] WARNING: found a 'neurogolf_utils' missing {_missing}; "
            f"using local fallback scorer instead."
        )
        _OFFICIAL = None
except Exception:
    # import itself failed (e.g. missing IPython/matplotlib/onnx_tool) -> fallback
    _OFFICIAL = None

_BATCH, _CH, _H, _W = 1, 10, 30, 30
_GRID_SHAPE = [_BATCH, _CH, _H, _W]


# ----------------------------------------------------------------------------
# Faithful fallback re-implementations (used only if official import fails).
# Kept byte-for-byte consistent with the 2026-05-14 neurogolf_utils.py.
# ----------------------------------------------------------------------------
def _convert_to_numpy(example: dict[str, Any]) -> dict[str, np.ndarray] | None:
    benchmark = {}
    shape = (1, _CH, _H, _W)
    for mode in ["input", "output"]:
        benchmark[mode] = np.zeros(shape, dtype=np.float32)
        grid = example[mode]
        if max(len(grid), len(grid[0])) > 30:
            return None
        for r, _ in enumerate(grid):
            for c, color in enumerate(grid[r]):
                benchmark[mode][0][color][r][c] = 1.0
    return benchmark


def _sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto | None:
    for node in model.graph.node:
        node.name = node.output[0]
        if "kernel_time" in node.output[0]:
            return None
    name_map: dict[str, str] = {}
    counter = [0]

    def safe(old: str) -> str:
        if not old or old in ["input", "output"]:
            return old
        if old not in name_map:
            name_map[old] = f"safe_name_{counter[0]}"
            counter[0] += 1
        return name_map[old]

    for inp in model.graph.input:
        inp.name = safe(inp.name)
    for init in model.graph.initializer:
        init.name = safe(init.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = safe(node.input[i])
        for i in range(len(node.output)):
            node.output[i] = safe(node.output[i])
        if len(node.output) > 0 and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = safe(out.name)
    for vi in model.graph.value_info:
        vi.name = safe(vi.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def _calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += int(np.prod(init.dims))
    for sp in model.graph.sparse_initializer:
        if any(d <= 0 for d in sp.values.dims):
            return None
        params += int(np.prod(sp.values.dims))
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d <= 0 for d in attr.t.dims):
                    return None
                params += int(np.prod(attr.t.dims))
            elif attr.name == "sparse_value":
                if any(d <= 0 for d in attr.sparse_tensor.values.dims):
                    return None
                params += int(np.prod(attr.sparse_tensor.values.dims))
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


_EXCLUDED = ["LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"]


def _calculate_memory(model: onnx.ModelProto, trace_path: str) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {i.name for i in graph.initializer}
    init_names.update(i.name for i in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None
    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None
        node_outputs[node.name] = list(node.output)
        for o in node.output:
            if o:
                tensor_names.add(o)
    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, Any] = {}
    tensor_map = {
        t.name: t
        for t in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    tensor_names.update(tensor_map.keys())
    for tn in tensor_names:
        item = tensor_map.get(tn)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tt = item.type.tensor_type
        if not tt.HasField("shape"):
            return None
        n = 1
        for dim in tt.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            n *= dim.dim_value
        if tn in ["input", "output"]:
            continue
        npd = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        tensor_memory[tn] = n * np.dtype(npd).itemsize
        tensor_dtypes[tn] = npd
    seen: set[str] = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for o in node.output:
            if o and o != "output":
                item = tensor_map.get(o)
                if item is None or not item.type.HasField("tensor_type"):
                    return None
    with open(trace_path) as f:
        trace = json.load(f)
    for ev in trace:
        if ev.get("cat") != "Node" or "args" not in ev:
            continue
        if "output_type_shape" not in ev["args"]:
            continue
        nm = ev.get("name").replace("_kernel_time", "")
        if nm not in node_outputs:
            continue
        for i, sd in enumerate(ev["args"]["output_type_shape"]):
            if i >= len(node_outputs[nm]):
                continue
            on = node_outputs[nm][i]
            if on not in tensor_dtypes:
                continue
            isz = np.dtype(tensor_dtypes[on]).itemsize
            mem = isz * sum(int(np.prod(d)) for d in sd.values())
            tensor_memory[on] = max(tensor_memory[on], mem)
    return sum(tensor_memory.values())


# Bind to official versions when available -----------------------------------
if _OFFICIAL is not None:
    convert_to_numpy = _OFFICIAL.convert_to_numpy
    sanitize_model = _OFFICIAL.sanitize_model
    calculate_params = _OFFICIAL.calculate_params
    calculate_memory = _OFFICIAL.calculate_memory
    SOURCE = "official neurogolf_utils"
else:
    convert_to_numpy = _convert_to_numpy
    sanitize_model = _sanitize_model
    calculate_params = _calculate_params
    calculate_memory = _calculate_memory
    SOURCE = "local fallback (official utils not importable)"


# ----------------------------------------------------------------------------
def _load_task(task_dir: str, task_num: int) -> dict[str, Any] | None:
    p = os.path.join(task_dir, f"task{task_num:03d}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        task: dict[str, Any] = json.load(f)
    return task


def _all_valid_examples(
    examples: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    out = []
    for key in ["train", "test", "arc-gen"]:
        for ex in examples.get(key, []):
            b = convert_to_numpy(ex)
            if b is not None:
                out.append((ex, b))
    return out


def audit_one(
    onnx_path: str,
    examples: dict[str, Any] | None,
    run_correctness: bool = True,
) -> dict[str, Any]:
    """Returns dict with params, memory, cost, points, n_pass, n_fail, status."""
    res: dict[str, Any] = dict(
        onnx=os.path.basename(onnx_path),
        params=None,
        memory=None,
        cost=None,
        points=None,
        n_pass=0,
        n_fail=0,
        status="ok",
        filesize=os.path.getsize(onnx_path),
    )

    if res["filesize"] > 1.44 * 1024 * 1024:
        res["status"] = "FILESIZE_OVER_LIMIT"
        res["points"] = 0.0
        return res

    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        res["status"] = f"load_error:{e}"
        res["points"] = 0.0
        return res

    # banned op check (mirror score_network)
    for node in model.graph.node:
        if node.op_type.upper() in _EXCLUDED or "Sequence" in node.op_type:
            res["status"] = f"BANNED_OP:{node.op_type}"
            res["points"] = 0.0
            return res

    # sanitize + session w/ profiling, exactly like the official validator
    try:
        sanitized = sanitize_model(onnx.load(onnx_path))
        if not sanitized:
            res["status"] = "sanitize_failed"
            res["points"] = 0.0
            return res
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        # unique prefix avoids profile clobbering across tasks
        opts.profile_file_prefix = f"audit_{res['onnx']}"
        sess = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
    except Exception as e:
        res["status"] = f"session_error:{e}"
        res["points"] = 0.0
        return res

    # correctness over ALL valid examples (exact array equality, >0 decode)
    if run_correctness and examples is not None:
        valid = _all_valid_examples(examples)
        for _ex, b in valid:
            try:
                out = sess.run(["output"], {"input": b["input"]})[0]
                pred = (out > 0.0).astype(np.float32)
                if np.array_equal(pred, b["output"]):
                    res["n_pass"] += 1
                else:
                    res["n_fail"] += 1
            except Exception:
                res["n_fail"] += 1

    trace_path = sess.end_profiling()
    try:
        mem = calculate_memory(sanitized, trace_path)
        params = calculate_params(sanitized)
    except Exception as e:
        res["status"] = f"score_error:{e}"
        res["points"] = 0.0
        try:
            os.remove(trace_path)
        except Exception:
            pass
        return res
    try:
        os.remove(trace_path)
    except Exception:
        pass

    if mem is None or params is None or mem < 0 or params < 0:
        res["status"] = "unscorable"
        res["points"] = 0.0
        return res

    res["params"] = int(params)
    res["memory"] = int(mem)
    res["cost"] = int(params + mem)
    # If any example fails, the network earns ZERO for that task on the real LB.
    if run_correctness and res["n_fail"] > 0:
        res["status"] = "INCORRECT"
        res["points"] = 0.0
    else:
        res["points"] = max(1.0, 25.0 - math.log(max(1.0, params + mem)))
    return res


def audit_dir(
    onnx_dir: str,
    task_dir: str | None = None,
    out_csv: str | None = None,
    run_correctness: bool = True,
    limit: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Score every ``task*.onnx`` in ``onnx_dir``; return rows sorted worst-first."""
    onnx_files = sorted(glob.glob(os.path.join(onnx_dir, "task*.onnx")))
    if limit:
        onnx_files = onnx_files[:limit]
    if verbose:
        print(f"[audit] scoring source: {SOURCE}")
        print(f"[audit] found {len(onnx_files)} onnx files in {onnx_dir}")
        print(f"[audit] correctness check: {'ON' if run_correctness else 'OFF'}")

    rows: list[dict[str, Any]] = []
    for i, op in enumerate(onnx_files):
        base = os.path.basename(op)
        try:
            tnum: int | None = int("".join(ch for ch in base if ch.isdigit())[:3])
        except Exception:
            tnum = None
        examples = _load_task(task_dir, tnum) if (task_dir and tnum) else None
        r = audit_one(
            op, examples, run_correctness=run_correctness and examples is not None
        )
        r["task"] = tnum
        rows.append(r)
        if verbose and (i + 1) % 25 == 0:
            print(f"  .. {i + 1}/{len(onnx_files)} done")

    # Sort by points ASC (worst first) so the worklist is the top of the file.
    rows.sort(
        key=lambda x: (
            x["points"] if x["points"] is not None else -1,
            x["cost"] if x["cost"] is not None else 1e18,
        )
    )

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "task",
                    "onnx",
                    "points",
                    "cost",
                    "params",
                    "memory",
                    "filesize",
                    "n_pass",
                    "n_fail",
                    "status",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r["task"],
                        r["onnx"],
                        f"{r['points']:.4f}" if r["points"] is not None else "",
                        r["cost"],
                        r["params"],
                        r["memory"],
                        r["filesize"],
                        r["n_pass"],
                        r["n_fail"],
                        r["status"],
                    ]
                )
        if verbose:
            print(f"[audit] wrote {out_csv}")

    # Summary
    total = sum(r["points"] for r in rows if r["points"] is not None)
    n_incorrect = sum(1 for r in rows if r["status"] == "INCORRECT")
    n_problem = sum(1 for r in rows if r["status"] not in ("ok", "INCORRECT"))
    missing = 400 - len(rows)
    if verbose:
        print("\n================= SUMMARY =================")
        print(f"tasks scored        : {len(rows)}")
        print(f"missing (no onnx)   : {missing}  (these earn 0 on the real LB)")
        print(f"INCORRECT (fail>0)  : {n_incorrect}  <-- earning 0, highest priority")
        print(f"other problems      : {n_problem}")
        print(f"sum of points (this set): {total:.2f}")
        print(f"implied LB if missing=0 : {total:.2f} / 10000")
        print("\n----- BOTTOM 25 (attack these first) -----")
        print(f"{'task':>4} {'pts':>7} {'cost':>12} {'params':>9} {'mem':>9}  status")
        for r in rows[:25]:
            pts = f"{r['points']:.3f}" if r["points"] is not None else "n/a"
            print(
                f"{str(r['task']):>4} {pts:>7} "
                f"{str(r['cost']):>12} {str(r['params']):>9} "
                f"{str(r['memory']):>9}  {r['status']}"
            )
        print("===========================================")
    return rows
