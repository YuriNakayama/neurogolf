# 20260703 case567 task364 short confirmation

Scope: re-gate the known task364 near miss and exact repair paths only. No
Kaggle submit.

Baseline:

```text
data/output/onnx/task364.onnx
cost=23013 params=113 memory=22900 n_fail=0 status=ok
```

## Re-gate

Audited 115 task364 scratch candidates from:

- `/private/tmp/neurogolf_agent_case543_C/fresh_alias_dce`
- `/private/tmp/neurogolf_case353`
- `/private/tmp/neurogolf_parallel_U/candidates`
- `/private/tmp/neurogolf_parallel_AP/scan_candidates`
- `/private/tmp/neurogolf_parallel_AC/recheck`

Best low-fail rows:

```text
task364_PA7_PB7_to_6.onnx                                  cost=21253 gain=0.079561 n_fail=21
task364_runtime_equiv_alias_dce.onnx                        cost=22133 gain=0.038990 n_fail=21
task364_PA7_to_PA6.onnx                                     cost=22133 gain=0.038990 n_fail=21
task364_PB7_to_PB6.onnx                                     cost=22133 gain=0.038990 n_fail=21
0245_task364_cost19493_gain0.166004_task364_PA7PB7_TO_PA5PB5.onnx cost=19493 gain=0.166004 n_fail=67
```

No candidate in this bounded set had `n_fail=0` with cost below `23013`.

## Exact Repair Paths

Additional case548 repair paths:

```text
/private/tmp/neurogolf_agent_case548_C/task364_repaired_exact.onnx
  cost=23013 gain=0.000000 n_fail=0 status=ok

/private/tmp/neurogolf_agent_case548_C/task364_repair_pb7_from_pb6.onnx
  cost=22573 gain=0.019305 n_fail=171 status=INCORRECT

/private/tmp/neurogolf_case548_task364_repairs/task364_repair_maxpool_pb6.onnx
  cost=22573 gain=0.019305 n_fail=171 status=INCORRECT

/private/tmp/neurogolf_case548_task364_repairs/task364_repair_mul_pb6_gu.onnx
  cost=22573 gain=0.019305 n_fail=21 status=INCORRECT

/private/tmp/neurogolf_agent_case548_C/task364_exact_sparse_initializers.onnx
  unscorable/session_error: missing graph input/initializer for safe_name_12
```

## Decision

No lower-cost exact candidate exists in the checked task364 PA/PB and repair
family. The only exact repair restores the full baseline-cost propagation path.
The lower-cost variants all remove required seventh-step propagation logic,
especially the `PB7 = MaxPool(PB6 * Gu)` branch documented in case548, and fail
local arc-gen/binary boundary cases.

Artifacts written:

```text
/private/tmp/neurogolf_case567_task364/candidates/task364_case567_gate_summary.json
/private/tmp/neurogolf_case567_task364/candidates/task364_case548_repair_regate.json
```
