# 20260701 submit-saving policy

## Context

The current accepted Public Score floor is `7185.05` after case336. Case337
showed that a large local-equivalence bundle can pass local `audit_one` but
still fail hidden/public validation badly:

```text
task286 local: 47013 -> 43985, n_fail=0
Public Score: 7185.05 -> 7170.81
```

The competition score is logarithmic:

```text
points = max(1, 25 - ln(cost))
cost = params + memory_bytes
gain = ln(old_cost / new_cost)
```

Functional correctness is still a hard gate. Kaggle Public Score is the source
of truth for adoption.

## Risk Classes

### Low Risk

Submit-safe only when `n_fail=0` and cost decreases, but still bank if tiny:

- duplicate initializer removal
- unused initializer/node removal
- constant broadcast compression where ONNX broadcasting exactly restores shape
- dtype/index initializer cleanup with unchanged consumer semantics

### Medium Risk

Use single-task submit only when expected gain is material:

- small alias bundles, roughly 1-5 rewires
- single-consumer rewires
- short branch pruning with simple value equivalence

### High Risk

Do not spend standalone submits unless the expected gain is very large and no
safer candidate is available:

- broad local-equivalence bundles
- bitwise topology rewrites
- selector/topology rewrites
- Scatter/shape-sensitive rewires
- any family already marked hidden-unsafe in `micro_backlog.md`

## Submit Thresholds

Use these as default gates before spending a Kaggle submit:

```text
low risk:    submit if gain >= 0.020, otherwise bank
medium risk: submit if gain >= 0.030, otherwise bank
high risk:   submit only if gain >= 0.080 and isolated to one task
```

Micro candidates below threshold should be added to `micro_backlog.md`, not
discarded. When enough low-risk micro candidates accumulate, submit them as a
small same-risk batch rather than individually.

## Micro Batching

To save submits while preserving diagnosability:

1. Bank every exact local micro candidate with task, transformation, cost delta,
   gain, temp path, and risk class.
2. Batch only low-risk candidates together first.
3. Target a batch expected gain of at least `0.030`.
4. If a batch regresses, bisect the batch rather than retesting each candidate
   one-by-one.
5. Do not mix broad local-equivalence candidates with low-risk cleanup
   candidates in the same submit.

## 30-Minute Submit Cadence

Default cadence after 2026-07-01 23:49 JST:

```text
submit window: about once every 30 minutes
minimum standalone candidate: same thresholds as above
minimum low-risk batch: total expected gain >= 0.030
minimum mixed-risk batch: total expected gain >= 0.050 and no hidden-unsafe family
```

The clock is a budget guard, not a reason to submit weak material. At each
30-minute window:

1. Submit immediately if there is one isolated threshold-clearing candidate.
2. Otherwise submit a low-risk micro batch if accumulated expected gain clears
   `0.030` and every changed task has `audit_one n_fail=0`.
3. Otherwise skip the Kaggle submit, write/update the case result, and keep
   banking exact local candidates in `micro_backlog.md`.

For batched submits, keep the batch small enough to bisect in one follow-up:

- Prefer 2-5 low-risk tasks per submit.
- Avoid combining task families with known hidden/public sensitivity.
- Keep medium/high-risk candidates isolated unless their standalone gain is
  strong enough to justify the submit by itself.
- If a batch regresses, restore baseline, then test only the largest-gain half
  first; do not spend one submit per micro candidate unless the batch total was
  large enough to justify the bisection.

## Scheduler Policy

Keep automatic scheduled submits disabled unless explicitly running a controlled
degrade check. Manual case refs must remain easy to attribute to a specific
plan/result pair.
