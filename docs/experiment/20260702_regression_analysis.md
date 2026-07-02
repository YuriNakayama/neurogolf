# 20260702 regression / no-move analysis

## Problem

Recent cycles after the `7186.71` floor produced either unchanged Public Score
or sharp regressions:

| case | submit type | local signal | Public result | lesson |
|---:|---|---|---|---|
| 411 | isolated `task342` micro | gain `0.00569`, `n_fail=0` | unchanged | below display/movement threshold |
| 412 | isolated `task338` micro | gain `0.00671`, `n_fail=0` | unchanged | below movement threshold |
| 413 | isolated `task392` micro | gain `0.00514`, `n_fail=0` | unchanged | below movement threshold |
| 414 | five-task unisolated micro bundle | combined gain `0.02629`, all `n_fail=0` | `7186.71 -> 7169.75` | hidden-unsafe member in bundle |

Earlier case403 and case406 show the same failure mode:

- case403 `task017` local candidate-row pruning regressed sharply;
- case406 filler bundle regressed and made attribution ambiguous.

## Root Causes

1. Local exactness is only a filter, not proof of hidden safety.

   `n_fail=0` covers train/test/arc-gen, but hidden/private examples still
   reject local-usage pruning and broad equivalence aliases. This is especially
   true when a candidate removes rows, columns, labels, candidate branches, or
   table entries selected only outside the generated coverage.

2. Submit cadence pushed below-threshold candidates into weak submits.

   The score gain is logarithmic, but Kaggle Public is displayed at two
   decimals. Isolated gains around `0.005-0.007` often do not visibly move the
   board, even when they may be technically valid.

3. Bundling unisolated micros saves submissions only when each member is already
   Kaggle-safe.

   Case414 bundled five locally exact candidates and regressed by `-16.96`.
   The problem was not the combined gain formula; it was treating unisolated
   local-equivalence aliases as independent low-risk changes.

4. Old scratch candidates become stale after later accepted changes.

   Candidate costs and graph semantics must be re-derived against the current
   accepted baseline. Reusing old paths without isolated Kaggle evidence mixes
   different risk contexts.

## Updated Submit Policy

Use this gate before every future submit.

### Hard Reject Locally

Never submit candidates in these categories unless the change is specifically
repaired and isolated with a new plan:

- known hidden-unsafe tasks/families: `task017`, `task101`, broad `task286`;
- known Kaggle ERROR families: `task233` uint8 TopK, `task285` uint8 TopK;
- contaminated filler families from rejected bundles: `task305`, `task366`;
- case414 unisolated candidate set: `task090`, `task209`, `task370`,
  `task308`, `task293` from the listed scratch paths;
- case411-413 unchanged standalone micros: `task342`, `task338`, `task392`
  unless a same-task bundle materially changes expected gain.

### Submit Thresholds

- `gain >= 0.020`: submit if isolated, locally exact, and mechanically
  explainable.
- `0.010 <= gain < 0.020`: submit only if the task or exact pattern has prior
  Kaggle-valid evidence, or if it is the single best cadence candidate and is
  isolated.
- `gain < 0.010`: bank only. Do not submit standalone.
- Bundles: allowed only when every member is already Kaggle-validated or the
  bundle is explicitly a bisection experiment. Otherwise bundling is disabled.

### Risk Classification

Low risk:

- removing unused initializers;
- replacing duplicate constants or shape/index tensors;
- local graph cleanup that does not change selector, candidate, or table
  coverage.

Medium risk:

- aliasing one computed tensor to another within a selector path;
- dtype or shape rewires in value paths;
- changes derived from old scratch graphs.

High risk:

- candidate row/column pruning;
- lookup/table/hash pruning;
- output-path aliases;
- uint8 TopK or runtime-sensitive dtype changes;
- any bundle containing a not-yet-isolated member.

## Operational Changes

1. Keep the 15-minute cadence as a polling cadence, not a forced-submit cadence.
   If no candidate clears the gate, write a no-submit result and keep searching.

2. During Kaggle pending time, search for the next isolated candidate, but do
   not stage it until the previous submission is accepted or restored.

3. For rejected bundles, do not bisect immediately unless no stronger isolated
   candidate exists. Bisection spends multiple submissions to recover tiny
   micros, which is poor value after a large regression.

4. Record every candidate as one of:

```text
adopted
rejected-hidden
rejected-unchanged
rejected-error
banked-low-gain
blocked-local-fail
```

5. Prefer one clear `gain >= 0.02` structural candidate over many
   unvalidated micros. The scoring formula rewards relative reduction, but
   correctness and hidden safety dominate.

## Immediate Next Step

Do not submit the subagent's `task035` candidate by itself:

```text
task035: 2262 -> 2255, gain 0.003099, n_fail=0
```

It is below the standalone threshold and came from the same broad recovery
candidate pool. Keep it banked only. The next cycle should search for a fresh
isolated structural reduction with current-baseline `gain >= 0.020`.
