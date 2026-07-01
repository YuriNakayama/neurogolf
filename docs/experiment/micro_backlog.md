# Micro Improvement Backlog

This file tracks exact local cost reductions that are too small, mutually
exclusive, or not yet worth a submit cycle by themselves. They should be
revisited when they can be bundled without breaking `n_fail=0`.

## 20260701

| task | candidate | local cost | est gain | status | notes |
|---|---|---:|---:|---|---|
| 219 | residual alias bundle: `mul_174 -> gat_164`, `gat_468 -> gat_392`, `red_497 -> red_345`, `red_501 -> red_425`, `red_505 -> red_429`, `pad_521 -> pad_445`, `sli_540 -> gat_537__h16` | `17240 -> 15964` | `0.0770` | submit-worthy queued | Found by case313 explorer after `task158` was already submitted. Strongest queued next-cycle candidate; avoid late `Max` aliases that failed correctness. |
| 174 | `mismatch_inside_43 -> mismatch_42` | `9326 -> 8978` | `0.0380` | submit-worthy queued | Found by case313 explorer after `task158` was already submitted. Rewire the `Cast` consumer and prune the dynamic inside-mask chain. Use as a next-cycle candidate if current floor allows. |
| 158 | `mask_b_u8 -> nonbg_u8` | `33717 -> 32415` | `0.0399` | exact but not selected | Mutually exclusive with case313 `mask_a_u8 -> nonbg_u8`; applying both collapses pair channels and gives `n_fail=152`. Keep as an alternative if case313 is rejected. |
| 023 | case3 surgery cleanup | `11992 -> 11982` | `0.0008` | exact micro | Params/index cleanup from case301; too small alone. |
| 066 | case3 surgery cleanup | `16838 -> 16835` | `0.0002` | exact micro | Too small alone. |
| 117 | case3 surgery cleanup | `4158 -> 4157` | `0.0002` | exact micro | Too small alone. |
| 222 | case3 surgery cleanup | `7597 -> 7595` | `0.0003` | exact micro | Too small alone. |
| 361 | case3 surgery cleanup | `4787 -> 4779` | `0.0017` | exact micro | Best known params-only micro from case301. |
| 379 | case3 surgery cleanup | `10774 -> 10773` | `0.0001` | exact micro | Too small alone. |
