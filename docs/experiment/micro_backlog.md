# Micro Improvement Backlog

This file tracks exact local cost reductions that are too small, mutually
exclusive, or not yet worth a submit cycle by themselves. They should be
revisited when they can be bundled without breaking `n_fail=0`.

## 20260701

| task | candidate | local cost | est gain | status | notes |
|---|---|---:|---:|---|---|
| 219 | residual alias bundle: `mul_174 -> gat_164`, `gat_468 -> gat_392`, `red_497 -> red_345`, `red_501 -> red_425`, `red_505 -> red_429`, `pad_521 -> pad_445`, `sli_540 -> gat_537__h16` | `17240 -> 15964` | `0.0770` | adopted case314 | Found by case313 explorer after `task158` was already submitted. Avoid late `Max` aliases that failed correctness. |
| 174 | `mismatch_inside_43 -> mismatch_42` | `9326 -> 8978` | `0.0380` | adopted case315 | Found by case313 explorer after `task158` was already submitted. Rewired the `Cast` consumer and pruned the dynamic inside-mask chain. |
| 271 | packed two live float16 channels, then one-hot `Conv` channel projection and final pad | `1187 -> 1090` | `0.0853` | adopted case321 | Same sparse-tail pattern as cases 318-320. Public Score improved `7183.64 -> 7183.72`. |
| 030 | packed live tail channels, then one-hot `QLinearConv` channel projection and final pad | `2396 -> 2230` | `0.0718` | adopted case322 | Public Score improved `7183.72 -> 7183.79`. |
| 023 | packed live tail channels, then one-hot `QLinearConv` channel projection and final pad | `11992 -> 11222` | `0.0664` | adopted case323 | Public Score improved `7183.79 -> 7183.86`. |
| 266 | packed live tail channels, then projection/pad replacement | `350 -> 334` | `0.0468` | exact candidate | Temporary artifact: `/tmp/task266_packed_tail.onnx`. Small absolute delta, but keep because small gains can accumulate. |
| 158 | `mask_b_u8 -> nonbg_u8` | `33717 -> 32415` | `0.0399` | exact but not selected | Mutually exclusive with case313 `mask_a_u8 -> nonbg_u8`; applying both collapses pair channels and gives `n_fail=152`. Keep as an alternative if case313 is rejected. |
| 080 | `isol -> nonb`, `ecells0 -> eblk`, `ccells -> cblk` | `17136 -> 16836` | `0.0176` | adopted case275 | Bundled exact mask aliases; moved Public LB. |
| 080 | remove redundant `hascorner` / `cblk2` guard | `16836 -> 16734` | `0.0061` | adopted case300 | Semantic guard removal; Public LB moved despite small gain. |
| 080 | `sepmask -> lineOrOut` | `16734 -> 15650` | `0.0670` | adopted case312 | Removed full separator predicate chain. |
| 064 | bundled high-bound aliases, including `h_high_p1 -> h_high24` / `v_high_p1 -> v_high24` family | `12478 -> 12430` | `0.0039` | exact micro | Mentioned in cases274/286/301; too small standalone, keep only for a larger same-task bundle. |
| 338 | `tr_pair -> tr_pair_0` or `bl_pair -> bl_pair_0` | `17939 -> 17819` | `0.0067` | rejected standalone case276 | Single exact alias did not move Public LB; applying both gives `n_fail=9`, so only revisit with a larger compatible bundle. |
| 101 | local alias bundle from case287 | `17755 -> 17706` | `0.0028` | rejected hidden-unsafe case287 | Local `n_fail=0` but Public Score regressed sharply; record to avoid rediscovery, not as an actionable candidate. |
| 023 | duplicate initializer cleanup: `Ksc -> Ksb`, `Khc -> Khb`, `Kvc -> Kvb` | `11992 -> 11982` | `0.0008` | rejected standalone case299; keep for bundle | Exact locally; case299 Public LB did not move. Same cleanup is the case301 surgery candidate. |
| 066 | case3 surgery cleanup | `16838 -> 16835` | `0.0002` | exact micro | Too small alone. |
| 117 | case3 surgery cleanup | `4158 -> 4157` | `0.0002` | exact micro | Too small alone. |
| 222 | case3 surgery cleanup | `7597 -> 7595` | `0.0003` | exact micro | Too small alone. |
| 361 | case3 surgery cleanup | `4787 -> 4779` | `0.0017` | exact micro | Best known params-only micro from case301. |
| 379 | case3 surgery cleanup | `10774 -> 10773` | `0.0001` | exact micro | Too small alone. |
| 133 | `sg_gate_2 -> sg_eqi_2` | `33573 -> 33572` | `0.00003` | exact micro | Found by case316 source-crop/single-consumer scan; one-byte equivalent `ReduceMin` alias. |
| 133 | `sg_gate_3 -> sg_eqi_3` | `33573 -> 33572` | `0.00003` | exact micro | Found by case316 source-crop/single-consumer scan; independent one-byte equivalent `ReduceMin` alias. |
| 133 | `sg_gate_4 -> sg_eqi_4` | `33573 -> 33572` | `0.00003` | exact micro | Found by case316 source-crop/single-consumer scan; independent one-byte equivalent `ReduceMin` alias. |
