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
| 266 | packed live tail channels, then projection/pad replacement | `350 -> 334` | `0.0468` | adopted case324 | Public Score improved `7183.86 -> 7183.91`; small exact candidates can move LB. |
| 274 | broadcast-compress `zero_small` from `uint8[1,1,3,3]` to `uint8[1,1,1,1]` | `177 -> 169` | `0.0463` | adopted case333 | Public Score improved `7184.88 -> 7184.93`. |
| 334 | packed uint8 live channels, then one-hot `QLinearConv` projection and final pad | `197 -> 131` | `0.4080` | adopted case330 | Public Score improved `7184.33 -> 7184.74`; distinct from the earlier rejected `197 -> 230` variant. |
| 348 | packed bool live channels, cast to uint8, then one-hot `QLinearConv` projection and final pad | `2273 -> 1968` | `0.1440` | adopted case325 | Public Score improved `7183.91 -> 7184.05`. |
| 348 | remove dead `zero_b` initializer left after packed-tail rewrite | `1968 -> 1858` | `0.0575` | adopted case332 | Public Score improved `7184.82 -> 7184.88`; params-only cleanup on current graph. |
| 003 | packed uint8 live channels, then one-hot `QLinearConv` projection and final pad | `302 -> 263` | `0.1383` | adopted case329 | Public Score improved `7184.19 -> 7184.33`; small absolute task but high relative gain. |
| 052 | packed bool strip tail, cast to uint8, then one-hot `QLinearConv` projection and final pad | `227 -> 209` | `0.0826` | adopted case331 | Public Score improved `7184.74 -> 7184.82`. |
| 325 | packed bool live channels, cast to uint8, then one-hot `QLinearConv` projection and final pad | `2133 -> 1932` | `0.0991` | adopted case326 | Public Score improved `7184.05 -> 7184.15`. |
| 250 | packed bool live channels, cast to uint8, then one-hot `QLinearConv` projection and final pad | `3218 -> 3142` | `0.0239` | adopted case327 | Public Score improved `7184.15 -> 7184.17`; confirms that small exact candidates can still accumulate. |
| 365 | packed bool live channels, cast to uint8, then one-hot `QLinearConv` projection and final pad | `4028 -> 3945` | `0.0208` | adopted case328 | Public Score improved `7184.17 -> 7184.19`; remaining validated packed-tail queue is exhausted. |
| 131 | compatible alias pair: either `old_rows -> green_rows` plus `special_col_valid -> special_col`, or `old_cols -> green_cols` plus `special_col_valid -> special_col` | `3987 -> 3977` | `0.0025` | marginal fallback | Third alias cannot join (`old_rows + old_cols` fails); already submitted once with no Public move, keep only if a larger same-task change appears. |
| 368 | `h4_b -> h4_raw_b` plus `w_probe_idx_i64 -> w_probe_raw_i64` | `6225 -> 6216` | `0.0014` | marginal fallback | Nearby probe aliases break correctness; previous submit did not move Public. |
| 378 | `outer_bottom_f16 -> outer_bottom_raw` plus `outer_right_f16 -> outer_right_raw` | `3728 -> 3722` | `0.0016` | marginal fallback | Larger bbox/selector aliases failed; previous submit did not move Public. |
| 092 | `h_right_idx_active -> h_right_idx` plus `v_bottom_idx_active -> v_bottom_idx` | `6992 -> 6982` | `0.0014` | marginal fallback | Other guard/branch removals failed; previous submit did not move Public. |
| 234 | `rect_h -> rect_extent` plus `rect_w -> rect_extent`, prune dead `Where` nodes | `6853 -> 6845` | `0.0012` | marginal fallback | Other terminal grid aliases failed; previous submit did not move Public. |
| 334 | packed bool/uint8 projection tail, earlier variant | `197 -> 230` | `-0.1547` | rejected locally | Worsened cost; superseded by adopted case330's direct uint8 live-channel projection. |
| 245 | packed bool/uint8 projection tail | `2743 -> 2868` | `-0.0446` | rejected locally | Worsened cost in subagent probe; record to avoid rediscovery. |
| 103 | packed bool/uint8 projection tail | `60 -> 72` | `-0.1823` | rejected locally | Worsened very small task; projection overhead dominates. |
| 002 | packed terminal sparse-tail probe | `21089 -> 21113` | `-0.0011` | rejected locally | Projection added params without reducing memory; record to avoid rediscovery. |
| 226 | packed terminal sparse-tail probe | `1633 -> 1978` | `-0.1917` | rejected locally | Projection overhead and memory increase dominate. |
| 338 | packed terminal sparse-tail probe | `17939 -> 20662` | `-0.1413` | rejected locally | Packed-tail candidate increases memory substantially; do not submit. |
| 395 | packed terminal sparse-tail probe | `165 -> 182` | `-0.0981` | rejected locally | Very small task; projection overhead dominates. |
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
