# 20260628 case3 result — floor 7172.43 を独立再確認、全自動レバー枯渇を本セッションでも実証

## 実 LB / 採否

- **採用なし（提出なし）**。floor 7172.43 を上回る正答かつ cost 減の変更を発見できず。
- 退行ゼロ最優先のため、不正答候補は破棄。floor bundle（data/output/onnx, LB 7172.43）を維持。

## 検証した内容と結果

| レバー | 手法 | 結果 |
|---|---|---|
| faithful 再ランキング | onnx 1.20.0/ort 1.24.1 で全400 cost-only | sum 7172.74。floor と整合、0点タスク無し |
| fp32 一括 narrowing | 自前トレース計測 | **偽の機会**（自前計測バグ。faithful では task400=1866 等、既に小） |
| whole-graph fp16 | onnxconverter_common keep_io_types | 上位は「already fp16」拒否。変換可分も **cost 増**（Cast 挿入、054:27133→61313, 187:40446→58446） |
| surgery（高 param 標的） | index/conv1x1/bcast | 高 param は専用 conv 重み（t367 rect_w[25,2,9,9]=4050 等）で削減不可 |
| surgery（全パス・全400） | cleanup/identity/index/bcast/conv1x1 | cost-only で 5 タスク減（005/084/099/133/185, 計 -2066B ≈+0.60点）に見えたが… |
| ↑ の faithful 正答検証 | 全 example audit | **5件すべて INCORRECT**（n_fail 175-267）。identity除去/bcast が floor グラフの意味を壊す → 全 reject |

## 確定した教訓

1. **cost-only の「win」は必ず faithful 正答検証する**。意味保存とされる surgery パス
   （eliminate_identity_elementwise / broadcast_compress）も floor の特定グラフでは出力を変える。
   正規の `apply_surgery`（run_correctness=True）なら正しく reject される。私の cost-only 近道が偽 win を生んだ。
2. **floor = combined_best は全自動レバーで局所最適**: surgery（正答付き）/ fp16 / fp32 / 高 param 削減すべて 0 有効 win。
   E42（全 worktree 横断で combined_best が最適）を本セッションが独立に再確認。
3. **この隔離コンテナでは dsl-harvest 不可**（sibling worktree 無し）。歴史的主力レバーが使えない。

## 追検証（第2フェーズ）: 幾何変換・cheap-solver も枯渇

- メモリの「task002 8連結 flood-fill +0.70」案を numpy 全例検証 → **完全に誤り**:
  4連結==8連結 は 268 例すべて不一致、8連結は **0/268**（真ルールは4連結, 268/268）。撤回。
- cheap-solver 検出（recolor/transpose/flip/rotate, cost>60）= 3 件のみ:
  t150(158, fliplr) t155(158, flipud) t380(99, rot90)。いずれも floor が既に最小付近:
  - t150/t155 は**可変サイズ**で静的 flip 不可。floor は ReduceL2 で動的にサイズ検出する 4 ノード版（158）。
    solver bank の静的 flip は `_const_dim` 前提で適用不可。
  - t380（全 3×3 定数）は floor が**単一 Einsum 99 params/0 memory**。bank の build_rot270 は正答だが
    **cost 36030**（[1,10,30,30] 中間テンソル）で完敗。floor が圧倒的に優秀。
- → 幾何変換タスクも floor が bank より遥かに golf 済み。**検証した全 10 レバーで positive win 0**。

## 追検証（第3フェーズ）: 中コスト帯 per-task redesign — 候補も全て near-optimal

ユーザー方針「per-task redesign 継続」に従い中コスト帯(1000-5000)を精査:
- **task196**(4538, 5 MaxPool): 閉ループ(完全な矩形枠)を検出して 1→3 recolor。内側 enclosure 判定の
  **非局所** flood で、step 削減は大きい private 図形で退行リスク → 不採用。
- **task302**(1774, 6ノード, QLinearConv 5×5): 最小一致 **k=5**（k=3 では不一致）→ 5×5 conv は正当、縮小不可。
- **task287**(1994): 最小一致 k=9（対称性 Einsum で実装）。task246/168/051: 純粋局所でない(k=None)。
- → 中コスト帯も bundle が最小 conv/最小 step/専用アルゴで実装済み。redesign win 無し。

**系統的 conv-shrink 検査**（中コスト同形状・全純粋局所タスク）: 各タスクの真の最小一致 k と
bundle の実カーネルサイズを比較。**ほぼ全タスクで bundle_k ≤ min_k**（bundle は naive k×k lookup
ではなく 1×1 conv + shift/ReduceMax 等の多段で既に安く実装）。bundle_k>min_k は t032/t139
（共に cost~910, 9→7/5 縮小可だが gain +0.04 と無視可能）のみ。→ conv 縮小の余地も systematic に無し。

## 追検証（第4フェーズ）: 系統的検出を計6種に拡張 — 全て win 0

| 系統的検出 | 結果 |
|---|---|
| cheap-solver (recolor/transpose/flip/rotate) | 3件のみ、全て floor が最小（t380=Einsum99 vs bank36030） |
| conv-shrink (真の最小k vs bundleカーネル) | bundle_k≤min_k がほぼ全て。縮小候補 t032/t139 のみ gain+0.04 |
| symmetry/tile/scale 補完 | 該当 1件(t249, 既に287)のみ |
| 固定距離 dilation（step削減余地） | growth型2件(t187/t110)とも **VARIABLE**（grid依存）で削減不可 |
| surgery 全パス (faithful) | win 0（cost減5件は全 INCORRECT） |
| fp16 whole-graph | cost 増 |

→ 単純パターン全クラス（幾何/局所/対称/タイル/dilation/recolor）と意味保存変換を網羅検査し、
floor が全クラスで最小実装と確定。**6 系統的検出 + 15 deep-dive で positive win 0**。

## 追検証（第5フェーズ）: lookup/residual ソルバの faithful 再評価 — win 0

仮説「元 blend は repo-default(false-negative) audit で安価 lookup を誤却下したかも」を faithful で検証:
- 純粋局所タスク全て(63候補)に solve_small_lookup/lookup/residual3/5 を構築し faithful audit。
- **全て正答(n_fail==0)だが floor より桁違いに高コスト**（例 t015: floor=900 vs 候補 324,062〜3,550,131,
  t192: floor=8621 vs 10,987,500）。bank の lookup は巨大テーブル/中間テンソルを作る。
- → 元 blend の却下は**正当**（false-negative 起因の取りこぼし無し）。floor の極小実装(t015=900)が圧勝。
- **7 系統的検証すべてで positive win 0**。floor 7172.43 が最適であることが完全に確定。

## 第6フェーズ: degrade-guard + 並行プロセス検出

- **floor bundle 健全性確認**（`dev/submit --dry-run`）: 全400タスク validation 通過、submission.zip
  生成成功、constraint/size 違反なし。floor は intact かつ submission-ready（退行なし）。提出枠は未消費。
- **並行 autonomous プロセスを検出**: 同一作業ツリーで `floodfill.build_floodfill_8conn` +
  `solvers.solve_floodfill`（8連結 flood-fill）を実装中（未コミット WIP）。
- **その solve_floodfill を faithful 検証 → 退行確定**: task002 cost 33317→17224 だが **n_fail=268/268**、
  task251 cost 4920→4344 だが **n_fail=144/266**。cost だけ「win」に見えるが全滅。SOLVERS の audit が
  正しく reject するため solve のまま回せば安全。メモリに「手動 override 厳禁」警告を追加。並行 WIP は不触。

## 現状と次の一手

- floor 7172.43 は全公開手法の上限付近。LB トップは非公開チーム 7942.46（≒目標 7950）。
- 7665 でも E16 の定量分析どおり ~230 タスクを各 2-4× golf する専門手作業（多人数・数週間規模）が必要で、
  自動反復・単独コンテナでは構造的に到達不能。
- 唯一の残レバーは **per-task の専門 redesign**（歴史的に 6 タスク・計~+0.35）。ただし上位高コストは
  すべて非局所の専門ソルバ（可変出力抽出/連結伝播/bit-pack）で、安価な厳密ルールが存在しない。
- 次サイクル: redesign 候補を「同形状・局所(k×k)・naive net」に厳密に絞って 1 タスク試行する。
  該当が無ければ floor 維持が最善（退行リスクを取らない）。
