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

## 現状と次の一手

- floor 7172.43 は全公開手法の上限付近。LB トップは非公開チーム 7942.46（≒目標 7950）。
- 7665 でも E16 の定量分析どおり ~230 タスクを各 2-4× golf する専門手作業（多人数・数週間規模）が必要で、
  自動反復・単独コンテナでは構造的に到達不能。
- 唯一の残レバーは **per-task の専門 redesign**（歴史的に 6 タスク・計~+0.35）。ただし上位高コストは
  すべて非局所の専門ソルバ（可変出力抽出/連結伝播/bit-pack）で、安価な厳密ルールが存在しない。
- 次サイクル: redesign 候補を「同形状・局所(k×k)・naive net」に厳密に絞って 1 タスク試行する。
  該当が無ければ floor 維持が最善（退行リスクを取らない）。
