# case2 — DSL-primitive ONNX solvers (override on the case1 baseline)

タスクごとに **最小の手書き ONNX**（幾何 / 色 / タイル等の DSL プリミティブ）を生成し、
case1 baseline（boristown bundle, LB 7159.44）の `taskNNN.onnx` を **「厳密に正答 かつ
strictly cheaper」のときだけ上書き(override)** するビルドケース。正答は
`src/evaluate.audit_one`（公式スコアラーミラー）で train+test+arc-gen 全例を検証する。

## 構成

```
case2/
├── onnx_ops.py    # プリミティブ → 最小 ONNX グラフ（param-free 優先、anchored top-left）
├── dsl.py         # numpy 参照変換（grid->grid）。solver が例に対して候補を照合する
├── solver.py      # タスク JSON から該当プリミティブを推定し最小 ONNX を emit
├── build.py       # 全 400 タスクで solver を回し、baseline を override（exact かつ cheaper のみ）
├── downcast.py    # float32→float16 ロスレスダウンキャスト試行（補助）
├── submit_loop.py # 目標スコアまで提出（case1 から copy: case 独立）
└── __main__.py    # typer CLI: build / package / submit
```

## プリミティブと cost（`src/evaluate` 実測, cost = params + memory）

| プリミティブ | ONNX | cost | pts |
|---|---|---|---|
| identity | `Identity` | 0 | 25.00 |
| transpose | `Transpose perm=[0,1,3,2]` | 0 | 25.00 |
| flip_h / flip_v | `Gather`(axis=3/2, reverse-index) | ~30 | 21.6 |
| rot180 | `Gather`×2 | ~60 | — |
| recolor | `Gather`(axis=1, perm[10]) | 10 | 22.7 |
| rot90 / rot270 | `Transpose`+`Gather`（中間 36000） | ~36k | 14.5 |
| tile | `Gather`×2（繰り返しインデックス） | ~60 | — |

要点（実測で確立）:
- **cost に MAC は寄与しない**（`cost = params + memory` のみ）。
- 1 ノードで `input→output` 直結なら中間テンソルが memory に乗らず最小。
- 反転/回転は **占有領域内のみを Gather で並べ替える**（全軸 reverse は枠外へずれて不正答）。

## 調査で確定した重要な結論（精度向上の限界）

case1 baseline（LB 7159.44, ローカル合計 6993.61）は **既に専門家により限界まで golf 済み**で、
以下の自動・半自動手法はいずれも **このベースラインを上回れない**ことをデータで確認した:

1. **DSL プリミティブ再解**: 単純な幾何/色変換タスク（9 件）は baseline が既に最小コストで解いており、
   override 利得 = **0**。
2. **dtype ダウンキャスト**（float32→fp16/uint8）: 高コストタスクは既に per-tensor で dtype 最適化済み
   （uint8/bool/int8 混在）。一律 fp16 化は Cast ノード増で **コスト増 or セッション破綻**。
3. **グラフ最適化**（onnxslim / onnxoptimizer）: baseline は最適化済みで利得 **≈0**（400 タスクで 0 wins）。
4. 高コストタスク（cost 30k–70k）の大半は flood-fill / 連結成分 / トポロジー等の **本質的に反復が必要な
   変換**で、baseline の数十〜数千ノード解は near-optimal。安く厳密に解く一般手法は存在しない。

**目標 7665（+506）には ~100 件の hard task を個別に手作業で golf する必要があり**（上位 Kaggle チームが
数週間かけている作業に相当）、本ケースの自動パイプラインだけでは到達できない。

## 実提出で判明した決定的事実（2026-06-25）

1. **より良い base が存在**: `franksunp-7166` bundle（Kokinn continuation 390 + Biohack Mix 10）が
   **Public Score 7166.63**（case1 の 7159.44 より +7）。`data/lake/case2-base7166/` に取得済み。
   case2 の override base はこれに切替（boristown-7159 ではない）。
2. **ローカルスコアラーは Kaggle より「弱い」**: local fallback scorer の onnxruntime は一部 op
   （**MaxUnpool / 負パッド ConvTranspose**）を実行できず、それを使う 8 タスク
   （347, 45, 127, 135, 146, 149, 240, 384）を **INCORRECT / score_error と誤判定**する。
   これらは **Kaggle 本番では正常採点・正答している**（= false negative）。
3. **override は危険**: 上記 false-negative の task347 を「0点だから」と自前 ONNX で上書き提出したところ、
   全 269 visible 例で exact だったにもかかわらず **LB が 7166.63 → 7161.55 に低下（−5.08）**。
   base の task347 は hidden set で正しく、私の解は visible≠hidden で overfit していた。
   → **base が hidden set で機能しているタスクは、ローカルで失敗判定でも絶対に上書きしない**。
   `build.py` は base と候補が両方 local-ok かつ候補が strictly cheaper のときだけ override する。

**結論**: ローカルで特定可能な「勝てるタスク」は存在しない（base は hidden set で全タスク near-optimal）。
安全な floor は **pure 7166.63**。それ以上は hidden set を盲目的に submit で探る高分散探索になり、
task347 が示した通り **むしろ低下する**ため、自動手法での 7665 到達は不可能と確認した。

## 公開解空間の飽和（実提出 + kernel ランキングで確定）

複数の独立した検証で、**公開されている全手法の上限が 7166.63** であることを確認:

| 検証 | スコア |
|---|---|
| case2 floor (pure 7166 bundle) | **7166.63** ← 全手法の上限 |
| case2 v1 (task347 override) | 7161.55（−5, false-negative 上書き） |
| 先行 case3 blend8safe (36 safe swaps) | 7166.63（**swap しても無利得**） |
| 先行 case3 blend8 (16バンドル best-of, 44 swaps + 8 zero-recovery) | 7150.70（−16, 退行） |
| Kaggle 公開 kernel ランキング最上位（seddik/kokinn/franksunp） | 全て **7166.63** |

- **公開バンドルは全て同一系譜**でほぼ byte 一致（seddik=base と 400/400 一致、kokinn は 3 タスクのみ差・うち安いのは task076 の 29 byte 差のみ）。blend する多様性が無い。
- LB 上位（7942: Matheus & Fritz & Tony 等）の bundle は **非公開**。公開解空間には 7166.63 超は存在しない。
- → **7665 は公開手法の到達範囲外**。上位チーム非公開の hard-task hand-golf（数週間規模）が必須。
  case2 は公開上限 **7166.63**（case1 比 +7）を達成し、override/blend が hidden set で逆効果になる
  ことを実提出データで実証した。

## 実行

```bash
cd backend
# baseline に override をかけてバンドル生成（override 0 件 = baseline 等価）
uv run python -m pipeline.case2 build \
    --baseline-dir ../data/lake/case1-baseline/onnx \
    --task-dir ../data/lake/neurogolf-2026 \
    --out-dir ../data/output/onnx/case2
uv run python -m pipeline.case2 package --onnx-dir ../data/output/onnx/case2
uv run python -m pipeline.case2 submit --zip-path ../data/output/onnx/case2/submission.zip
```

## 設計メモ

- **case 独立**: 他 case を import しない。共有は `src/evaluate` / `src/submit` のみ（`submit_loop.py` は copy）。
- **override は安全側**: exact（n_fail==0）かつ strictly cheaper のときだけ baseline を差し替えるので、
  ローカル合計が baseline を下回ることはない。arc-gen 数百例で検証するため private split でも崩れにくい。
- 次の一手（精度向上）は **hard task の個別手 golf**（stamp/dilation を ConvTranspose、対称補完、
  局所 stencil を小 Conv 等）。各々 `audit_one` で exact 検証してから採用する。
