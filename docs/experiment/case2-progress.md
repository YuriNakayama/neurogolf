# case2 実験ログ — Public Score 7665 を目指す

目標: **Public Score 7665**。base = `franksunp-7166.65` bundle（standing best **7169.91**, 27 win, +3.25）。
方針: golf エンジン(並列 hand-golf ワークフロー)で多数タスクを cost 削減、Kaggle-safe gate で安全統合。

### 実験10: research による golf 技法の精密化（2026-06-27）
4本並列 web 調査 + scorer source 確認で確立した actionable 技法（workflow brief に反映済み）:
- **Scatter→mask blend 書き換え**: `out=data*(1-mask)+value*mask`。ScatterElements/ND の
  bounds/opset エラーを構造的に回避 → TopK+Scatter combo ERROR 問題が消える可能性。
- **静的形状ゲート（最重要）**: scorer は `infer_shapes(strict_mode=True)`、中間 dim が symbolic だと
  **cost=None→0点**。data-dependent Slice crop は失格。**固定長 Gather**（index=Add(top_const,[0..H-1])）
  で shape 固定・値動的にする。crop 窓はビルド時オフライン検出して定数 bake。
- **整数1ch保持**: `ArgMax(axis=1)→[1,1,H,W]` で中間 1/10、最終 op で 10ch 展開→output。
- **float32 統一**: Kaggle CPU EP は float16 kernel を拒否しうる（ERROR 源）。opset13/ir8 が IR 互換安全。
- **Compress も禁止**（scorer `_EXCLUDED`）。確認: `InferenceSession()` 生成まで検証（checker だけでは不足）。
- **cost に MAC なし**（2026-05-04 廃止、scorer 確認）。レバーは params + memory のみ。
- 訂正: private LB なし。**ミッドコスト帯(1k-15k)に新規 win 余地大**（task055 13836→4537 等実績）。
  天井再測定: 全 cost>300 タスクを 300 化で +660→7829（**7665 到達可能**）。

### 実験11: round7-8 + quarantine 回収試行（逓減確認）
- **round7（5k-8k帯, 12タスク）**: win は task005(7100→7020)/task014(8078→8069) のみ、**+0.026**。
  この帯は base が良く golf 済みで逓減。→ v9 submit = **Public 7169.90**（回帰なし）。
- **round8 quarantine 回収（mask-blend 書き換え）**: task173 を combo 無しで再構築できたが
  **cost 3046851（base 23883 の 128倍）**。mask-blend は combo ERROR を回避するが**遥かに高コスト**。
  task285 はまだ combo、task133/338 も heavier/incorrect。→ **quarantine 回収は cost 壁で失敗**。
- **account best = 7169.95**（並列セッション）。回帰なし。
- **逓減の構造**: 大 win（task055 +1.115 等）は base 過剰設計タスク。残りは ① mid-cost=near-optimal
  ② high-cost=複雑 object-assembly で cheap golf 不可。各 round +0.01〜0.03 に低下。
- **現実的到達点の再評価**: 天井 7829 は「全タスクを専門レベルで golf」した場合。実際には
  各タスクの golf 成功率と cost 削減幅が逓減し、**本セッションの自動 golf では ~7170-7175 が実効上限**。
  7665 には専門チーム級の per-task hand-golf（数百タスク）が必須。

### 実験12: gate のバグ修正 + round7/9 win 統合 → v10
- **gate のバグ発見**: merge_submit の gate が `opset<14` を誤って除外していた（docstring は
  「opset<14 は base が証明済みで安全」と書いてあるのにコードは除外）。これで round9 の
  opset13 win（task196/201/206/265/335/358）と task338 を取りこぼしていた。
- **修正**: opset<14 チェックを削除。残す除外は **TopK+ScatterElements combo のみ**
  （唯一の実証済み ERROR 信号、task173/285）。
- **round7 完走（11.7分）= 5 win**: task005/014/089/377/383（全て SAFE, no risky op,
  sharpened brief で「no TopK」を遵守）。
- **round8 = task338 のみ safe**（133/285 は combo + 133 は public overfit で除外）。
- **round9（1.5k-5k帯）= task196(+0.048)/201(+0.025) 等**。
- **v10 submit = base + 34 safe win, local +3.369**（gate 修正で +0.12 回収）。

### 実験9: クラウド並列 hand-golf + fp16 safe-narrow（進行中, 2026-06-26）
ユーザー指示「クラウド並列 hand-golf」に基づき Workflow を起動:
- **並列 golf ワークフロー**: top20 expensive タスク（task233-173, 各~10pt headroom）を
  1タスク1エージェントで symbolic golf（真ルール numpy 全例検証→専門プリミティブで base 未満
  cost の ONNX 構築→ローカル自己採点）。各勝ち主張を独立エージェントが**敵対的再採点**で確認。
- **fp16 safe-narrow**: 7166.65 base の float32→float16 を、**出力 byte 一致 + cheaper** の
  二重ゲートで採用。**全400タスクで 0 win, +0.0**（確定）。専門 base は安全変換可能な float を
  既に全て fp16 化済み。残る float32 は精度必須で narrow すると出力が変わり byte 不一致。
  → submit 候補にならず（tie は提出しない）。dtype レバーは完全枯渇を再確認。
- 確定 win は 7166.65 base に重ね、再検証→package→submit（overlay 有時のみ）。

#### 🎯 実験9 結果: 初の base 超え win を達成（cost 壁を突破）
並列 hand-golf ワークフロー（20タスク, 22エージェント, 1.3M tokens, 22分）が
**専門 base より安い exact 解を初めて産出**。敵対的再採点で確認:
| task | base cost | golf cost | Δpts |
|---|---|---|---|
| task002 (enclosed flood-fill) | 40116 | **39316** | +0.020 |
| task187 (border-reach flood) | 46809 | **41699** | +0.116 |
| task349 | 38996 | **38987** | +0.000 |

- **submit 結果: Public Score 7166.66 → 7166.79（+0.13, ローカル予測 +0.136 と一致）**。
- = **cost 壁は突破可能**と実証。専門 base ですら flood-fill 系で削れる余地があった
  （task187: 13-step MaxPool flood を 25×25 crop 上で実行し中間縮小 → −5110 cost）。
- ワークフローは API 切断6 + session limit 5 で **20中9タスクのみ完走**。
  未完タスク（133/219/209/191/233/173/367/255/286/364/349）は**未開拓の機会**。
- **教訓**: per-task hand-golf は機能する。並列化で多数タスクを同時攻略すれば +点が積算可能。
  1タスク ~+0.02-0.12 → 残り expensive ~40タスク全勝で +1〜5、さらに中コスト帯へ展開。

#### 実験9 第2バッチ: 6 win 追加 → Public 7167.62
12 golf 可能タスクで2回目ワークフロー → **6 win 確認**（敵対的再採点済み）:
| task | base | golf | Δpts |
|---|---|---|---|
| task209 (crop+key upscale) | 51149 | **35838** | +0.356 |
| task205 (row/col marker paint) | 19863 | **15557** | +0.244 |
| task191 (8-orient matched-filter stamp) | 35425 | **31217** | +0.126 |
| task367 (closed-box interior fill) | 26903 | **25127** | +0.068 |
| task364 (line-topology recolor) | 26974 | 26050 | +0.035 |
| task077 (bbox+col-support recolor) | 14874 | 14834 | +0.003 |

- **9 win 累計（round1+2）を 7166.65 base に重ね submit → Public Score 7167.62**
  （7166.66 → 7166.79 → **7167.62**, 累計 +0.96, ローカル予測 +0.968 と一致）。
- **手法確立**: 並列 hand-golf で多数タスクを同時攻略 → 確定 win を base に overlay → submit。
  crop/recolor/stamp/flood 系で専門 base を 1-30% 削減可能。object-assembly 系（76/118/158/285）は
  まだ未陥落だが、crop/topology/flood 系は高勝率。次バッチで残り expensive + 中コスト帯へ展開。

#### 実験9 第3バッチ: 5 win → ただし v3 で Kaggle ERROR（op 互換性問題）
- 第3バッチ 5 win 確認（task076/173/219/243/255, local +0.826）。
- **14 win 全部を載せた v3 submit が Kaggle ERROR**。原因: 新 win の一部に
  **local ORT は通すが Kaggle scorer が拒否する op**（TopK/ScatterElements/Mod/MatMul）。
- **op 互換性 bisect**: v2(7167.62)で実証済み Kaggle-OK op = {Conv,ConvTranspose,GridSample,
  QLinearConv,CumSum,ArgMax,MaxPool,Gather,Where,Slice,Pad,...39種}。
  v3-new の追加 op: task076=Mod/ScatterElements/TopK, task173=ScatterElements/TopK,
  task255=MatMul（いずれも危険）。task219/243 は v2-proven op のみ→安全。
- **対策**: v2-proven op のみの **11 win 安全 bundle（v4）を再構築 → submit → Public 7167.94**。
  task219(+0.32)/243 を回収。076/173/255 は op を Kaggle-safe に書き換えるまで保留。
- **重要教訓**: golf ONNX は **Kaggle-safe op allowlist 内**で構築必須。local 合格≠Kaggle 合格。
  TopK/ScatterElements/Mod/MatMul は要回避 or 検証。

#### 実験9 第4バッチ + ERROR culprit bisection
- 第4バッチ（中コスト14）で **3 safe win**: task055(**13836→4537, +1.115!**), task396(13144→9608, +0.304),
  task138(14332→13765, +0.04)。+ task064/080/202 等。
- **ERROR 原因の再特定**: v3/v5 ERROR は op 語彙でも opset<14 でもなかった
  （base は TopK/Scatter/Mod/MatMul/MaxUnpool を多用、opset12 の task076 単体でもない）。
  → **二分探索で特定**: BISECT-A(v4+055/064/080/138)=**7169.26 OK**、
  BISECT-B(v4+173/202/255/396)=**ERROR**。culprit は {173,202,255,396} 内。
  さらに B1{173,202} / B2{255,396} に分割して submit 中。
- **重要教訓**: golf ONNX の Kaggle 互換性は **submit による実証が必須**。local ORT 1.27 で
  通っても Kaggle scorer（別 ORT version）で特定 op instance が落ちうる。
  → 安全運用: 新 win は **小グループ submit で Kaggle 検証してから** 本 bundle に統合。

#### 累計スコア推移
| submit | Public | win |
|---|---|---|
| standing | 7166.66 | — |
| v1 | 7166.79 | 2 |
| v2 | 7167.62 | 9 |
| v3 | ERROR | 14（culprit 混入）|
| v4 安全 | 7167.94 | 11 |
| v5 | ERROR | 19（culprit 混入）|
| BISECT-A | 7169.26 | 15 |
| BISECT-B2 | 7169.59 | 17 |
| **BISECT-C202** | **7169.68** | **18**（最新 floor）|

#### ERROR culprit 完全特定（二分探索完了）
- **culprit = task173 単体**（TopK **AND** ScatterElements 併用, opset16）→ Kaggle ERROR。
- **TopK 単体は安全**（task064 が BISECT-A 7169.26 で稼働）。ScatterElements 単体も恐らく安全。
  → 危険なのは **TopK+ScatterElements の組み合わせ**。merge gate に該当検出を追加。
- task173 は quarantine（+0.347 を放棄）。task076/285 も同 combo のため予防的に隔離。
- standing best = 7169.68（18 win）→ 第5バッチで task222/044/029/340/350/158 追加。

#### 実験9 第5バッチ + 回帰確認 → Public 7169.81（最新 best）
- round5（中コスト14）で task222(9179→8321,+0.098)等 +6 win。
- **v6 = base + 24 safe win, local +3.158 → submit → Public 7169.81（ERROR なし, 回帰なし）**。
- Kaggle-safe gate（TopK+ScatterElements combo 除外 + opset≥14 + checker）が ERROR を自動回避。
- 並列 worktree の cherrypick(7169.72)を上回り **LB 最上位**。floor bundle を floor_7169_81.zip に保存。
- **累計 +3.15（7166.66 → 7169.81）, 24 win**。回帰確認: 全 submit が n_fail=0 + Kaggle COMPLETE。

#### 実験9 第6バッチ + v7 + 重要な gate 理論修正
- round5 完了で task174(9890→9349)/task204(11818→11499) 追加 → **v7 = 25 win, Public 7169.87**。
- round6 は **session limit**（5:10pm Tokyo reset）で 0 win。golf エンジン一時 rate-limited。
- **重要修正: gate 理論が誤りだった**。base bundle 自体が opset<14（task007-015等）と
  TopK+ScatterElements（task012/037/092等）を多用し **Kaggle で 7166.66 稼働**。
  → opset<14 も TopK+ScatterElements combo も **Kaggle-safe**。task173 の ERROR は
  **task173 固有のグラフ欠陥**（op 語彙でも opset でもない）。
  現 gate は保守的に task076/173/285 を除外中（過剰だが ERROR は確実回避）。
- **回帰確認結果**: 並列 worktree が **7169.91** に到達（t174/204 harvest）。
  account best = 7169.91、**回帰なし**（v7 7169.87 COMPLETE, poll の ERROR は transient 誤読）。
  並列セッションと account 共有で協調的にスコア上昇中。
- floor を floor_7169_87.zip に保存。standing best = **7169.91**（account 全体）。

#### v8: task173/076 recovery 失敗（TopK+ScatterElements combo は opset18 でも ERROR）
- task173 の ERROR 根本原因を調査: `ScatterElements reduction=max(3)` は **opset18 で導入**だが
  task173 は **opset16 宣言**（local ORT は寛容、Kaggle ORT は厳格 → ERROR）。
- version_converter で opset18 化 → local OK (cost 16486<23271)。だが **v8 submit は再び ERROR**。
- → opset 変換だけでは不十分。**TopK+ScatterElements combo の golf graph は Kaggle で確実に
  ERROR**（task173/076 とも opset18 化後も失敗）。根本機序は不明だが combo が信頼できる除外信号。
- task173/076 を再隔離。gate の combo 除外を維持。**回帰なし**（account best 7169.91 維持）。
- **教訓**: 回帰確認 = poll の transient ERROR に注意（v7 も一度 ERROR 誤読、実際は COMPLETE 7169.87）。
  必ず submissions リストで最終 status を直接確認すること。

## ⚠️ 重要な事実修正（実験7 web research, 2026-06-26）

過去の結論の前提に**誤りがあった**ことが判明。ホスト公式コメント + 公開ノートブック調査で修正:

| 旧（誤）認識 | 正しい事実 |
|---|---|
| top LB 7942 は非公開、公開上限 7166 が ceiling | 公開 LB は **7800-7950 に密集**。私の base(7166)は**凡庸**、ceiling ではない |
| hidden private split で override が落ちる（hidden 壁） | **private LB は存在しない**（ホスト明言: "No private LB, no shakeup"）。correctness 用の小さい hidden benchmark のみ。**全 local example(train+test+arc-gen)を exact に通れば Kaggle でも score する** |
| base bundle は最適 | base(7166)は**公開フロンティア(~7950)より遥かに下**、headroom 大 |
| gap は新タスク解決が必要 | gap = **既解決タスクを安くする**（dtype 縮小 / crop-early / output直書き） |

ただし追加検証で判明した制約:
- **7166.65 超の公開バンドルは存在しない**（franksunp 7166.65 が公開最高、+0.02のみ）。
  ~7950 は非公開チームで配布なし。→ +499 は **base 上の自前 per-task golf が必須**。
- **dtype 縮小の大半は実現不可**: fat float32 中間（task233 con1 等）は Conv が graph input
  (f32) を直接消費するため f16 化に input の Cast→f16([1,10,30,30]=18000B)が必要で**純損**。
  base が f32 のままなのは構造的必然。379KB の f32 headroom はほぼ実現不可。
- **task347 −5 の再解釈**: hidden 壁ではなく、おそらく local で捕捉漏れの arc-gen 失敗か
  Kaggle/local scorer 差。no-hidden-set 下では local 全 exact なら安全。

→ **方針修正**: hidden 壁を恐れる必要はない。local 全 example exact + cost 減 = 確実な LB gain。
残る課題は純粋に **cost 壁**（自前構成が専門 base より重い）のみ。

### 実験7-b: cost 壁の再確認（no-hidden-set 下でも自前 < base）
no-hidden-set の自由を得て、pixel-permutation で base より安くできるタスクを探索:
- **single-Gather 解ける pixel-perm タスク 21件**を発見、うち base_cost>500 が 4件
  (task108=971, 194=949, 214=767, 152=629)。
- しかし自前 Gather/separable 構成は **全て base の 13-40倍重い**（task108: 自前35049 vs base971）。
  原因: 自前は `Cast(input→uint8)` で [1,10,30,30]=9000B 全体を materialize。
- **base の専門 trick を解析**: task142(282) = `Slice(input)→x[1,4,3,3]` (色ch も crop, 144B)
  + **単一 Einsum** `bcrs,oc,hr,ws->bohw`（色remap+行remap+列remap を 1 op で融合し
  `output` に直書き）。中間は tiny な sliced x のみ。output(30×30)は無料。
- task194(949)= base が既に Reshape+Gather+Pad で **900B 中間**の最適形。task108= Conv+Resize で最小。
- **結論**: 幾何タスクですら base は **crop-channels + 単一融合 Einsum/Gather→output** の
  専門最適形を使用済み。自前は再現できず 13-40倍重い。**cost 壁は easy task でも成立**。

### 実験8: per-task hand-golf 本格投資（ユーザー指示「hand-golf に長時間投資」）
local-rule タスク（output cell = 小近傍の関数）を 11件検出し、安い Conv で golf を試行:
- **task222(9179)**: 「solid 単色矩形を残す」= 実は最大矩形検出の**大域**タスク（radius-2
  local 判定は sparse data の偽陽性）。cheap local 化不可。
- **task192(8621)**: 3×3 lookup が **265例で決定的**（真に local）。
  - 単層 3×3 Conv 線形 fit = acc 99.98%（**not linearly separable**, 1セル違いで score 0）。
  - **2層 3×3 Conv (H=16) で acc 100%（265/265 exact）を達成**。
  - ONNX 構築 → **cost 205027（base 8621 の 24倍）**, pass 265/268 → exact だが大敗。
    [1,16,30,30]f32 hidden(57600B)+logits(36000B)が支配。
  - = research の警告「学習する MLP は symbolic graph の 10-40倍重い」を**実測で確認(24倍)**。
- 他 9件: 単層線形 acc 0.91-0.998（exact 不可）、2層は exact 可能だが同様に cost 爆発見込み。
- **hand-golf 投資の結論**: 正しい exact ルールを導出・構築できても（task192 で実証）、
  base の symbolic 構成（CC+property logic 等で 8621）には dense conv では 24倍負ける。
  base を下回るには base と同じ **symbolic 構成を再発見**する必要があり、それは専門 base が
  既に最適実装済み。**5つの完全構築（flood 4-8倍, 幾何 ~等倍, dense conv 24倍）全てで
  自前 ≥ base を実証**。cost 壁は構造的。

## ⚠️ 現実的到達点の確定
- 全レバー（自動 7系統 + hand-golf 実投資）で base 7166.65 を**下回る submit 候補ゼロ**。
- +499 到達には base と同等以上の symbolic golf を ~90 タスクで実装する必要があり、
  これは専門トップチーム（非公開, ~7950）の数週間〜数ヶ月の作業に相当。
- 本セッションの手法・時間枠では **7166.65 が達成可能上限**。

## 確立した事実（公式スコアラー source 確認済み）

- `cost = memory + params`（MAC は 2026-05-04 に廃止）。`points = max(1, 25 - ln(max(1,cost)))`。
- `output` テンソルは memory から除外 → **最終ノードが `output` に書けば無料**。
- params = 全 initializer/Constant の numel（**dtype 非依存**, int8 でも float32 でも numel）。
- memory = 中間テンソルの `numel × dtype_itemsize`（**uint8/bool=1B は float32=4B の 1/4**）。
- 単一 op→output（Identity/Transpose）= cost 0 = **25.0 点**。`Add(input, init[1])` = cost 1 = 25.0。

## hidden-split ルール（最重要・octavi 検証）

- cost が計上されるのは **hidden private + 全 arc-gen で真に正答**したときのみ。
- visible だけ通る脆い override は **0 LB**（octavi v16: 予測+116→実測−308）。
- **既存 op 語彙内の dtype 縮小・ノード削除は grader-faithful**（安全）。
- **新規 op チェーン導入**は visible 通過でも 0 LB のことがある（危険）。
- → 安全策: 既存 base ONNX の **dtype を uint8/bool へ縮小**（op 構成は変えない）。

## 実提出ログ

| 日時 | 施策 | Public Score | 判定 |
|---|---|---|---|
| - | case1 baseline (boristown) | 7159.44 | — |
| - | case2 floor (franksunp-7166) | **7166.63** | best floor |
| - | case2 v1 (+task347 自前override) | 7161.55 | ❌ −5（hidden overfit） |
| - | 先行 case3 blend8 (16バンドル) | 7150.70 | ❌ −16 |
| - | 先行 case3 blend8safe (36 swaps) | 7166.63 | 無利得 |
| - | case2 v3 (uint8 narrow task037/319) | **ERROR** | ❌ Kaggle scorer 拒否 |
| - | 先行 case3 surgery (15 tasks cheaper) | 7166.11 | 無利得（≈floor） |

## これから（research が示した +499 の道）

1. **dtype 縮小パス**: base の高コストタスク（task018=66k に float16 41,900B 残存等）の
   中間を uint8/bool へ縮小。op 語彙不変 → grader-faithful。top 40 を 25 点化で +400。
2. 各候補は **全 arc-gen + train + test で exact 検証**してから submit。
3. crop-to-bbox + final→output で中間サイズ削減。

## 進捗

### 実験1: uint8 narrowing パス（grader-faithful dtype 縮小）
- 高コストタスクの float16/float32 中間 Cast を、値が [0,255] 整数に収まる場合 uint8 へ retarget。
- **結果: 400タスクで gain = 0.05（実質ゼロ）**。task037/319 のみ narrow、各 +0.0。
- task349 等は narrow すると downstream op が uint8 を拒否し `session_error`。
- **理由**: 専門家は既に各 op が許す最狭 dtype を使用済み。大きい index 中間（CC ラベル
  = row*30+col, 最大 900）は uint8 に収まらず float16 が必須。
- **結論**: dtype 縮小に余地なし。base は dtype も最適化済み。

### 全自動レバーの最終状態（実証済み, 全て利得≈0 or 退行）
| レバー | 利得 |
|---|---|
| DSL プリミティブ再解（22種, 18タスク一致） | 0（base が全て cost 最適） |
| uint8/fp16 dtype 縮小 | +0.05 |
| グラフ最適化 onnxslim/optimizer | ≈0 |
| マルチバンドル blend | 0（バンドル同一系譜） |
| override 不正解タスク（task347） | **−5（hidden overfit）** |
| 高コストタスク cheap exact 形探索 | 0 hits |

→ research 結論: **+499 は top40-60 高コストタスクの個別 re-architecting（真のルール導出
→ 最小ソルバ手作り）でのみ可能**。自動ショートカットは全て専門家が適用済みで余地なし。
これは LLM 支援の per-task hand-golf（数週間規模）に相当。

### 実験2: 2つの「壁」を実提出で確認
1. **cost 壁**: 自前構成は専門家構成に及ばない。einsum_remap でも mosaic cost 1857 vs base 282。
2. **hidden 壁（二重）**:
   - ローカル合格でも **Kaggle で ERROR**（v3 uint8 narrow が Kaggle scorer 拒否）。
     私のローカル scorer は時に Kaggle より**寛容** → 改変は Kaggle で落ちうる。
   - novel op chain は visible 合格でも **hidden で 0 LB**（octavi v16 −308, 自前 task347 −5）。
3. ルール導出は可能（task002 enclosed-fill 268/268, task347 OR-halves 269/269 等を導出）が、
   **grader が期待する構成で安く実装する**のが専門知識の壁。novel chain は危険、専門家構成は最適。

### 最終確認: 公開・自動手法の上限 = 7166.63
複数の独立検証（自前 v0-v3 + 先行 case3 全提出 + 公開 kernel 最上位 + バンドル byte 一致）で、
**全手法が 7166.63 に収束**。改変は ERROR/退行/無利得のいずれか。
**安全な standing best = pure 7166.63 floor**。+499 到達には非公開の専門 hand-golf が必須。

### 実験3: 高コストタスクの re-architecting 試行（research 推奨の唯一の道）
全 expensive タスクに対し多数のルール仮説を網羅検証:
- size-reducing block downscale/mode/any: **0 hits**
- symmetry-repair / majority-fill: **0 hits**
- output = crop of input: **10 タスク該当**（task014/029/036/079/091/174/216/310/355/365）
  だが crop 位置は **data-dependent**（task091 = 角8+側5 の特殊枠を検出して抽出、
  task355 = 特定オブジェクト）。静的 ONNX で安く位置特定できず → base の object 検出が必須。
- ルール導出に成功した例（task002 enclosed-fill, task347 OR-halves）は導出可能だが、
  ① 安く実装できない（cost 壁）② novel chain で hidden 0 LB（hidden 壁）。

**結論（実験1-3 + ~15 提出で確定）**: expensive タスクは全て data-dependent な
object/topology 計算が必須で、静的 cheap ONNX に落ちない。base はそれを最小に近い
構成で実装済み。自動・半自動手法での 7665 到達は構造的に不可能。残るは専門家による
per-task hand-golf（数週間規模、grader 期待の構成での厳密実装）のみ。

### 実験4: FLOAT→FLOAT16 safe-narrowing（進行中）
- uint8 narrow は Kaggle ERROR だったが、FLOAT16 は bundle が既に使用（Kaggle 受理）。
- FLOAT Cast→FLOAT16 に retarget、**出力 byte 一致 + cheaper** の二重検証で採用。
- 全タスク FLOAT 中間 = 378,484 B（323 タスク）。半減で ~189K B 削減ポテンシャル（薄く分散）。
- final-op-fusion 調査: expensive タスクの最終 op は Equal/Pad（compact label grid を
  10ch one-hot に展開する安い op）。big intermediate は展開前の genuine 計算で、fuse 不可。
- expensive タスクの最終構造 = label grid 計算（469 nodes 等）→ Equal で 10ch 展開。
  cost は label 計算そのもので irreducible。
- **結果: 2 safe wins, gain +0.01（実質ゼロ）**。専門家は安全に変換可能な FLOAT を
  既に全て FLOAT16 化済み。残る FLOAT は division/modulo 等で float 精度が必須
  （変換すると出力が変わり byte 一致検証で却下）。dtype レバーは完全に枯渇。

## 全実験の総括（自動レバー利得一覧）

| レバー | local gain | LB |
|---|---|---|
| DSL 22プリミティブ再解 | 0 | — |
| uint8 narrow | +0.05 | **Kaggle ERROR** |
| FLOAT16 narrow（出力一致検証） | +0.01 | 無利得見込み |
| グラフ最適化 onnxslim/optimizer | ≈0 | — |
| マルチバンドル blend | 0（同一系譜） | — |
| 不正解 override（task347 全例 exact） | +15 local | **LB −5（hidden）** |
| ルール導出（block/symmetry/fill/crop/bbox/remap/conv/local 10+ 系統） | crop 10件のみ該当も data-dependent | — |
| final-op-fusion | 不可（big intermediate は genuine 計算） | — |

→ **全自動・半自動レバーが利得 ≤0.05 or 退行/ERROR**。専門家 base は dtype・グラフ・
構成すべて最適化済み。+499 は data-dependent な object/topology 計算を grader 期待の
構成で安く厳密実装する per-task hand-golf（数週間規模）が必須で、本セッションの手法群
では構造的に到達不可能。standing best = **pure 7166.63**。

### 実験5: re-architecting 具体実装（task002 enclosed-fill, rule 268/268 既導出）
uint8 単一チャネル flood-fill を手作り（30 step × 4-shift dilation + bg-mask）:
- **結果: cost 329284（base 40116 の 8倍）, fail 72/268, INCORRECT**。
- 失敗原因: ① 30 step × 6 op × 900B = 中間が膨大（base の MaxPool 構成より遥かに重い）
  ② グリッド枠問題: 枠外セルも channel0=0 → bg に含まれ、30-grid 境界から flood すると
  out-of-grid のゼロ経由で grid 内部へ誤侵入（base は Slice で grid 抽出してから処理）。
- → ルールを導出できても、**grid-extent 処理 + tight 構成**という専門技術なしには
  base より 8倍重く・不正確にしかならない。re-architecting は専門知識の壁が本質。
- 補足: numpy sim（30-grid framing, 30 step）は **268/268 一致**＝ロジックは正しい。
  ONNX も train[0] は exact。だが cost 329284（base 40116 の 8.2倍）で**勝ち目なし**。
  base の MaxPool 構成（40116）の方が遥かに tight。専門家の golf を再現できない。
- **最終確認**: rule 導出（268/268）→ 正しい ONNX 構築まで到達しても、cost で 8倍負ける。
  これが「専門 hand-golf でのみ +499 可能」の具体的実証。

### 実験5b: task002 を separable MaxPool で再最適化
- 4-conn flood を vpool[3,1]+hpool[1,3] MaxPool で実装。numpy sim 268/268 一致。
- ONNX: cost 162003（329284→改善も **base 40116 の 4倍**）。
- base task002 = 91 nodes（MaxPool 39 + Max 21 + Mul 21 = **20 iteration**、単一 MaxPool で
  4-conn を実現する trick）。私 = 129 nodes（MaxPool 60 = 30 iter × 2 pool）。
- **専門家との差**: ① 正確な iteration 数（20, object 上限を把握）② 単一 MaxPool 4-conn trick
  ③ 中間最小化。これらを再現できず 4倍重い。
- **重要**: 仮に完全再現できても cost 40116 = **tie（利得0）**。勝つには世界最高の golf を
  上回る必要があり、専門家が既に最小化済み。
- ONNX に 72 fail のバグ残（uint8 MaxPool の ORT 挙動差と推定）が、cost 4倍で**修正しても
  勝ち目なし**のため深追いせず。

## 総合結論（実験1-5b, ~17 提出, 12+ rule系統, 完全な re-architecting 試行で確定）

7665（base 7166.63 から +499）は、本セッションの自動・半自動・手動 re-architecting の
いずれでも到達不可能。理由は2つの壁:
1. **cost 壁**: expensive タスクの自前再実装は専門 base の 4-8倍重い（flood-fill 実証）。
   完全再現しても tie（利得0）。勝つには世界トップの golf を上回る必要。
2. **hidden 壁**: novel chain は visible 合格でも hidden で 0 LB/退行（task347 −5 実証）。
   local 合格でも Kaggle ERROR（uint8 narrow 実証）。

→ +499 は上位チーム非公開の ~100 タスク専門 hand-golf（grader 期待構成での厳密最小実装、
数週間規模）が必須。**達成可能な最良 = pure 7166.63**（case1 比 +7）。

### 実験6: cheap-program 全探索 + 理論天井マップ（本セッション）
- **cheap-program 全探索**（depth≤2 の幾何/recolor 合成を全400タスクで exact 照合）:
  23 タスクが cheap program で解けるが、うち expensive(>200) は task152/083/142/249 のみ。
  しかし base はこれらを **Einsum separable remap**（task142 = Slice+Einsum 2ノードで cost 282）
  で既に golf 済み。自前の Slice+Concat+Pad 構成は **cost 4347**（30×30 への Pad が
  36000B float32 中間を生む）で **base の 7倍**。→ cheap-win は全て base が先に最小化済み、0 hits。
- **理論天井マップ**: cost→pts = max(1, 25-ln(cost))。全 392 costed タスクを cost=1 化で
  最大 +2785（天井 ~9950）。+499 到達には **top ~50 expensive タスク（cost>10000 が 46件、
  各 ~10pt headroom）を near-minimal 化**する必要。これは可能だが各タスクが
  data-dependent computation（flood-fill / object / symmetry-repair）。
- **expensive タスクの cost 内訳分析**: 支配的中間は ~3600-9000B のみで、総コストは
  **多数の小中間（flood-fill の reach_0..9, label propagation 等）の総和**。
  大きい中間は **既に float16/int 化済み**（task286/187 = inp_u float16, task209 = int onehot）。
  残る float32 中間（task233 con1 [1,1,30,30]）は downstream の Conv/Einsum/Scatter が
  float 必須で narrow 不可（uint8 narrow は Kaggle ERROR 実証済み）。
- **7 ShapeInferenceError タスク**（045/127/135/146/149/240/384）= ConvTranspose/Conv の
  負パッド等で local onnx-checker が落ちるだけ（Kaggle は正常 score）。base は tiny
  （task135 = 単一 Conv 997B, task149 = 2 Conv）で golf 済み、headroom なし。
- **結論**: base は dtype・グラフ・op 構成・全中間が per-task 最適化済み。cost は genuine
  multi-step computation に irreducible に分散。自動・半自動の全レバーで余地ゼロを再確認。
  +499 は ~50 タスクの専門 hand-golf（base より tight な独自アルゴリズム）が必須。

### 実験13: 並列 hand-golf round 9-10（本セッション継続）
- **round 9 (12タスク golf)**: 6勝確定 → v11 submit = **Public 7169.98 COMPLETE**（ERROR無し）。
  新規clean勝利 089/196/201/377/383 を golf_wins にコミット（335 は base tie でスキップ）。
  - task089: 7252→7210, task196: 4761→4538(+0.05), task201: 4774→4658, task377: 8505→8379, task383: 7320→7311
- **ゲート修正の確認**: TopK+ScatterElements combo のみ除外、opset<14 除外は誤りだったため撤廃。
  これにより round-9 の clean 勝利が回収可能に。ScatterND/Mod 単独でも念のため除外（v10 ERROR の疑い）。
- **round 10 起動**: 最高コスト未着手16タスク
  (233:70606, 18:66319, 118:42418, 366:37446, 54:27133, 23:22386, 319:22242, 66:19807,
   101:17925, 157:12707, 96:8953, 192:8621, 182:8030, 165:6456, 368:6225, 208:6083)
  に対し cloud-parallel golf + adversarial verify を実行中。高コスト帯は1勝 +2pt 級の headroom。
- **アカウント best**: 7170.06（並列セッション）。floor 保護下で未着手高コスト帯を攻略中。

### 実験14: 危険op単独ERROR の決定的検証（本セッション）
- **背景**: round-9 で task378(Mod, 4489→3733), task265(ScatterND, 4362→4073) 等が local 合格。
  これらは combo (TopK+ScatterElements) を含まないため「Mod/ScatterND 単独なら Kaggle-safe か?」を検証。
- **PROBE 実験**（floor33 = v11と同じ proven-clean bundle を土台に、危険opタスクだけ追加）:
  | probe | 追加タスク | op | Kaggle結果 |
  |-------|-----------|-----|-----------|
  | PROBE | 378+265 | Mod+ScatterND | **ERROR** |
  | PROBE-MOD | 378 のみ | Mod | **ERROR** |
  | PROBE-SCAT | 265 のみ | ScatterND | **ERROR** |
- **決定的結論**: floor33 は単独では clean(=7169.98)。そこに **Mod 1タスク追加 → ERROR**、
  **ScatterND 1タスク追加 → ERROR**。つまり golf生成の **Mod / ScatterND グラフは
  combo 不要で単独 Kaggle-ERROR**。ゲートの RISKY 除外 (Mod/ScatterND/TopK/ScatterElements) は正当。
  → これらopを使う golf 勝利は **全て破棄**。BRIEF も Mod 全面禁止に更新。
- **影響**: round-9/10 の Mod/ScatterND 勝利 (378/265/358/206 等) は採用不可。
  Einsum/Gather/Where ベースの再構成のみ採用可能。

### 実験15: round 10 高コスト未着手攻略 → v12 = 新アカウントbest 7170.28（本セッション）
- **round 10 (16高コストタスク golf)**: 4勝確定、うち3つが risky-op-free:
  | task | base→new | gain | op |
  |------|----------|------|-----|
  | 023 | 22386→16337 | **+0.315** | safe (2x2 tiling) |
  | 165 | 6456→6370 | +0.013 | safe |
  | 182 | 8030→7585 | +0.057 | safe |
  | 208 | 6083→5569 | +0.088 | **TopK+Mod → 破棄** |
- **v12 submit = Public 7170.28 COMPLETE** = **本セッション初の新アカウントbest**（旧 7170.06 を更新）。
  task023 が単独 +0.315 で今セッション最大の勝利。floor_7170_28.zip 保存。
- **golf_wins クリーンアップ**: 危険op混入の 064/076/173 (TopK/ScatterElements/Mod) を削除。
  golf_wins = 33ファイル全て Kaggle-safe に。bundle gate も risky を二重に除外。
- **超高コストタスクの irreducibility 確認**: 233(70606)/18/118/366/54/319/66/101/157/96/192/368 は
  beat:false。理由は dynamic crop/reposition 必須（static-shape 違反）or base が既に tight。
  → cheap golf は **mid-high band (1500-6100, 113タスク未着手)** が主戦場。round 11 起動済み。
- **BRIEF 強化**: Mod 単独ERROR・Scatter単独ERROR を明記し全面禁止。modular index は定数Gather化を指示。

### 実験16: round 11 mid-high攻略 → v13 = 7171.27（新best）（本セッション）
- **round 11 (16タスク golf, mid-high band 5000-6100)**: **10勝確定 全て risky-op-free**:
  | task | base→new | gain |
  |------|----------|------|
  | 161 | 5217→3733 | **+0.335** |
  | 359 | 5187→4348 | +0.176 |
  | 279 | 5555→4827 | +0.140 |
  | 251 | 5640→4920 | +0.137 |
  | 004 | 5904→5378 | +0.093 |
  | 363 | 5126→4757 | +0.075 |
  | 277 | 5360→5147 | +0.041 |
  | 008/382/398 | 微減 | +0.002 |
- **v13 submit = Public 7171.27 COMPLETE = 新best**（v12 7170.28 から +0.99）。
  local +0.999 が LB にほぼ 1:1 反映。mid-high band golf は変換効率が高い。
- **知見**: flood-fill(MaxPool反復)/16x16 uint8 crop/separable majority-vote が high-yield パターン。
  mid-high band (1500-6100) は round11で 10/16 勝率。round 12 起動済み（99タスク未着手）。
- floor_7171_27.zip 保存。golf_wins=43（全safe）。
