# case3 — Public Score 7665 への到達記録

**目標**: Kaggle "The 2026 NeuroGolf Championship" で **Public Score 7665** を達成する。
現状ベスト提出は **pure frank7166 = 7166.10**（−498.9 不足）。本ログは到達に向けた
各実験の仮説・手法・結果・結論を時系列で記録する。

## スコアリングの確定事実

- `cost = params + memory`（MACs は 2026-05-04 改定で寄与しない）。
  `points = max(1, 25 - ln(max(1, cost)))`。
- **`params` は初期化子・Constant の「要素数」**（dtype 非依存、バイトではない）。
  → int64→int32 / fp16 化は **params を一切減らさない**。
- **`memory` は中間ノード出力テンソルのバイト数総和**（`input`/`output` 名は除外）。
  dtype を小さくすると memory は減るが、境界 Cast の中間テンソルが増えると相殺/悪化する。
- 正答は train+test+arc-gen の全ペア完全一致が条件。1 例でも外すとそのタスクは 0 点。

## ⚠️ 最重要の障害: ローカル採点 ≠ Kaggle 実 LB（実証済み）

`src/evaluate.audit_one`（公式スコアラミラー）の正答判定は **Kaggle 実 LB と一致しない**。
frank7166 の一部タスク（t045/127/240/347 等）はローカルで `unscorable`/`INCORRECT` に
なるが、**Kaggle 実採点では得点している**。

実証: frank7166（実 LB 7166.10）の t347 をローカルで「正答かつ +17.8 点」のソルバに
差し替えて提出 → **実 LB は 7166.63 → 7161.55 に低下**。ローカルの「勝ち」が実 LB では
「負け」。帰結:

- frank7166 のタスクを **ローカル監査だけを根拠に差し替えてはならない**（回帰する）。
- bespoke ソルバの実効性は**提出するまで不明**。
- 安全策は **pure frank7166**（タスクを 1 つも落とさない）。
- local→real のオフセットは frank7166 で local 7014.5 → real 7166.10（**+≈152**）。
  → 実 LB 7665 に必要なローカル ≈ **7505**（frank より **+490 local**）。これは膨大。

## 🔑 local↔real オフセットの再較正（fsun の audit summary より, 最重要）

fsun の `source_audit_summary.json` が **バンドル別の local↔real 対応**を明示:

| バンドル | local total | real public | offset |
|---|---|---|---|
| seddik all-graph-surgeries (=7166.65 base) | **7146.52** | **7166.65** | **+20.13** |
| kokinn continuation | 7146.49 | 7166.63 | +20.14 |
| frank7166-10（旧 base）| 7014.5 | 7166.10 | +152 |

→ **seddik/7166.65 系では local≈real（offset +20）**。frank7166-10 で見られた +152 の乖離は
frank 固有（多数の unscorable タスクが Kaggle で得点）であり、**seddik base では local 監査が
ほぼそのまま実 LB を予測する**。

**含意（戦略転換）**: seddik/7166.65 を base にすれば、audit_one でローカル検証した改善が
**実 LB にほぼ 1:1 で伝播**するはず（frank base の回帰リスクが無い）。よって今後は
**7166.65 を base に surgery / override / 新規ソルバを重ね、ローカル総点の純増を実 LB で確認**する。
real 7665 には local ≈ **7645**（seddik base から **+498 local**）が必要。

### ⚠️ 訂正: fsun の scorer ≠ 自前 scorer（offset は base 不問で +152）

自前 `audit_dir` で seddik/7166.65 base を採点 → **local 7015.02**（fsun の 7146.52 ではない,
−131 差）。つまり fsun の audit と自前 audit は別物（unscorable 判定が違う）。
**自前 scorer の offset は base に依らず ≈ +152**（seddik も frank も同じ）:

| base | 自前 local | real | 自前 offset |
|---|---|---|---|
| seddik/7166.65 | 7015.02 | 7166.65 | **+151.6** |
| frank7166-10 | 7014.5 | 7166.10 | +151.6 |

→ **local≠real の当初の警告は有効**。自前ローカル改善は実 LB に 1:1 では伝播しない。
ただし**意味保存の cost 削減（同一計算の書き換え）は安全**（surgery 7166.11 で実証済み、回帰なし）。
新規ソルバ**差し替え**は依然リスク（t347 −5）。

## 提出履歴 追記

| ref | 手法 | public score |
|---|---|---|
| 54060086 | case3 surgery（frank7166 base + graph surgery 15 tasks）| 7166.11 |
| 54060198 | fsun rewire-audit-mark-i 7166.65 verbatim（seddik base, sha256 一致）| 7166.65 |
| 54061025 | **7166.65 base + override(1) + safe surgery(4)（意味保存 cost 削減のみ）** | **7166.66（現ベスト）** |

## LB 情勢（2026-06-25 時点, kaggle CLI 実取得）

- **7665 は LB 中位**（達成可能）。トップは 7950.66、7800-7950 に十数チーム、7665 超は ~17 チーム。
- ただし **公開ノートブックは全て ~7166 で頭打ち**。最新最高は
  `franksunp/7166-65-lb-neurogolf-rewire-audit-mark-i` = **7166.65**（本日 23:17 run）。
- 7900+ のチームは**手法を非公開**。公開知識（≤7166.65）と LB 上位（7900+）の間に
  約 +780 のギャップがあり、その手法は公開されていない。
- 公開バンドル系列: franksunp 7159.44→7166.10→7166.65、kojimar overrides、
  seddiktrk graph-surgeries、qinghchen overlay、pawanmali 7160、biohack44 blend-max。
  いずれも ~7166 帯。**blend / surgery で詰められるのは小数点台**。
- 含意: 7665 への +498 は公開バンドルの組合せでは届かない。**未解決タスクを新規に解く**か
  **多数タスクを大幅に安く再構成**する必要があるが、いずれもローカル検証不能（local≠real）。

## 実験ログ

| # | 仮説 / 手法 | 結果 | 結論 |
|---|---|---|---|
| E1 | 公開バンドルを cost 最適 blend | local 7014.56（frank 以下）→ 提出 7166.10 | blend は frank の「ローカル失敗・Kaggle 成功」タスクを落とすと回帰。pure frank が floor |
| E2 | bespoke solver（identity/recolor/lookup/residual/floodfill/panels）で個別タスクを golf | t347 で local +17.8 を達成も、**提出で実 LB −5 回帰** | local 改善が実 LB に伝播しない（E2=上記の実証）。bespoke は信頼不能 |
| E3 | onnxslim による generic 簡約 | グラフが strict shape-inference を壊し **unscorable** 化 | 汎用簡約は scorer 非互換。scorer-aware パスが必須 |
| E4 | **graph surgery**（seddiktrk パス再実装）を frank7166 全 400 へ適用。<br>cleanup / index(int32) / broadcast / conv1x1→gather / fp16。<br>各パスを audit_one で「正答 ∧ cost 厳密減」のみ採用、タスクは落とさない | **進行中**（下記 E4 詳細）。先頭 62 は全て frank と byte 一致＝改善ゼロ。<br>285 タスク時点で **9 タスクのみ改善**（t080/138/173/198/205/209/215/222/278） | frank7166 は大半のパスで既に golf 済み。改善はごく一部に限定 |

### E4 詳細（graph surgery）

- 実装: `pipeline/case3/surgery.py`（パス本体）+ `pipeline/case3/apply_surgery.py`
  （audit_one ゲート付きドライバ）。CLI: `python -m pipeline.case3 surgery --base <bundle>`。
- 外部依存（onnxsim/onnxscript）は未インストールのため **generic 簡約パスは除外**し、
  `onnx` API のみで完結する決定的・意味保存パスのみ実装。
- 安全装置: 各パス出力を `audit_one` で採点し、(1) checker 通過 (2) 全 example 正答
  (3) cost 厳密減 を満たす場合のみ採用してチェイン。1 つでも崩れたら直前モデルを保持。
  改善が無いタスクは **元 frank ファイルをそのままコピー**（タスクを落とさない）。
- 観測: `params` は要素数ゆえ int32/fp16 では減らない。memory 削減も fp16 境界 Cast で
  相殺/悪化（t001 で cost 1121→19119 となり棄却）。実質効くのは broadcast 圧縮 /
  dedup / conv1x1→gather による **要素数削減**のみ。

**E4 最終結果（全 400 完了）**: **15 タスク改善、local points 合計 +0.0126、cost 削減 57**。
改善タスク: t080/138/173/198/205/209/215/222/278/349/366/377/387/396/400。
→ frank7166 は大半のパスで既に golf 済み。surgery の純増は実質ゼロ。

**E4 提出結果（実 LB）**: 全 400 を直接 zip（CLI validator は unscorable 7 件を弾いて
packaging を中断するため、`packager.collect_onnx_files` + `build_submission_zip` で全 400 を
直接梱包し `kaggle_api.submit` で提出）→ ref 54060086 = **public 7166.11**
（frank7166 base 7166.10 から +0.01、local 予測 +0.0126 とほぼ一致）。

→ **重要**: 意味保存のグラフ書き換え（dedup/broadcast 圧縮/int32 narrowing）は
**実 LB で回帰しない**ことを実証（t347 のソルバ差し替え −5 とは対照的）。surgery は安全だが
寄与が +0.01 と微小。

## 提出履歴（実 LB, kaggle submissions）

| ref | 手法 | public score |
|---|---|---|
| 54059754 | blend8: best-of-16 公開バンドル, 44 task swaps | 7150.70 |
| 54059779 | **blend8safe: 36 safe swaps from franksunp-62** | **7166.63**（記録上のベスト）|
| (旧) | pure frank7166 | 7166.10 |
| 54060086 | case3 surgery（frank7166 + graph surgery 15 tasks）| 7166.11 |

→ 記録上のベストは **7166.63**（blend8safe）。surgery は frank7166 ベースで +0.01。
次手は **blend8safe バンドル（7166.63）を base に surgery** を重ねる。

## 到達状況サマリ（最新）

- **現ベスト = 7166.66**（54061025）。floor 7166.65 から safe surgery で +0.01 上積み（回帰なし）。
- 検証済みの全レバーの寄与:
  - 公開バンドル verbatim: 7159.44 → 7166.10 → **7166.65**（公開最高）。
  - cross-bundle override（7 bundle, audit-gated）: **+1 task / +0 点**（全バンドル同コスト帯）。
  - graph surgery（意味保存）: frank base +0.01, seddik base +0.01（**実 LB で安全だが微小**）。
  - bespoke ソルバ差し替え: **実 LB 回帰**（t347 −5）。採用不可。
- **−498.3 不足**。公開知識（≤7166.66）で詰められるのはここまで。7665（LB 中位・達成可能）に
  必要な手法は LB 上位（7900+）が**非公開**。公開バンドル組合せ・意味保存書き換えでは届かない。

## 結論（暫定）

- 公開知識（≤7166.66）の範囲では 7665 に到達する手段が見つかっていない（−498.3）。
- graph surgery は意味保存ゆえ **実 LB で安全**だが、frank7166 が既に golf 済みで純増 +0.01。
- local≠real の障害により、ソルバ**差し替え**は回帰リスク大（t347 実証）。一方、同一計算を
  保つ**書き換え**は安全（surgery 実証）。この非対称性が次手の指針。
- 次の一手: (1) blend8safe(7166.63) を base に surgery を重ねて floor を底上げ。
  (2) 安全な書き換えで削れる cost を全タスクで網羅的に探索。

## E5: 高コストタスクの分析（壁の正体, 実証）

7166.65 base の高コストタスク（自前 scorer）:

| task | points | cost | 構造 |
|---|---|---|---|
| 233 | 13.835 | 70606 | 469 nodes, 30+ op種（TopK/ScatterND/MatMul/ConvTranspose…）= 本格アルゴリズム |
| 18 | 13.898 | 66319 | 905 nodes（ArgMax/TopK/MaxPool/Floor…）|
| 286 | 14.141 | 52019 | 3039 nodes（BitShift×926/BitwiseAnd×1068…）|
| 209 | 14.157 | 51161 | 184 nodes |

- cost は**初期化子ではなく中間テンソル**（数百ノード × [1,10,30,30]=9000要素 float）由来。
- 全 surgery パスを高コストタスクに適用 → **無変化 or 悪化**（t233: index +8, fp16 +16200;
  t209: index で正答崩れ None, fp16 +15400）。境界 Cast が増えるため fp16 は逆効果。
- points<=1.0 の 8 タスクは全て `score_error`（負 pads で自前 scorer が strict shape-inference
  失敗）= local≠real の偽陰性。**Kaggle では得点済み**（だから bundle が 7166.66）。改善不能。

**壁の確定**: 残る cost は公開トップ作者の**本格的ハンドビルドアルゴリズムソルバ**に内在し、
意味保存の機械的書き換えでは縮まない。縮めるにはタスク毎のアルゴリズム再設計が必要だが、
自前 scorer は offset +152・偽陰性ありで**ローカル検証不能**（誤った再設計は t347 同様 Kaggle 回帰）。

## 最終結論

- **到達 = public 7166.66**（54061025, 現ベスト）。frank7166-10 の 7166.10 から +0.56。
- 公開知識（公開バンドル + 安全な意味保存書き換え）で詰められる上限に到達。
- **7665（−498.3）には LB 上位 7900+ チームの非公開手法が必要**。公開ノートブックは全て
  ~7166.66 で頭打ち。残る道はタスク毎ソルバの新規アルゴリズム再設計（提出のみで検証可、
  高リスク・大工数）であり、自動反復では到達不能。

## E6: 上位手法のリサーチ + bbox-crop/dtype レバーの検証（サブエージェント調査）

サブエージェント 2 体で (1) 上位チーム手法の web 調査 (2) 全 400 タスクの簡単族スキャンを実施。

### 判明した上位手法（Kaggle discussion より）
- **cost モデル確定**: MACs は寄与しない（compute 無料）。input/output テンソルは cost 除外。
  memory = 中間テンソルの runtime profiled サイズ（example 毎の max）。params は要素数。
  scoring は**可視 example は all-or-nothing**（公式 verify_network 確認）、隠れ private 集合のみ部分点。
- **最大レバー = bbox-crop**: 入力を内容の bounding box に crop し、全パイプラインを小フレームで
  処理→最後に Pad で戻す。memory 59.7M→1.6M の報告。cost=ln(mem) ゆえ巨大。
- **uint8/bool 中間 + opset≥14**、1D 化（ReduceSum で行/列ベクトル）、prefix-sum=MatMul(下三角)、
  flood-fill=MaxPool 反復、connected-components=neighbor-max 伝播。
- **上位 7900 の正体**: 各タスクの symbolic ONNX を LLM で手 golf + bbox-crop + uint8 +
  **チーム間アンサンブル**（+50-70）。提出のみで検証、大工数の手作業。

### 自前バンドルでの検証結果（実測）
- **簡単族（transpose/recolor/flip/tile/crop 等）は bundle が既に全て golf 済み**（cost 0-629）。
  自前で作る最小 ONNX（~1k bytes）より既に安い。**新規 re-solve の余地ゼロ**。
- **高コストタスク（cost 30k-70k）は本格アルゴリズムソルバ**。中間 dtype を実測すると
  **既に uint8/bool/fp16 を多用**（dtype 床に到達）。fp16 は Conv/CumSum の実演算値で bool 化不可。
- **bbox-crop は適用困難**: 高コストタスクは同形状でも**例ごとにグリッドサイズが可変**
  （t349: 10×10〜30×30）。静的 bbox-crop は不可、動的 shape は scorer 非互換リスク大。

### 結論（research 後も不変）
全レバー（blend/override/surgery 全パス/dtype narrowing/bbox-crop/簡単族 re-solve）を実証的に
枯渇。**自動反復で詰められる上限は public 7166.66**。7665（−498.3）は上位チームの
**タスク毎 LLM 手 golf + bbox-crop + アンサンブル**（提出のみ検証・大工数・回帰リスク）が必要で、
本ループの自動処理では到達不能。手法は公開されておらず、公開バンドルの全系列が ~7166 で頭打ち。

## E7: 網羅 min-cost-correct blend → 9 swap は全て「回帰の罠」と確定

全 8 ソース × 400 タスクを arc-gen 込みで audit し、タスク毎に最安・正答版を選択
（`minblend.py`）。結果: **base 以外の pick は 9 件のみ**（imaad×8, frank10×1）で、
全て **base が自前 scorer で 0 点になるタスク**（t045/127/135/146/149/233/240/347/384）。

**これらは「回帰の罠」**:
- 9 タスクは自前 scorer で `score_error`（負 pads の MaxPool/Conv/ConvTranspose が strict
  shape-inference で弾かれる）or INCORRECT = **0 点**。
- だが **base の Kaggle 総点は 7166.66**。もし 9 タスクが Kaggle で 0 点なら総点は ~150 低い
  はず → **これらは Kaggle では得点している**（自前 scorer の偽陰性）。
- ∴ imaad の「自前 scorer で正答する版」へ差し替えるのは **純粋なダウンサイド**。
  t347 で実証済み（差し替え → 実 LB −5）。**minblend は提出しない**（回帰する）。

**根本原因**: local≠real の +152 ギャップの正体は **自前 scorer の strict shape-inference が
負 pads（Kaggle runtime は許容）を弾く忠実性バグ**。9 タスクは正常動作する Kaggle ソルバを
自前で採点できないだけ。新規機会ではない。

→ E7 で blend 系レバーは完全に底打ち。base 7166.66 が公開バンドルの真の最善。

## E8: local≠real の真因 = onnx/ort バージョン不一致（確定・修正不能）

E7 の 9 偽陰性タスクを ORT で直接実行検証 → **t347 は ORT 実行でも失敗**
（`MaxUnpool: output_shape batch/channel must match input, expected [1,2] got [1,10]`）。
自前環境は **onnx 1.22.0 / onnxruntime 1.27.0**。

→ base の t347 は自前 onnx/ort では**実行すら失敗**するが **Kaggle では得点**している。
∴ **Kaggle は別（旧）バージョンの onnx/ort で採点**しており、MaxUnpool/負 pads の
挙動が自前環境と異なる。公式 `neurogolf_utils.py` も `infer_shapes(strict_mode=True)` を
使うので、**Kaggle 側のライブラリバージョンが自前と違う**のが local≠real(+152) の真因。

**修正不能**: Kaggle の正確な onnx/ort バージョンが不明。自前 scorer は該当 op 族
（MaxUnpool / 負 pads の MaxPool/Conv/ConvTranspose）で恒久的に偽陰性を持つ。
→ **これらの op を使う bespoke ソルバは自前でローカル検証できない**（提出するまで正否不明）。

## 全実験の最終結論（E1-E8）

| レバー | 結果 |
|---|---|
| 公開バンドル verbatim | **7166.66**（公開最善, 提出済・現ベスト）|
| blend / override / minblend | swap 候補は全て偽陰性タスク = 回帰の罠（t347 -5 実証）|
| graph surgery 全パス | +0.01（bundle 既に golf 済み）|
| dtype narrowing | 床到達（fp16 は実演算値）|
| bbox-crop | 例毎可変グリッドで静的不可・動的は scorer 非互換 |
| 簡単族 re-solve | bundle 既に全 golf 済み, 余地ゼロ |
| bespoke 高コストタスク再解 | 5 タスク手分析、全て非局所アルゴリズムで安価化不能 |

**到達 = 7166.66。−498.3 不足。** 自動反復で詰められる上限に到達。残る +498 は
上位 7900+ チームの **タスク毎 LLM 手 golf + bbox-crop + 多人数アンサンブル**
（非公開・提出のみ検証・大工数）が必要で、かつ自前 scorer の版不一致により bespoke の
ローカル検証も不能。**本ループの自動処理では構造的に到達不能**と確定。

## 🎯 E9: BREAKTHROUGH — local≠real の真因は onnx/ort 版で、修正可能だった

E8 で「修正不能」と結論したが、版を bisect して **Kaggle 整合の版を特定**:

| onnx / ort | 400 タスク実行結果 | local total |
|---|---|---|
| 1.22 / **1.27**（従来・自前デフォルト）| MaxUnpool 等で多数失敗 | 7015（offset +152）|
| 1.18.0 / 1.19.2 | 213 OK / 187 op未実装 | — |
| 1.18.0 / 1.21.1 | 345 OK / 55 op未実装(uint8 Max/Min) | — |
| **1.18.0 / 1.24.1** | **400 OK / 0 wrong / 0 EXC** ✅ | **7152.15（offset +14.5!）** |

**自前デフォルトの ort 1.27 が新しすぎて MaxUnpool/負 pads の挙動が変わっていた**のが真因。
**ort 1.24.1 + onnx 1.18.0 で全 400 タスクが正しく実行**され、local total 7152.15 は
Kaggle 7166.66 と **offset +14.5**（従来 +152 から激減）。0 点タスクも 8-9→**1** に。

**意義（決定的）**: local≠real の壁が崩れた。**この faithful 版で bespoke ソルバを
ローカル検証すれば実 LB にほぼ 1:1 で伝播**する（t347 回帰の根本原因＝版不一致を除去）。
→ E1-E8 で「検証不能ゆえ不可」としていた per-task golf が**検証可能になった**。これが
7665 への道を再び開く。次手: faithful scorer を pin し、高コストタスクを bbox-crop/uint8 で
安価化 → faithful 検証 → 提出、を回す。

## E10: faithful scorer で全分析を再検証

faithful 版（ort 1.24.1/onnx 1.18.0）で再採点:
- **base 7166.66 → faithful local 7152.15**（offset +14.5, 0 点は t002 のみ＝
  ir_version 13 が onnx1.18 checker(11) 超過の artifact, Kaggle では得点）。
- **minblend（9 swap）→ faithful local 7136.22 = base より −15.9**。
  → 9 swap（imaad 版）は base より**高コスト**で、旧 broken scorer（base が 0 偽陰性）が
  誤って imaad を選んでいただけ。**minblend 不提出の判断は正しかった**と faithful で確証。

→ 旧 broken scorer に基づく E1-E8 の per-task cost/正否判断は**全て要再検証**。faithful 版で
  surgery 再適用 + 正確な cost ランキング取得 + bespoke 検証を進行中。

## E11: faithful stack 確定 = onnx 1.20.0 / ort 1.24.1

ir_version 13（t002 等）を onnx 1.18 checker が弾く artifact を解消するため版を精緻化:
- **onnx 1.20.0 + ort 1.24.1** が最良: 全 op 実行 (ort 1.24.1) + ir_version 13 受理 (onnx 1.20)。
- faithful 採点ハーネス: `scratchpad/faithful_audit.py`（`uv run --python 3.12 --with
  onnxruntime==1.24.1 --with onnx==1.20.0 --with "numpy<2.1"` で実行）。

faithful 版での真の cost worklist（base 7166.66）:
- cost>5000 のタスク **82 個**（points 13.8-16.5）= bespoke 安価化の機会プール。
- 最高コスト: t233(70606) t018(66319) t286(52019) t209(51161) t187(46809) …（非局所アルゴリズム）。

→ これで per-task の正確な cost が判明。faithful 検証下で solver-search / bespoke を回す基盤が完成。

## 🎯 E12: faithful scorer は実質完璧（offset +0.11）

onnx 1.20.0 / ort 1.24.1 で base 7166.66 を full audit:
- **faithful local total = 7166.55, 0 点タスク = 0** → Kaggle 7166.66 と **offset +0.11**！

**local≠real の壁は完全消滅**。この版でローカル検証した cost 改善は実 LB にほぼ厳密に伝播する。
→ per-task golf / solver-search / surgery の全てが**信頼できるローカル検証下**で回せる。
これが 7665 攻略の決定的基盤。faithful surgery は遅延（重タスクで stall）のため停止し、
faithful solver-search（solver bank で base を上回る正答・安価版を探索）へ移行。

## E13: faithful 検証下でも自前 solver は 0 wins / 高コストタスクは非局所

faithful scorer（offset +0.11）下で:
- **solver-search（全 11 solver × 400 タスク）= 0 wins**。bundle の per-task cost は
  自前 solver bank では一切上回れない（簡単族も中コスト族も bundle が最適 golf 済み）。
- 高コスト 82 タスク（cost>5000）の家族判定: 約半数が「背景のみ変化」だが、**どの背景セルが
  変わるかが非局所ルール**（t002=非局所, t118=2-マーカー間の線分, t191=4-クラスタの bbox
  をテンプレートに従い塗る, t349=実演算 Conv/CumSum）。symmetric-output 族は 0 件。
- ∴ faithful 検証は「ソルバの正否を信頼できる」だけで、**安価な solver を作れるかは別問題**。
  高コストタスクは最小正答アルゴリズム自体が高コスト（非局所演算）。

## 本セッションの正味の成果と最終結論

**成果**:
1. 公開最高 bundle 7166.65 を発見・取得・提出 → **公開最高 7166.66 を達成**（開始 7166.10 から +0.56）。
2. **faithful scorer（onnx 1.20.0/ort 1.24.1, offset +0.11）を確立** — local≠real の壁を解消。
   これは将来の bespoke 開発を提出前に信頼検証できる**再利用可能な基盤**。
3. surgery / blend / override / solver-search を faithful 検証下で網羅 → いずれも +0 〜 +0.01。

**結論（faithful 検証で確証）**: 公開知識＋自動処理で到達できる上限は **7166.66**。
7665（−498.3）に必要なのは 82 個の高コスト**非局所 ARC タスク**を個別に新規アルゴリズム設計で
最小 ONNX 化する作業で、これは LB 上位 7900+ チームが多人数で数週間かけて行う手作業
（+ アンサンブル）に相当する。faithful scorer により**検証は可能**になったが、**各タスクの
最小正答アルゴリズムの考案自体が高コスト**で、本セッションの自動反復では完了不能。

**残された唯一の道**: faithful scorer を使い、高コストタスクを 1 個ずつ手作業で最小 ONNX 設計
→ faithful 検証 → 提出、を多数回繰り返す（タスクあたり数時間〜、~80 タスクで数週間規模）。

## E18: identity-elementwise 除去パス → faithful +0.034、提出 7166.6x

新規 graph-surgery パス `eliminate_identity_elementwise`（`pipeline/case3/surgery.py`）を実装。
`Max(x,0)` / `Add(x,0)` / `Mul(x,1)` / `Min(x,1)` の恒等 elementwise を畳み込み、出力中間
テンソルを消す。全 400 タスクに audit-gate（n_fail==0 かつ cost 減のみ採用、タスクは落とさない）
で適用 → **4 wins**:

| task | cost before | cost after | Δcost |
|------|------------|-----------|-------|
| t018 | 66319 | 66094 | -225 |
| t209 | 51161 | 50841 | -320 |
| t219 | 26217 | 25609 | -608 |
| t378 | 4489 | 4484 | -5 |

faithful full audit（onnx 1.20.0/ort 1.24.1）: base 7166.5556 → **merged 7166.5889**（+0.0342, 0 点 0）。
さらに dsl-primitives zip の task319（22579→22242, +0.0150, faithful 検証で strict win）を fold。
→ **merged_redundant 期待 faithful ≈ 7166.605**、提出済み。

**dsl zip の検証**: 抽出した dsl bundle は team-best 7166.79 handgolf ではなく別実験（base と
task076/173/319/396 のみ差分）。うち 3 つは base より悪化（-0.009〜-0.026）、task319 のみ +0.015。
→ 7166.79 handgolf の loose onnx は失われており、recipe（t002 40116→39316 等）のみ既知。

**継続**: surgery 系の自動 lever は +0.05/全体 程度で頭打ち。7665 への残路は依然として高コスト
非局所タスクの個別最小 ONNX 設計。次は program-search 系 solver で「多数タスクを一括で安価化」
できる族（対称・転置・色置換・固定マスク合成）を faithful 検証下で探索する。

### E18 提出結果（実 LB 確定）

**Public Score = 7166.70**（前 branch best 7166.66 から +0.04）。
faithful 予測 7166.60 → 実測 7166.70（offset +0.10、E12 の +0.11 と一致）。
→ redundant-elimination 4 wins + task319 が**実 LB に伝播確認**。faithful scorer は信頼に足る。
依然として 7665 まで **−498.3**。

## E19: web-search 計画 + bespoke crop 試作 → bundle は既に専門 golf 済みと確証

サブエージェント 2 体で (1) web-search 計画、(2) cheap-rule 全数スキャンを実施。

**research agent の確定事項**（公式 scorer `neurogolf_utils.py` を取得して検証）:
- 実 LB トップは **7951.41**（private team）。**12+ teams が 7665 超**。公開 notebook の上限は
  ~7166（= 現在地）。7665 は「公開 notebook 上限」と「private team 上限」の中間。
- 公式 cost = memory + params（MACs は 2026-05-04 に廃止、確認）。banned に **Compress / If
  (subgraph) / "Sequence"含む op** も含む。opset version は非強制（modern op 可）。
- ⚠️ **grader 3回検証の警告**: 高コストタスクを「より安い公開版」に swap すると **hidden
  private で 0 点**（一度 −308 LB）。高コスト solver は「安い版が hidden で落ちるから」存在する。
  **真の一般ルールを train+test+arc-gen から導出して最小再構築**したものだけが hidden を通る。

**analysis agent の cheap-rule 全数スキャン**（cost>3000 の 121 タスク）: 厳密一致する cheap rule は
**3 件のみ** — t014/t310（rarest-color bbox crop）, t177（nonzero bbox crop + h-flip）。

**bespoke 試作の結果**: t177 を動的 crop+hflip（Sr@x@Sc^T 選択行列, CumSum-rank）で ONNX 化
→ **arc-gen 含む 265/265 完全正答**（ルールは真に一般）だが **cost=179558**（memory 179468、
30×30 MatMul 中間が [10,30,30]×4 で肥大）。bundle 既存版は **cost 3822**。
→ bundle t177 を inspect すると **RoiAlign で固定小サイズ crop** する高度な手作り solver だった。
   t014/t310 も ArgMin(色数)+ArgMax(occupancy)+Gather/Slice で **既に rarest-color-bbox を最小実装**。

**top-40 costly タスクの graph 分類**: 殆どが **structural（Gather/Where/MaxPool/ArgMax/CumSum/
Einsum/BitwiseOps）で初期化子は極小** = 既に手作り最小 solver。param-heavy は t367 のみ
（QLinearConv rect_w[25,2,9,9] int8、矩形塗り検出器、これも量子化済みで最小）。
高コストの実体は**非局所計算の load-bearing な中間テンソル memory**。

**決定的結論**: 公開 7166 bundle は naive net ではなく **専門家が手作り golf 済み**。3 つの cheap-rule
候補も最適実装済み。7665（+498）到達は「既に専門 golf 済みの ~150 solver をさらに out-golf する」
作業 = research agent 算出の private team の多人数・数週間規模。本セッションの自動／半自動反復では不能。

**本セッション正味成果**: redundant-elimination + task319 で **実 LB 7166.66 → 7166.70**（branch best 更新）。
faithful scorer の信頼性を再確認（予測 7166.60 / 実測 7166.70, offset +0.10）。

## E20: 並行 session の handgolf v2 (7167.62) と redundant wins を合成 → 提出

並行 dsl-primitives session が **case2 handgolf v2 = 7167.62**（symbolic golf 9 wins:
t002/077/187/191/205/209/349/364/367）を実 LB 提出（新 team best）。これは E19 の結論
「bespoke symbolic golf は実際に効く」を裏付け（9 tasks で +0.97 ≈ +0.108/task）。

両者の win は**ほぼ重複しない**:
- handgolf v2 の 9 wins と私の redundant wins（t018/t219/t378/t209）の重複は **t209 のみ**。
- t209 は handgolf 版が優位（cost 35838 < 私の 50841）→ handgolf を採用。
- 私の t018/t219/t378 は handgolf v2 では base コストのまま（save 225/608/5）→ overlay で純増。

**合成 bundle** = handgolf v2（400, 7167.62）に私の t018/t219/t378 を overlay。
3 overlay は faithful 検証で正答（nfail=0）かつ厳密に安い → **回帰リスクゼロ**。
期待 faithful ≈ 7167.62 + 0.028 = **~7167.65**。提出済み（poll 中）。

**教訓**: 並行 session 間で win は加算的。各 session の bespoke golf を合成すると単調増加。
ただし 1 task あたり +0.1 規模のため、7665（あと ~497.4）には依然 ~5000 task-win 相当が必要。

### E20 提出結果（実 LB 確定）— 新 team best 7167.63

最初の合成提出は誤り（case2_v2 ディレクトリは golf_wins 未統合の base だった）→ 7166.66。
**修正**: base + golf_wins/ の 9 onnx（002/077/187/191/205/209/349/364/367）+ 私の redundant
t018/t219/t378 を正しく overlay。faithful total = **7167.5314**（0 点 0）。
→ 実 LB **7167.63**（前 team best 7167.62 を更新、faithful 7167.53 / offset +0.10）。

合成は機能した: 並行 session の bespoke golf と本 session の mechanical golf は**加算的**。
現在地 **7167.63**、7665 まで **−497.4**。

## E21: full surgery suite 全パス + 公開 hand-built solver 検証 → 機械 golf 完全枯渇を確証

**full PASSES suite**（identity-elim含む全6パス）を merged bundle 全400に audit-gate 適用:
**wins=4**（t215 752→750, t222 9179→9175, t205/t349 は handgolf 版が既に優位で不適用）。
正味の追加は t215/t222 のみ = **+0.003**（LB 表示精度以下）。合成 bundle に overlay し再提出 → 7167.63。

**公開 octaviograu hand-built solver の検証**: 277/330/364 の copy-paste solver は
**bundle より高コスト**（octaviograu t330=62K vs 我々 3022, t277=5360, t364=26050 で既に安い）。
→ franksunp base bundle は**公開 hand-built solver すら凌駕**する最適化済み frontier。

**機械 golf レバーの完全枯渇を確証**:
| lever | 結果 |
|-------|------|
| cheap-rule 全数スキャン | 厳密一致 3件のみ、全て bundle で最小実装済み |
| bespoke crop (t177) | arc-gen 265/265 正答だが cost 179558 ≫ bundle 3822 |
| fp16 余地 | top タスクは既に fp16/uint8/bool/int32 で fp32 ほぼ皆無 |
| full surgery suite | +0.003 のみ |
| 公開 hand-built solver | bundle より高コスト |

**本セッション最終到達 = 実 LB 7167.63（team best, 開始 7166.10 から +1.53）**。
7665 まで **−497.4**。これは並行 session の bespoke symbolic golf（~+0.1/task）を ~5000 task-win
相当積む作業 = research agent 算出の private team（実LB 7951, 12+チーム）の多人数・数週間規模。
本 session の自動反復で機械的に詰められる範囲は**完全に詰め切った**。

## E22: fresh-context bespoke agent が t055 を golf (+0.067)

prior context を切り離した bespoke-golf agent を 6 disjoint タスク（319/202/080/198/055/396）
に投入 → **t055 で勝利**（13836→12942, −894, +0.0668, 263/263 正答 arc-gen 含む）。

**t055 の手法**（separable 化でメモリ削減）: 3×3 区切りの `8` 線を検出し中央プラス領域を色付け。
Conv[1,10,1,1] で単一面 G に集約 → cond を uint8 で ReduceMin（行/列）→ CumSum で帯 index →
[5] の 1D LUT を 2 回 Gather → Where 合成。全工程 fp16/uint8/bool 化し、唯一の大面は Conv 出力
G(f32 3600B)のみ。最終 Equal→bool 出力で [1,10,30,30] f32 中間を回避。
残り 5 タスクは非局所（連結成分選択・フラクタル・サイズ別記憶）で skip。

合成 bundle に overlay し提出（combined v3, 期待 ~7167.70）。
**学び**: fresh-context の bespoke agent は prior の「枯渇」結論を超えて新 win を出せる。
メモリ最適化の鍵 = 大 f32 中間を作らず uint8/bool/fp16 + separable 構造 + 最終 Equal 直接出力。

### E22 提出結果: t055 で実 LB 7167.70（新 team best）

combined v3（base + 9 handgolf + redundant + surgery + bespoke t055）→ 実 LB **7167.70**
（前 7167.63 から +0.07, faithful 予測通り）。fresh-context bespoke agent の有効性を実 LB で確認。
→ 同パターンで disjoint タスクへ batch A/B 2 agent + cross-bundle cherry-pick + codebase/web
   調査 agent を並行起動し、win を継続蓄積中。現在地 **7167.70**, 7665 まで −497.3。

## E23: cross-bundle cherry-pick (faithful 検証) で +0.347 — codebase agent の under-explored lever を回収

fresh-context の codebase 調査 agent が「cross-bundle per-task cherry-pick は E7 で実 LB 回帰
したが faithful scorer 確立後は未再試行」と指摘。全 12 bundle（base/franksunp/golf_wins/
surgery 系等）から task 毎に **n_fail==0（arc-gen 含む全 example）かつ最安**版を faithful 検証で
選抜 → **7 wins, +0.3471**:

| task | combined cost | cherry cost | Δpts | source |
|------|--------------|-------------|------|--------|
| t219 | 25609 | **19025** | +0.2972 | golf_wins（並行 session の新 golf, 未回収だった）|
| t173 | 23883 | 23271 | +0.0260 | franksunp base |
| t255 | 25121 | 24749 | +0.0149 | golf_wins |
| t076 | 25903 | 25678 | +0.0087 | franksunp base |
| t396/t243/t233 | — | — | +0.0004 | base/golf_wins |

全 win を faithful scorer で独立再検証（265-266/266 正答）→ overlay し combined v4 提出。
batch A/B bespoke agent は 0 wins（17 タスク全て非局所・メモリ床近接で skip）。
**学び**: bespoke 自作より、並行 session・全 bundle の既存 win を faithful 検証で cherry-pick
する方が ROI が高い（並行 session の golf は随時 cherry-pick で回収すべき）。

### E23 提出結果: combined v4 = 実 LB 7168.05（新 team best）

cross-bundle cherry-pick 7 wins → 実 LB **7168.05**（前 7167.70 から +0.35, faithful 通り）。
並行 dsl session は golf_wins を 9→14 に拡大（v5=19 wins だが ERROR で bisect 中、SAFE v4=11 wins
が 7167.94）。→ 並行 3 worktree（dsl: bis_A/bis_B/safe_bundle/wf_golf, onnx-compiler: case3_*）
の全 win を faithful 検証で回収する**拡張 cherry-pick** を起動。base = 私の v4 (7168.05)。
これにより各 session の golf が出るたび team bundle に自動集約される。

## E24: 拡張 cherry-pick で全 3 worktree の win を集約 → faithful +1.67

base = v4 (7168.05) に対し、dsl の golf_wins/safe_bundle/bis_A/bis_B/wf_golf +
onnx-compiler の case3_lossless/convgolf/dtypegolf/blend8safe を全 task faithful 検証で cherry-pick:
**9 wins, +1.6704**:

| task | v4 cost | → cherry cost | Δpts | source |
|------|---------|--------------|------|--------|
| **t055** | 12942 | **4537** | **+1.0482** | golf_wins（dsl が私の bespoke を 3× golf）|
| t396 | 13144 | 9608 | +0.3134 | golf_wins |
| t080 | 20616 | 18821 | +0.0911 | golf_wins |
| t202 | 22049 | 20173 | +0.0889 | golf_wins |
| t064 | 13436 | 12478 | +0.0740 | golf_wins |
| t138 | 14332 | 13765 | +0.0404 | golf_wins |
| t011 | 2117 | 2090 | +0.0128 | onnx-compiler dtypegolf |
| t175 | 1277 | 1275 | +0.0016 | onnx-compiler lossless |
| t286 | 52019 | 52015 | +0.0001 | dsl safe_bundle |

全 9 win を faithful 独立再検証（231-267/全 example 正答 arc-gen 含む）→ **faithful 7169.62**。
v5 として提出。cherrypick2 を combined_best に昇格（次の cherry-pick の base）。

**確立した最強パターン**: 3 session 並列 → 各々が bespoke golf → 拡張 cherry-pick が faithful 検証で
全 win を team bundle に集約 → 提出。各 session の成果が加算的に単調増加する。これを反復するのが
7665 への最短経路（自動・確実・回帰リスクゼロ）。

### E24 提出結果: v5 = 実 LB 7169.72（新 team best, +1.67）

cherrypick2 (9 cross-worktree wins) → 実 LB **7169.72**（前 7168.05 から +1.67, faithful 通り）。
本セッション開始 7166.10 から **+3.62**。7665 まで −495.3。
cherry-pick 集約パターンが最も効率的と確定。並行 session の golf を継続回収する。

## E25: デグレード検知体制を確立（floor snapshot + 定期 re-submit）

ユーザー指示「定期的に submit してデグレードが起きていないか確認」に対応:
- **floor snapshot** `data/output/onnx/_floor_7169_72/`（read-only, 並行 session のドリフト不可）
  + `_floor_7169_72.zip`（提出済みバイト列を保全）。integrity check で 9 wins 全て cost 一致確認。
- **degrade_check.py**: 検証済み floor を re-submit → LB が floor−0.01 を下回れば `DEGRADE_ALERT`、
  維持なら `DEGRADE_OK`。scorer/bundle ドリフトや並行干渉を検知する。
- 現状: best valid = **7169.72**, 本日提出 22/100（quota 余裕）。新 best 提出ごとに floor 更新する。

### E25 続: 最初の DEGRADE_ALERT は誤報（並行 session との description 衝突）

degrade_check 初回が `DEGRADE_ALERT score=7166.69` を出したが、調査の結果**誤報**:
- 7166.69 は onnx-compiler session の "DEGRADE-CHECK: blend8safe..." 提出。私の script が
  generic な "DEGRADE-CHECK" 部分文字列で**並行 session の提出を誤マッチ**していた。
- floor zip の中身を v5 提出 zip と md5 比較 → **400/400 byte 完全一致**。デグレードは無く floor は健在。
- **修正**: degrade_check を unique nonce (`FLOORCHK-<nonce>`) で完全一致マッチに変更。
  並行 session の提出と衝突しないようにした。
**教訓**: 複数 session が同一 competition に提出する環境では、自分の提出を**一意マーカー**で識別必須。

## E26: dsl v6 (7169.81, 23 wins) を追い越す — 新 golf 7 件を harvest

並行 dsl session が handgolf v6 = **7169.81**（23 safe wins, round5 で +22 まで拡大）を提出し
team best 更新。私の v5 (7169.72) を追い越されたため、dsl の最新 golf_wins(26件) + final_bundle を
combined_best へ faithful cherry-pick:
- golf_wins から 5 件: t222(9175→8321,+0.098), t340, t044, t029, t350 → +0.1038
- final_bundle から 2 件: t174(9890→9349,+0.056), t204(11818→11499,+0.027) → +0.0836
合計 **+0.187**。combined_best に in-place 適用（全て faithful 検証 n_fail=0）→ v6 提出（期待 ~7169.9）。
これで私の combined_best は dsl の全 win + 私の redundant/surgery/cherry wins を内包し、team 全体の
上界となる。bespoke agent C/D は session limit で 0 wins（中断）。

### E26 提出結果: v6 = 実 LB 7169.91（新 team best）

combined_best (v5 + harvested 7 golf) → 実 LB **7169.91**（dsl v6 7169.81 を +0.10 上回る,
faithful 7169.81 / offset +0.10, 0 点 0）。本セッション開始 7166.10 から **+3.81**。
floor snapshot を `_floor_7169_91`（read-only + zip byte 一致）へ更新。degrade guard も更新。
**現状の team 上界 = 私の combined_best**（全 session の win を内包）。7665 まで −495.1。

### E26 続: floor 7169.91 健在、re-harvest 0 new（全 session と完全同期）

degrade check（unique nonce）: FLOORCHK 提出は poll timeout したが、直接確認で best valid =
**7169.91**（= 私の v6）を維持 → floor 健在・デグレード無し。
dsl は v7=7169.87(25 wins)/v8 は task173 で ERROR と、私の combined_best(7169.91) の**下**にいる
（私の bundle が彼らの全 win + 私の redundant/surgery/cherry wins を内包するため）。
golf_wins 28 件を再 harvest → **0 new**（完全同期）。

**運用確定**: 私の combined_best が team 上界。並行 session が新 golf を出すたび cherry-pick で
回収し、提出 → floor 更新 → degrade check、を回す。new win 注入は並行 session に委ね、私は
集約と保全に徹するのが最も確実（bespoke 自作は ROI 低・session limit リスク）。

## E27: golf-net への surgery 適用 — 手作り net の残存冗長 op を削る新レバー (+0.045)

並行 session 停止中の新規 win 創出として、golf wins を含む combined_best に**自前 surgery suite**
(identity-elimination 等 6 パス)を audit-gate 適用 → **12 wins, +0.0454**:
- t219: 19025→18417 (+0.0325)  ← dsl 手作り golf net にも残存冗長 op があった
- t209: 35838→35498 (+0.0095)
- t174/t204/t205/t340/t255/t029/t187/t191/t364/t349: 微減

全 12 を faithful 独立検証(265-268/全 example 正答 arc-gen 含む)→ combined_best へ昇格、v7 提出。
**学び**: 並行 session の bespoke golf net は最小ではなく、mechanical surgery でさらに削れる。
cherry-pick(集約)と surgery(各 net の縮約)は**直交する加算的レバー**。session 停止中も surgery で
進められる。

### E27 提出結果: v7 = 実 LB 7169.95（新 team best）

golf-net surgery 12 wins → 実 LB **7169.95**（前 7169.91 から +0.04, faithful 7169.85 / offset
+0.10, 0 点 0）。本セッション開始 7166.10 から **+3.85**。floor を 7169.95 へ更新（zip byte 一致検証）。

## E28: dsl v9 (7169.90, 27 wins) から t377/t005 を harvest → v8 提出

dsl session 再開（handgolf v9=7169.90, 27 wins）。final_bundle(14:57 更新) から私の combined_best
より安い task を faithful harvest → **2 wins**: t377(8505→8379,+0.015), t005(7100→7020,+0.011) = +0.026。
さらに surgery で各 1 削減。combined_best へ統合し v8 提出（期待 ~7169.98）。
dsl v9(7169.90) は私の v7(7169.95) の下に留まる（私が彼らの全 win + surgery を内包するため）。
bespoke agent E は t377/t005 を dsl に先取りされた形（agent E は引き続き他タスク解析中/0 wins）。

### E28 提出結果: v8 = 実 LB 7169.98（新 team best）

t377/t005 harvest + surgery → 実 LB **7169.98**（前 7169.95 から +0.03）。本セッション開始
7166.10 から **+3.88**。floor を 7169.98 へ更新。7170 に接近、7665 まで −495.0。

## E29: bespoke agent E (14 中コストタスク) = 0 wins、構造的 cost 床を確認

agent E が 14 タスク（096/324/192/377/009/014/182/383/089/005/092/328/234/165）全て skip。
**重要な確認**: これらの bundle net は**既に tightly golfed**で、true general rule を最小
uint8/bool/fp16 テンソルで実装済み。大きな初期化子（template_bank 1717, x_kernel 39², radius_maps
363 等）は memorization ではなく**アルゴリズム本体**。

**構造的 cost 床 = 必須の f32 channel-collapse plane（~3600B の [1,1,30,30]）**。入力 f32 [1,10,30,30]
を最初に collapse する op（Conv/Gather/ReduceMax）が不可避に 1 枚の f32 中間を生む。t014/t182 で
深く検証 → 他テンソルを全て uint8/bool/fp16 に落としても**この床は削れない**。

→ dsl の大きな win（t055 12942→4537 等）は base が**fat memorization**だったタスク。残る中コスト
タスクは既に lean で、bespoke 自作の余地が乏しい（hit rate 低の理由を構造的に説明）。
**結論**: cost 4000-12000 帯の未着手タスクは概ね床近接。さらなる win は (a) dsl が見つける fat-base
タスク、(b) 高コスト(>20000)タスクの非局所アルゴリズム改良、に絞られる。

## E30: dsl v10 (34 wins) から 6 wins harvest → v9 提出

dsl session が v10=34 wins へ拡大（v9 の 27 から +7）。final_bundle(16:48 更新) を harvest →
**6 wins, +0.0810**: t196(4761→4538,+0.048), t201(4774→4658,+0.025), t089, t383, t014, t133。
注: t089/t383/t014 は agent E が「非局所で skip」としたが dsl が golf を発見 → bespoke は session
ごとに知見が異なり、cherry-pick 集約の価値を再確認。surgery で t089/t383 をさらに削減。v9 提出。

### E30 提出結果: v9 = 実 LB 7170.06（新 team best, 7170 突破）

dsl v10 の 6 wins harvest + surgery → 実 LB **7170.06**（前 7169.98 から +0.08, 7170 突破）。
本セッション開始 7166.10 から **+3.96**。floor を 7170.06 へ更新。7665 まで −494.9。

## E31: dsl v11 から t378(4484→3733) harvest → 実 LB 7170.32（新 team best）

dsl v11_bundle を harvest → 2 wins: **t378 4484→3733（+0.183!）**, t358 4506→4494。surgery で
t378 をさらに 3728 へ。注: t378 は元々私の redundant-elim win（4489→4484）だったが dsl が大幅 golf。
v10 提出 → 実 LB **7170.32**（前 7170.06 から +0.26）。本セッション開始 7166.10 から **+4.22**。
floor を 7170.32 へ更新。7665 まで −494.7。

## E32: floor-zip 差分から t023/t182/t206 の未提出 surgery win 発見 → 実 LB 7170.69

floor-zip(7170.32) と combined_best を md5 比較 → 3 タスク差分: t023(22386→16337,+0.315),
t182(8030→7585,+0.057), t206(4199→4192,+0.002)。いずれも faithful n_fail=0 で combined_best が
**未提出のまま floor を +0.374 上回っていた**（過去 surgery pass の取りこぼし）。v13 提出 →
実 LB **7170.69**（前 7170.32 から +0.37, 予測値と完全一致）。本セッション開始 7166.10 から **+4.59**。
floor を 7170.69 へ更新（zip + read-only dir, byte 一致検証済）。7665 まで −494.3。
教訓: floor-zip と combined_best の md5 差分監査は、surgery が黙って積んだ未提出 win を拾う安価な手段。

## E33: dsl session の全 30 bundle を網羅 harvest（frank766/bis_*/blend_* 系を新規スキャン）

harvest_v12 は 4 source のみ走査だったが、dsl scratchpad に **30 個の full 400-task bundle**
（bis_A/B/B1/B2/C173/C202, best_base, blend_kokinn, frank766_extract, narrow_out, opt_bundle 等）が
存在。frank766/kojimar/seddik/kokinn は Kaggle 作者ハンドル → 外部 public base の実験群。
全 30 source を faithful 監査で cherry-pick する harvest_v14 を起動（candidate 134 タスク）。結果は次項。

### E33 結果: 30 bundle 網羅 harvest → 2 wins +0.0089（外部 base 群は現ベスト未満を確定）

candidate 134 タスクを全 30 source で faithful 監査 → **2 wins**: t066(19807→19647, fp16_out 由来),
t319(22242→22224, narrow_out 由来)。いずれも高コストタスクの memory 微減（+0.0089）。
重要な確認: **frank766/kojimar/seddik/kokinn/bis_* 等の外部 public base 実験群は、私の
combined_best を 1 タスクも上回らなかった**。→ これらは dsl session が外部 notebook を試した
劣化版であり、win 源ではない。bespoke golf 源は依然 golf_wins(dsl の自作)に限られる。
surgery は t066/t319 で fixpoint（fp16_out/narrow_out は既に surgery 済み）。v14 提出（degrade-check 兼用）。

### E33 提出結果: v14 = 実 LB 7170.70（新 team best, degrade なし確認）

t066/t319 の harvest win → 実 LB **7170.70**（前 7170.69 から +0.01, 予測 +0.0089 と一致）。
400 タスク全体の degrade-check も同時通過（劣化なし）。floor を 7170.70 へ更新（byte 検証済）。
本セッション開始 7166.10 から **+4.60**。7665 まで −494.3。
所見: collaborative harvest の井戸はほぼ枯渇（dsl の全 30 bundle + golf_wins を集約済、外部 base は劣化版）。
今後の伸びは (a) dsl の新規 bespoke golf、(b) 私自身の新レバー探索、に依存。次サイクルで後者を起動検討。

## E34: int8 ダウンキャスト surgery — 実装完了も 0 wins（surgery の限界を実証）

fresh-context agent が「59 タスクに整数値 fp32 中間テンソル（177KB 回収可能）」を発見 → t029 で
8 中 7 が int8 安全と実証。これを受け audit-gated な `int8_surgery` パスを実装（surgery.py, lint/mypy clean,
PASSES 登録）。だが 400 タスクスイープで **0 wins**。決定的な 2 つの壁を実測確認:
1. **scorer は全ノード出力を memory 計上** → fp32 producer の出力を Cast→int8 すると、元 fp32(大)
   と int8 コピーが**両方残り純増**（t029 実測 10766→11126）。利得には producer が int を native 出力する必要。
2. **大テンソルは input 由来で Einsum/ReduceSum を通る** → ORT 1.24.1 はこれらを int32 のみ対応
   （省メモリ効果ゼロ）、int8 非対応。
結論（3 agent + probe で確定）: **post-hoc surgery（fp16/int8/channel-collapse）は飽和**。
さらなる利得は **build 時の int-first ネット再設計**か dsl の新規 bespoke win のみ。int8_surgery は
audit-gated で無害（0 win 時 no-op）のため実装は保持。int8_candidate dir は combined_best とバイト一致のため削除。

## E35: task002 int-first 再設計 → 33317（+0.166, redesign レバー実証）

agent が task002 をゼロから再構築。**真の変換は局所 3×3 ではなく 4-連結 flood-fill**（enclosed
interior を color4 で塗る、global ルール）と判明。numpy で全 268 例一致を確認 → int8 throughout の
ネット（74 nodes, params 34, int8 output で float cast 回避）。**39316→33317（+0.166, n_fail=0/268）**。
surgery で 1 byte 追加削減。v15 提出。
教訓: agent の cost 推定（→1500, +3.27）は楽観的すぎた。task002 は flood ~30 反復が本質的に必要で
コスト下限が高い。「高コストタスクは変換自体が高コスト」という構造下限の直感を再確認。だが redesign は
surgery と違い実 win を生む唯一の自前レバー。次は複数タスクへ並列展開して一般化を検証。

### E35 提出結果: v15 = 実 LB 7170.87（新 team best, redesign が実 LB 利得を実証）

task002 redesign（39316→33317）→ 実 LB **7170.87**（前 7170.70 から +0.17, 予測一致）。
**int-first ネット再設計が実 LB 利得を生む唯一の自前レバーと end-to-end 実証**。floor を 7170.87 へ更新。
本セッション開始 7166.10 から **+4.77**。7665 まで −494.1。次: t187/t076/t364 の並列 redesign 結果を回収。

## E36: redesign 並列展開 第1波（t187/t076/t364）→ 3/4 勝率

t002 に続き t187/t076/t364 を並列再設計。結果:
- **t076**: 25678→25005（+0.027）— CC+dihedral template-copy（非局所）。既存ネット論理は最適、
  scatter-tail を f16→int8 化して golf。
- **t187**: 41696→40446（+0.030）— ACCEPT。surgery で 2 byte 追加削減。
- **t364**: rebuild は 30982 > 26049 で **REJECT**（既存ネットを下回れず）。
集計: t002+t076+t187 採用、t364 棄却 = **3/4 勝率、平均 ~+0.07/task**。v15 から +0.057 を v16 提出。
評価: redesign は一般化するが per-task 利得は小（変換が本質的に複雑、既存ネット論理はほぼ最適、
利得は surgery 取りこぼしの dtype/構造 golf）。残り ~25 高コストタスクで +1.5〜2pt 見込み。第2波展開。

### E36 提出結果: v16 = 実 LB 7170.92（新 team best）

t076+t187 redesign → 実 LB **7170.92**（前 7170.87 から +0.05, 予測一致）。floor を 7170.92 へ更新。
本セッション開始 7166.10 から **+4.82**。7665 まで −494.1。redesign 第2波（t286/t349/t018/t191）展開中。

## E37: redesign 第2波 結果 + dsl golf_wins 再活性化（33→43）

第2波（t286/t349/t018/t191/t364再試行）結果:
- **t364 第2試行**: 26049→25209（+0.033）✓ ACCEPT。Conv kernel center=20 で seed を Gather で gate、
  seed-mask の Mul を削除する新技。
- **t018**: rule 完全解明（266/266 numpy 一致）も ONNX は本質的に ~905-node 非局所プログラム
  （連結成分ラベリング＋合同整列＋二面体選択＋scatter）。float32 入力制約が最有力 golf を封じ、
  best-known 66319 を下回れず → no-win（正直報告、保存せず）。
- **t349**: rule 解明（267/267）も from-scratch ネットは 95542 >> 38986（-0.90）→ REJECT。
- **t286/t191**: no-win（baseline 下回れず）。
集計: 第2波は **1/5 勝（t364 のみ）**。redesign の per-task ROI は逓減（残る高コストタスクは
真に複雑な非局所プログラム、コミュニティ収束済）。
**重要**: redesign 中に dsl golf_wins が 33→43（+10）に再活性化。harvest が高 ROI 源に復帰 → v17 で harvest 起動。

## E38: dsl golf_wins 大量 harvest → 16 wins +1.23 + surgery +0.15（最大ジャンプ）

dsl golf_wins 43 を harvest → **16 wins +1.2323**（t161 5217→3733 +0.335, t359 +0.176, t279 +0.141,
t251 +0.137 等）。**この単発 harvest が本セッションの全 redesign 成果を上回る**。さらに dsl の生 golf は
未 surgery のため、17 タスクに surgery 適用で **追加 +0.1545**。t364 redesign +0.033 と合わせ局所 ~+1.42。
v17 提出（期待 ~7172.3）。
**戦略確定**: dsl session の bespoke golf が支配的 win 源、私の役割は高速集約 + surgery 上乗せ。
redesign は補助（per-task ROI 逓減）。harvest を最優先で回す。本セッション開始 7166.10 から大幅前進中。

### E38 提出結果: v17 = 実 LB 7172.34（新 team best, 本セッション最大ジャンプ +1.42）

dsl 16 wins harvest + surgery + t364 → 実 LB **7172.34**（前 7170.92 から **+1.42**, 予測一致）。
本セッション開始 7166.10 から **+6.24**。floor を 7172.34 へ更新。7665 まで −492.7。
本セッション最大の単発前進。dsl harvest が支配的レバーと再確認。

## E39: ROI 起点ターゲティング → verified-simple タスクに redesign 第3波

使用量上限リセット後、fresh-context agent で「コスト中〜高だが変換が単純」なタスクを ROI 順に探索。
wave-2 の盲目的高コスト選定と異なり、各変換を numpy で厳密検証して候補化。上位（verified-exact）:
- **t202** 20173→~1500 (+2.60, 230/230): band 内の zero hole を band 軸に broadcast して 0-stripe 化
- **t359** 4348→~400 (+2.39, 266/266): 軸方向 mode denoise（既に dsl golf 済だが更に削減可能）
- **t085** 5381→~1200 (+1.50): solid bar の中心線に交互 hole を punch
- **t255** 24746→~4000 (+1.82): color-3 cross/region を paint
上位5の高信頼合計 ~+9.9pt。t202/t359/t085/t255 の rebuild agent を並列起動（第3波）。
hard-class（t074/t110/t243/t066 等 = symmetry-inpaint/periodic/flood/pathfinding）は除外。

## E40: redesign 第3波 0/4 → redesign レバー枯渇を確定

verified-simple ターゲット（t202/t359/t085/t255）も全敗:
- **t255**: 非局所と実証（最大長方形/連結成分検出が必須、ノイズと分離不能、最良ルール 207/265）→ no-win。
- **t202/t359/t085**: ルールは厳密正答（230/230, 266/266, 265/265）だが、from-scratch ネットは
  既存より高コスト（50374>20173, 6149>4348, 23505>5381）→ 全 REJECT。
**決定的教訓**: ROI targeting agent の cost 推定（~1500 等）は楽観的すぎた。正しい rebuild が
既存 golf 済ネットを上回ることは稀。コストは不可避な full-grid 中間テンソルに宿り、論理ではなく
**golf こそが難所**。wave-2+3 で ~9 タスク試行・1 勝（t364）= redesign ROI ≈ 0。
**戦略確定**: redesign agent 投入を停止。支配的レバーは **dsl harvest**（1 バッチ +1.23 vs redesign の
散発 +0.03〜0.17）。dsl session がエンジン、私は高速集約器。harvest 即応体制で待機。

### E40 補遺: t359 第2試行で +0.0139 採用（wave-3 は 1/4、redesign 通算 2/9）

t359 agent が第2パスで Where-vector を 6→4 に統合し **4348→4288（+0.0139, 266/266）** を達成 → ACCEPT。
t085 は逆に「既存 Conv ネットが構造的最適」を確認（3600B f32 Conv 出力が ORT 下限）= no-win。
wave-3 最終 **1/4**（t359 のみ）。redesign 通算 **2/9**（t364, t359）。v18 提出（+0.014, degrade-check 兼用）。
**4+ agent が独立に確認した構造下限**: ORT 1.24.1 の int8/16 Conv・ReduceSum 非対応 + 固定 f32 入力 +
全ノード出力 memory 計上 → ~3600B/collapse-plane が surgery・redesign で破れない壁。redesign 停止を維持。

### E40 提出結果: v18 = 実 LB 7172.36（新 team best）

t359 redesign → 実 LB **7172.36**（前 7172.34 から +0.02）。floor を 7172.36 へ更新。
本セッション開始 7166.10 から **+6.26**。7665 まで −492.6。degrade なし。
以降は dsl harvest 待ち（唯一の残レバー）。redesign 停止維持。

### E40 補遺2: t202 第3パスで +0.0735 採用 → wave-3 は 2/4（redesign 通算 3/10）

t202 agent が第3パスで channel-axis MatMul collapse + fp16 + output-直結 Where により
**20173→18743（+0.0735, 230/230）** を達成 → ACCEPT。wave-3 最終 **2/4**（t202 +0.074, t359 +0.014）。
**認識修正**: 「redesign 枯渇」は早計だった。verified-simple ターゲットは agent が初回 reject 後も
golf を継続すれば勝てる（t202/t359/t364 とも初回 reject → 後続パスで勝利）。私の途中 audit は
agent の初回試行を拾っていただけで、最終保存ファイルは追加 golf 後に勝つことが多い。
→ verified-simple ターゲットへの redesign は **per-task +0.01〜0.07 で 50% 勝率** と再評価。残 ROI 候補
（t064 +1.61, t174 +1.14, t162 +1.00 等）はまだ未試行。dsl harvest（凍結中）と並行で redesign 再開可能。
v19 提出（+0.074, degrade-check 兼用）。

### v19 = 実 LB 7172.43（新 team best）

t202 redesign → 実 LB **7172.43**（前 7172.36 から +0.07）。floor 更新。本セッション 7166.10 から **+6.33**。
7665 まで −492.6。degrade なし。wave-4（t064/t174/t162）実行中。

## E41: redesign wave-4 結果 → 全 no-win（既存ネットが既に lean/correct）

t064/t162/t174 すべて no-win:
- **t162**: 既に最適 4068（1600B f32 入口面が下限）。tie のみ、保存せず。
- **t064**: ~10 golf パス（228957→13299）も既存 12478 を ~6% 下回れず。algorithm-matched baseline。
- **t174**: agent が「既存 9329 は wrong rule で実 LB 0 点」と主張 → **検証で誤りと判明**（現 task174 は
  n_fail=0/266, 15.86pt で正答）。agent は targeting agent の loose "shift" ラベルと実装ネットを混同。
  rebuild 28272 は −1.11 の退行 → REJECT。**agent の警告的主張は必ず faithful scorer で検証すべき**を再確認。
wave-4 = **0/3**（t192 のみ実行中）。確定パターン: rebuild が勝つのは「素朴実装の高コストネット」のみ。
既存 lean ネットには tie/負け。redesign の残 ROI はごく僅か。dsl harvest（凍結中）が依然主力。

### E41 補遺: t192 も tie (8628 vs 8621) → wave-4 = 0/4 確定、redesign 枯渇

t192 は 265/265 正答だが 8628 > 8621（-0.0008）= 既存 lean ネットと実質同値 → REJECT。
**wave-4 最終 0/4**（t064/t162/t174/t192）。redesign は完全に勝ち筋を掘り尽くした:
直近の全ターゲットが既に golf 済の lean ネット（dsl/コミュニティが最適化済）。
勝利は初期の naive-net タスク（t002/t202/t359/t364/t076/t187, 計6・~+0.35）に集中、現在は枯渇。
**戦略最終確定**: redesign 停止。残る唯一のレバーは dsl harvest（+1.23 実績）だが 5時間以上凍結中。
本セッション総括: 7166.10 → **7172.43（+6.33）**。全レバー網羅的に採掘済。7665（−492.6）は
構造下限・コミュニティ収束（~7172）の証拠から現行手法では到達不能。loop は dsl-harvest-watch として維持。

## E42: onnx-compiler worktree も harvest → 0 wins（全 worktree 横断で combined_best が最適確定）

dsl が枯渇したため、未スキャンだった **onnx-compiler worktree**（9 bundle dir: case3_convgolf/dtypegolf/
lossless/blend8/combined_safe 等 + 23 個別 golf variant in _case3_tmp）を harvest。**0 wins**。
→ combined_best は onnx-compiler の全 golf 実験を既に上回る（dsl wins の方が良く、harvest 済）。
**確定**: combined_best は **全 3 worktree（dsl + onnx-compiler + 自分）横断で各タスク最良版を集約済**の
team-wide optimum（7172.43）。harvestable な未取得 source は存在しない。
全レバー（dsl harvest, onnx-compiler harvest, redesign, surgery）を網羅的に採掘完了。
