# Handgolf 対象タスク worklist（faithful 採点ベース）

公開最高 bundle（7166.66 / 別セッション handgolf v1 で 7166.79）の cost 削減候補を、
**faithful scorer（onnx 1.20.0/ort 1.24.1, offset +0.11）で実測した cost** と
**ルールの単純さ指標**で整理したもの。古い版は高 cost を強く見ていたが、score は
対数なので、今後の handgolf ROI は `old_cost / new_cost` の期待相対削減率で見る。
生データ: `handgolf-targets.json`。

## 読み方

- `cost` = faithful 実測（params+memory）。`pts` = 25-ln(cost)。
- cost 削減の点差は `ln(old_cost / new_cost)`。同じ 10% 削減なら cost の大小に関係なく
  同じ点差で、同じ固定 cost 削減なら小さい task の方が点効率は高い。
- `change_frac` = 変化セル率。**低いほど局所的＝ルールが単純で安価化しやすい**。
- `bg_only` = 変化が背景(0)セルのみか。`sameshape` = 入出力同形状か。
- `ncolors` = 入力の色数（少ないほど単純）。

## 優先度（改定）

1. `n_fail=0` を絶対条件にする。不正解・unscorable は 0 点。
2. 絶対 cost ではなく、期待 `old_cost / new_cost` が大きい候補を優先する。
3. 低 cost 単純 task は既に最小化済みが多く、高 cost hard task も near-optimal が多い。
   狙い目は「中〜高 cost で、構造的に過剰な中間テンソル・dtype 境界・branch・table を
   丸ごと削れる task」。

## 旧候補（高コスト × 低 change_frac）

以下は「高 cost かつ見た目が単純」という旧 worklist。現在は最優先リストではなく、
相対削減率の大きい具体的な削減仮説がある場合だけ着手する。

| task | cost | pts | change_frac | colors | 期待 ROI |
|---|---|---|---|---|---|
| **118** | 42418 | 14.34 | 0.026 | 3 | 高（2.6% のみ変化, 5→8 をマーカー間で）|
| **002** | 40116 | 14.40 | 0.063 | 2 | 高（背景に 4 を局所充填）|
| **191** | 35425 | 14.52 | 0.058 | 3 | 高（4-クラスタ bbox をテンプレ充填）|
| 367 | 26903 | 14.80 | 0.066 | 2 | 中 |
| 219 | 26217 | 14.83 | 0.093 | 2 | 中 |
| 173 | 23271 | 14.95 | 0.035 | 8 | 中 |
| 367/76/285 | ~26k | ~14.85 | ~0.06 | — | 中 |

例: t118 を cost 42418→~500 に安価化できれば pts 14.34→18.79 = **+4.45**。
ただし後続実験で、高 cost 非局所 task は bundle が bbox-crop/uint8/反復数まで
near-optimal に golf 済みの例が多いと確認済み。大 cost だけを理由に選ばない。

## 検証手順（必須）

handgolf 候補は必ず faithful scorer で提出前検証する（`docs/experiment/faithful-scorer.md`）:
```bash
cd backend && uv run --python 3.12 \
  --with "onnxruntime==1.24.1" --with "onnx==1.20.0" --with "numpy<2.1" \
  python -c "import sys; sys.path.insert(0,'src'); from evaluate.scorer import audit_one; \
    print(audit_one('cand.onnx', __import__('json').load(open('../data/lake/neurogolf-2026/task118.json'))))"
```
status=='ok' かつ n_fail==0 かつ cost<bundle のときのみ採用。

## 確定したタスク・ルール（reverse-engineering 済み）

faithful 検証で**全 example 一致を確認した厳密ルール**。handgolf 時はこれを最小 ONNX 化する:

| task | ルール（numpy で全例一致を確認）| 注意点 |
|---|---|---|
| **002** | **flood-fill**: 背景(0)セルのうち外周から 4 連結で到達不能（=囲まれた内側）を色 4 で塗る。268/268 一致 | `floodfill.py` は 222/268 正答（46 fail のバグ）+ cost 314819（bundle 40116 より高い）。要: (1) 正答バグ修正 (2) bbox-crop+bool+大カーネル MaxPool で安価化。flood-fill は ~grid 径回の反復が必要で、bundle 既存実装に勝つのは非自明 |
| 191 | 色 4 クラスタの bbox をテンプレに従い色 1 で塗る（非局所・テンプレ照合）| クラスタ検出が必要で安価化困難 |
| 118 | 色 2 マーカー間の線分上の色 5 を色 8 に（非局所・マーカー対検出）| 同上 |

→ 教訓: change_frac が低くても**ルールが非局所（flood-fill/クラスタ/テンプレ照合）なら最小 ONNX も高コスト**。
bundle はこれらを既に効率実装済みで、勝つには bbox-crop+uint8 等の更なる golf が要る。
最有望は「真に局所（k×k 近傍で決まる）かつ bundle が冗長」なタスクだが、solver-search 0 wins より
そうしたタスクは bundle が既に最小化済み。

## E14: t002 を実際に handgolf → bundle は既に bbox-crop+uint8 で golf 済みと判明

t002 の flood-fill ルール（囲まれた背景→色4, 268/268）を最小 ONNX 化し faithful 検証:
- 自前 flood-fill（4連結 cross-dilation, k=3×20 steps）: float32 → cost 353715 / uint8 → **133216**（正答）。
- **bundle t002 = cost 40116**。グラフ調査: 39 MaxPool + uint8 中間 + **2 Slice/2 Pad = bbox-crop 済み**。
  つまり bundle は「入力を内容 bbox に crop → 小空間で uint8 flood-fill → Pad 復元」を既に実装。

**含意（決定的・実装レベルで確証）**: bundle の高コストソルバは**既に bbox-crop + uint8 + 最小反復で
golf 済み**。ルールを再現しても全 30×30 で処理すると bundle に負ける。**bundle を上回るには
専門家の手 golf（bbox-crop+uint8）を更に上回る必要**があり、ルール特定だけでは不十分。

→ faithful scorer は検証を可能にしたが、高コストタスクは bundle が golf 最前線にあり、
ルール再現＋標準的 golf では並ぶのが精一杯。**正味の上積みは bundle がまだ crop していない
タスク or より賢い構成を見つけた場合に限られる**（稀）。並行 handgolf セッションの +0.14 が
その実例（数タスクで僅かに上回った）。

## E15: bbox-crop も適用したが bundle に届かず（実装で確証）

t002 flood-fill を更に golf:
- 全30×30 uint8: 133216 → 静的 20×20 crop + uint8: **111230**（正答, n_fail=0）。
- **bundle 40116 には依然 2.8× 届かない**。bundle は**例ごとの動的 content bbox crop**
  （静的 20×20 より遥かに tight）+ より少ない反復で実装しており、標準的手法（静的 crop+uint8）
  では追いつけない。

**最終確証（実装レベル）**: bundle の高コストソルバは専門家が動的 bbox-crop + uint8 + 最適反復で
golf 済み。ルール再現 + 標準 golf（uint8/静的 crop）では cost が bundle を**上回る**（負ける）。
faithful scorer で検証可能になっても、**個々の高コストタスクで bundle を下回るには専門家の
手 golf を更に上回る必要**があり、これは 1 タスクあたり数時間〜の精密 ONNX 設計を ~80 回繰り返す
作業（上位チームの数週間・多人数規模）。本セッションの自動反復では到達不能。

到達可能だった上限 = **7166.66**（私の提出）/ チーム **7166.79**（並行 handgolf で数タスク僅増）。
faithful scorer・worklist・確定ルール・flood-fill 実装は次段の handgolf の土台として全てコミット済み。

## E16: 7665 到達の定量分析 — 全タスク golf が必要（並行3セッションで確証）

faithful cost 分布（base 7166.6）と必要 gain の精密計算:

| cost 帯 | タスク数 |
|---|---|
| 0-100 | 32 | 100-500 | 81 | 500-1000 | 58 |
| 1000-2000 | 65 | 2000-5000 | 81 | 5000+ | 83 |

- **7665 には avg +1.25/task（全 400）が必要**。高コスト 83 個だけを cost 1000 に golf しても +213→7380 で**届かない**。
- 全タスク（cost>300）を cost 300 に golf できれば +665→**7831**（到達可）。**gain の主役は中コスト 146 タスク（1000-5000）**で、各 2-4× の cost 削減が要る。
- だが中コストタスクの中間テンソルは**既に大半が uint8/int**（fp32-heavy は 60 中 5 のみ）。更なる削減は**タスク毎のアルゴリズム再構成/bbox-crop**が必要で、bundle 作者が概ね実施済み。

**並行 3 セッションの独立確証**:
- case2-dsl-primitives: 本格投資で **2 タスクのみ base 超え**（t002 40116→39316, t187 46809→41699 = 計 +0.13）。
- case2-onnx-compiler: **43 タスク win-sweep で WINS=0**、「franksunp は局所最適」と実証。
- case2-program-search（本セッション）: solver-search 0 wins, t002 再現も bundle 未達。

**確定**: 観測 win レート ~+0.06-0.13/task。7665（+498）には ~230 タスクを各 2-4× golf する必要があり、
bundle はほぼ局所最適。これは上位チームの数週間・多人数 handgolf に相当し、**自動反復では構造的に到達不能**。
到達上限: 7166.79（チーム手 golf 2 タスク）/ 7166.66（本セッション提出）。

## E17: 並行 win の正体 = 冗長 Max ノード 2 個削除（+800 cost = +0.02）

dsl セッションの t002 win（40116→39316）を bundle と diff:
- **Max ノード 21→19（2 個削除）** = uint8 [1,1,20,20] 中間 2 個削除 = 800 bytes。
- flood-fill の単調増加 reach 鎖で `Max(reach_k, reach_0)` 型の冗長 Max を除去したもの。
- **新ソルバではなく bundle グラフの task 固有 micro 最適化**（= 意味保存 surgery の一種）。

自前 surgery 全パスは bundle t002 で**無変化**（この task 固有冗長を検出できない）。
汎用化は単調性証明が要り危険、かつ得 800 cost（+0.02）。flood-fill 全 task に効いても ~+0.4。

**最終確定（全レベル検証済み）**: 並行 win は task 固有の冗長ノード削除で **+0.02-0.11/task**。
E16 の通り 7665 には ~230 task の 2-4× golf が必要だが、観測 win レートは桁違いに小さい。
bundle は専門家により near-optimal に golf 済みで、各 task の上積みは microscopic かつ手作業。
**7665 は自動反復・本セッションでは到達不能。到達上限 7166.79（team）/ 7166.66（自分）。**

全成果（faithful scorer offset+0.11 / worklist / 定量分析 / t002・flood-fill 実装 / win 解析）は
コミット済みで、継続的分散 handgolf の土台。
