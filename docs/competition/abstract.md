# The 2026 NeuroGolf Championship — コンペ概要

> ⚠️ **出典について**: 本書は Web 検索（Kaggle / competehub / IJCAI / Kaggle discussion）で取得した情報を基に作成しています。Kaggle 公式ページは JS レンダリングのため一次取得できず、**正式な数値・日付・規約は必ず [公式ページ](https://www.kaggle.com/competitions/neurogolf-2026) で確認**してください。特にタイムラインは出典間で差異があります（後述）。

## 1. 概要

- **正式名**: The 2026 NeuroGolf Championship
- **主催**: Neurosynthetic Research Institute（Kaggle ホスト）
- **位置づけ**: IJCAI-ECAI 2026 Competitions Track の一環。上位提出は IJCAI-ECAI 2026（ドイツ・ブレーメン、2026-08-15〜21）の特別セッションで発表に招待される可能性。
- **URL**: <https://www.kaggle.com/competitions/neurogolf-2026>
- **テーマ ("neural golf")**: ARC-AGI のグリッド変換タスクを、**できる限り小さい ONNX ニューラルネットで「厳密に」解く**。各ネットは「パズルの仕組みを記述する合成プログラム（synthesized program）」とみなされる。正答を保ちつつパラメータ数・計算量・メモリを最小化する＝スコアを縮める "golf" 競技。

> 動機（公式説明より）: "Today's AI systems perform well on familiar tasks but often struggle to generalize to new ones." — 汎化と最小構成を問う。

## 2. タスク定義

- **ベースデータ**: ARC-AGI-1 の公開トレーニングサブセット。
- **タスク数**: `task001` 〜 `task400`（最大 400 タスク）。**1 タスク = 1 つの ARC グリッド変換ルール**。
- 各タスクは複数の example サブセットを持つ:
  - `train` — ARC-AGI のトレーニング例ペア
  - `test` — ARC-AGI のテスト例ペア
  - `arc-gen` — ARC-GEN-100K 由来の追加例ペア
- **正答（correct）の定義**: あるタスクの **train / test / arc-gen の全ペア**について、入力グリッドから期待される出力グリッドを **全セル完全一致**で構築できること。1 セルでも不一致なら不正解。
- 機能的正答性は次の 3 つで検証される:
  1. オリジナル ARC-AGI ベンチマーク
  2. ARC-GEN-100K データセット
  3. 小規模な**非公開ベンチマークスイート**（private benchmark）

## 3. ONNX モデル I/O 仕様

タスクごとに 1 つの ONNX ネットワークを作り、そのネットが「入力グリッド → 出力グリッド」の変換を表す。

### 入力

- グリッドを次の形状のテンソルに変換: **`[BATCH=1, CHANNELS=10, HEIGHT=30, WIDTH=30]`**
- 各ピクセルの色を **one-hot チャネル**（10 色 → 10 チャネル）で表現。
- 元グリッドの枠外にある「clear（空）」ピクセルは **zero-hot**（全チャネル 0）。
- グリッドサイズは **1×1 〜 30×30** の範囲。

### 出力

- 各セルについて、**正しい色チャネルに `1`、それ以外のチャネルに `0`**。
- そのセルが画像の枠外なら**全チャネル `0`**。
- 上記を全セルで満たし、期待出力と完全一致した場合のみ correct。

### 形状制約

- **全テンソル・全パラメータは静的形状（statically-defined shapes）必須**。動的形状は不可。

## 4. スコアリング

各タスクについて、以下のスコアを得る:

```
score = max(1, 25 - ln(cost))
```

- **`cost`** = そのネットワーク実行に要する
  - 総パラメータ数（total number of parameters）
  - 総メモリフットプリント（total memory footprint）
  - 総 MAC 数（total multiply-accumulate operations）

  の**合計**。
- `cost` が小さいほど `25 - ln(cost)` が大きくなり、高得点。下限は 1。
- **前提**: そのタスクを「厳密に解けている（correct）」ネットのみが得点対象。正答していないネットはスコアにならない（＝まず厳密に解き、その上で cost を削るのが戦略）。

## 5. 提出形式

- 提出物は **`submission.zip`** 1 つ。
- zip 内に **タスクごとに最大 1 つの ONNX ファイル**を含める。
- ファイル名は **`task001.onnx`, `task002.onnx`, …, `task400.onnx`**。
- 解けるタスクだけ含めればよい（全 400 を埋める必要はない）。

## 6. 制約

| 制約 | 内容 |
|---|---|
| ファイルサイズ | 各 ONNX ファイルは **最大 1.44 MB** |
| 静的形状 | 全テンソル・パラメータは静的形状であること |
| 禁止 ONNX 演算 | **`Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`** |

> 禁止演算は「制御フロー・データ依存ループ・サブグラフ呼び出し」を排し、純粋な静的テンソル演算グラフに限定する意図と解釈できる。

## 7. データセット

| データ | 用途 |
|---|---|
| ARC-AGI-1 公開トレーニングサブセット | タスク定義の基盤（train / test 例ペア） |
| ARC-GEN-100K | 各タスクの追加検証例（`arc-gen` サブセット）。Google の手続き的 ARC ベンチ生成器 [google/arc-gen](https://github.com/google/arc-gen) 由来 |
| 非公開ベンチスイート | 最終的な機能的正答性の検証（汎化チェック） |

## 8. タイムライン & 賞金

> ⚠️ **タイムラインは出典間で差異あり。公式ページで要確認。**

| 項目 | 値（出典差異あり） |
|---|---|
| Start Date | 2026-04-15（Kaggle/competehub 構造化ページ） |
| Entry / Team Merger Deadline | 2026-07-08（同上） |
| Final Submission Deadline | 2026-07-15（同上） |
| 別表記（不確実） | 一部スニペット（LinkedIn 由来）に「2026-06-09 GMT+8」 |
| 賞金プール | **$50,000**（+ top student team / longest leader への追加賞） |

（参考: 本書作成時点の日付は 2026-06-25。上記が正しければコンペは開催期間中。）

## 9. 出典（Sources）

- [The 2026 NeuroGolf Championship | Kaggle](https://www.kaggle.com/competitions/neurogolf-2026)
- [The 2026 NeuroGolf Championship - CompeteHub](https://www.competehub.dev/en/competitions/kaggleneurogolf-2026)
- [Competitions – IJCAI 2026](https://2026.ijcai.org/competitions/)
- [Kaggle discussion: Issues in onnx-tool](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827)
- [Kaggle discussion 697079](https://www.kaggle.com/competitions/neurogolf-2026/discussion/697079)
- [Kaggle on X — NeuroGolf 告知](https://x.com/kaggle/status/2044494732504522780)
- [google/ARC-GEN（ARC-GEN-100K 生成器）](https://github.com/google/arc-gen)

> 本書の数値・日付・規約は二次情報に基づく暫定版です。実装・提出の前に Kaggle 公式の Overview / Rules / Data ページで最新の正規仕様を確認してください。実装 TODO は [`../develop/MIGRATION.md`](../develop/MIGRATION.md) を参照。
