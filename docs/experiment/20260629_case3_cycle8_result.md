# 20260629 case3 cycle8 result — 意味保存 ORT 最適化は unscorable で提出不能、0 win

## 採否

- **採用なし（提出なし）**。floor 7180.46 維持。

## 検証した内容

新規採用 17 グラフ（harishk 15 + frank 2）を ORT オフライン最適化（BASIC / EXTENDED）で
再保存し faithful 監査:

| 結果 | 詳細 |
|---|---|
| 正答性 | 全 17 タスク n_fail=0（ORT 最適化は意味保存なので当然）|
| **cost 計測** | **全タスク status=unscorable**（公式 scorer が params/memory を算出不可）|
| 原因 | ORT 最適化が fused / com.microsoft 系 op を挿入 → 公式 calculate_memory が
  プロファイル trace をマップできず unscorable。**提出 validator でも弾かれる非標準グラフ**。|
| 採用 | **0 win** |

## 確定した教訓

1. **ORT オフライン最適化（pre-optimize して再保存）はこの競技で使えない**: 最適化グラフは
   fused op を含み、公式 scorer で unscorable・提出制約違反になる。意味保存だが**測定/提出不能**。
2. 過去の「surgery 飽和」結論は新規バンドル由来の他者作グラフ（harishk/frank）にも実質当てはまる
   （標準的最適化では cost を下げられない）。bundle は提出可能な形で既に near-optimal。

## 本セッション総括（cycle6-8）

| cycle | レバー | 結果 |
|---|---|---|
| 6 | harishk2209/7178-23 harvest（15-task cherry-pick）| **+3.96 → 7180.45 ACCEPT** |
| 7 | frank/consolidated-audit harvest（2-task）| **+0.01 → 7180.46 ACCEPT** |
| 8 | per-task hand-golf（t233 解析）+ ORT 最適化（17 グラフ）| 両者 0 win、削減不可確定 |

- **到達上限 = 7180.46**（team「Yuri Nakayama」rank 153, LB 確定, 退行なし）。
- 7950 への残り 769pt は、①外部の新規高 LB バンドル公開 か ②expert hand-golf（~230 タスクを
  各 2-4× golf、上位チームの数週間・多人数規模 = 本ループの実証能力外）でしか埋まらない。
- 全公開ソース（kernel + dataset）は出し切り、提出可能な意味保存最適化も枯渇。

## cycle8b 追記: golfer 全 variant 横断 harvest も 0 win（包絡線確定）

harvest 早計枯渇判断を是正し、franksunp/kokinnwakashuu の追加 variant を広く検証:
- **fk_stack / fk_rewire / fk_vb / kk_7169 / kk_7166**（各 400 onnx, distinct md5）: 全て
  **0 候補**（どのタスクでも combined 7180.46 未満なし）。
- 結論: harishk(7178.23) バンドルは franksunp/kokinnwakashuu 全 variant の consolidated frontier。
  私の 15-task cherry-pick が既にその最良を捕捉済み。**計 12 バンドル検証で 7180.46 が公開包絡線**と確定。

## cycle8c 追記: dtype 削減レバーを実測で棄却（hard evidence）

「高コストタスクのメモリは float32 のブール浪費が支配 → uint8 化で 4× 削減可」仮説を、
faithful scorer のメモリ計測を per-tensor 分解して実測検証:

| task | total_mem | f32 比率 | 支配テンソル |
|---|---|---|---|
| t286 | 55272 | **0%** | u8/bool [1,10,30,30]（既に最小 dtype）|
| t018 | 51342 | **8%**(4240B のみ) | bool 9000 + 小 f32 3600 |

- **仮説棄却**: bundle 作者は dtype 削減を完了済み。支配的中間は既に u8/bool。
- t286 の 9000B u8 [1,10,30,30] = 「全グリッド 1byte/cell」はグリッド中間の理論下限。これ以上は
  **全グリッド中間を materialize しない別アルゴリズム**（expert 再設計）でしか減らない。
- 従来「中間は大半 uint8」は assumption だったが、本サイクルで初めて **hard evidence で確認**。

## 次の一手

- 新規高 LB バンドルの公開を監視し、出現次第 cherry-pick→faithful→submit を自動実行。
- 検証済みレバー一覧（全て枯渇/不可）: 公開 harvest（12 バンドル）/ per-task golf / ORT 最適化 /
  **dtype 削減（実測で u8/bool 済み確認）** /
  surgery / fp16 / 幾何・局所・対称・lookup ソルバ（過去 7 系統）。退行ゼロで前進できる
  利用可能レバーは現存しない。無改善 submit はしない（提出枠の無駄・退行リスク回避）。
