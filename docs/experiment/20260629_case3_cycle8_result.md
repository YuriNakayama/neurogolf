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

## 次の一手

- 新規高 LB バンドルの公開を監視し、出現次第 cherry-pick→faithful→submit を自動実行。
- それ以外に退行ゼロで前進できる利用可能レバーは現存しない（無改善 submit はしない）。
