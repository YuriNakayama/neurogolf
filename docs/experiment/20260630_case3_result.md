# 20260630 case3 result — 新規バンドル harvest、live-best floor 同期 + lucifer t233 で 7180.58（ACCEPT, +0.03）

## 実 LB / 採否

- **ACCEPT**。**Public Score 7180.58**（前 best 7180.55 から **+0.03**）。新 team best。
- pick = **lucifer19/neurogolf-agi-circuit-forge** の t233（58327→56707, −1620, +0.028pts）。
  faithful n_fail=0/266（train/test/arc-gen 全通過）。graded 公開バンドルなので hidden 安全。

## 重要な発見: ローカル floor が live-best より stale だった

- ローカル main HEAD = cycle5（`d2c08c9`, DVC `4f81c7b3`, **7176.49**）。
- だが infra ループは別ブランチ `auto/20260629-102313`（cycle6-12）で **7180.55** に到達済み
  （DVC `def34b37`）。cycle13 ブランチ（`auto/20260629-cycle13`, cycle14 まで）は cycle5 から
  再 fork し 7176.49 のまま per-task 設計を継続（0 win）。**main には未マージ**。
- 教訓: **採否比較は必ず live-best floor に対して行う**。stale floor 上の harvest は誤った gain を出す。

## サイクル経過（2 提出）

| 提出 | floor | 内容 | 実 LB | 採否 |
|---|---|---|---|---|
| 1a | stale 7176.49 | harish+lucifer 70 wins 再構築（cost-only +8.10）| **7180.53** | REJECT（live-best 7180.55 に −0.02。cycle10/11 の micro-win を欠く）|
| 1b | live-best 7180.55 | DVC で live-best 同期 + lucifer t233 のみ | **7180.58** | **ACCEPT（+0.03）** |

- 提出 1a の失敗で「stale floor harvest は live-best に劣りうる」を実証 → live-best DVC を pull
  （`dvc get . data/output/onnx --rev auto/20260629-102313`）して正しい floor を復元。
- live-best に対する cross-bundle cherry-pick（{harish,losist,lucifer} vs live-best）は **t233 1 件のみ** win。
  cycle6-11 が harish 全部 + lucifer t002/t191 を既に取込み済みのため。lucifer t233 のみ未取込みだった。

## 検証内容

| 項目 | 結果 |
|---|---|
| 新規公開バンドル | harishk2209/7178-23, losist/717421, lucifer/circuit-forge, 他を取得 |
| live-best 同期 | DVC `def34b37`（7180.55, cycle6-11 harvest 込み）を /tmp に pull |
| cross-bundle cherry-pick vs live-best | **lucifer t233 のみ** win（save 1620）。他は live-best が既に同等以上 |
| t233 faithful | n_fail=0/266, cost 56707, +0.028pts |
| 実 LB | **7180.58**（予測 7180.55+0.028 と一致）|

## 結論 / 次の一手

- 新 floor = **7180.58**（live-best 7180.55 + lucifer t233）を data/output/onnx へ同期、dvc add/push。
- 公開バンドル harvest はほぼ飽和（live-best に対し 1 task/+0.03 のみ）。harish/lucifer/losist は
  cycle6-11 floor とほぼ同包絡線。次の主レバーは **per-task 真アルゴリズム手 golf**（cycle13-14 で
  「整数コード符号化が公開 net の安さの核心」と特定済み）。
- 7950 まで −769.4。harvest 残量は薄く、手 golf 主導へ移行が必要。
