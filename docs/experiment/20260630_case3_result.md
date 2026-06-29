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

## cycle2 追記: 残り全公開バンドル harvest → 実質 0 win（harvest 飽和確定）

新 floor(7180.58, DVC `713fc67a`) に対し未取得バンドルを総当たり:
- **kojimar/audited-neurogolf-onnx-overrides（283 votes!）**: 0 win。
- **mirzayasirabdullah07/top-score（94 votes）**: 0 win。
- **losist/717421**: 0 win。
- **boristown v51 最新 / franksunp-consolidated**: frank t054（27102→27094, save 8 = +0.0003pts）の
  **LB 分解能未満の 1 件のみ**。提出枠を使う価値なしと判断しスキップ。

**結論**: 公開バンドル harvest は **完全飽和**。現 floor 7180.58 が公開 net の包絡線。
高 vote バンドル（kojimar 283）すら全タスクで floor 以上。次の主レバーは
**per-task 真アルゴリズム手 golf**（cycle13-14 で「整数コード符号化が公開 net の安さの核心」と特定）。

## 高コスト現状（手 golf 標的, 新 floor）

| task | cost | mem | 性質 |
|---|---|---|---|
| t233 | 56707 | 56151 | memory 支配, 非局所 algorithmic net |
| t286 | 47013 | 46272 | 同上 |
| t018 | 45211 | 43124 | 同上 |
| t187/t366/t158/t133/t209 | 32-37k | mem 支配 | 全て非局所、floor が near-optimal |

いずれも memory 支配の手作り net で、過去 8+ session が floor 近最適と確認済み。
per-task 手 golf は expert-scale・低確率だが、harvest 枯渇のため次はこちらに集中する。

## 次の一手

- 新 floor = **7180.58** を data/output/onnx へ同期、dvc add/push 済み（`713fc67a`）。
- harvest は飽和。次サイクルは **整数コード符号化**を使った per-task net 再設計を 1 タスク試す
  （cycle13-14 の知見: 公開 net の安さの核心は色を整数コードで符号化し channel 数を圧縮する点）。
- 7950 まで −769.4。

## cycle3 追記: per-task 手 golf 着手（t367 flood-fill 解析）→ 規則未確定、defer

harvest 飽和を受け hand-golf へ移行。高コスト同形状・少色タスクから **t367**（cost 25127,
2色, 7%変化）を選定。
- 変換は「5 で囲まれた閉領域の内部 0-cell を 4 で埋める」per-glyph interior-fill。
- 試した規則と arc-gen 一致率: enclosed(4/8-conn) 108/266, pad-flood 108/266,
  largest-0-comp=outside 37/266。**いずれも未一致**。失敗は主にグリッド端に接する glyph の
  内部判定（端を壁として扱うか開放とするかが glyph 形状依存）。
- 結論: t367 floor net は per-shape interior-fill の真アルゴリズムを実装済み（25127 で正答）。
  正確な規則は glyph 分解を要し **1 サイクルでは確定困難**。floor 維持（退行ゼロ）。
- **defer**: t367 は複数サイクル規模の reverse-engineering。次は floor net が単純規則に対し
  過剰構築なタスク（recolor/translate 系）を優先し、確実な golf を狙う。

## cycle3 結論

提出なし（floor 7180.58 維持）。hand-golf 第一手の t367 は規則未確定で defer。
教訓: 高コストタスクの多くは flood-fill / shape-fill の真アルゴリズムで floor が既に最小付近。
hand-golf の EV を上げるには「単純規則 × 過剰 net」のタスク選定が鍵。次サイクルで再選定する。
