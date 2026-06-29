# 20260629 case3 cycle6 plan — 新規公開バンドル harishk2209/neurogolf-7178-23 を harvest

## 現状把握

- floor = `data/output/onnx`（400 onnx, 実 LB **7176.49**, cycle4 の boristown ベース cherry-pick）。
- Kaggle 公開 kernel を再スキャン → **`harishk2209/neurogolf-7178-23`（LB 7178.23）** が floor を
  **+1.74** 上回る新規バンドルを発見。`kokinnwakashuu/neurogolf-graph-golf-blend` は md5 一致＝同一物。
- lucifer19/biohack44 は kernel output に submission.zip 無し or DL ハングで今回は除外。

## 仮説

graded 公開バンドル（実 LB を持つ）の cross-bundle cherry-pick は hidden 安全に転送する
（cycle4 の grader-faithfulness law で実証済み）。harishk は floor より高 LB なので、
各タスクで min(floor, harishk) を取れば **≥7178.23**、cherry-pick 相乗で更に上積み期待。

## 本サイクルで回す 1 改善

- 各タスク 001-400 で floor と harishk の cost-only を計測 → harishk が安いタスクのみ
  faithful 正答検証（n_fail=0, train/test/arc-gen 全 example）→ 通れば harishk net を採用。
- combined_best を構築 → submission.zip → Kaggle submit → Public Score。

## 採否

- 実 LB が 7176.49 を上回れば採用（combined を data/output/onnx へ昇格, dvc add/push）。
  下回れば破棄。faithful 環境 = onnx 1.20.0 / ort 1.24.1 / py3.12（docs/experiment/faithful-scorer.md）。
