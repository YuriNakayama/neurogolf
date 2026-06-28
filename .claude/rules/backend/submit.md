---
paths:
  - "backend/src/submit/**"
  - "dev/submit"
---

# Submit Rules (`backend/src/submit/**`, `dev/submit`)

Kaggle NeuroGolf 2026 への提出（packaging / validation / Kaggle CLI 呼び出し）の規約。一般的な Python 規約は `.claude/rules/python.md`、提出仕様（zip 構造・禁止 op・サイズ上限）は `docs/competition/abstract.md` を参照。

`dev/submit` は `backend/` で `uv run python -m submit` を呼ぶラッパー。パスは `backend/` 起点（`src/submit/...`、`data/output/onnx`）。

## 提出ファイル名（厳守）

- 提出 zip の **basename は厳密に `submission.zip`**。別名は Kaggle に拒否される（Web/CLI とも `"Submission files must be named submission.zip"`）。
- 正準名は `submit.packager.SUBMISSION_NAME` 定数。`build_submission_zip()` は常に `out_dir/submission.zip` を生成するため、このパッケージャ経由なら自動的に正しい。
- 任意 zip を直接 submit する経路では、`kaggle_api.submit()` が basename を検査し、不一致なら `SubmissionNameError`（`KaggleCLIError` のサブクラス）を **Kaggle へ送る前に** 送出する。別名 zip を作って渡してはならない。

## 提出経路（厳守）

- submit は **kaggle CLI 経由**（`src/submit/kaggle_api.py` の `subprocess` で `kaggle competitions submit`）で行う。
- Kaggle Python SDK の `api.competition_submit()` は **使わない**。participate スコープの quirk で `403 PERMISSION_DENIED ('competitions.participate')` を返すことがある（CLI 経路では発生しない）。
- 提出後は CLI 出力だけで成否を判断せず、`confirm_submission()` / `poll()` で履歴 API により最終確認する（CLI は成功時にも非ゼロ終了し得る）。

## 認証

- 資格情報は環境変数（`KAGGLE_USERNAME` / `KAGGLE_KEY`）または `~/.kaggle/kaggle.json` から解決（`auth.ensure_credentials()`）。コードに秘密情報をハードコードしない。ログ・stdout に出さない。
- 公開メタ取得（`competitions_list` 等）は participate 不要だが、submit・自分の提出一覧取得は participate スコープが必要。

## 採否フロー（LB ゲート）

- ローカル scorer（`src/evaluate.audit_one`）は Kaggle 本番より弱く、一部 op（MaxPool/ConvTranspose の負パッド等）で **false-negative** を出す。**ローカル単独 gating で override 採否を決めない**。
- バンドル変更の採否は **実 LB の degrade-check**（submit して Public Score を確認）を正とする。改善なし・退行なら不採用、退行ゼロかつ private に微益な機械最適化（audit-gated で exact かつ strictly cheaper）のみ例外的に採用可。

## dev/submit の使い方

```bash
dev/submit -m "メッセージ"                  # data/output/onnx を検証→submission.zip→提出
dev/submit -m "確認" --dry-run              # 検証 + zip 生成のみ（提出しない）
dev/submit -m "待機" --wait                 # 提出後に validation をポーリング
dev/submit validate --onnx-dir DIR          # コスト/スコアの検証だけ
```
