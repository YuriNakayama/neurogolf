# 接続後に手で貼るプロンプト集（manual loop / submit）

`dev/ecs-connect` で稼働中の **loop コンテナ**に入った後、自律ループ相当の作業を
**人が手で1回だけ起動**するためのコピペ用プロンプト/コマンド集。
（自動で回す本体のプロンプトは `infra/scripts/prompts/*.prompt`、ECS 全体像は
`infra/README.md` を参照。）

- 接続すると cwd は作業ツリー `/work/neurogolf`（実コード + git + dvc）。
- ループ用プロンプトはイメージに同梱され `/app/loop/prompts/*.prompt` にある。
- `git` / `gh` / `kaggle` / `dvc` / `uv` / `claude` はすべて利用可能
  （`entrypoint.sh` が初期化済み。env に各シークレットが注入されている）。

> ⚠️ 手動起動は**常駐ループと競合し得る**。同時に走らせると同じ `auto/<ts>`
> 帯や同一 PR を二重に作る恐れがある。手で回す前にループを一時停止するのが安全:
> `aws ecs update-service --cluster neurogolf --service neurogolf-loop --desired-count 0`
> （手動作業後に `--desired-count 1` で戻す）。同一コンテナ内で `claude` を
> 直叩きする分にはループ本体（`loop_runner.sh`）とは別プロセスになる。

---

## 0. 前提確認（最初に1回）

```bash
cd /work/neurogolf
git -C . ls-remote origin HEAD >/dev/null && echo "git OK"
kaggle competitions list -s neurogolf >/dev/null && echo "kaggle OK"
uv --project backend run dvc remote list
test -n "$CLAUDE_CODE_OAUTH_TOKEN" && echo "claude token present"
```

---

## 1. 実装を1サイクル手動起動（implement → PR → セルフマージ）

`loop_runner.sh` の1サイクルと同じ流れを手で踏む。**ブランチを自分で切ってから**
`claude` に実装させ、緑なら PR を出す。プロンプト本体は同梱の
`/app/loop/prompts/implement.prompt` をそのまま使う。

```bash
cd /work/neurogolf
git fetch origin main && git checkout -B main origin/main
TS=$(date -u +%Y%m%d-%H%M%S); git checkout -b "manual/${TS}"
uv --project backend run dvc pull || true

# 実装（ループと同じ制限: ツール限定・acceptEdits・max-turns）
claude -p "$(cat /app/loop/prompts/implement.prompt)" \
  --max-turns "${CLAUDE_MAX_TURNS:-60}" \
  --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
  --permission-mode acceptEdits \
  --output-format stream-json --verbose

# 生成 ONNX を remote へ（ポインタ onnx.dvc が git 差分に乗る）
uv --project backend run dvc add data/output/onnx || true
uv --project backend run dvc push data/output/onnx || true

# ローカル検証 → 差分があれば PR
./dev/lint && ./dev/test-bot
git add -A && git commit -m ":robot: manual: 手動実装 ${TS}"
git push -u origin "manual/${TS}"
gh pr create --base main --head "manual/${TS}" \
  --title ":robot: manual: 手動実装 ${TS}" \
  --body-file /app/loop/prompts/pr_body.prompt
```

CI green を確認してからマージ（`loop_runner.sh` の `ci_passed` 相当を手で）:

```bash
PR=$(gh pr view --json url -q .url)      # 直前に作った PR
gh pr checks "$PR" --watch --interval 30 || true
gh pr checks "$PR" --json name,state     # 全 SUCCESS を目視
gh pr merge "$PR" --squash --delete-branch
```

### 1-bis. 「特定の改善」をピンポイントで指示したい場合

同梱プロンプトの代わりに、自分の指示文を直接渡す（**正答最優先・静的形状・
禁止演算なし・各 onnx ≤ 1.44MB・TDD・`dev/lint`+`dev/test-bot` 緑**は厳守させる）:

```bash
claude -p "$(cat <<'EOF'
あなたは NeuroGolf 2026 の実装エージェント。リポジトリ規約は .claude/CLAUDE.md と
.claude/rules/python.md に従う。今回のタスク: <ここに具体的な1改善を書く。例:
「上下反転ソルバを backend/src/solvers に追加し、最小チャネル構成で cost を抑える」>。
厳守: train/test/arc-gen 全ペア完全一致を維持。静的形状・禁止演算(Loop/Scan/NonZero/
Unique/Script/Function)なし・各 onnx ≤ 1.44MB。先にテストを書き、backend で
uv run pytest を緑に。最後に ./dev/lint と ./dev/test-bot をローカルで通す。
差分は小さく、不要物と print ログを残さない。
EOF
)" \
  --max-turns 60 --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
  --permission-mode acceptEdits --output-format stream-json --verbose
```

---

## 2. リサーチ/プランニングを手動起動（main を汚さない）

差分は破棄前提。次サイクルの種を出させるだけ。

```bash
cd /work/neurogolf
claude -p "$(cat /app/loop/prompts/planning.prompt)" \
  --max-turns "${CLAUDE_MAX_TURNS:-60}" \
  --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
  --permission-mode acceptEdits --output-format stream-json --verbose
git reset --hard origin/main && git clean -fd     # 生成物を破棄
```

---

## 3. Kaggle submit を手動起動（submit ループ相当）

`submit_once.sh` と同じ「最新 ONNX を取得 → 検証 → 提出」を手で行う。
**zip 名は `submission.zip` 固定・CLI 経路**（`python -m submit` が担保）。

```bash
cd /work/neurogolf
uv --project backend run dvc pull data/output/onnx || true

# まず提出せず検証だけ（dry-run）
cd backend
uv run python -m submit validate --onnx-dir ../data/output/onnx

# 問題なければ提出（--wait で validation をポーリング）
uv run python -m submit submit \
  -m "manual-submit $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --onnx-dir ../data/output/onnx --wait
```

> 短命 submit タスクとの二重提出に注意。15分毎スケジューラが動いている間は
> fingerprint 一致なら自動側が skip するが、手動で内容を変えた直後は重複し得る。

---

## 4. ループ本体をフォアグラウンドで観察したい場合

`loop_runner.sh` をそのまま手で起動するとサイクルが回り始める（**常駐サービスと
競合するので、必ずサービスを desired=0 にしてから**）。ログは標準出力に出る。

```bash
# 事前にサービス停止（別端末/ローカルから）:
#   aws ecs update-service --cluster neurogolf --service neurogolf-loop --desired-count 0
WORK_DIR=/work/neurogolf /app/loop/loop_runner.sh
```

---

## 付録: ループ既定値（`loop_runner.sh` 由来）

| 変数 | 既定 | 意味 |
|------|------|------|
| `CLAUDE_MAX_TURNS` | 60 | claude -p の最大ターン |
| `CLAUDE_IMPL_TIMEOUT` | 3600 | 実装フェーズ上限(秒) |
| `CLAUDE_PLAN_TIMEOUT` | 1200 | planning 1回の上限(秒) |
| `CLAUDE_ALLOWED_TOOLS` | `Read,Edit,Write,Bash,Grep,Glob` | 許可ツール |
| `BRANCH_INTERVAL_SECONDS` | 10800 | 1サイクル長(秒) |
