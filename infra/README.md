# infra/ — ECS 上の Claude 自律ループ + Kaggle submit（Terraform/AWS）

NeuroGolf の精度改善を **AWS ECS Fargate 上の Claude Code CLI（headless）** で自律的に回す基盤。

- **15 分毎**: EventBridge Scheduler → ECS RunTask（短命）で Kaggle submit
- **常駐ループ**: 3 時間毎に `main` からブランチを切り、Claude が実装 → PR → CI green 待ち → セルフマージ。合間は精度測定・改善案・Web 調査プランニングを継続
- **ログ**: 全実行ログを S3（`logs/`）へ常時書き出し
- **コスト最小化**: Fargate **Spot** 常駐、NAT GW なし（public IP egress）、CloudWatch Insights 無効

## ディレクトリ構成

```
infra/
  bootstrap/        Terraform state 用 S3 + DynamoDB（local state で先に apply）
  envs/prod/        本番スタック（backend "s3"）。modules を結線
  modules/
    network/        既定 VPC + egress-only SG（NAT GW なし）
    ecr/            loop イメージのリポジトリ
    s3_logs/        bootstrap バケットを参照（logs/ prefix）
    ssm_secrets/    SecureString パラメータの箱（値は手動投入）
    iam/            execution / task / scheduler の最小権限ロール
    ecs_loop/       常駐 service（Fargate Spot）+ loop task definition
    scheduler_submit/ EventBridge Scheduler + submit task definition
  docker/Dockerfile loop/submit 共用イメージ
  scripts/          コンテナ内 /app/loop/ のループ実装
    entrypoint.sh   モード分岐（loop|submit）+ SSM/git/gh 初期化
    loop_runner.sh  3h ブランチ→実装→PR→CI green→セルフマージ + 継続作業
    submit_once.sh  短命 submit タスク（fingerprint で重複提出を抑制）
    s3_logger.sh    ログ整形 + S3 同期 + シークレットマスキング
    prompts/        Claude headless に渡すプロンプト（implement/planning/pr_body）
```

## セットアップ手順

### 前提
- AWS CLI が認証済み（`aws sts get-caller-identity` が通る）
- Terraform >= 1.10、Docker、`gh`（ローカルからのイメージ push 用）

### 1. state バックエンドを作る（bootstrap, local state）

```bash
cd infra/bootstrap
cp terraform.tfvars.example terraform.tfvars
# account_id を埋める:
#   aws sts get-caller-identity --query Account --output text
terraform init
terraform apply
# 出力の state_bucket（neurogolf-infra-<account_id>）を控える
```

> bootstrap の state は **local**。`terraform.tfstate` はコミットしない（`.gitignore` 済み）。

### 2. prod スタックを init（backend に bootstrap バケットを指定）

```bash
cd ../envs/prod
cp terraform.tfvars.example terraform.tfvars   # 必要に応じ値を調整
terraform init \
  -backend-config="bucket=neurogolf-infra-<account_id>"
```

### 0. ⚠️ 必須の前提: GitHub ブランチ保護（セルフマージの安全弁）

ループは PR を**セルフマージ**する。`loop_runner.sh` の `ci_passed` は「チェックが 1 件以上実際に走り全て成功」を確認してからマージするが、**最終的な強制力は GitHub 側のブランチ保護**にある。稼働前に必ず設定すること:

- `main` に branch protection / ruleset を設定し **Require status checks to pass** を有効化（必須チェックに `ci` を指定）。
- **Do not allow bypassing the above settings**（管理者にも適用）を有効化。これが無いと bot トークンが保護を素通りする。
- 任意: CI の `paths` フィルタ（現状 `backend/**`）に依存せず全 PR で必須ジョブが走るようにすると、`infra/` のみ変更の PR もゲートされる。

`GH_TOKEN` は **fine-grained PAT** で当該リポジトリのみ・`Contents: RW` / `Pull requests: RW` に最小化する（`Workflows` 権限は付与しない＝ワークフロー改変を防止）。

### 3. シークレットを SSM に投入（値は手動・コミット厳禁）

`ssm_secrets` モジュールが箱（SecureString, 値はプレースホルダ）を作る。apply 後に実値を投入する:

```bash
NAME=neurogolf
aws ssm put-parameter --overwrite --type SecureString \
  --name "/$NAME/ANTHROPIC_API_KEY" --value "sk-ant-..."
aws ssm put-parameter --overwrite --type SecureString \
  --name "/$NAME/KAGGLE_USERNAME" --value "<kaggle user>"
aws ssm put-parameter --overwrite --type SecureString \
  --name "/$NAME/KAGGLE_KEY" --value "<kaggle key>"
aws ssm put-parameter --overwrite --type SecureString \
  --name "/$NAME/GH_TOKEN" --value "<github PAT: repo + PR 権限>"
```

> `ssm_secrets` は `ignore_changes = [value]` のため、以後の `terraform apply` がこの実値を上書きしない。

### 4. イメージをビルドして ECR へ push

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-northeast-1
REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/neurogolf-loop"
TAG=$(git rev-parse --short HEAD)

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# context はリポジトリルート
docker build -f infra/docker/Dockerfile -t "$REPO:$TAG" .
docker push "$REPO:$TAG"
```

### 5. apply（image_tag を push したタグに）

```bash
cd infra/envs/prod
terraform apply -var="image_tag=$TAG"
```

これで DVC remote バケット `neurogolf-dvc-<account_id>` も作成される（`terraform output dvc_remote_url` で確認）。`.dvc/config` の URL と一致していること。

### 6. 初回の ONNX を DVC remote へ push（データ受け渡しの起点）

submit タスクは `dvc pull` で ONNX を取得する。空だと毎回「提出物なし」で空振りするため、手元の検証済み ONNX を一度 push しておく:

```bash
# repo root で（~/.aws 認証）
uv --project backend run dvc push
```

以後はループが各サイクルで `dvc add`+`dvc push` し、submit が `dvc pull` する。**認証**: ローカルは `~/.aws`、ECS は task role の IAM を `dvc[s3]` が自動使用（キー注入不要）。

## 段階導入（推奨）

1. **submit だけ先に稼働**: `terraform apply` 後、EventBridge Scheduler が 15 分毎に submit を起動。`logs/<date>/submit/` と Kaggle 提出履歴で動作確認。
2. **ループを稼働**: `ecs_loop` の service は desired_count=1 で常駐。最初の数サイクルは PR がセルフマージされる挙動を `gh pr list` と S3 ログで監視。

## 運用メモ

- **品質ゲート**: PR は `ci.yml`（ruff/mypy/pytest）が green の場合のみセルフマージ。`main` のブランチ保護で同 CI を必須チェックに設定しておくこと。
- **暴走防止**: `loop_runner.sh` は `--max-turns` / 各フェーズ timeout / サイクル間 sleep を持つ。Claude の許可ツールは `Read,Edit,Write,Bash,Grep,Glob` に限定。
- **コスト監視**: AWS Budgets でアラートを設定。Claude API 課金が主コストになり得る。
- **停止**: ループを止めるなら `aws ecs update-service --cluster neurogolf --service neurogolf-loop --desired-count 0`。submit を止めるなら scheduler を無効化。
- **シークレットマスキング**: `s3_logger.sh` が既知トークンをログから伏字化する。ログに新たな秘匿値を出さないこと。

## ログ / state の S3 レイアウト

```
s3://neurogolf-infra-<account_id>/
  state/prod/terraform.tfstate     # prod スタックの state（DynamoDB lock）
  state/submit/last_fingerprint.txt # submit 重複判定
  logs/<YYYY/MM/DD>/loop/<run_id>.log
  logs/<YYYY/MM/DD>/submit/<run_id>.log
```
