# ---------------------------------------------------------------------------
# iam: 最小権限ロールを 3 つ定義する。
#
#   1. execution role  — ECS エージェントが ECR pull / SSM 取得 / ログ出力に使う
#   2. task role       — コンテナ内プロセス（Claude / submit）が AWS API を呼ぶ
#   3. scheduler role  — EventBridge Scheduler が ECS RunTask を起動する
#
# task role の権限は意図的に絞る:
#   - ssm:GetParameter* … 当該 /<name>/* パラメータのみ（復号 kms 含む）
#   - s3:PutObject       … ログバケットの logs/ prefix のみ
#   - ecs:RunTask        … （submit 起動はスケジューラ側で行うため task role には付与しない）
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

# --- execution role ---------------------------------------------------------

resource "aws_iam_role" "execution" {
  name               = "${var.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# execution role は SSM secrets を解決してコンテナに注入するため read 権限が要る。
data "aws_iam_policy_document" "execution_ssm" {
  statement {
    sid       = "ReadTaskSecrets"
    actions   = ["ssm:GetParameters"]
    resources = var.secret_arns
  }

  statement {
    sid       = "DecryptSecrets"
    actions   = ["kms:Decrypt"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "execution_ssm" {
  name   = "${var.name}-execution-ssm"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_ssm.json
}

# --- task role --------------------------------------------------------------

resource "aws_iam_role" "task" {
  name               = "${var.name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task" {
  # 実行ログを logs/ prefix に書く。
  statement {
    sid     = "WriteLogs"
    actions = ["s3:PutObject"]
    resources = [
      "${var.log_bucket_arn}/${var.log_prefix}*",
    ]
  }

  # 重複提出判定 fingerprint の読み書き（state/submit/ 配下）。
  statement {
    sid       = "SubmitFingerprint"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${var.log_bucket_arn}/state/submit/*"]
  }

  # NOTE: シークレットは execution role 経由で task definition の `secrets` として
  # コンテナに注入される。task role には SSM 読取 / kms:Decrypt を **付与しない**
  # （headless Claude が Bash で任意実行できるため、攻撃面を最小化する）。
  # entrypoint.sh の SSM フォールバックは、注入済み env がある正常系では no-op。
}

resource "aws_iam_role_policy" "task" {
  name   = "${var.name}-task"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

# --- scheduler role ---------------------------------------------------------

resource "aws_iam_role" "scheduler" {
  name               = "${var.name}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

data "aws_iam_policy_document" "scheduler" {
  statement {
    sid     = "RunSubmitTask"
    actions = ["ecs:RunTask"]
    # 循環依存を避けるため task definition ARN を命名規約から組み立てる
    # （family = "<name>-submit"、全 revision を ArnLike で許容）。
    resources = [
      "arn:aws:ecs:${var.region}:${var.account_id}:task-definition/${var.name}-submit:*",
    ]

    condition {
      test     = "ArnLike"
      variable = "ecs:cluster"
      values = [
        "arn:aws:ecs:${var.region}:${var.account_id}:cluster/${var.name}",
      ]
    }
  }

  # RunTask に渡す execution / task ロールを PassRole する。
  statement {
    sid     = "PassTaskRoles"
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.execution.arn,
      aws_iam_role.task.arn,
    ]
  }
}

resource "aws_iam_role_policy" "scheduler" {
  name   = "${var.name}-scheduler"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler.json
}
