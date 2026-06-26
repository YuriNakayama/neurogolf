# ---------------------------------------------------------------------------
# ssm_secrets: ECS タスクが必要とするシークレットの「箱」を SecureString として
# 定義する。値そのものは Terraform 管理対象外（コミット禁止）で、apply 後に
# 手動投入する（infra/README.md 参照）。
#
#   ANTHROPIC_API_KEY  : Claude Code CLI の課金キー
#   KAGGLE_USERNAME    : Kaggle 提出
#   KAGGLE_KEY         : Kaggle 提出
#   GH_TOKEN           : セルフマージ用 GitHub トークン（repo / PR スコープ）
#
# `lifecycle.ignore_changes = [value]` で、手動投入した値を terraform apply が
# プレースホルダで上書きしないようにする。
# ---------------------------------------------------------------------------

locals {
  secret_keys = [
    "ANTHROPIC_API_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "GH_TOKEN",
  ]
}

resource "aws_ssm_parameter" "secret" {
  for_each = toset(local.secret_keys)

  name        = "/${var.name}/${each.key}"
  description = "${each.key} for ${var.name} ECS tasks (value set manually)"
  type        = "SecureString"
  value       = "PLACEHOLDER_SET_MANUALLY"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Key = each.key
  }
}
