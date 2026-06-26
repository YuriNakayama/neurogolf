output "parameter_arns" {
  description = "全シークレットパラメータの ARN（IAM ポリシーの Resource に使用）"
  value       = [for p in aws_ssm_parameter.secret : p.arn]
}

output "parameter_names" {
  description = "値を手動投入する SSM パラメータ名一覧"
  value       = [for p in aws_ssm_parameter.secret : p.name]
}

# ECS の container secrets は { 環境変数名 = valueFrom(ARN) } 形式で渡す。
output "parameter_map" {
  description = "環境変数名 -> SSM パラメータ ARN のマップ"
  value       = { for k, p in aws_ssm_parameter.secret : k => p.arn }
}
