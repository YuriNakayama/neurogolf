output "subnet_ids" {
  description = "タスクを起動するサブネット ID（既定 VPC のパブリックサブネット）"
  value       = data.aws_subnets.default.ids
}

output "security_group_id" {
  description = "egress-only セキュリティグループ ID"
  value       = aws_security_group.egress.id
}
