output "repository_url" {
  description = "ECR リポジトリ URL"
  value       = aws_ecr_repository.loop.repository_url
}

output "repository_arn" {
  description = "ECR リポジトリ ARN"
  value       = aws_ecr_repository.loop.arn
}
