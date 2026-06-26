output "cluster_arn" {
  description = "ECS クラスタ ARN"
  value       = aws_ecs_cluster.this.arn
}

output "cluster_name" {
  description = "ECS クラスタ名"
  value       = aws_ecs_cluster.this.name
}

output "service_name" {
  description = "常駐ループの ECS サービス名"
  value       = aws_ecs_service.loop.name
}
