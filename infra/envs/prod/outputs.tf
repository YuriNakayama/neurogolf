output "ecr_repository_url" {
  description = "loop イメージを push する ECR リポジトリ URL"
  value       = module.ecr.repository_url
}

output "ecs_cluster_name" {
  description = "ECS クラスタ名"
  value       = module.ecs_loop.cluster_name
}

output "loop_service_name" {
  description = "常駐ループの ECS サービス名"
  value       = module.ecs_loop.service_name
}

output "submit_task_definition" {
  description = "submit 短命タスクの task definition family"
  value       = module.scheduler_submit.task_definition_family
}

output "log_bucket" {
  description = "実行ログ書き出し先 S3 バケット"
  value       = module.s3_logs.bucket_name
}

output "ssm_parameter_names" {
  description = "値を手動投入する必要がある SSM パラメータ名"
  value       = module.ssm_secrets.parameter_names
}
