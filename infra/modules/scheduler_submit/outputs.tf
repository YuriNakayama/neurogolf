output "task_definition_arn" {
  description = "submit task definition ARN"
  value       = aws_ecs_task_definition.submit.arn
}

output "task_definition_family" {
  description = "submit task definition family"
  value       = aws_ecs_task_definition.submit.family
}

output "schedule_name" {
  description = "EventBridge Scheduler 名"
  value       = aws_scheduler_schedule.submit.name
}
