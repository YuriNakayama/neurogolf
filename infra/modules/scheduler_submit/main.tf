# ---------------------------------------------------------------------------
# scheduler_submit: 15 分毎に EventBridge Scheduler が短命の submit タスクを
# ECS RunTask で起動する。submit タスクは ONNX 群を集めて submission.zip を
# 作り Kaggle に提出し、完了で停止する（常駐しない）。
#
# task definition family は "<name>-submit"（IAM scheduler ポリシーの ArnLike と
# 一致）。
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "submit" {
  name              = "/ecs/${var.name}-submit"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "submit" {
  family                   = "${var.name}-submit"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "submit"
      image     = var.image
      essential = true
      command   = ["/app/loop/entrypoint.sh", "submit"]

      environment = [
        { name = "AWS_REGION", value = var.region },
        { name = "LOG_BUCKET", value = var.log_bucket },
        { name = "LOG_PREFIX", value = var.log_prefix },
        { name = "RUN_MODE", value = "submit" },
      ]

      secrets = [
        for k, arn in var.secret_arns : { name = k, valueFrom = arn }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.submit.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "submit"
        }
      }
    }
  ])
}

resource "aws_scheduler_schedule" "submit" {
  name = "${var.name}-submit"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = "UTC"

  target {
    arn      = var.cluster_arn
    role_arn = var.scheduler_role_arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.submit.arn
      launch_type         = "FARGATE"
      task_count          = 1

      network_configuration {
        subnets          = var.subnet_ids
        security_groups  = var.security_group_ids
        assign_public_ip = true
      }
    }

    # 多重起動を抑える: 前回の submit がまだ走っていれば retry しない。
    retry_policy {
      maximum_retry_attempts = 0
    }
  }
}
