# ---------------------------------------------------------------------------
# ecs_loop: 常駐の Claude 自律ループを Fargate Spot で 1 タスク回す。
#
#   - cluster 名は "<name>"（IAM の ArnLike 条件と一致させる）
#   - capacity provider は FARGATE_SPOT 100% でコスト最小化。Spot 中断時は
#     service が desiredCount=1 を維持して自動再起動する（作業は逐次 PR に push
#     されるので冪等再開できる）
#   - container はループ用 entrypoint を起動し、ANTHROPIC / KAGGLE / GH の
#     シークレットは SSM から secrets で注入する
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "this" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "disabled" # コスト抑制。可観測性は S3 ログ + awslogs で担保
  }
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = ["FARGATE_SPOT", "FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "loop" {
  name              = "/ecs/${var.name}-loop"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "loop" {
  family                   = "${var.name}-loop"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "loop"
      image     = var.image
      essential = true
      command   = ["/app/loop/entrypoint.sh", "loop"]

      environment = [
        { name = "AWS_REGION", value = var.region },
        { name = "LOG_BUCKET", value = var.log_bucket },
        { name = "LOG_PREFIX", value = var.log_prefix },
        { name = "GITHUB_REPO", value = var.github_repo },
        { name = "BRANCH_INTERVAL_SECONDS", value = tostring(var.branch_interval_seconds) },
        { name = "RUN_MODE", value = "loop" },
      ]

      secrets = [
        for k, arn in var.secret_arns : { name = k, valueFrom = arn }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.loop.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "loop"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "loop" {
  name            = "${var.name}-loop"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.loop.arn
  desired_count   = 1

  # ECS Exec（aws ecs execute-command）でコンテナへ対話接続できるようにする。
  enable_execute_command = true

  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = true # NAT GW を立てないため public IP でアウトバウンド
  }

  # ループは状態を持たず、新リビジョンに即時切替してよい。
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
}
