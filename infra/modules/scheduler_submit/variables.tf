variable "name" {
  description = "リソース名の接頭辞"
  type        = string
}

variable "region" {
  description = "AWS リージョン"
  type        = string
}

variable "image" {
  description = "submit コンテナイメージ（loop と同一イメージ）"
  type        = string
}

variable "cpu" {
  description = "submit タスク CPU"
  type        = number
}

variable "memory" {
  description = "submit タスクメモリ (MiB)"
  type        = number
}

variable "schedule_expression" {
  description = "EventBridge Scheduler の式（例 rate(15 minutes)）"
  type        = string
}

variable "subnet_ids" {
  description = "タスクを起動するサブネット"
  type        = list(string)
}

variable "security_group_ids" {
  description = "タスクのセキュリティグループ"
  type        = list(string)
}

variable "execution_role_arn" {
  description = "ECS execution role ARN"
  type        = string
}

variable "task_role_arn" {
  description = "ECS task role ARN"
  type        = string
}

variable "scheduler_role_arn" {
  description = "EventBridge Scheduler role ARN"
  type        = string
}

variable "cluster_arn" {
  description = "submit task を起動する ECS クラスタ ARN"
  type        = string
}

variable "secret_arns" {
  description = "環境変数名 -> SSM パラメータ ARN マップ"
  type        = map(string)
}

variable "log_bucket" {
  description = "実行ログ書き出し先 S3 バケット"
  type        = string
}

variable "log_prefix" {
  description = "ログ prefix"
  type        = string
}
