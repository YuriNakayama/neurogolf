variable "name" {
  description = "リソース名の接頭辞（cluster 名と一致）"
  type        = string
}

variable "region" {
  description = "AWS リージョン"
  type        = string
}

variable "image" {
  description = "loop コンテナイメージ（ECR URL:tag）"
  type        = string
}

variable "cpu" {
  description = "タスク CPU ユニット"
  type        = number
}

variable "memory" {
  description = "タスクメモリ (MiB)"
  type        = number
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

variable "github_repo" {
  description = "セルフマージ対象リポジトリ (owner/name)"
  type        = string
}

variable "branch_interval_seconds" {
  description = "ブランチ→PR→マージのサイクル間隔（秒）"
  type        = number
}
