variable "region" {
  description = "AWS リージョン"
  type        = string
  default     = "ap-northeast-1"
}

variable "image_tag" {
  description = "ECR にプッシュした loop イメージのタグ（通常は git sha）"
  type        = string
  default     = "latest"
}

variable "github_repo" {
  description = "セルフマージ対象の GitHub リポジトリ (owner/name)"
  type        = string
  default     = "YuriNakayama/neurogolf"
}

variable "log_prefix" {
  description = "実行ログを書き出す S3 prefix"
  type        = string
  default     = "logs/"
}

variable "branch_interval_seconds" {
  description = "ブランチ→PR→セルフマージのサイクル間隔（既定 3 時間）"
  type        = number
  default     = 10800
}

variable "submit_schedule" {
  description = "Kaggle submit を起動する EventBridge Scheduler の式（既定 15 分毎）"
  type        = string
  default     = "rate(15 minutes)"
}

variable "loop_cpu" {
  description = "常駐ループタスクの CPU ユニット"
  type        = number
  default     = 1024
}

variable "loop_memory" {
  description = "常駐ループタスクのメモリ (MiB)"
  type        = number
  default     = 2048
}

variable "submit_cpu" {
  description = "submit 短命タスクの CPU ユニット"
  type        = number
  default     = 512
}

variable "submit_memory" {
  description = "submit 短命タスクのメモリ (MiB)"
  type        = number
  default     = 1024
}
