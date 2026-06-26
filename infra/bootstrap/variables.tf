variable "region" {
  description = "AWS リージョン"
  type        = string
  default     = "ap-northeast-1"
}

variable "account_id" {
  description = "AWS アカウント ID（state バケット名の一意化に使用）"
  type        = string
}

variable "log_retention_days" {
  description = "S3 logs/ prefix の保持日数"
  type        = number
  default     = 30
}
