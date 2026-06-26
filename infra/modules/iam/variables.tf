variable "name" {
  description = "リソース名の接頭辞"
  type        = string
}

variable "region" {
  description = "AWS リージョン（kms ViaService 条件に使用）"
  type        = string
}

variable "account_id" {
  description = "AWS アカウント ID"
  type        = string
}

variable "log_bucket_arn" {
  description = "ログバケット ARN"
  type        = string
}

variable "log_prefix" {
  description = "ログ書き込みを許可する prefix"
  type        = string
}

variable "secret_arns" {
  description = "タスクが読む SSM パラメータ ARN 一覧"
  type        = list(string)
}
