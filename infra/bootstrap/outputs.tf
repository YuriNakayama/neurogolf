output "state_bucket" {
  description = "Terraform state + 実行ログを格納する S3 バケット名"
  value       = aws_s3_bucket.infra.bucket
}

output "lock_table" {
  description = "Terraform state lock 用の DynamoDB テーブル名"
  value       = aws_dynamodb_table.tf_lock.name
}

output "state_key_prefix" {
  description = "envs/prod の backend で使う state key の prefix"
  value       = "state/"
}

output "logs_prefix" {
  description = "実行ログを書き出す prefix"
  value       = "logs/"
}
