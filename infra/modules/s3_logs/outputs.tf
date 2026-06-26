output "bucket_name" {
  description = "ログバケット名"
  value       = data.aws_s3_bucket.infra.bucket
}

output "bucket_arn" {
  description = "ログバケット ARN"
  value       = data.aws_s3_bucket.infra.arn
}
