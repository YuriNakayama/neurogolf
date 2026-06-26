output "bucket_name" {
  description = "DVC remote バケット名"
  value       = aws_s3_bucket.dvc.bucket
}

output "bucket_arn" {
  description = "DVC remote バケット ARN"
  value       = aws_s3_bucket.dvc.arn
}

output "remote_url" {
  description = ".dvc/config に設定する remote URL"
  value       = "s3://${aws_s3_bucket.dvc.bucket}/remote"
}
