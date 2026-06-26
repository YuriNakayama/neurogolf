# ---------------------------------------------------------------------------
# dvc_remote: DVC のリモートストレージ用 S3 バケット（data/ 層の実体を保管）。
#
# state/logs と分離した専用バケット `neurogolf-dvc-<account_id>`。
# ループが生成した ONNX を `dvc push` し、submit タスクが `dvc pull` で取得する
# 受け渡し経路の中核。バケット名は .dvc/config の remote URL と一致させること。
# ---------------------------------------------------------------------------

locals {
  bucket_name = "neurogolf-dvc-${var.account_id}"
}

resource "aws_s3_bucket" "dvc" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_versioning" "dvc" {
  bucket = aws_s3_bucket.dvc.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "dvc" {
  bucket = aws_s3_bucket.dvc.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc" {
  bucket = aws_s3_bucket.dvc.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}
