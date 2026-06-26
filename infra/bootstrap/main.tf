# ---------------------------------------------------------------------------
# Bootstrap: Terraform state バックエンドの「箱」だけを作る最小構成。
#
# 鶏卵問題（state を S3 に置きたいが、その S3 を作る Terraform の state を
# どこに置くか）を避けるため、この bootstrap だけは **local state** で apply し、
# 生成された `terraform.tfstate` をリポジトリにコミットせず手元保管する
# （`infra/bootstrap/.gitignore` 参照）。
#
# 生成物:
#   - S3 バケット: state（envs/prod 用）と実行ログを prefix 分離で同居
#   - DynamoDB テーブル: state lock
#
# apply 後、envs/prod/main.tf の backend "s3" がこのバケットを参照する。
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "neurogolf"
      ManagedBy = "terraform"
      Component = "bootstrap"
    }
  }
}

locals {
  bucket_name = "neurogolf-infra-${var.account_id}"
}

resource "aws_s3_bucket" "infra" {
  bucket = local.bucket_name

  # state を含むため、誤削除を防ぐ。破棄時は手動で空にする必要がある。
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "infra" {
  bucket = aws_s3_bucket.infra.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "infra" {
  bucket = aws_s3_bucket.infra.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "infra" {
  bucket = aws_s3_bucket.infra.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# 古い実行ログは自動失効させてコストを抑える（state prefix は対象外）。
resource "aws_s3_bucket_lifecycle_configuration" "infra" {
  bucket = aws_s3_bucket.infra.id

  rule {
    id     = "expire-logs"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = var.log_retention_days
    }
  }
}

resource "aws_dynamodb_table" "tf_lock" {
  name         = "neurogolf-tf-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
