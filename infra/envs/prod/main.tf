# ---------------------------------------------------------------------------
# prod 環境: ECS Fargate 上で Claude 自律ループ + 15分毎 Kaggle submit を回す。
#
# state は bootstrap で作った S3 バケットに置く（backend "s3"）。
# `terraform init` 前に infra/bootstrap を apply し、bucket 名を backend に
# 反映すること（infra/README.md 参照）。
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # bootstrap 出力の bucket 名（neurogolf-infra-<account_id>）を埋めること。
  # 値は変数化できないため、init 時に -backend-config で渡すか直接書き換える。
  backend "s3" {
    key            = "state/prod/terraform.tfstate"
    region         = "ap-northeast-1"
    dynamodb_table = "neurogolf-tf-lock"
    encrypt        = true
    # bucket = "neurogolf-infra-<account_id>"  ← init 時に -backend-config で指定
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "neurogolf"
      ManagedBy = "terraform"
      Env       = "prod"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  name         = "neurogolf"
  account_id   = data.aws_caller_identity.current.account_id
  infra_bucket = "neurogolf-infra-${local.account_id}"
}

module "network" {
  source = "../../modules/network"

  name = local.name
}

module "ecr" {
  source = "../../modules/ecr"

  name = local.name
}

module "s3_logs" {
  source = "../../modules/s3_logs"

  bucket_name = local.infra_bucket
}

module "dvc_remote" {
  source = "../../modules/dvc_remote"

  account_id = local.account_id
}

module "ssm_secrets" {
  source = "../../modules/ssm_secrets"

  name = local.name
}

module "iam" {
  source = "../../modules/iam"

  name           = local.name
  region         = var.region
  account_id     = local.account_id
  log_bucket_arn = module.s3_logs.bucket_arn
  log_prefix     = var.log_prefix
  secret_arns    = module.ssm_secrets.parameter_arns
  dvc_bucket_arn = module.dvc_remote.bucket_arn
}

module "ecs_loop" {
  source = "../../modules/ecs_loop"

  name                    = local.name
  region                  = var.region
  image                   = "${module.ecr.repository_url}:${var.image_tag}"
  cpu                     = var.loop_cpu
  memory                  = var.loop_memory
  subnet_ids              = module.network.subnet_ids
  security_group_ids      = [module.network.security_group_id]
  execution_role_arn      = module.iam.execution_role_arn
  task_role_arn           = module.iam.task_role_arn
  secret_arns             = module.ssm_secrets.parameter_map
  log_bucket              = module.s3_logs.bucket_name
  log_prefix              = var.log_prefix
  github_repo             = var.github_repo
  branch_interval_seconds = var.branch_interval_seconds
}

module "scheduler_submit" {
  source = "../../modules/scheduler_submit"

  name                = local.name
  region              = var.region
  image               = "${module.ecr.repository_url}:${var.image_tag}"
  cpu                 = var.submit_cpu
  memory              = var.submit_memory
  schedule_expression = var.submit_schedule
  subnet_ids          = module.network.subnet_ids
  security_group_ids  = [module.network.security_group_id]
  execution_role_arn  = module.iam.execution_role_arn
  task_role_arn       = module.iam.task_role_arn
  scheduler_role_arn  = module.iam.scheduler_role_arn
  cluster_arn         = module.ecs_loop.cluster_arn
  secret_arns         = module.ssm_secrets.parameter_map
  log_bucket          = module.s3_logs.bucket_name
  log_prefix          = var.log_prefix
}
