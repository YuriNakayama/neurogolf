# ---------------------------------------------------------------------------
# network: コスト最小化のため既定 VPC のパブリックサブネットを使い、NAT GW を
# 立てない（NAT GW は ~$32/月 + 転送課金）。Fargate タスクには public IP を
# 付与してアウトバウンド（Kaggle / GitHub / Anthropic API）を確保し、
# セキュリティグループは egress のみ許可・ingress 全閉でガードする。
# ---------------------------------------------------------------------------

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "egress" {
  name        = "${var.name}-egress"
  description = "Egress only for ${var.name} ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All outbound (Kaggle / GitHub / Anthropic / AWS APIs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name}-egress"
  }
}
