resource "aws_ecr_repository" "loop" {
  name                 = "${var.name}-loop"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# 直近イメージのみ保持してストレージコストを抑える。
resource "aws_ecr_lifecycle_policy" "loop" {
  repository = aws_ecr_repository.loop.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
