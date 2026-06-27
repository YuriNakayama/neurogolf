# ---------------------------------------------------------------------------
# neurogolf-loop イメージ: ECS Fargate 上で Claude Code CLI を headless 実行し、
# 自律的に精度改善ループ（実装→PR→セルフマージ）と 15 分毎 Kaggle submit を回す。
#
# 同梱物:
#   - Python 3.13 + uv（backend を uv sync）
#   - git / gh CLI（ブランチ作成・PR・セルフマージ）
#   - Node.js + @anthropic-ai/claude-code（claude -p headless）
#   - awscli v2（SSM フォールバック取得・S3 ログ書き出し）
#   - リポジトリの backend/ と .claude/（CLAUDE.md / rules / agents / skills）
#   - /app/loop/ 配下のループ用スクリプト群
#
# context はリポジトリルート（infra/docker/Dockerfile を -f で指定してビルド）。
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# --- OS パッケージ + gh CLI + Node.js + awscli ------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gnupg unzip jq tini \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends gh nodejs \
    && rm -rf /var/lib/apt/lists/*

# awscli v2（アーキ依存。amd64/arm64 を判定）
RUN ARCH="$(uname -m)" \
    && case "$ARCH" in \
         x86_64)  AWS_ARCH=x86_64 ;; \
         aarch64) AWS_ARCH=aarch64 ;; \
         *) echo "unsupported arch: $ARCH" && exit 1 ;; \
       esac \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_ARCH}.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# uv（Python パッケージ管理）と Claude Code CLI
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /app

# --- backend 依存を uv キャッシュへ事前解決（レイヤキャッシュ + 起動高速化） -
# 実行時は entrypoint が repo を fresh clone し、そのワークツリーで
# `uv sync` する（git 履歴 + DVC ポインタを得るため）。ここで依存を
# ダウンロード済みにしておくことで、clone 後の sync を高速化する。
# backend/pyproject.toml は readme = "../README.md" を参照するため、
# ローカルパッケージ build 用に repo root の README.md も配置する。
COPY README.md /app/README.md
COPY backend/pyproject.toml backend/uv.lock backend/.python-version /app/backend/
RUN cd /app/backend && uv sync --locked --no-dev

# --- ループ用スクリプト（不変。実行時はここから起動）------------------------
COPY infra/scripts/ /app/loop/
RUN chmod +x /app/loop/*.sh

# tini で PID1 シグナル処理（長時間ループの安全な停止）。
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/loop/entrypoint.sh", "loop"]
