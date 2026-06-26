#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# entrypoint.sh — コンテナ起動時の共通初期化 + モード分岐。
#
# 使い方:  entrypoint.sh <loop|submit>
#
#   loop   : 常駐ループ（3h ブランチ→実装→PR→CI green→セルフマージ + 合間の
#            継続作業）を回す。ECS Service が起動。
#   submit : 15 分毎の短命タスク。最新 ONNX を Kaggle に提出して終了。
#            EventBridge Scheduler が RunTask で起動。
#
# シークレットは ECS task definition の `secrets`（SSM 由来）で環境変数として
# 注入される前提。万一未注入の場合は SSM から直接フォールバック取得する。
# ---------------------------------------------------------------------------
set -uo pipefail

readonly MODE="${1:-loop}"
readonly REPO_DIR="/app"
readonly LOOP_DIR="${REPO_DIR}/loop"
readonly SSM_PREFIX="/neurogolf"

# shellcheck source=infra/scripts/s3_logger.sh
source "${LOOP_DIR}/s3_logger.sh"

# --- シークレットのフォールバック解決 ---------------------------------------
# task definition の secrets で注入済みなら何もしない。未設定なら SSM から引く。
resolve_secret() {
  local key="$1"
  local current="${!key:-}"
  if [[ -n "${current}" ]]; then
    return 0
  fi
  local value
  value="$(aws ssm get-parameter \
    --name "${SSM_PREFIX}/${key}" \
    --with-decryption \
    --query 'Parameter.Value' \
    --output text 2>/dev/null || true)"
  if [[ -n "${value}" && "${value}" != "None" ]]; then
    export "${key}=${value}"
  fi
}

init_secrets() {
  for key in ANTHROPIC_API_KEY KAGGLE_USERNAME KAGGLE_KEY GH_TOKEN; do
    resolve_secret "${key}"
  done
}

# --- git / gh の初期化 ------------------------------------------------------
init_git_identity() {
  git config --global user.name "neurogolf-bot"
  git config --global user.email "bot@neurogolf.local"
  git config --global --add safe.directory "${REPO_DIR}"
  # gh はトークンを GH_TOKEN 環境変数から自動的に拾う。
  if [[ -n "${GH_TOKEN:-}" ]]; then
    git config --global \
      url."https://x-access-token:${GH_TOKEN}@github.com/".insteadOf \
      "https://github.com/"
  fi
}

main() {
  init_secrets
  log_init "${MODE}"
  log_info "entrypoint mode=${MODE} starting"

  case "${MODE}" in
    loop)
      init_git_identity
      exec "${LOOP_DIR}/loop_runner.sh"
      ;;
    submit)
      exec "${LOOP_DIR}/submit_once.sh"
      ;;
    *)
      log_error "unknown mode: ${MODE} (expected loop|submit)"
      exit 64
      ;;
  esac
}

main "$@"
