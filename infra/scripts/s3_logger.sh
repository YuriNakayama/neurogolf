#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# s3_logger.sh — 実行ログの整形 + S3 への継続書き出し。各スクリプトから source。
#
# ログは 2 系統:
#   1. stdout/stderr（CloudWatch awslogs が拾う）
#   2. ローカルファイル ${LOG_FILE} に追記し、定期的に S3 へ同期
#      （logs/<date>/<mode>/<run_id>.log）
#
# シークレットがログに混入しないよう mask_secrets で既知トークンを伏字化する。
# ---------------------------------------------------------------------------
set -uo pipefail

: "${LOG_BUCKET:=}"
: "${LOG_PREFIX:=logs/}"
: "${AWS_REGION:=ap-northeast-1}"

LOG_RUN_ID=""
LOG_FILE=""
LOG_S3_KEY=""

# run_id は時刻ベースだが Date は外部依存なのでコンテナの date を使う。
log_init() {
  local mode="$1"
  local ts date_part
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  date_part="$(date -u +%Y/%m/%d)"
  LOG_RUN_ID="${mode}-${ts}-$$"
  LOG_FILE="/tmp/${LOG_RUN_ID}.log"
  LOG_S3_KEY="${LOG_PREFIX}${date_part}/${mode}/${LOG_RUN_ID}.log"
  : >"${LOG_FILE}"
}

# 既知シークレットを伏字化する。
#   1. 環境変数の実値の完全一致置換
#   2. トークン形状の正規表現フォールバック（値が加工・部分露出した場合の保険）
mask_secrets() {
  local line="$1"
  local key
  for key in ANTHROPIC_API_KEY KAGGLE_KEY GH_TOKEN KAGGLE_USERNAME; do
    local val="${!key:-}"
    if [[ -n "${val}" ]]; then
      line="${line//${val}/***${key}***}"
    fi
  done
  # トークン形状のフォールバック（Anthropic / GitHub PAT）。
  line="$(printf '%s' "${line}" | sed -E \
    -e 's/sk-ant-[A-Za-z0-9_-]+/***ANTHROPIC_KEY***/g' \
    -e 's/gh[pousr]_[A-Za-z0-9]+/***GH_TOKEN***/g' \
    -e 's/github_pat_[A-Za-z0-9_]+/***GH_TOKEN***/g')"
  printf '%s' "${line}"
}

# 1 行をマスクして stdout + LOG_FILE に書く（ログ書き込みの単一チョークポイント）。
mask_line() {
  local masked
  masked="$(mask_secrets "$1")"
  printf '%s\n' "${masked}"
  if [[ -n "${LOG_FILE}" ]]; then
    printf '%s\n' "${masked}" >>"${LOG_FILE}"
  fi
}

# パイプ入力を 1 行ずつマスクして stdout + LOG_FILE に流す。
# 生ログを書く全経路はこれを通すこと（`... 2>&1 | mask_stream`）。
mask_stream() {
  local line
  while IFS= read -r line; do
    mask_line "${line}"
  done
}

_log() {
  local level="$1"
  shift
  local msg
  msg="$(mask_secrets "$*")"
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) [${level}] ${msg}"
  printf '%s\n' "${line}"
  if [[ -n "${LOG_FILE}" ]]; then
    printf '%s\n' "${line}" >>"${LOG_FILE}"
  fi
}

log_info()  { _log INFO  "$@"; }
log_warn()  { _log WARN  "$@"; }
log_error() { _log ERROR "$@" >&2; }

# ローカルログファイルを S3 に同期（best-effort、失敗してもループは継続）。
log_sync() {
  if [[ -z "${LOG_BUCKET}" || -z "${LOG_FILE}" || ! -f "${LOG_FILE}" ]]; then
    return 0
  fi
  aws s3 cp "${LOG_FILE}" "s3://${LOG_BUCKET}/${LOG_S3_KEY}" \
    --region "${AWS_REGION}" >/dev/null 2>&1 || \
    printf '%s [WARN] log_sync to s3 failed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
}

# 任意コマンドを実行し、出力をマスクして stdout + LOG_FILE に取り込みつつ
# S3 同期する。使い方: log_run "<label>" <command...>
log_run() {
  local label="$1"
  shift
  log_info "RUN ${label}: $*"
  "$@" 2>&1 | mask_stream
  local rc=${PIPESTATUS[0]}
  log_info "DONE ${label} rc=${rc}"
  log_sync
  return "${rc}"
}

# コマンドを実行し stdout/stderr をマスクして LOG_FILE + stdout に流す。
# S3 同期はしない軽量版（git/gh の細かい操作向け）。終了コードを保持する。
run_logged() {
  "$@" 2>&1 | mask_stream
  return "${PIPESTATUS[0]}"
}
