#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# submit_once.sh — 15 分毎に起動される短命タスク。
#
#   1. 最新の ONNX 群を取得（DVC pull / S3 同期、best-effort）
#   2. submission.zip を検証・生成して Kaggle に提出（既存 python -m submit）
#   3. 直近の提出と内容が同一なら skip（無駄な提出枠消費を避ける）
#   4. ログを S3 へ書き出して終了
#
# entrypoint.sh から source 済みの s3_logger.sh のヘルパを利用する。
# ---------------------------------------------------------------------------
set -uo pipefail

readonly BACKEND_DIR="/app/backend"
readonly ONNX_DIR="${ONNX_DIR:-data/output/onnx}"
readonly FINGERPRINT_S3="s3://${LOG_BUCKET:-}/state/submit/last_fingerprint.txt"

cd "${BACKEND_DIR}" || exit 1

# --- 1) 最新 ONNX 取得（best-effort） --------------------------------------
fetch_onnx() {
  if uv run dvc pull "${ONNX_DIR}" >/dev/null 2>&1; then
    log_info "dvc pull ${ONNX_DIR} ok"
  else
    log_warn "dvc pull failed or no dvc tracking; using existing ${ONNX_DIR}"
  fi
}

# ONNX 群のフィンガープリント（ファイル名+サイズ+sha のまとめハッシュ）。
compute_fingerprint() {
  if [[ ! -d "${ONNX_DIR}" ]]; then
    printf 'no-onnx-dir'
    return 0
  fi
  find "${ONNX_DIR}" -name 'task*.onnx' -type f -print0 2>/dev/null \
    | sort -z \
    | xargs -0 sha256sum 2>/dev/null \
    | sha256sum \
    | awk '{print $1}'
}

# 直近提出時の fingerprint を S3 から取得。
load_last_fingerprint() {
  [[ -z "${LOG_BUCKET:-}" ]] && return 0
  aws s3 cp "${FINGERPRINT_S3}" - 2>/dev/null || true
}

save_fingerprint() {
  local fp="$1"
  [[ -z "${LOG_BUCKET:-}" ]] && return 0
  printf '%s' "${fp}" | aws s3 cp - "${FINGERPRINT_S3}" >/dev/null 2>&1 || \
    log_warn "failed to persist submit fingerprint"
}

main() {
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    log_error "KAGGLE credentials missing; aborting submit"
    log_sync
    exit 3
  fi

  fetch_onnx

  local onnx_count
  onnx_count="$(find "${ONNX_DIR}" -name 'task*.onnx' -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${onnx_count}" -eq 0 ]]; then
    log_warn "no taskNNN.onnx found under ${ONNX_DIR}; nothing to submit"
    log_sync
    exit 0
  fi
  log_info "found ${onnx_count} onnx files"

  local fp last
  fp="$(compute_fingerprint)"
  last="$(load_last_fingerprint)"
  if [[ -n "${last}" && "${fp}" == "${last}" ]]; then
    log_info "fingerprint unchanged (${fp:0:12}); skip submit to save quota"
    log_sync
    exit 0
  fi

  local msg
  msg="auto-submit $(date -u +%Y-%m-%dT%H:%M:%SZ) ${onnx_count}tasks"
  log_run "kaggle-submit" \
    uv run python -m submit submit -m "${msg}" --onnx-dir "${ONNX_DIR}" --wait
  local rc=$?

  if [[ ${rc} -eq 0 ]]; then
    save_fingerprint "${fp}"
    log_info "submit success; fingerprint saved"
  else
    log_error "submit failed rc=${rc}"
  fi

  log_sync
  exit "${rc}"
}

main "$@"
