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

readonly LOOP_DIR="/app/loop"
readonly REPO_DIR="${WORK_DIR:-/work/neurogolf}"
readonly BACKEND_DIR="${REPO_DIR}/backend"
# ONNX_DIR は backend cwd 相対。DVC は repo root の data/output/onnx に pull
# するため、backend から見ると ../data/output/onnx を指す必要がある（python
# -m submit も find/fingerprint も backend cwd で動くため一貫してこの相対で解決）。
readonly ONNX_DIR="${ONNX_DIR:-../data/output/onnx}"
readonly SUBMIT_GATE_TASK_DIR="${SUBMIT_GATE_TASK_DIR:-../data/lake/neurogolf-2026}"
# DVC は repo root に init 済み。pull 対象は root 相対パス。
readonly DVC_TARGET="data/output/onnx"
readonly FINGERPRINT_S3="s3://${LOG_BUCKET:-}/state/submit/last_fingerprint.txt"

# entrypoint は exec で本スクリプトを起動するため、ロガー関数を再度 source する。
# shellcheck source=infra/scripts/s3_logger.sh
source "${LOOP_DIR}/s3_logger.sh"
log_init "submit"

cd "${BACKEND_DIR}" || exit 1

# --- 1) 最新 ONNX を DVC remote から取得 ------------------------------------
# DVC は repo root で動かす（.dvc/ が /app にある）。dvc[s3] は task role の
# IAM 認証を boto3 経由で自動使用するため、明示のキー注入は不要。
fetch_onnx() {
  if (cd "${REPO_DIR}" && uv --project "${BACKEND_DIR}" run dvc pull "${DVC_TARGET}") \
      >/dev/null 2>&1; then
    log_info "dvc pull ${DVC_TARGET} ok"
  else
    log_warn "dvc pull failed; using existing ${ONNX_DIR} if any"
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

  if [[ -z "${SUBMIT_GATE_BASELINE_DIR:-}" ]]; then
    log_error "SUBMIT_GATE_BASELINE_DIR missing; aborting ungated submit"
    log_sync
    exit 3
  fi

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
    uv run python -m submit submit -m "${msg}" --onnx-dir "${ONNX_DIR}" \
      --gate-baseline-dir "${SUBMIT_GATE_BASELINE_DIR}" \
      --gate-task-dir "${SUBMIT_GATE_TASK_DIR}" \
      --wait
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
