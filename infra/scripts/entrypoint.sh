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
# イメージに焼いたスクリプト群の場所（不変）。
readonly LOOP_DIR="/app/loop"
# 実作業ツリー。git 履歴 + remote + DVC ポインタを持つ clone を毎回作る
# （イメージには .git を含めないため）。loop/submit 双方ここで動く。
readonly WORK_DIR="${WORK_DIR:-/work/neurogolf}"
readonly GITHUB_REPO="${GITHUB_REPO:-YuriNakayama/neurogolf}"
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
  for key in CLAUDE_CODE_OAUTH_TOKEN KAGGLE_USERNAME KAGGLE_KEY GH_TOKEN; do
    resolve_secret "${key}"
  done
  # サブスク OAuth トークンを使うため、万一 API キーが環境に紛れ込んでいても
  # 優先順位で勝たないよう明示的に除去する（API キー > OAuth トークンのため）。
  unset ANTHROPIC_API_KEY 2>/dev/null || true
}

# --- git / gh の初期化 ------------------------------------------------------
init_git_identity() {
  git config --global user.name "neurogolf-bot"
  git config --global user.email "bot@neurogolf.local"
  git config --global --add safe.directory "${WORK_DIR}"
  # 任意の git 操作（clone 先以外の作業ツリーも含む）で dubious-ownership を
  # 回避し、git が常に使えるようにする。
  git config --global --add safe.directory '*'
  # gh はトークンを GH_TOKEN 環境変数から自動的に拾う。
  if [[ -n "${GH_TOKEN:-}" ]]; then
    git config --global \
      url."https://x-access-token:${GH_TOKEN}@github.com/".insteadOf \
      "https://github.com/"
  fi
}

# --- kaggle CLI を常に直接使えるようにする ----------------------------------
# kaggle / dvc は backend venv にインストールされる（uv sync）。bare `kaggle`
# `dvc` でも動くよう venv を PATH に通し、さらに env 由来の資格情報を
# ~/.kaggle/kaggle.json にも書き出す（CLI と `ensure_credentials()` の双方が
# env / file のどちらでも解決できるようにする）。値はマスキング対象。
init_kaggle_config() {
  # backend venv を PATH 先頭へ（clone 後に存在）。
  local venv_bin="${WORK_DIR}/backend/.venv/bin"
  if [[ ":${PATH}:" != *":${venv_bin}:"* ]]; then
    export PATH="${venv_bin}:${PATH}"
  fi
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    mkdir -p "${HOME}/.kaggle"
    printf '{"username":"%s","key":"%s"}' \
      "${KAGGLE_USERNAME}" "${KAGGLE_KEY}" >"${HOME}/.kaggle/kaggle.json"
    chmod 600 "${HOME}/.kaggle/kaggle.json"
    log_info "kaggle credentials configured (~/.kaggle/kaggle.json + PATH)"
  else
    log_warn "KAGGLE_USERNAME/KAGGLE_KEY missing; kaggle CLI will be unauthenticated"
  fi
}

# --- 作業ツリーの準備（fresh clone）----------------------------------------
# イメージには .git を含めないため、起動毎に最新 main を clone する。
# これにより git 履歴・remote・DVC ポインタ(*.dvc)・.dvc/config が揃い、
# git self-merge と dvc pull/push の双方が成立する。
prepare_workspace() {
  if [[ -z "${GH_TOKEN:-}" ]]; then
    log_error "GH_TOKEN missing; cannot clone ${GITHUB_REPO}"
    return 1
  fi
  rm -rf "${WORK_DIR}"
  mkdir -p "$(dirname "${WORK_DIR}")"
  local url="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPO}.git"
  if git clone --depth 50 "${url}" "${WORK_DIR}" >/dev/null 2>&1; then
    log_info "cloned ${GITHUB_REPO} into ${WORK_DIR}"
  else
    log_error "git clone failed for ${GITHUB_REPO}"
    return 1
  fi
  # backend 依存をワークツリーで解決（イメージの uv キャッシュを再利用）。
  (cd "${WORK_DIR}/backend" && uv sync --locked --no-dev >/dev/null 2>&1) || \
    log_warn "uv sync in workspace failed; relying on image deps"
  trust_workspace
}

# Claude Code は未信頼ワークスペースの .claude/settings.json の permissions を
# 無視する（headless では trust dialog を出せない）。clone 先を信頼済みとして
# ~/.claude.json に登録し、permissions.allow を有効化する。
trust_workspace() {
  local cfg="${HOME}/.claude.json"
  local existing="{}"
  [[ -f "${cfg}" ]] && existing="$(cat "${cfg}")"
  printf '%s' "${existing}" | python3 -c '
import json, sys, os
work = os.environ["WORK_DIR"]
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
d.setdefault("projects", {}).setdefault(work, {})["hasTrustDialogAccepted"] = True
json.dump(d, sys.stdout)
' >"${cfg}.tmp" 2>/dev/null && mv "${cfg}.tmp" "${cfg}" \
    && log_info "trusted workspace ${WORK_DIR}" \
    || log_warn "failed to mark workspace trusted"
}

# --- dvc 到達性の軽量チェック ----------------------------------------------
# remote 'storage' の S3 URL を .dvc/config から引き、その prefix を
# `aws s3 ls`（task role IAM）で bounded に叩く。remote 未設定 / S3 到達不可なら
# 非ゼロ。`dvc status -c` の全走査（数分）を避けるための軽量版。
dvc_preflight() {
  local url bucket prefix bp
  # 1) remote URL を取得（config 操作のみ・即時）。
  url="$(cd "${WORK_DIR}" && timeout 30 \
    uv --project backend run dvc remote list 2>/dev/null \
    | awk '$1=="storage"{print $2; exit}')"
  if [[ -z "${url}" || "${url}" != s3://* ]]; then
    return 1
  fi
  # 2) s3://bucket/prefix を分解し、IAM で bounded に到達確認。
  bp="${url#s3://}"
  bucket="${bp%%/*}"
  prefix="${bp#*/}"
  timeout 30 aws s3 ls "s3://${bucket}/${prefix}/" \
    --region "${AWS_REGION:-ap-northeast-1}" >/dev/null 2>&1
}

# --- preflight: git / kaggle / dvc が実際に使えるか起動時に検証 -------------
# 「ECS を立ち上げると常に git / kaggle submit / dvc が使える」ことを保証する。
# clone 済み WORK_DIR を前提に各機能を能動的に叩く。
#
# fail-fast の対象は git / kaggle のみ（loop の必須前提）。dvc は warn 止まりに
# する: dvc_preflight は `uv run dvc remote list` + `aws s3 ls` を timeout 付きで
# 叩くが、初回 uv 解決や S3 レスポンスの揺らぎで bounded timeout を超えやすく、
# false-negative で loop 全体を exit 71 → 再起動ループに陥れていた。dvc の実体は
# loop_runner / submit_once が pull/push 時に毎回叩くため、起動時の到達確認が
# 多少滑っても実害は無い（失敗すればそのサイクルが warn を出すだけ）。
preflight_check() {
  local ok=0

  # git: remote 認証が通り fetch 可能か。【致命】
  if git -C "${WORK_DIR}" ls-remote origin HEAD >/dev/null 2>&1; then
    log_info "preflight git: OK (remote reachable + authenticated)"
  else
    log_error "preflight git: FAIL (cannot ls-remote origin — check GH_TOKEN)"
    ok=1
  fi

  # kaggle: CLI が PATH にあり、認証付きでコンペにアクセスできるか。【致命】
  if (cd "${WORK_DIR}/backend" && kaggle competitions list -s neurogolf \
      >/dev/null 2>&1); then
    log_info "preflight kaggle: OK (CLI on PATH + authenticated)"
  else
    log_error "preflight kaggle: FAIL (kaggle CLI missing or unauthenticated)"
    ok=1
  fi

  # dvc: remote 'storage' が設定され、IAM(task role)で S3 に到達できるか。【非致命】
  # 失敗しても ok=1 にせず warn のみ。実 pull/push はサイクル内で再確認される。
  if dvc_preflight; then
    log_info "preflight dvc: OK (remote 'storage' configured + S3 reachable via IAM)"
  else
    log_warn "preflight dvc: WARN (remote unconfigured or S3 unreachable — 起動は継続。pull/push 時に再確認)"
  fi

  return "${ok}"
}

main() {
  init_secrets
  log_init "${MODE}"
  log_info "entrypoint mode=${MODE} starting"

  export WORK_DIR
  init_git_identity
  if ! prepare_workspace; then
    log_error "workspace preparation failed; aborting"
    log_sync
    exit 70
  fi
  # clone 後に kaggle / venv-PATH を整え、git / kaggle / dvc を検証する。
  init_kaggle_config
  if ! preflight_check; then
    log_error "preflight failed: git/kaggle のいずれかが利用不可（dvc は非致命）"
    # loop は壊れた環境で空回りさせず終了（ECS が再起動を試みる）。
    # submit は呼び出し側で kaggle 必須を再判定するため継続する。
    if [[ "${MODE}" == "loop" ]]; then
      log_sync
      exit 71
    fi
  fi

  case "${MODE}" in
    loop)
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
