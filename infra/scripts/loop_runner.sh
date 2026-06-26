#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# loop_runner.sh — 常駐の自律精度改善ループ本体。
#
# 1 サイクル（既定 3 時間）ごとに:
#   1. main を最新化し、auto/<timestamp> ブランチを切る
#   2. Claude Code CLI を headless 実行して改善を実装
#      （MIGRATION.md の残タスクからソルバ拡充 / cost 精緻化を 1 つ選ぶ）
#   3. 変更があれば commit → PR 作成（PR テンプレ準拠）
#   4. CI（ci.yml: ruff/mypy/pytest）green を待つ
#   5. green ならセルフマージ（squash, ブランチ削除）。red ならマージせず記録
#   6. 残り時間は継続作業（精度測定 / 改善案 / Web 調査プランニング）を回す
#
# クラッシュ耐性: 各サイクルは独立。例外を捕捉し S3 に記録してループを継続する。
# ---------------------------------------------------------------------------
set -uo pipefail

readonly REPO_DIR="/app"
readonly LOOP_DIR="${REPO_DIR}/loop"
readonly PROMPTS_DIR="${LOOP_DIR}/prompts"
readonly GITHUB_REPO="${GITHUB_REPO:-YuriNakayama/neurogolf}"
readonly BRANCH_INTERVAL="${BRANCH_INTERVAL_SECONDS:-10800}"

# Claude headless の安全弁。
readonly CLAUDE_MAX_TURNS="${CLAUDE_MAX_TURNS:-60}"
readonly CLAUDE_IMPL_TIMEOUT="${CLAUDE_IMPL_TIMEOUT:-3600}"   # 実装フェーズ上限 1h
readonly CLAUDE_PLAN_TIMEOUT="${CLAUDE_PLAN_TIMEOUT:-1200}"   # 継続作業 1 回 20m
readonly CLAUDE_ALLOWED_TOOLS="Read,Edit,Write,Bash,Grep,Glob"

cd "${REPO_DIR}" || exit 1

# --- Claude headless 実行ラッパ ---------------------------------------------
# プロンプトファイルを渡し、claude -p を制限付きで実行する。
run_claude() {
  local label="$1" prompt_file="$2" timeout_s="$3" workdir="$4"
  if [[ ! -f "${prompt_file}" ]]; then
    log_error "prompt file not found: ${prompt_file}"
    return 1
  fi
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    log_error "ANTHROPIC_API_KEY missing; cannot run claude"
    return 1
  fi
  log_info "claude[${label}] start (timeout=${timeout_s}s, max-turns=${CLAUDE_MAX_TURNS})"
  timeout "${timeout_s}" bash -c '
    cd "$1" && claude -p "$(cat "$2")" \
      --max-turns "$3" \
      --allowedTools "$4" \
      --permission-mode acceptEdits \
      --output-format stream-json --verbose
  ' _ "${workdir}" "${prompt_file}" "${CLAUDE_MAX_TURNS}" "${CLAUDE_ALLOWED_TOOLS}" \
    2>&1 | mask_stream
  local rc=${PIPESTATUS[0]}
  log_info "claude[${label}] done rc=${rc}"
  log_sync
  return "${rc}"
}

# --- ローカル検証（CI を待つ前の早期失敗検出） ------------------------------
local_checks() {
  log_info "running local lint + tests"
  run_logged bash -c "cd '${REPO_DIR}' && ./dev/lint && ./dev/test-bot"
}

# --- CI ゲート（H-1 対策） ---------------------------------------------------
# `gh pr checks --watch` は「チェックが 0 件」のとき success で即時返るため、
# それだけに依存するとパスフィルタで CI が起動しない PR（infra/ や .claude/ のみ
# 変更等）がノーゲートでセルフマージされる。これを防ぐため:
#   1. --watch で完了を待つ（rc は判定に使わない）
#   2. --json で実際のチェック結果を取得
#   3. チェックが 1 件以上あり、かつ pass/skipping 以外（fail/pending/cancel）が
#      ゼロの場合のみ green と判定する
# 必須チェックが 1 件も走っていなければ green とみなさず、マージしない。
ci_passed() {
  local pr_url="$1"

  # 完了待ち（タイムアウトや 0 件 success の rc は無視して結果を JSON で見る）。
  run_logged gh pr checks "${pr_url}" --repo "${GITHUB_REPO}" \
    --watch --interval 30 || true

  local checks_json total bad passed
  checks_json="$(gh pr checks "${pr_url}" --repo "${GITHUB_REPO}" \
    --json name,state 2>/dev/null || echo '[]')"

  total="$(printf '%s' "${checks_json}" | jq 'length')"
  if [[ "${total}" -eq 0 ]]; then
    log_warn "no CI checks ran for ${pr_url}; refusing to merge (ungated)"
    return 1
  fi

  # SUCCESS / SKIPPED / NEUTRAL を許容、それ以外（FAILURE/PENDING/...）を不可。
  bad="$(printf '%s' "${checks_json}" \
    | jq '[.[] | select(.state|test("SUCCESS|SKIPPED|NEUTRAL")|not)] | length')"
  passed="$(printf '%s' "${checks_json}" \
    | jq '[.[] | select(.state=="SUCCESS")] | length')"

  if [[ "${bad}" -eq 0 && "${passed}" -ge 1 ]]; then
    log_info "CI gate ok: ${passed}/${total} success, 0 failing/pending"
    return 0
  fi
  log_warn "CI gate failed: ${passed}/${total} success, ${bad} not-passing"
  return 1
}

# --- 1 サイクル -------------------------------------------------------------
run_cycle() {
  local cycle_start branch ts
  cycle_start="$(date -u +%s)"
  ts="$(date -u +%Y%m%d-%H%M%S)"
  branch="auto/${ts}"

  log_info "===== cycle start branch=${branch} ====="

  # main 最新化 + ブランチ作成
  if ! run_logged git -C "${REPO_DIR}" fetch origin main; then
    log_error "git fetch failed; skip cycle"
    return 1
  fi
  run_logged git -C "${REPO_DIR}" checkout -B main origin/main
  run_logged git -C "${REPO_DIR}" checkout -b "${branch}"

  # 実装フェーズ
  run_claude "implement" "${PROMPTS_DIR}/implement.prompt" "${CLAUDE_IMPL_TIMEOUT}" "${REPO_DIR}"

  # 変更が無ければサイクル終了（空 PR を作らない）
  if git -C "${REPO_DIR}" diff --quiet && git -C "${REPO_DIR}" diff --cached --quiet; then
    log_info "no changes produced; ending cycle without PR"
    run_logged git -C "${REPO_DIR}" checkout main
    return 0
  fi

  # ローカル検証（落ちても PR は作るが、CI が最終ゲート）
  if ! local_checks; then
    log_warn "local checks failed; PR will rely on CI gate"
  fi

  # commit + push + PR
  run_logged git -C "${REPO_DIR}" add -A
  run_logged git -C "${REPO_DIR}" commit -m ":robot: auto: 精度改善ループ ${ts}

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  if ! run_logged git -C "${REPO_DIR}" push -u origin "${branch}"; then
    log_error "git push failed; abort cycle"
    run_logged git -C "${REPO_DIR}" checkout main
    return 1
  fi

  local pr_url
  pr_url="$(gh pr create \
    --repo "${GITHUB_REPO}" \
    --base main --head "${branch}" \
    --title ":robot: auto: 精度改善ループ ${ts}" \
    --body-file "${PROMPTS_DIR}/pr_body.prompt" 2> >(mask_stream))"
  if [[ -z "${pr_url}" ]]; then
    log_error "gh pr create failed; abort cycle"
    run_logged git -C "${REPO_DIR}" checkout main
    return 1
  fi
  log_info "PR created: ${pr_url}"
  log_sync

  # CI ゲート（実際にチェックが走り全て成功した場合のみセルフマージ）。
  if ci_passed "${pr_url}"; then
    log_info "CI gate passed; self-merging ${pr_url}"
    if run_logged gh pr merge "${pr_url}" --repo "${GITHUB_REPO}" \
        --squash --delete-branch; then
      log_info "merged ${pr_url}"
    else
      log_error "merge failed for ${pr_url}"
    fi
  else
    log_warn "CI gate not passed; leaving PR open for next cycle: ${pr_url}"
  fi

  run_logged git -C "${REPO_DIR}" checkout main
  run_logged git -C "${REPO_DIR}" fetch origin main
  run_logged git -C "${REPO_DIR}" reset --hard origin/main
  log_sync

  # --- 残り時間は継続作業（精度測定 / 改善案 / Web 調査プランニング） -------
  local now elapsed remaining
  now="$(date -u +%s)"
  elapsed=$((now - cycle_start))
  remaining=$((BRANCH_INTERVAL - elapsed))
  log_info "cycle impl phase took ${elapsed}s; ${remaining}s for continuous work"

  while [[ ${remaining} -gt ${CLAUDE_PLAN_TIMEOUT} ]]; do
    run_claude "planning" "${PROMPTS_DIR}/planning.prompt" "${CLAUDE_PLAN_TIMEOUT}" "${REPO_DIR}"
    # 継続作業の生成物（メモ/改善案）は次サイクルの実装入力になる。差分は捨てる
    # （PR はサイクル先頭でのみ作る）。main を汚さないようリセット。
    run_logged git -C "${REPO_DIR}" reset --hard origin/main
    run_logged git -C "${REPO_DIR}" clean -fd
    now="$(date -u +%s)"
    remaining=$((BRANCH_INTERVAL - (now - cycle_start)))
  done

  if [[ ${remaining} -gt 0 ]]; then
    log_info "sleeping remaining ${remaining}s until next cycle"
    sleep "${remaining}"
  fi
  log_info "===== cycle end ====="
}

main() {
  log_info "loop_runner starting (interval=${BRANCH_INTERVAL}s repo=${GITHUB_REPO})"
  while true; do
    run_cycle || log_error "cycle failed; continuing after backoff"
    # サイクルが早期 return した場合の最低待機（暴走防止）。
    sleep 30
  done
}

main "$@"
