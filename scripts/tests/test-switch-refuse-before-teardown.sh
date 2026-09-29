#!/usr/bin/env bash
# Gate: a launch that switch.sh REFUSES must leave the running slug serving.
#
# switch.sh used to tear the running slug down FIRST and refuse afterwards, so a
# refused launch left the rig serving nothing. Hit twice on the reference rig
# (2026-09-27): once for an experimental slug run without --force, once from a
# worktree whose model files weren't on the host (no MODEL_DIR). A mistyped slug
# did the same. check_variant now runs every refusal that doesn't need the old
# slug's GPUs or RAM back BEFORE down_running.
#
# Drives the REAL switch.sh end to end against a `docker` shim that reports one
# running club container and logs every teardown and `compose up`. The positive
# control (a launch that passes) must still tear the old slug down, and before
# the new one comes up — so a switch.sh that never tears anything down fails it.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUTF8="${PYTHONUTF8:-1}"

fail=0
ok()  { echo "  ok   — $*"; }
bad() { echo "  FAIL — $*" >&2; fail=1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
# a launch writes the slug-label override (data dir) and cache dirs: never into your real ones
export CLUB3090_DATA_DIR="$T/data" CLUB3090_CACHE_DIR="$T/cache"
mkdir -p "$T/bin" "$T/empty-models" "$T/running"

cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
sub="${1:-}"; shift || true
case "$sub" in
  ps) printf '%s\n' "$MOCK_RUNNING" ;;
  inspect)
    fmt=""
    while [[ $# -gt 0 ]]; do case "$1" in --format) fmt="$2"; shift 2 ;; *) shift ;; esac; done
    if [[ "$fmt" == *working_dir* ]]; then printf '%s\n' "$MOCK_WORKDIR"
    elif [[ "$fmt" == *config_files* ]]; then printf 'running.yml\n'; fi ;;
  stop|rm|kill) printf 'TEARDOWN %s %s\n' "$sub" "$*" >> "$MOCK_LOG" ;;
  compose)
    case " $* " in
      *" down "*) printf 'TEARDOWN compose %s\n' "$*" >> "$MOCK_LOG" ;;
      *" up "*)   printf 'UP compose %s\n' "$*" >> "$MOCK_LOG" ;;
    esac ;;
esac
exit 0
EOF
chmod +x "$T/bin/docker"

export MOCK_LOG="$T/calls.log"
export MOCK_WORKDIR="$T/running"
# A container switch.sh manages (it is in the registry), so down_running takes it down.
export MOCK_RUNNING="vllm-qwen38-27b-dual-fast"
# No network, no /dev/shm cleanup, no OWUI / gateway sync — only the launch path.
export PREFLIGHT_NO_FETCH=1 CLUB3090_SHM_CLEANUP=0
unset FORCE

# --- refusals: each must exit non-zero and tear NOTHING down ------------------
refused() {   # <label> <expected message> -- <env/args…>
  local label="$1" want="$2"; shift 3
  : > "$MOCK_LOG"
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if (( rc == 0 )); then
    bad "$label: launch was NOT refused (rc=0)"
  elif command grep -q TEARDOWN "$MOCK_LOG"; then
    bad "$label: refused, but tore the running slug down first: $(tr '\n' ' ' < "$MOCK_LOG")"
  elif [[ "$out" != *"$want"* ]]; then
    bad "$label: refused for another reason (wanted '$want'): $(printf '%s' "$out" | tail -3)"
  else
    ok "$label: refused, running slug left up"
  fi
}

refused "experimental slug without --force" "Re-run with --force" -- \
  env PATH="$T/bin:$PATH" bash scripts/switch.sh --no-wait --no-owui vllm/qwen38-27b-dual-fast
refused "unknown slug" "unknown variant" -- \
  env PATH="$T/bin:$PATH" bash scripts/switch.sh --no-wait --no-owui vllm/no-such-slug-here
refused "model files missing on this host" "[preflight]" -- \
  env PATH="$T/bin:$PATH" MODEL_DIR="$T/empty-models" bash scripts/switch.sh --no-wait --no-owui --force vllm/qwen38-27b-dual-fast

# --- positive control: a launch that passes still replaces the running slug ---
: > "$MOCK_LOG"
out="$(env PATH="$T/bin:$PATH" PREFLIGHT_NO_COMPOSE_DEPS=1 bash scripts/switch.sh --no-wait --no-owui --force sgl/qwen38-27b-dual-fast 2>&1)"; rc=$?
down_line="$(command grep -n '^TEARDOWN' "$MOCK_LOG" | head -1 | cut -d: -f1)"
up_line="$(command grep -n '^UP' "$MOCK_LOG" | head -1 | cut -d: -f1)"
if (( rc == 0 )) && [[ -n "$down_line" && -n "$up_line" ]] && (( down_line < up_line )); then
  ok "a launch that passes tears the running slug down, then brings the new one up"
else
  bad "positive control: rc=$rc, calls: $(tr '\n' ' ' < "$MOCK_LOG")— $(printf '%s' "$out" | tail -3)"
fi

[[ $fail -eq 0 ]] && echo "test-switch-refuse-before-teardown: ok" || echo "test-switch-refuse-before-teardown: FAIL"
exit $fail
