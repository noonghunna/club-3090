#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-pickers-follow-url — guards #1590: report.sh and spec-sweep.sh pick their container by URL.
#
# #1587 / #1588 put endpoint autodetect, soak-test.sh and concurrency-probe.sh on one rule — with
# URL= set, the container is the one publishing that URL on this host (club_container_for_url),
# and nothing publishing it means no container. These two still picked their own:
#   report.sh     three sections (per-card VRAM, KV calibration, "Active container") each took
#                 the stack default / first of ours, whatever URL said — while its endpoint
#                 resolver honoured URL, so one bundle described two servers;
#   spec-sweep.sh took the first container whose NAME carried the engine family (grep -m1) to read
#                 an arm's acceptance rate — another engine of the same family, or a local one for a
#                 remote URL, landed in the row.
#
# 1. spec-sweep's _arm_container, extracted from the script and run against a stubbed docker.
# 2. report.sh end to end through the report harness: the Active container follows URL; a remote
#    URL names no local container; with no URL the stack-default ladder is unchanged (positive).
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

free_port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'; }

# ---------------------------------------------------------------------------
echo "--- 1. spec-sweep: _arm_container ---"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
P_RIGHT="$(free_port)"; P_WRONG="$(free_port)"
cat > "$tmp/docker" <<DOCK
#!/usr/bin/env bash
[[ "\$1" == "ps" ]] && printf '%s\n' \
  "vllm-qwen38-27b-wrong|0.0.0.0:${P_WRONG}->8000/tcp" \
  "vllm-gemma-4-31b-right|0.0.0.0:${P_RIGHT}->8000/tcp"
exit 0
DOCK
chmod +x "$tmp/docker"
fn="$(sed -n '/^_arm_container() {/,/^}/p' scripts/spec-sweep.sh)"
[[ -n "$fn" ]] || fail "spec-sweep.sh no longer defines _arm_container — the arm's container pick moved; re-point this test"
command grep -q '_arm_container' <(sed -n '/^_measure_vllm_arm() {/,/^}/p' scripts/spec-sweep.sh) \
  || fail "_measure_vllm_arm no longer asks _arm_container for the arm's container"
arm() {   # arm URL [CONTAINER]
  PATH="$tmp:$PATH" URL="$1" CONTAINER="${2:-}" FN="$fn" \
    bash -c 'source scripts/lib/club-containers.sh; eval "$FN"; _arm_container'
}
got="$(arm "http://localhost:${P_RIGHT}")"
[[ "$got" == "vllm-gemma-4-31b-right" ]] || fail "spec-sweep: the URL's container, not the first vllm-*: got '$got'"
pass "the container serving URL, though another vllm-* is listed first"
got="$(arm "http://remote.invalid:${P_RIGHT}")"
[[ -z "$got" ]] || fail "spec-sweep: a remote URL must give no container (an honest '—'): got '$got'"
pass "a remote URL → no container"
got="$(arm "http://localhost:${P_RIGHT}" mine)"
[[ "$got" == "mine" ]] || fail "spec-sweep: an explicit CONTAINER= wins: got '$got'"
got="$(arm "http://localhost:${P_RIGHT}" none)"
[[ -z "$got" ]] || fail "spec-sweep: CONTAINER=none means no container: got '$got'"
pass "explicit CONTAINER= wins; CONTAINER=none → none"

# ---------------------------------------------------------------------------
echo "--- 2. report.sh: one container, chosen by URL ---"
# shellcheck source=fixtures/report-harness/report-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/report-harness/report-env.sh"
report_env_init "$ROOT_DIR"
trap 'report_env_cleanup; rm -rf "$tmp"' EXIT
report_stub_start                      # a live endpoint on 127.0.0.1:<port> (REPORT_STUB_URL)
STUB_PORT="${REPORT_STUB_URL##*:}"
OTHER_PORT="$(free_port)"
# The stack default (vllm-qwen36-*) is listed FIRST and serves another port — the old ladder's pick.
report_docker_stub <<STUB
#!/usr/bin/env bash
NAMES=("vllm-qwen36-27b-default" "sglang-qwen38-27b-other")
PORTS=("0.0.0.0:${OTHER_PORT}->8000/tcp" "0.0.0.0:${STUB_PORT}->30000/tcp")
case "\${1:-}" in
  info) exit 0 ;;
  ps)
    filter=""; fmt=""
    while [[ \$# -gt 0 ]]; do
      case "\$1" in --filter) filter="\${2#name=}"; filter="\${filter#^}"; filter="\${filter%\\$}"; shift 2 ;;
                    --format) fmt="\$2"; shift 2 ;; *) shift ;; esac
    done
    for i in 0 1; do
      [[ -n "\$filter" && "\${NAMES[\$i]}" != *"\$filter"* ]] && continue
      case "\$fmt" in
        *Ports*Image*) echo "\${NAMES[\$i]}|\${PORTS[\$i]}|img" ;;
        *Ports*)       if [[ "\$fmt" == *Names* ]]; then echo "\${NAMES[\$i]}|\${PORTS[\$i]}"; else echo "\${PORTS[\$i]}"; fi ;;
        *Status*)      echo "Up 5 minutes" ;;
        *Image*)       echo "example/engine:1" ;;
        *)             echo "\${NAMES[\$i]}" ;;
      esac
    done
    exit 0 ;;
  *) exit 1 ;;
esac
STUB

report_run --no-redact                 # URL = the stub, which the SECOND container publishes
name_line="$(command grep -m1 -- '- \*\*Name:\*\*' <<<"$REPORT_OUT" || true)"
[[ "$name_line" == *'`sglang-qwen38-27b-other`'* ]] \
  || { printf '%s\n' "$REPORT_OUT" | command grep -A4 -i 'active container' >&2; fail "report: Active container should be the one serving URL — got: ${name_line:-<none>}"; }
pass "URL → the container publishing it (not the stack default listed first)"

REPORT_STUB_URL="http://remote.invalid:${STUB_PORT}" report_run --no-redact
command grep -q "No local container serves the endpoint (CONTAINER=none)" <<<"$REPORT_OUT" \
  || { printf '%s\n' "$REPORT_OUT" | command grep -A4 -i 'active container' >&2; fail "report: a remote URL should name no local container"; }
if command grep -q -- '- \*\*Name:\*\*' <<<"$REPORT_OUT"; then fail "report: a remote URL was described by a local container"; fi
pass "a remote URL → no local container, said so"

report_run_unprobeable --no-redact     # no URL at all: the stack-default ladder (positive control)
name_line="$(command grep -m1 -- '- \*\*Name:\*\*' <<<"$REPORT_OUT" || true)"
[[ "$name_line" == *'`vllm-qwen36-27b-default`'* ]] \
  || fail "report: with no URL the stack default should still be picked — got: ${name_line:-<none>}"
pass "no URL → the stack default, as before"

echo "test-pickers-follow-url: ok"
