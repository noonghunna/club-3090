#!/usr/bin/env bash
# Test for switch.sh down_running() closed-world teardown (orphan cleanup).
#
# Contract: switch.sh tears down ONLY containers it manages — i.e. whose name
# is in the registry-derived VARIANT_CONTAINER set — via each container's own
# compose file WITH --remove-orphans. Anything not in that set (the services
# stack, estate instances with distinct ESTATE_CONTAINER names, unrelated user
# containers) is left strictly alone.
#
# Validates:
#   1. A managed vLLM container is torn down.
#   2. A managed ik-llama container is torn down (the bug: old RUNNING_PATTERN
#      ^(vllm-|llama-cpp-) missed ik-llama-/sglang-/beellama-).
#   3. An unrelated container (not in the managed set) is left alone.
#   4. A services-stack container (litellm) is left alone.
#   5. Teardown passes --remove-orphans.
#   6. When only unmanaged containers run, nothing is torn down and the
#      "no club-3090 container running" path is taken.
#
# Harness: extract down_running() via sed, source it, inject a VARIANT_CONTAINER
# managed set + a mock `docker` on PATH that records compose-down / stop calls.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
assert_contains() {
  local haystack="$1" needle="$2" label="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label}: expected output to CONTAIN: ${needle}" >&2
    echo "--- teardown log ---" >&2
    printf '%s\n' "$haystack" >&2
    FAIL=$((FAIL + 1))
  fi
}
assert_not_contains() {
  local haystack="$1" needle="$2" label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label}: expected output to NOT contain: ${needle}" >&2
    echo "--- teardown log ---" >&2
    printf '%s\n' "$haystack" >&2
    FAIL=$((FAIL + 1))
  fi
}

# --- Extract the function under test (avoids running switch.sh's main) --------
HELPERS_FILE="$(mktemp --suffix=.sh)"
sed -n '/^down_running()/,/^}/p' scripts/switch.sh > "$HELPERS_FILE"

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir" "$HELPERS_FILE"; }
trap cleanup EXIT

# --- Mock `docker`: records compose-down / stop to $DOCKER_MOCK_LOG -----------
# Labels are returned non-empty so the compose-down branch is taken; the
# config_files label embeds the container name so the log identifies which
# container was brought down.
cat > "${tmp_dir}/docker" <<'EOF'
#!/usr/bin/env bash
sub="$1"; shift || true
case "$sub" in
  ps)
    for n in $DOCKER_MOCK_RUNNING; do printf '%s\n' "$n"; done
    ;;
  inspect)
    fmt=""; cname=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --format) fmt="$2"; shift 2 || true ;;
        *) cname="$1"; shift ;;
      esac
    done
    if [[ "$fmt" == *working_dir* ]]; then
      printf '%s\n' "$DOCKER_MOCK_WORKDIR"
    elif [[ "$fmt" == *config_files* ]]; then
      printf 'compose-%s.yml\n' "$cname"
    fi
    ;;
  stop)
    printf 'STOP %s\n' "$1" >> "$DOCKER_MOCK_LOG"
    ;;
  compose)
    printf 'COMPOSE %s\n' "$*" >> "$DOCKER_MOCK_LOG"
    ;;
  *) : ;;
esac
exit 0
EOF
chmod +x "${tmp_dir}/docker"

export DOCKER_MOCK_LOG="${tmp_dir}/calls.log"
export DOCKER_MOCK_WORKDIR="${tmp_dir}"   # real dir so (cd "$lbl_dir") succeeds
export PATH="${tmp_dir}:$PATH"
export COMPOSE_BIN="docker compose"

# Managed set, as derive_switch_variant_tables would populate it (default
# container names, ESTATE_CONTAINER unwrapped). Spans all engine prefixes.
declare -A VARIANT_CONTAINER=(
  ["vllm/default"]="vllm-qwen36-27b"
  ["vllm/dual"]="vllm-qwen36-27b-dual"
  ["ik-llama/iq4ks-mtp"]="ik-llama-qwen36-27b"
  ["beellama/dflash"]="beellama-qwen36-27b"
  ["llamacpp/default"]="llama-cpp-qwen36-27b"
)

# shellcheck source=/dev/null
source "$HELPERS_FILE"

# --- Scenario 1: mixed managed + unmanaged containers running -----------------
: > "$DOCKER_MOCK_LOG"
export DOCKER_MOCK_RUNNING="vllm-qwen36-27b unrelated-db ik-llama-qwen36-27b litellm"
( set +u; down_running ) >/dev/null 2>&1 || true
log="$(cat "$DOCKER_MOCK_LOG")"

assert_contains     "$log" "compose-vllm-qwen36-27b.yml"     "managed vLLM container torn down"
assert_contains     "$log" "compose-ik-llama-qwen36-27b.yml" "managed ik-llama container torn down (the bug)"
assert_not_contains "$log" "unrelated-db"                    "unrelated container left alone"
assert_not_contains "$log" "litellm"                         "services-stack container left alone"
assert_contains     "$log" "--remove-orphans"                "teardown passes --remove-orphans"

# --- Scenario 2: only unmanaged containers running ----------------------------
: > "$DOCKER_MOCK_LOG"
export DOCKER_MOCK_RUNNING="litellm openwebui some-db"
out="$( set +u; down_running 2>&1 )" || true
log="$(cat "$DOCKER_MOCK_LOG")"

assert_not_contains "$log" "COMPOSE"                          "nothing torn down when only unmanaged running"
assert_not_contains "$log" "STOP"                             "nothing stopped when only unmanaged running"
assert_contains     "$out" "no club-3090 container running"   "reports no managed container running"

# --- Summary ------------------------------------------------------------------
echo "----------------------------------------"
echo "PASS: $PASS   FAIL: $FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
echo "OK: switch.sh closed-world teardown"
