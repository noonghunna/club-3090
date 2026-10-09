#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-sampling-stamp — guards #1579: every Quality: line says which
# sampler produced it, and a run against URL= is never described by a local
# container that does not serve that URL.
#
#   1. scripts/lib/quality_line.py — the sampler stamp per run shape, from
#      fixture results JSONs (no server):
#        server · explicit overrides · max_tokens-only overrides (NOT explicit —
#        the negative control for the default 4,096 budget) · extra-body ·
#        --thinking-sampler · canonical OFF / ON / mixed · pre-2026-05-24 JSON.
#   2. container_serves_url (scripts/lib/listen-scope.sh) against a fake
#      `docker port` / `hostname -I`, with a positive leg for every negative.
#   3. quality-test.sh end to end with a mocked docker: an auto-detected
#      container that does not publish URL's port is dropped (no --run-meta
#      reaches benchlocal), and the same container IS used when URL is its own
#      port (positive control) or when CONTAINER= names it explicitly.
#      Asserted on run_context's own `--run-meta engine=…`, not on any
#      --run-meta: the wrapper always records `budgets=` that way too.
#
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EMIT="scripts/lib/quality_line.py"
LIB="scripts/lib/listen-scope.sh"
WRAPPER="scripts/quality-test.sh"

fail() { echo "✗ $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }

tmp="$(mktemp -d)"
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp"
  rm -f "$before_list" "$after_list"
}
trap cleanup EXIT

python3 -m py_compile "$EMIT" || fail "py_compile: $EMIT"
bash -n "$LIB" || fail "bash -n: $LIB"

PACKS='"packs":[{"pack_id":"toolcall-15","status":"ok","passed":14,"total":15,"score":0.933,"version":"1.0.1"}]'

# stamp JSON_BODY [BENCHLOCAL_ARG ...] → the sampling=… stamp, or "<none>"
stamp() {
  local body="$1"; shift
  printf '{%s,%s}\n' "$body" "$PACKS" > "$tmp/r.json"
  local line
  line="$(python3 "$EMIT" "$tmp/r.json" --quick "$@")" || fail "emitter exited non-zero on {$body}"
  [[ "$line" == "Quality:   toolcall-15 14/15 (93%) ("* ]] || fail "unexpected Quality line: $line"
  local s
  s="$(sed -nE 's/.*, (sampling=[^,]*),.*/\1/p' <<<"$line")"
  printf '%s\n' "${s:-<none>}"
}

expect() {
  local label="$1" want="$2" got="$3"
  [[ "$got" == "$want" ]] || fail "$label: expected '$want', got '$got'"
  pass "$label → $got"
}

# ---------------------------------------------------------------------------
echo "--- 1. sampler stamp per run shape ---"
expect "server" "sampling=server" \
  "$(stamp '"thinking_mode":"force-off","sampling_source":"server"')"
expect "server with the default budget" "sampling=server" \
  "$(stamp '"thinking_mode":"force-on","sampling_source":"server","sampling_overrides":{"max_tokens":4096}')"

# The pre-#1579 mislabel: a 2026-09-23 run with the Qwen3.8 card's non-thinking
# sampler printed exactly the line a canonical run prints.
expect "explicit overrides" "sampling=explicit temperature=0.7/top_p=0.8/top_k=20/min_p=0" \
  "$(stamp '"thinking_mode":"force-off","sampling_overrides":{"min_p":0.0,"temperature":0.7,"top_k":20,"top_p":0.8}')"
expect "explicit drops max_tokens" "sampling=explicit temperature=1/top_p=0.95/top_k=20/min_p=0" \
  "$(stamp '"thinking_mode":"force-off","sampling_overrides":{"max_tokens":4096,"min_p":0.0,"temperature":1.0,"top_k":20,"top_p":0.95}')"

# NEGATIVE CONTROL: every default wrapper run since 2026-10-02 carries
# max_tokens=4096 in sampling_overrides. It is a budget, not a sampler.
expect "max_tokens-only overrides, OFF" "sampling=pack:greedy" \
  "$(stamp '"thinking_mode":"force-off","sampling_overrides":{"max_tokens":4096}')"
expect "max_tokens-only overrides, ON" "sampling=pack:thinking" \
  "$(stamp '"thinking_mode":"force-on","sampling_overrides":{"max_tokens":4096}')"

expect "canonical OFF" "sampling=pack:greedy" "$(stamp '"thinking_mode":"force-off"')"
expect "canonical ON" "sampling=pack:thinking" "$(stamp '"thinking_mode":"force-on"')"
expect "canonical mixed" "sampling=pack" "$(stamp '"thinking_mode":"pack-defaults"')"

# Flags benchlocal does not record in the results JSON come from its argv.
expect "--thinking-sampler, ON" "sampling=pack+thinking-sampler temperature=0.6/top_p=0.95/top_k=20" \
  "$(stamp '"thinking_mode":"force-on"' run --thinking-sampler '{"temperature":0.6,"top_p":0.95,"top_k":20}')"
expect "--thinking-sampler ignored on an OFF leg" "sampling=pack:greedy" \
  "$(stamp '"thinking_mode":"force-off"' run --thinking-sampler '{"temperature":0.6}')"
expect "--extra-body sampler keys" "sampling=pack+extra-body temperature=0.3/presence_penalty=1.5" \
  "$(stamp '"thinking_mode":"force-off"' run --extra-body '{"presence_penalty":1.5,"temperature":0.3}')"
expect "--extra-body=… form, last one wins" "sampling=pack+extra-body top_k=40" \
  "$(stamp '"thinking_mode":"force-off"' run --extra-body '{"temperature":0.3}' "--extra-body={\"top_k\":40}")"
expect "--extra-body without sampler keys (the thinking-budget one)" "sampling=pack:thinking" \
  "$(stamp '"thinking_mode":"force-on"' run --extra-body '{"thinking_token_budget":8192}')"
expect "extra_body recorded in the JSON wins over argv" "sampling=pack+extra-body temperature=0.2" \
  "$(stamp '"thinking_mode":"force-off","extra_body":{"temperature":0.2}' run --extra-body '{"temperature":0.9}')"

# Before 2026-05-24 benchlocal recorded no thinking_mode, and a canonical run is
# then indistinguishable from an unrecorded one — say nothing rather than guess.
expect "pre-2026-05-24 JSON" "<none>" "$(stamp '"schema_version":"1"')"

# The suffix is comma-joined: a multi-key stamp must not split it.
printf '{"thinking_mode":"force-off","sampling_overrides":{"temperature":0.7,"top_p":0.8},"run_meta":{"tp":"2"},%s}\n' "$PACKS" > "$tmp/r.json"
line="$(python3 "$EMIT" "$tmp/r.json" --full)"
[[ "$line" == *"(--full, thinking OFF, sampling=explicit temperature=0.7/top_p=0.8, tp=2, packs tc1.0.1, "* ]] \
  || fail "suffix order / comma-free stamp: $line"
pass "suffix order intact: ${line#*(}"

# ---------------------------------------------------------------------------
echo "--- 2. container_serves_url ---"
fake="$tmp/bin"; mkdir -p "$fake"
cat > "$fake/docker" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "port" ]]; then
  case "$2" in
    srv) printf '8000/tcp -> 0.0.0.0:8020\n8000/tcp -> [::]:8020\n' ;;
    lo)  printf '8000/tcp -> 127.0.0.1:8031\n' ;;
  esac
fi
exit 0
EOF
cat > "$fake/hostname" <<'EOF'
#!/usr/bin/env bash
echo "192.168.1.5 172.17.0.1 "
EOF
chmod +x "$fake/docker" "$fake/hostname"

serves() {
  PATH="$fake:$PATH" bash -c 'source "$1"; container_serves_url "$2" "$3"' _ "$LIB" "$1" "$2"
}
yes_() { serves "$1" "$2" || fail "expected '$1' to serve $2"; pass "serves:  $1 ← $2"; }
no_()  { if serves "$1" "$2"; then fail "expected '$1' NOT to serve $2"; fi; pass "refuses: $1 ← $2"; }

yes_ srv http://localhost:8020
no_  srv http://localhost:8021
yes_ srv http://127.0.0.1:8020/v1
yes_ srv "http://[::1]:8020"
yes_ srv http://192.168.1.5:8020        # this host's own LAN address
no_  srv http://10.9.9.9:8020           # same port, another machine
yes_ srv http://172.17.0.1:8020         # the bridge gateway is this host too
no_  srv http://localhost               # no port → 80
yes_ lo  http://localhost:8031
no_  nobody http://localhost:8020       # a container that publishes nothing
no_  "" http://localhost:8020

# ---------------------------------------------------------------------------
echo "--- 3. quality-test.sh drops an auto-detected container that does not serve URL ---"
wbin="$tmp/wbin"; mkdir -p "$wbin"
cat > "$wbin/docker" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  ps)      printf 'vllm-mock|0.0.0.0:8020->8000/tcp\n' ;;
  port)    [[ "$2" == "vllm-mock" ]] && printf '8000/tcp -> 0.0.0.0:8020\n' ;;
  inspect) [[ "${2:-}" == "vllm-mock" || "${3:-}" == "vllm-mock" ]] || exit 1
           if [[ "${2:-}" == "--format" ]]; then echo "vllm/vllm-openai:v0.0.0-mock"
           else printf '[{"Name":"/vllm-mock","Config":{"Image":"vllm/vllm-openai:v0.0.0-mock","Cmd":[]},"State":{"RestartCount":0},"HostConfig":{}}]\n'; fi ;;
  logs)    printf "INFO non-default args: {'tensor_parallel_size': 2}\n" ;;
esac
exit 0
EOF
cat > "$wbin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "$wbin/curl" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    */v1/models) printf '{"data":[{"id":"mock-model"}]}'; exit 0 ;;
  esac
done
exit 0
EOF
cat > "$wbin/benchlocal-cli" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [[ "$a" == "--help" ]] && { echo "--run-meta --reasoning-effort --progress"; exit 0; }; done
printf '%s\n' "$*" >> "$MOCK_ARGV"
json_out=""
while [[ $# -gt 0 ]]; do
  case "$1" in --save-json) json_out="$2"; shift 2 ;; *) shift ;; esac
done
[[ -n "$json_out" ]] && { mkdir -p "$(dirname "$json_out")"; printf '{"thinking_mode":"force-off","sampling_overrides":{"max_tokens":4096},"packs":[{"pack_id":"toolcall-15","status":"ok","passed":14,"total":15,"score":0.933,"version":"1.0.1"}]}\n' > "$json_out"; }
exit 0
EOF
chmod +x "$wbin"/*

wrun() {   # wrun LOG [VAR=value ...] — the wrapper with only the mocks on PATH's front
  local log="$1"; shift
  : > "$tmp/argv"
  env PATH="$wbin:$PATH" MOCK_ARGV="$tmp/argv" MODEL=mock-model "$@" \
    bash "$WRAPPER" --quick --pack toolcall-15 > "$log" 2>&1 || { cat "$log" >&2; fail "wrapper exited non-zero ($*)"; }
}

# (a) URL= another machine, container auto-detected → dropped.
wrun "$tmp/a.log" URL=http://10.9.9.9:8020
command grep -q "auto-detected container 'vllm-mock' does not serve http://10.9.9.9:8020" "$tmp/a.log" \
  || { cat "$tmp/a.log" >&2; fail "(a) no 'does not serve' notice for a remote URL"; }
if command grep -q -- "--run-meta engine=" "$tmp/argv"; then fail "(a) the local container's rig reached benchlocal for a remote URL"; fi
command grep -q "^Quality:   .*sampling=pack:greedy" "$tmp/a.log" || fail "(a) Quality line lost its stamp"
pass "(a) remote URL: local container ignored, no engine/rig recorded"

# (b) POSITIVE CONTROL: no URL= → the container's own port → kept and read.
wrun "$tmp/b.log"
if command grep -q "does not serve" "$tmp/b.log"; then cat "$tmp/b.log" >&2; fail "(b) the serving container was dropped"; fi
command grep -q -- "--run-meta engine=vllm" "$tmp/argv" || { cat "$tmp/b.log" "$tmp/argv" >&2; fail "(b) run_context never ran for the serving container — (a) proves nothing"; }
pass "(b) autodetected URL: container kept, engine/rig recorded"

# (c) CONTAINER= named explicitly → trusted even for a URL it does not publish.
wrun "$tmp/c.log" URL=http://10.9.9.9:8020 CONTAINER=vllm-mock
if command grep -q "does not serve" "$tmp/c.log"; then fail "(c) an explicit CONTAINER= was second-guessed"; fi
command grep -q -- "--run-meta engine=vllm" "$tmp/argv" || fail "(c) explicit CONTAINER= was not read"
pass "(c) explicit CONTAINER=: trusted"

echo "test-quality-sampling-stamp: ok"
