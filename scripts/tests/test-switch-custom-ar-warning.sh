#!/usr/bin/env bash
# test-switch-custom-ar-warning — switch.sh warns after boot when the engine's custom
# all-reduce setup failed and it fell back to NCCL (club-3090#1462), and stays silent
# otherwise. Runs the real function against a docker shim; the decision itself is
# the shared classifier's (test-p2p-state.sh covers its rules).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
sed -n '/^warn_if_custom_ar_setup_failed()/,/^}/p' scripts/switch.sh > "$T/fn.sh"
[[ -s "$T/fn.sh" ]] || { echo "✗ warn_if_custom_ar_setup_failed not found in switch.sh" >&2; exit 1; }
command grep -qE '^\[\[ \$WAIT -eq 1 \]\] && warn_if_custom_ar_setup_failed$' scripts/switch.sh \
  && ok "switch.sh calls the check after a waited launch" || bad "switch.sh never calls warn_if_custom_ar_setup_failed"

mkdir -p "$T/bin"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
[ "$1" = logs ] && cat "$MOCK_LOG"
exit 0
EOF
chmod +x "$T/bin/docker"

run() {  # <fixture text> → the function's stderr
  printf '%s\n' "$1" > "$T/log"
  PATH="$T/bin:$PATH" MOCK_LOG="$T/log" ROOT_DIR="$ROOT" bash -c '
    declare -A VARIANT_CONTAINER=([sgl/x]=sglang-x); VARIANT=sgl/x
    . "$1"; warn_if_custom_ar_setup_failed' _ "$T/fn.sh" 2>&1
}

FAILED="[t] server_args={'tp_size': 2, 'disable_custom_all_reduce': False}
[t TP0] sglang is using nccl==2.29.7
[t TP0] Setup Custom allreduce failed with invalid literal for int() with base 10: 'GPU-0e72'. To silence this warning, specify --disable-custom-all-reduce explicitly."
CLEAN="[t] server_args={'tp_size': 2, 'disable_custom_all_reduce': False}
[t TP0] All Reduce config: symmetric_memory = 20.01 MB, local_buffer = 2.00 MB, multicast = False, pull = True"
OPERATOR="[t] server_args={'tp_size': 2, 'disable_custom_all_reduce': True}"

out="$(run "$FAILED")"
[[ "$out" == *"SETUP FAILED"* && "$out" == *"invalid literal for int()"* ]] \
  && ok "failed setup: warns and quotes the engine's line" || bad "failed setup did not warn: '$out'"
out="$(run "$CLEAN")";    [[ -z "$out" ]] && ok "clean custom-AR boot: silent"   || bad "clean boot warned: '$out'"
out="$(run "$OPERATOR")"; [[ -z "$out" ]] && ok "operator-disabled boot: silent" || bad "operator-disabled boot warned: '$out'"

[[ $fail -eq 0 ]] && echo "test-switch-custom-ar-warning: ok" || echo "test-switch-custom-ar-warning: FAIL"
exit $fail
