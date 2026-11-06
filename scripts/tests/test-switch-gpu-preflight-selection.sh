#!/usr/bin/env bash
# test-switch-gpu-preflight-selection — switch.sh's free-memory preflight checks the cards
# a selection names, whether it names them by index or by UUID.
#
# WHY THIS EXISTS
# ---------------
# gpu_preflight() refuses a boot when a selected card is still holding memory. It matched the
# selection against host INDICES only, but CLUB3090_GPU and launch.sh --gpus both export GPU
# UUIDs (#610). A UUID selection matched no card, so the check passed with the pinned card full:
# found 2026-11-06 while making c3's Catalog Start name its cards next to running pods (#1939).
# This guard runs the REAL function against a stub nvidia-smi with card 1 busy.
#   PREFLIGHT_SWITCH=<path>  run it against another switch.sh (the negative control).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SWITCH="${PREFLIGHT_SWITCH:-scripts/switch.sh}"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
sed -n '/^gpu_preflight()/,/^}/p' "$SWITCH" > "$T/fn.sh"
[[ -s "$T/fn.sh" ]] || { echo "✗ gpu_preflight not found in $SWITCH" >&2; exit 1; }

# stub nvidia-smi: three 24 GB cards, card 1 (GPU-bbb) holding 23 GB. It answers both the
# old index-only query and the index,uuid one.
mkdir -p "$T/bin"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *--query-gpu=index,uuid,memory.free,memory.total*)
    printf '0, GPU-aaa, 24000, 24576\n1, GPU-bbb, 1000, 24576\n2, GPU-ccc, 24000, 24576\n' ;;
  *--query-gpu=index,memory.free,memory.total*)
    printf '0, 24000, 24576\n1, 1000, 24576\n2, 24000, 24576\n' ;;
  *--query-compute-apps*) echo "4242, llama-server, 23000 MiB" ;;
  *) exit 6 ;;
esac
EOF
chmod +x "$T/bin/nvidia-smi"

# run <NVIDIA_VISIBLE_DEVICES|-unset-> → "pass" or "refused", stderr in $T/err
run() {
  PATH="$T/bin:$PATH" SEL="$1" bash -c '
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES FORCE
    [ "$SEL" = "-unset-" ] || export NVIDIA_VISIBLE_DEVICES="$SEL" CUDA_VISIBLE_DEVICES="$SEL"
    VARIANT=test/slug
    . "$1"
    gpu_preflight
    echo pass
  ' _ "$T/fn.sh" 2>"$T/err" | tail -1 | command grep -qx pass && echo pass || echo refused
}

# Positive controls first: the stub's busy card is caught, and free cards pass.
[[ "$(run -unset-)" == refused ]] && command grep -q 'GPU 1: 1000 / 24576' "$T/err" \
  && ok "no selection: every card checked, busy GPU 1 refused" || bad "no selection: not refused"
[[ "$(run 1)" == refused ]] && ok "index 1 (busy): refused" || bad "index 1: not refused"
[[ "$(run 0)" == pass ]] && ok "index 0 (free): passes" || bad "index 0: refused ($(cat "$T/err"))"
[[ "$(run 0,2)" == pass ]] && ok "indices 0,2 (free): pass" || bad "indices 0,2: refused"

# The bug: UUIDs.
[[ "$(run GPU-bbb)" == refused ]] && command grep -q 'GPU 1: 1000 / 24576' "$T/err" \
  && ok "UUID of the busy card: refused, naming GPU 1" || bad "UUID GPU-bbb: passed with the card full"
[[ "$(run GPU-aaa,GPU-bbb)" == refused ]] && ok "two UUIDs, one busy: refused" || bad "GPU-aaa,GPU-bbb: passed"
[[ "$(run GPU-aaa)" == pass ]] && ok "UUID of a free card: passes (the busy card isn't selected)" \
  || bad "UUID GPU-aaa: refused ($(cat "$T/err"))"
[[ "$(run 'GPU-aaa, GPU-ccc')" == pass ]] && ok "UUIDs with spaces, both free: pass" || bad "'GPU-aaa, GPU-ccc': refused"

# A token that names no card: every card is checked rather than none.
[[ "$(run GPU-zzz)" == refused ]] && command grep -q "names no card here" "$T/err" \
  && ok "unknown token: says so and checks every card" || bad "unknown token: passed or silent"

[[ $fail -eq 0 ]] && echo "test-switch-gpu-preflight-selection: ok" \
  || { echo "test-switch-gpu-preflight-selection: FAIL" >&2; exit 1; }
