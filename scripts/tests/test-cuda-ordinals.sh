#!/usr/bin/env bash
# test-cuda-ordinals — scripts/lib/cuda-ordinals.sh (club-3090#1462).
#
# The helper runs at the top of every SGLang entrypoint, under `set -euo pipefail`,
# so two things matter as much as the remap itself: it must NEVER fail the boot, and
# it must leave CUDA_VISIBLE_DEVICES exactly as it was whenever it cannot resolve
# every entry — a half-remapped list would put ranks on the wrong cards.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$ROOT/scripts/lib/cuda-ordinals.sh"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/smi" "$T/nosmi"
# A container that sees two GPUs, listed the way nvidia-smi prints them.
cat > "$T/smi/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '0, GPU-aaaaaaaa-0000\n1, GPU-bbbbbbbb-1111\n'
EOF
chmod +x "$T/smi/nvidia-smi"
# A PATH with the tools the helper needs but no nvidia-smi at all.
for tool in awk bash; do ln -s "$(command -v $tool)" "$T/nosmi/$tool"; done

# run <path-dir> <CUDA_VISIBLE_DEVICES or __unset__>  → prints "CVD|ORDER|rc"
run() {
  local pathdir="$1" cvd="$2"
  env -i PATH="$pathdir" HOME="$T" bash -c '
    set -euo pipefail
    [ "$1" = "__unset__" ] || export CUDA_VISIBLE_DEVICES="$1"
    . "$2"
    club3090_cuda_ordinals
    echo "${CUDA_VISIBLE_DEVICES-__unset__}|${CUDA_DEVICE_ORDER:-unset}|0"
  ' _ "$cvd" "$LIB" 2>"$T/stderr" || echo "boot-killed|-|$?"
}
expect() {  # <label> <pathdir> <input> <want "CVD|ORDER|rc"> [stderr must contain]
  local got; got="$(run "$2" "$3")"
  if [[ "$got" != "$4" ]]; then bad "$1: got '$got', want '$4'"; return; fi
  if [[ -n "${5:-}" ]] && ! command grep -qF -- "$5" "$T/stderr"; then bad "$1: warning missing ('$5')"; return; fi
  ok "$1"
}

S="$T/smi:/usr/bin:/bin"
expect "unset stays unset, no order change"          "$S" "__unset__"                          "__unset__|unset|0"
expect "numeric list untouched"                      "$S" "0,1"                                "0,1|unset|0"
expect "UUIDs → this container's ordinals"           "$S" "GPU-aaaaaaaa-0000,GPU-bbbbbbbb-1111" "0,1|PCI_BUS_ID|0" "-> 0,1"
expect "requested order is kept"                     "$S" "GPU-bbbbbbbb-1111,GPU-aaaaaaaa-0000" "1,0|PCI_BUS_ID|0"
expect "unknown UUID: whole list left as is"          "$S" "GPU-aaaaaaaa-0000,GPU-zzzzzzzz-9999" \
       "GPU-aaaaaaaa-0000,GPU-zzzzzzzz-9999|unset|0" "is not a GPU this container can see"
expect "MIG instance: left as is"                     "$S" "MIG-12345678-abcd"                   "MIG-12345678-abcd|unset|0" "MIG instances"
expect "no nvidia-smi: left as is, boot continues"    "$T/nosmi" "GPU-aaaaaaaa-0000,GPU-bbbbbbbb-1111" \
       "GPU-aaaaaaaa-0000,GPU-bbbbbbbb-1111|unset|0" "cannot list this container's GPUs"

# Positive control for the test itself: a helper that exits non-zero must show up as
# a killed boot, or "the boot continues" above proves nothing.
printf 'club3090_cuda_ordinals() { return 3; }\n' > "$T/broken.sh"
got="$(LIB="$T/broken.sh"; run "$S" "0,1" 2>/dev/null)"
[[ "$got" == boot-killed* ]] && ok "self-test: a failing helper is detected as a killed boot" \
                             || bad "self-test: a failing helper was not detected (got '$got')"

# Wiring: every compose that launches SGLang mounts the helper and calls it. A compose
# copied from an upstream recipe would otherwise bring the silent fallback back.
wired() { command grep -qE 'scripts/lib/cuda-ordinals\.sh:/etc/club3090/cuda-ordinals\.sh:ro' "$1" \
          && command grep -qE '^[[:space:]]+if \[ -f /etc/club3090/cuda-ordinals\.sh \]; then \. /etc/club3090/cuda-ordinals\.sh; club3090_cuda_ordinals;' "$1"; }
printf 'services:\n  x:\n    volumes:\n      - ./a:/b\n    command:\n      - |\n        # club3090_cuda_ordinals mentioned only in a comment\n' > "$T/unwired.yml"
if wired "$T/unwired.yml"; then bad "self-test: the wiring check accepted an unwired compose"; fi
n=0; missing=0
while IFS= read -r f; do
  n=$((n + 1))
  wired "$ROOT/$f" || { bad "$f launches SGLang without mounting + calling cuda-ordinals.sh"; missing=1; }
done < <(cd "$ROOT" && command grep -rlE 'sglang\.launch_server|sglang serve' models --include='*.yml' | sort)
[[ $n -gt 0 ]] || bad "found no SGLang composes — the scan itself is broken"
[[ $missing -eq 0 && $n -gt 0 ]] && ok "all $n SGLang composes mount and call the helper"

[[ $fail -eq 0 ]] && echo "test-cuda-ordinals: ok" || echo "test-cuda-ordinals: FAIL"
exit $fail
