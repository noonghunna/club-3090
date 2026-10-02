#!/usr/bin/env bash
# test-switch-gpu-pin — CLUB3090_GPU pins a single-card slug to one host GPU, under --force too.
#
# WHY THIS EXISTS
# ---------------
# switch.sh advertised CLUB3090_GPU as the "single-card GPU index override … on a hetero rig",
# but the only code that applied it lived inside preflight_compose_hardware, which runs for vLLM
# slugs only, returns early on a compose without Requires-* metadata, and is skipped under
# --force. So on every 🧪 slug (they need --force) and every non-vLLM single-card slug the
# override did nothing and the slug landed on GPU 0. Found 2026-10-02 handing a 64 GB CMP
# 170HX (host GPU 1, between two 3090s) the single-fast slugs (club-3090#1537).
# apply_club3090_gpu_pin now runs once, before anything reads the GPU selection. This guard runs
# the REAL function against a stub nvidia-smi, and checks where switch.sh calls it.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
sed -n '/^apply_club3090_gpu_pin()/,/^}/p' scripts/switch.sh > "$T/fn.sh"
[[ -s "$T/fn.sh" ]] || { echo "✗ apply_club3090_gpu_pin not found in switch.sh" >&2; exit 1; }

# stub nvidia-smi: index N in 0..2 -> GPU-fake-N; anything else fails, like the real one
mkdir -p "$T/bin"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
idx=""; while [ $# -gt 0 ]; do [ "$1" = "-i" ] && idx="$2"; shift; done
case "$idx" in 0|1|2) echo "GPU-fake-$idx" ;; *) exit 6 ;; esac
EOF
chmod +x "$T/bin/nvidia-smi"

# run <CLUB3090_GPU value|-unset-> <slug> → "rc|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|stderr"
run() {
  local gpu="$1" slug="$2"
  PATH="$T/bin:$PATH" ROOT_DIR="$ROOT" GPU_VAL="$gpu" SLUG="$slug" bash -c '
    declare -A VARIANTS=(
      [sgl/one]="sglang|models/m/sglang/compose|single/q/mtp.yml"
      [vllm/two]="vllm|models/m/vllm/compose|dual/q/mtp.yml"
      [vllm/four]="vllm|models/m/vllm/compose|multi4/q/mtp.yml"
    )
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES CLUB3090_GPU
    [ "$GPU_VAL" = "-unset-" ] || export CLUB3090_GPU="$GPU_VAL"
    . "$1"
    apply_club3090_gpu_pin "$SLUG" 2>"$2"
    printf "0|%s|%s" "${CUDA_VISIBLE_DEVICES:-}" "${NVIDIA_VISIBLE_DEVICES:-}"
  ' _ "$T/fn.sh" "$T/err" 2>/dev/null
  local rc=$?
  [[ $rc -eq 0 ]] || printf "%s||" "$rc"
  printf "|%s" "$(cat "$T/err" 2>/dev/null)"
}

out="$(run -unset- sgl/one)"
[[ "$out" == "0|||" ]] && ok "unset: no-op, nothing exported, silent" || bad "unset: got '$out'"

out="$(run 1 sgl/one)"
[[ "$out" == "0|GPU-fake-1|GPU-fake-1|"*"pinned to GPU-fake-1"* ]] \
  && ok "single-card slug: GPU 1 pinned by UUID in both CUDA_ and NVIDIA_VISIBLE_DEVICES" || bad "single: got '$out'"

out="$(run 9 sgl/one)"
[[ "$out" == "0|9|9|"* ]] \
  && ok "index nvidia-smi cannot resolve: falls back to the raw index (gpu-select's rule)" || bad "unresolvable: got '$out'"

out="$(run 1 vllm/two)"
[[ "$out" == "0|||"*"ignored"*"launch.sh --variant vllm/two --gpus"* ]] \
  && ok "dual slug: left alone, says why and how to pick cards" || bad "dual: got '$out'"
out="$(run 1 vllm/four)"
[[ "$out" == "0|||"*"ignored"* ]] && ok "multi4 slug: left alone" || bad "multi4: got '$out'"

for v in "1,2" "a" "GPU-x"; do
  out="$(run "$v" sgl/one)"
  [[ "$out" == "1||"*"must be one GPU index"* ]] && ok "CLUB3090_GPU='$v': refused with exit 1" || bad "'$v': got '$out'"
done

# Where switch.sh calls it: after the slug is resolved, before anything reads the GPU selection.
n_resolve=$(command grep -n '^VARIANT="$(resolve_default_variant "$VARIANT")"$' scripts/switch.sh | head -1 | cut -d: -f1)
n_call=$(command grep -n '^apply_club3090_gpu_pin "\$VARIANT"$' scripts/switch.sh | head -1 | cut -d: -f1)
n_warn=$(command grep -n '^warn_if_default_arch_gated ' scripts/switch.sh | head -1 | cut -d: -f1)
n_check=$(command grep -n '^check_variant "\${VARIANT}"' scripts/switch.sh | head -1 | cut -d: -f1)
if [[ -n "$n_resolve" && -n "$n_call" && -n "$n_warn" && -n "$n_check" ]] \
   && (( n_resolve < n_call && n_call < n_warn && n_call < n_check )); then
  ok "switch.sh calls it after resolving the slug and before the arch warning / check_variant"
else
  bad "call order wrong or missing (resolve=$n_resolve call=$n_call warn=$n_warn check=$n_check)"
fi

[[ $fail -eq 0 ]] && echo "test-switch-gpu-pin: ok" || { echo "test-switch-gpu-pin: FAIL" >&2; exit 1; }
