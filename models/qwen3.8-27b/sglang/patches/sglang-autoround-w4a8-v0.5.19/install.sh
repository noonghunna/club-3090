#!/usr/bin/env bash
# Apply the v0.5.19-adapted W4A8 patch to SGLang inside the container.
#
# Re-cut of jb-seo's club-3090 0004-autoround-w4a8.patch (cut against v0.5.18)
# onto SGLang v0.5.19. The only adaptation: auto_round.py context line
#   `if isinstance(layer, (LinearBase, ParallelLMHead)):`
#   -> `if is_linear:`
# (v0.5.19 refactored that). gptq_kernels.py is byte-identical between the two
# versions, so its hunks apply verbatim. The 8 new files (w4a8.py + 7 CUDA
# sources) are new files with no context dependency.
#
# Run from the compose command BEFORE `sglang serve`. Idempotent: a restart
# re-runs it and detects the already-applied state instead of failing.
# Usage:  install.sh          (apply if not applied)
#         install.sh --verify (report state only, change nothing)

set -uo pipefail

SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
PATCH_DIR="${SGLANG_PATCH_DIR:-/etc/club3090/w4a8}"
PATCH="$PATCH_DIR/0004-autoround-w4a8.v0519.patch"

GPTQ="$SGLANG_DIR/python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py"
AR="$SGLANG_DIR/python/sglang/srt/layers/quantization/auto_round.py"
W4A8PY="$SGLANG_DIR/python/sglang/kernels/ops/quantization/gptq_marlin_w4a8.py"
CUDA="$SGLANG_DIR/python/sglang/kernels/jit/csrc/gemm/marlin_w4a8"

die() { echo "[w4a8] ERROR: $*" >&2; exit 1; }

[ -d "$SGLANG_DIR" ] || die "SGLang source not at $SGLANG_DIR (override with SGLANG_DIR=...)"
[ -f "$GPTQ" ] || die "gptq_kernels.py not found under $SGLANG_DIR — wrong SGLANG_DIR?"

# Per-component markers (content-based, so they work whether or not the tree
# is a git repo).
m_gptq() { grep -q 'self.use_w4a8 = False' "$GPTQ"; }
m_ar()   { grep -q 'quant_args_marlin.experimental_w4a8 = True' "$AR"; }
m_new()  { [ -f "$W4A8PY" ] && [ -f "$CUDA/entry.cuh" ] \
          && [ -f "$CUDA/marlin_template.h" ] && [ -f "$CUDA/repack.cuh" ]; }

if [ "${1:-}" = "--verify" ]; then
  if m_gptq; then echo "  gptq_kernels.py : applied";  else echo "  gptq_kernels.py : NOT applied";  exit 1; fi
  if m_ar;   then echo "  auto_round.py   : applied";  else echo "  auto_round.py   : NOT applied";  exit 1; fi
  if m_new;  then echo "  W4A8 new files  : present";  else echo "  W4A8 new files  : MISSING";      exit 1; fi
  echo "[w4a8] all checks passed"
  exit 0
fi

a=0
m_gptq && a=$((a+1))
m_ar   && a=$((a+1))
m_new  && a=$((a+1))

case "$a" in
  3) echo "[w4a8] already applied — skipping"; exit 0 ;;
  0) : ;;
  *) die "partial application detected ($a/3 markers present) — recreate the container and retry" ;;
esac

[ -f "$PATCH" ] || die "patch not found at $PATCH"

echo "[w4a8] applying v0.5.19-adapted W4A8 patch..."
if git -C "$SGLANG_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[w4a8]   tree is a git repo — using git apply"
  git -C "$SGLANG_DIR" apply --check "$PATCH" 2>&1 | sed 's/^/    /' || die "git apply --check failed (context drift?)"
  git -C "$SGLANG_DIR" apply "$PATCH" || die "git apply failed"
else
  echo "[w4a8]   not a git repo — using patch(1)"
  patch -p1 -d "$SGLANG_DIR" < "$PATCH" >/dev/null || die "patch apply failed"
fi

echo "[w4a8] verifying..."
bash "$0" --verify || die "post-apply verification failed"
echo "[w4a8] applied + verified OK"
