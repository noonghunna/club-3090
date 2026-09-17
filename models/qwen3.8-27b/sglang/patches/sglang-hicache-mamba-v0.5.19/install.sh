#!/usr/bin/env bash
# Apply the SGLang HiCache L2 / mamba radix-cache patches to the v0.5.19
# engine source. Two independent patches are applied in order, with DIFFERENT
# failure gates:
#
#   1. apply_39342.py  — sgl-project/sglang#39342  (FAIL-CLOSED)
#      Mixed-chunk corrupts the mamba radix cache: prepare_for_extend stamps
#      mamba_last_track_seqlen; merge_batch nulls the batch track tensor;
#      prepare_for_caching_req reads the stale stamp and donates a stale
#      mamba slot into the unified radix tree under a mismatched key.
#      FIX: 5-edit load-bearing unit (schedule_batch.py A1/A2/B +
#      mamba_component.py C/D). Edit C returns 0 (not None) and the rollback
#      (edit B) is scoped to the HEAD of mix_with_running so it touches only
#      the incoming prefill claims, not the running decode reqs. Apply is
#      two-pass all-or-nothing across both files.
#      STATUS: VALIDATED (90/90 clean under sustained mixed-chunk load; the
#      exact trigger that crashed the unpatched engine is neutralized).
#      => If it cannot apply/verify (anchor drift / half-patched), REFUSE BOOT.
#
#   2. apply_33713.py  — sgl-project/sglang#33713  (FAIL-OPEN)
#      MAMBA component nodes pruned instead of downgraded on device eviction,
#      breaking host-tier loadback. 2-edit match-side change to
#      unified_tree_core.py (traversal dead-node check + _is_host_leaf): a
#      node is a host leaf if ANY component (FULL or MAMBA) has a host copy,
#      not only FULL.
#      STATUS: HYPOTHESIS-ONLY / NOT VALIDATED on this GDN model (the
#      KV-level symptom does not reproduce here; the edit does not touch the
#      evict-fork the reporter's data points to; it reproduces on the issue's
#      KDA model). A drifted hypothesis anchor must NOT refuse a boot.
#      => If it cannot apply/verify, WARN + CONTINUE (serve unpatched).
#
# Both scripts are idempotent (re-running on a patched tree is a no-op) and
# atomic (per-edit tmp-file + os.replace, so a crash never leaves a
# half-patched file). apply_39342.py is also all-or-nothing ACROSS its two
# files, so it can never leave one patched and the other pristine.
#
# Usage (inside the container, before `sglang launch_server`):
#   bash install.sh          # apply both, verify 39342 (gate) + 33713 (warn)
#   bash install.sh --verify # read-only: 39342 gate, 33713 non-gating
#
# Mounted at /etc/club3090/hicache-mamba by the compose (volumes:).

set -uo pipefail

SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
PATCH_DIR="${SGLANG_PATCH_DIR:-/etc/club3090/hicache-mamba}"

P39342="$PATCH_DIR/apply_39342.py"
P33713="$PATCH_DIR/apply_33713.py"

die() { echo "[hicache-mamba] ERROR: $*" >&2; exit 1; }

[ -d "$SGLANG_DIR" ] || die "SGLang source not at $SGLANG_DIR (override with SGLANG_DIR=...)"
[ -f "$P39342" ]      || die "apply_39342.py not found at $P39342 — mount missing (39342 is load-bearing)"
# 33713 is optional/fail-open: if its mount is absent, skip it (baseline).
[ -f "$P33713" ] || echo "[hicache-mamba] apply_33713.py not mounted — skipping (hypothesis-only)."

# ── --verify mode: read-only status ────────────────────────────────────────
if [ "${1:-}" = "--verify" ]; then
  echo "[hicache-mamba] verify mode (read-only)"
  # 39342 is the gate: it must pass.
  if python3 "$P39342" --check 2>&1 | sed 's/^/  39342 /'; then :; else die "39342 --check failed (fail-closed)"; fi
  # 33713 is non-gating: report but never die.
  if [ -f "$P33713" ]; then
    if python3 "$P33713" --check 2>&1 | sed 's/^/  33713 /'; then :; else
      echo "  WARN: 33713 --check failed — non-gating (hypothesis-only)." >&2
    fi
  fi
  echo "[hicache-mamba] verify done (39342 PASS/gated, 33713 non-gating)"
  exit 0
fi

# ── apply mode ─────────────────────────────────────────────────────────────
# 39342 FAIL-CLOSED: must apply AND verify, else refuse boot.
echo "[hicache-mamba] applying #39342 (mixed-chunk mamba radix-cache fix; VALIDATED, fail-closed)..."
python3 "$P39342" || die "#39342 apply failed — refusing to boot"
if ! python3 "$P39342" --check; then
  die "#39342 post-apply --check failed — refusing to boot"
fi
echo "[hicache-mamba] #39342 applied + verified."

# 33713 FAIL-OPEN: apply + verify, but a failure WARNs and continues.
if [ -f "$P33713" ]; then
  echo "[hicache-mamba] applying #33713 (mamba host-leaf match-side fix; HYPOTHESIS-ONLY, fail-open)..."
  if ! python3 "$P33713"; then
    echo "[hicache-mamba] WARN: #33713 apply failed (anchor drift?) — continuing unpatched." >&2
  elif ! python3 "$P33713" --check; then
    echo "[hicache-mamba] WARN: #33713 post-apply --check failed — continuing (non-gating)." >&2
  else
    echo "[hicache-mamba] #33713 applied + verified (hypothesis-only, NOT validated on this model)."
  fi
fi

echo "[hicache-mamba] done: 39342=VALIDATED (fail-closed) + 33713=HYPOTHESIS-ONLY (fail-open)"
exit 0
