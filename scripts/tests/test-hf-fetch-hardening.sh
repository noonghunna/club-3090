#!/usr/bin/env bash
# The hardened hf() wrapper (services/comfyui/comfyui-paths.sh) — #715 gap 2 + #726.
#
# Three failure classes, each reproduced with a PATH-shimmed `hf`:
#   1. STALE LOCK (#726): an orphaned *.lock under the dest download cache (no live
#      holder, old mtime) is cleared before the attempt, so it can't wedge the run.
#   2. STALL (#726): an attempt that produces no byte growth (hf_hub's unbounded
#      "Still waiting to acquire lock" wait, or a dead peer) is killed by the
#      watchdog after HF_FETCH_STALL_TIMEOUT and the retry succeeds.
#   3. FALSE DONE (#715): rc=0 with leftover *.incomplete (huggingface_hub
#      offline-degrade) is treated as failure and retried.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "ASSERTION FAILED: $1" >&2; exit 1; }

mkdir -p "$TMP/bin" "$TMP/dest/.cache/huggingface/download"

# shim harness: behavior file selects the fake hf's mode per case
cat > "$TMP/bin/hf" <<SH
#!/usr/bin/env bash
mode="\$(cat "$TMP/mode")"
case "\$mode" in
  ok) exit 0 ;;
  hang) sleep 3600 ;;
  hang-then-ok)
    if [ -f "$TMP/second" ]; then exit 0; fi
    touch "$TMP/second"; sleep 3600 ;;
  falsedone-then-ok)
    if [ -f "$TMP/second" ]; then rm -f "$TMP/dest/.cache/huggingface/download/x.incomplete"; exit 0; fi
    touch "$TMP/second"; touch "$TMP/dest/.cache/huggingface/download/x.incomplete"; exit 0 ;;
esac
SH
chmod +x "$TMP/bin/hf"

run_wrapped() {  # <mode> [env...] — source the lib fresh + run one wrapped download
  local mode="$1"; shift
  echo "$mode" > "$TMP/mode"
  rm -f "$TMP/second"
  env -i PATH="$TMP/bin:/usr/bin:/bin" HOME="$TMP" "$@" bash -c '
    set -uo pipefail
    C3_PATHS_NO_ENV=1 MODEL_DIR="'"$TMP"'/models" . "'"$ROOT_DIR"'/services/comfyui/comfyui-paths.sh"
    hf download some/repo --local-dir "'"$TMP"'/dest"
  ' 2>&1
}

# --- 1. stale lock cleared (no live holder, old mtime) -----------------------
lock="$TMP/dest/.cache/huggingface/download/model.safetensors.lock"
touch "$lock"
touch -d '30 minutes ago' "$lock"
out="$(run_wrapped ok)" || fail "case 1: wrapper failed on a clean download: $out"
grep -q "clearing stale download lock" <<<"$out" || fail "case 1: stale lock not cleared: $out"
[ ! -f "$lock" ] || fail "case 1: stale lock file still present"

# --- 1b. a FRESH lock is spared (could be a live SoftFileLock holder) --------
touch "$lock"
out="$(run_wrapped ok)" || fail "case 1b: wrapper failed: $out"
grep -q "clearing stale download lock" <<<"$out" && fail "case 1b: FRESH lock was wrongly cleared"
[ -f "$lock" ] || fail "case 1b: fresh lock removed"
rm -f "$lock"

# --- 2. stall watchdog kills the hung attempt; retry succeeds ---------------
start=$(date +%s)
out="$(run_wrapped hang-then-ok HF_FETCH_STALL_TIMEOUT=15)" || fail "case 2: wrapper failed after stall: $out"
took=$(( $(date +%s) - start ))
grep -q "no byte growth" <<<"$out" || fail "case 2: watchdog never fired: $out"
grep -q "retry 1/" <<<"$out" || fail "case 2: no retry after the kill: $out"
[ "$took" -lt 120 ] || fail "case 2: took ${took}s — watchdog didn't bound the hang"

# --- 3. false DONE (rc=0 + leftover .incomplete) is retried ------------------
out="$(run_wrapped falsedone-then-ok)" || fail "case 3: wrapper failed: $out"
grep -q "false DONE" <<<"$out" || fail "case 3: false-DONE not detected: $out"
find "$TMP/dest" -name '*.incomplete' | grep -q . && fail "case 3: .incomplete still present after retry"

echo "test-hf-fetch-hardening: ok"
