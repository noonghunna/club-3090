#!/usr/bin/env bash
# test-preflight-thp.sh — the transparent-hugepage hint for CPU-offload slugs.
#
# WHY THIS EARNS ITS KEEP: the hint fires on a HOST condition we do not control,
# so on a correctly-configured rig it is never exercised and would rot unnoticed.
# The failure modes that would do real damage are the false ones:
#   • warning on a GPU-resident compose (pure noise for most users)
#   • warning when NO model is resident (0% coverage read as a finding)
#   • BLOCKING a boot over what is only a latency hint
#
# Fixture-driven: scripts/lib/thp.sh reads THP_SYSFS_DIR / THP_MEMINFO, so every
# case below is a real end-to-end run of the guard against a synthetic host.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
fail=0
ok()  { echo "  ok   — $1"; }
bad() { echo "  FAIL — $1" >&2; fail=1; }
echo "== test-preflight-thp =="

# shellcheck source=/dev/null
source scripts/lib/compose-meta.sh 2>/dev/null
# shellcheck source=/dev/null
source scripts/preflight.sh 2>/dev/null
export SCRIPT_DIR="$ROOT_DIR/scripts"

declare -F preflight_offload_thp >/dev/null 2>&1 \
  && ok "preflight_offload_thp defined" || { bad "preflight_offload_thp missing"; exit 1; }

OFFLOAD=models/glm-5.3-flash/llamacpp-club3090/compose/dual/unsloth-ud-iq3xxs/moecache.yml
NONOFF=models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml
[[ -f "$OFFLOAD" && -f "$NONOFF" ]] || { bad "fixture composes missing"; exit 1; }
is_cpu_offload_compose "$OFFLOAD" || bad "fixture OFFLOAD is not detected as offload"
is_cpu_offload_compose "$NONOFF"  && bad "fixture NONOFF is detected as offload"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkhost() { # $1=dir $2=shmem_enabled $3=Shmem_kB $4=ShmemHugePages_kB
  mkdir -p "$TMP/$1"
  printf 'always madvise [never]\n' > "$TMP/$1/enabled"
  case "$2" in
    never)  printf 'always within_size advise [never] deny force\n' > "$TMP/$1/shmem_enabled" ;;
    always) printf '[always] within_size advise never deny force\n'  > "$TMP/$1/shmem_enabled" ;;
  esac
  printf 'always defer [defer+madvise] madvise never\n' > "$TMP/$1/defrag"
  { echo "MemTotal:       201326592 kB"
    echo "Shmem:          $3 kB"
    echo "ShmemHugePages: $4 kB"
    echo "ShmemPmdMapped: $4 kB"
    echo "AnonHugePages:    210944 kB"; } > "$TMP/$1.meminfo"
}
run() { # $1=hostdir $2=compose -> stderr text
  THP_SYSFS_DIR="$TMP/$1" THP_MEMINFO="$TMP/$1.meminfo" \
    preflight_offload_thp "$2" 2>&1 >/dev/null
}
rc_of() { THP_SYSFS_DIR="$TMP/$1" THP_MEMINFO="$TMP/$1.meminfo" \
    preflight_offload_thp "$2" >/dev/null 2>&1; echo $?; }

# 110 GiB resident; 107 GiB huge = ~97% ; and 3 GiB huge = ~2%
mkhost never_host  never  115343360 0
mkhost good_host   always 115343360 112197632
mkhost frag_host   always 115343360   3145728
mkhost idle_host   always      65536         0   # <1 GiB Shmem: nothing loaded

# --- POSITIVE: the conditions that SHOULD warn -------------------------------
run never_host "$OFFLOAD" | command grep -q 'shmem_enabled=never' \
  && ok "warns when shmem_enabled=never" || bad "silent on shmem_enabled=never"
run never_host "$OFFLOAD" | command grep -qi 'latency' \
  && ok "never-case is worded as LATENCY" || bad "never-case must not imply throughput"
run never_host "$OFFLOAD" | command grep -q 'Fix:' \
  && ok "never-case carries a Fix: hint" || bad "no Fix: hint"
run frag_host "$OFFLOAD" | command grep -q 'coverage of Shmem is 2%' \
  && ok "warns on low coverage despite the knob being set" || bad "missed the fragmentation case"

# --- NEGATIVE CONTROLS: the conditions that must stay QUIET -------------------
[[ -z "$(run good_host "$OFFLOAD")" ]] \
  && ok "quiet at 97% coverage" || bad "false alarm on a correctly-configured host"
[[ -z "$(run idle_host "$OFFLOAD")" ]] \
  && ok "quiet when no model is resident (0% is not a finding)" \
  || bad "false alarm with no large Shmem — 0% read as low coverage"
[[ -z "$(run never_host "$NONOFF")" ]] \
  && ok "quiet on a NON-offload compose even with shmem_enabled=never" \
  || bad "fires on a GPU-resident compose — pure noise"

# --- it must NEVER block ------------------------------------------------------
for h in never_host frag_host good_host idle_host; do
  [[ "$(rc_of "$h" "$OFFLOAD")" == "0" ]] \
    || bad "returned non-zero on $h — this is a hint and must never block a boot"
done
ok "returns 0 on every host state (never blocks)"

# --- the trap this whole check exists for ------------------------------------
# `enabled` is NOT a substitute for `shmem_enabled`: AnonHugePages climbs while
# the experts stay on 4 KiB pages. Assert the guard reads the right knob.
mkhost trap_host never 115343360 0
printf '[always] madvise never\n' > "$TMP/trap_host/enabled"   # enabled=always, shmem=never
run trap_host "$OFFLOAD" | command grep -q 'shmem_enabled=never' \
  && ok "reads shmem_enabled, not enabled (the misdiagnosis trap)" \
  || bad "guard was fooled by enabled=always while shmem_enabled=never"

# --- the lib must resolve even under an unexpected SCRIPT_DIR ----------------
# A wrong SCRIPT_DIR must not silently DISABLE the check — that is the very
# failure shape this guard exists to catch, and it would look identical to a
# healthy host.
for sd in "$ROOT_DIR/scripts" "/nonexistent-dir" ""; do
  if [[ -z "$sd" ]]; then
    got="$(env -u SCRIPT_DIR THP_SYSFS_DIR="$TMP/never_host" THP_MEMINFO="$TMP/never_host.meminfo" \
            bash -c "cd '$ROOT_DIR'; source scripts/lib/compose-meta.sh 2>/dev/null; source scripts/preflight.sh 2>/dev/null; preflight_offload_thp '$OFFLOAD' 2>&1")"
  else
    got="$(SCRIPT_DIR="$sd" THP_SYSFS_DIR="$TMP/never_host" THP_MEMINFO="$TMP/never_host.meminfo" \
            bash -c "cd '$ROOT_DIR'; source scripts/lib/compose-meta.sh 2>/dev/null; source scripts/preflight.sh 2>/dev/null; preflight_offload_thp '$OFFLOAD' 2>&1")"
  fi
  printf '%s' "$got" | command grep -q 'shmem_enabled=never' \
    && ok "fires with SCRIPT_DIR=${sd:-<unset>}" \
    || bad "SILENT with SCRIPT_DIR=${sd:-<unset>} — lib resolution disabled the check"
done

[[ $fail -eq 0 ]] && echo "test-preflight-thp: ok" || echo "test-preflight-thp: FAILED" >&2
exit $fail
