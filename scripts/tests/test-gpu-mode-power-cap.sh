#!/usr/bin/env bash
# test-gpu-mode-power-cap — `gpu-mode power-cap on` must apply the wattage the
# systemd unit defines, or refuse. It must never apply one of its own.
#
# Why this test exists. The script documented the cap as 250 W and called
# nvidia-power-cap.service "the single source of truth for the 250W value" — but
# never read the unit. The value was restated in comments and then applied
# LITERALLY in a path that runs: when `systemctl restart` failed, the fallback
# executed `nvidia-smi -i 0 -pl 250 && nvidia-smi -i 1 -pl 250`. On the reference
# rig that unit's ExecStart lines say `-pl 230`, so the fallback capped 20 W away
# from the service it was standing in for — and reported success either way.
#
# That class of error is expensive here: a cap on this rig produced a −40%
# reading that looked exactly like an upstream speculative-decoding regression
# and was nearly filed as an engine bug. An unverified cap is worse than no cap,
# because the benchmark it suppresses still looks valid. Hence: resolve from the
# unit, or refuse loudly.
#
# Contract asserted here:
#   - unit missing / unparseable        -> REFUSE, non-zero exit, NOTHING applied;
#   - unit present + systemd restart OK -> systemd does it, no direct -pl at all;
#   - unit present + restart FAILS      -> replay EXACTLY the unit's own values;
#   - the `-i`/`-pl` pairs and the wattage in every message come from the unit;
#   - no power-limit wattage literal survives anywhere in executed code.
#
# No GPU is touched: nvidia-smi, sudo and systemctl are all recording stubs, and
# the harness refuses to run unless it has verified it shadowed the real ones.
set -uo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_MODE="$ROOT_DIR/scripts/gpu-mode.sh"

PASS=0; FAIL=0
ok() { PASS=$((PASS+1)); }
no() { echo "FAIL: $1" >&2; FAIL=$((FAIL+1)); }
eq()     { if [[ "$2" == "$3" ]];  then ok; else no "$1: expected '$3', got '$2'"; fi; }
has()    { if [[ "$2" == *"$3"* ]]; then ok; else no "$1: missing '$3' — output: $CURRENT_DUMP"; fi; }
hasnt()  { if [[ "$2" != *"$3"* ]]; then ok; else no "$1: must NOT contain '$3' — output: $CURRENT_DUMP"; fi; }

TMP="$(mktemp -d)"
BIN="$TMP/bin"; mkdir -p "$BIN"
DUMPS="$TMP/dumps"; mkdir -p "$DUMPS"
DUMP_IDX=0
CURRENT_DUMP="(none)"
trap 'rm -rf "$TMP"' EXIT

# ── recording stubs ──────────────────────────────────────────────────────────
# nvidia-smi: answers the query forms the script uses and RECORDS every
# invocation. `-pl` is recorded and NEVER applied — this rig's cap is
# deliberately cleared and a test must not change it.
cat > "$BIN/nvidia-smi" <<'EOS'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$NVSMI_REC"
case "$*" in
  *"index,power.limit,power.default_limit,power.min_limit,power.max_limit"*)
      printf '%s\n' "${MOCK_FULL_ROWS:-}" ;;
  *"index,power.min_limit,power.max_limit"*)
      printf '%s\n' "${MOCK_RANGE_ROWS:-}" ;;
  *"index,power.default_limit"*)
      printf '%s\n' "${MOCK_DEFAULT_ROWS:-}" ;;
  *"index,power.limit"*)
      printf '%s\n' "${MOCK_LIMIT_ROWS:-}" ;;
  *-pl*)
      exit "${MOCK_PL_RC:-0}" ;;
esac
exit 0
EOS

# systemctl: `cat` serves the unit fixture named by MOCK_UNIT_FILE (absent file
# = the unit does not exist, exit 1, like the real thing); `restart` returns
# MOCK_RESTART_RC so the fallback can be driven deterministically.
cat > "$BIN/systemctl" <<'EOS'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SYSTEMCTL_REC"
case "${1:-}" in
  cat)
      if [[ -n "${MOCK_UNIT_FILE:-}" && -f "${MOCK_UNIT_FILE}" ]]; then
        cat "$MOCK_UNIT_FILE"; exit 0
      fi
      echo "No files found for ${2:-}." >&2; exit 1 ;;
  restart) exit "${MOCK_RESTART_RC:-0}" ;;
esac
exit 0
EOS

# sudo: transparent, so `sudo nvidia-smi …` lands on the stub above.
printf '#!/usr/bin/env bash\nexec "$@"\n' > "$BIN/sudo"
chmod +x "$BIN/nvidia-smi" "$BIN/systemctl" "$BIN/sudo"

export PATH="$BIN:$PATH"
export NVSMI_REC="$TMP/nvidia-smi.calls"
export SYSTEMCTL_REC="$TMP/systemctl.calls"
export MOCK_FULL_ROWS MOCK_RANGE_ROWS MOCK_DEFAULT_ROWS MOCK_LIMIT_ROWS
export MOCK_PL_RC MOCK_UNIT_FILE MOCK_RESTART_RC
MOCK_LIMIT_ROWS="$(printf '0, 230.00\n1, 230.00')"
MOCK_DEFAULT_ROWS="$(printf '0, 370.00\n1, 420.00')"
MOCK_RANGE_ROWS="$(printf '0, 100.00, 390.00\n1, 100.00, 440.00')"
MOCK_FULL_ROWS="0, 230.00, 370.00, 100.00, 390.00"
MOCK_PL_RC=0

# ── harness integrity gate ───────────────────────────────────────────────────
# If PATH shadowing silently failed, every assertion below would still "pass"
# while the REAL sudo/nvidia-smi changed this rig's power limits. Refuse to run.
for tool in nvidia-smi systemctl sudo; do
  resolved="$(command -v "$tool" || true)"
  if [[ "$resolved" != "$BIN/$tool" ]]; then
    echo "FAIL: harness did not shadow $tool (resolved: ${resolved:-<none>}) — refusing to run" >&2
    exit 1
  fi
done

# ── unit fixtures ────────────────────────────────────────────────────────────
# What `systemctl cat` prints for the reference rig's unit: 230 W, not 250 W.
cat > "$TMP/unit-230.service" <<'EOS'
# /etc/systemd/system/nvidia-power-cap.service
[Unit]
Description=Cap both 3090s for quiet/cool operation
After=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -i 0 -pl 230
ExecStart=/usr/bin/nvidia-smi -i 1 -pl 230

[Install]
WantedBy=multi-user.target
EOS

# Long-form flag, no per-card index — must still resolve.
cat > "$TMP/unit-longform.service" <<'EOS'
# /etc/systemd/system/nvidia-power-cap.service
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi --power-limit=280
EOS

# A unit that exists but caps nothing — unparseable, must be refused, not guessed.
cat > "$TMP/unit-nopl.service" <<'EOS'
# /etc/systemd/system/nvidia-power-cap.service
[Service]
Type=oneshot
ExecStart=/bin/true
EOS

# run_powercap <label> <arg...> — full output captured to a file (never piped
# through head/tail at run time: that would swallow the diagnostic AND the exit
# status). Sets OUT / RC / CURRENT_DUMP, and truncates the call recorders first.
run_powercap() {
  local label="$1"; shift
  DUMP_IDX=$((DUMP_IDX+1))
  CURRENT_DUMP="$DUMPS/$(printf '%02d' "$DUMP_IDX")-${label}.log"
  : > "$NVSMI_REC"; : > "$SYSTEMCTL_REC"
  bash "$GPU_MODE" power-cap "$@" > "$CURRENT_DUMP" 2>&1
  RC=$?
  OUT="$(cat "$CURRENT_DUMP")"
  PL_CALLS="$(command grep -- '-pl' "$NVSMI_REC" || true)"
}

# ===========================================================================
# 1. Unit MISSING -> refuse. Nothing applied, non-zero exit.
#    Pre-fix this silently applied 250 W and exited 0.
# ===========================================================================
MOCK_UNIT_FILE="$TMP/does-not-exist.service"
MOCK_RESTART_RC=1
run_powercap "unit-missing" on
if [[ "$RC" -ne 0 ]]; then ok; else no "unit missing: expected non-zero exit, got $RC — output: $CURRENT_DUMP"; fi
has  "unit missing: says it is refusing"   "$OUT" "Refusing to re-apply a power cap"
has  "unit missing: names the unit"        "$OUT" "systemctl cat nvidia-power-cap.service"
has  "unit missing: offers the explicit escape hatch" "$OUT" "gpu-mode power-cap <WATTS>"
eq   "unit missing: NOTHING applied"       "$PL_CALLS" ""
hasnt "unit missing: no invented wattage"  "$OUT" "250"

# Same refusal when the unit exists but defines no power limit.
MOCK_UNIT_FILE="$TMP/unit-nopl.service"
run_powercap "unit-unparseable" on
if [[ "$RC" -ne 0 ]]; then ok; else no "unit unparseable: expected non-zero exit, got $RC — output: $CURRENT_DUMP"; fi
has "unit unparseable: refuses"            "$OUT" "Refusing to re-apply a power cap"
eq  "unit unparseable: NOTHING applied"    "$PL_CALLS" ""

# ===========================================================================
# 2. Unit present, systemd restart FAILS -> replay the unit's OWN values.
#    Pre-fix this applied 250 W against a unit that says 230 W.
# ===========================================================================
MOCK_UNIT_FILE="$TMP/unit-230.service"
MOCK_RESTART_RC=1
run_powercap "fallback-replays-unit" on
eq "fallback: exit 0"                      "$RC" "0"
eq "fallback: replays exactly the unit's ExecStart caps" \
   "$PL_CALLS" "$(printf -- '-i 0 -pl 230\n-i 1 -pl 230')"
has   "fallback: says what it is applying" "$OUT" "230W"
hasnt "fallback: no 250 W anywhere"        "$OUT" "250"

# Long-form `--power-limit=`, no `-i` -> applies to all cards at the unit's value.
MOCK_UNIT_FILE="$TMP/unit-longform.service"
run_powercap "fallback-longform" on
eq "fallback longform: applies the unit's value to all cards" "$PL_CALLS" "-pl 280"
has "fallback longform: label from the unit" "$OUT" "280W"

# ===========================================================================
# 3. Unit present, restart SUCCEEDS -> systemd applies it; no direct -pl.
#    The banner must quote the unit's value, not a hardcoded one.
# ===========================================================================
MOCK_UNIT_FILE="$TMP/unit-230.service"
MOCK_RESTART_RC=0
run_powercap "restart-ok" on
eq    "restart ok: exit 0"                 "$RC" "0"
eq    "restart ok: no direct -pl (systemd did it)" "$PL_CALLS" ""
has   "restart ok: restarted the unit"     "$(cat "$SYSTEMCTL_REC")" "restart nvidia-power-cap.service"
has   "restart ok: banner quotes the unit's value" "$OUT" "230W"
hasnt "restart ok: no 250 W anywhere"      "$OUT" "250"

# ===========================================================================
# 4. `off` and a custom wattage still work, and neither invents a cap value.
#    `off` reads each card's own default (they differ) — unchanged behaviour.
# ===========================================================================
run_powercap "off" off
eq    "off: exit 0"                        "$RC" "0"
eq    "off: applies each card's own default" \
      "$PL_CALLS" "$(printf -- '-i 0 -pl 370.00\n-i 1 -pl 420.00')"
has   "off: session-scope note names the unit's value" "$OUT" "230W"
hasnt "off: no 250 W anywhere"             "$OUT" "250"

run_powercap "custom" 300
eq    "custom: exit 0"                     "$RC" "0"
eq    "custom: applies the requested value" \
      "$PL_CALLS" "$(printf -- '-i 0 -pl 300\n-i 1 -pl 300')"
hasnt "custom: no 250 W anywhere"          "$OUT" "250"

# `off` with the unit absent must NOT refuse — uncapping needs no unit, and the
# message must degrade to a number-free phrase rather than invent one.
MOCK_UNIT_FILE="$TMP/does-not-exist.service"
run_powercap "off-no-unit" off
eq  "off without the unit: exit 0"         "$RC" "0"
has "off without the unit: number-free session note" "$OUT" "its configured cap"

# ===========================================================================
# 5. Static guard — no power-limit wattage literal survives in executed code.
#    Comment lines are stripped first: the comments deliberately record the old
#    250 W literal as history, and that history must not be what the gate reads.
# ===========================================================================
CURRENT_DUMP="(static scan of scripts/gpu-mode.sh)"
CODE="$(command grep -v '^[[:space:]]*#' "$GPU_MODE")"
lits="$(printf '%s\n' "$CODE" | command grep -nE '(-pl|--power-limit)[= ]+[0-9]' || true)"
eq "no -pl literal in executed code" "$lits" ""
wlits="$(printf '%s\n' "$CODE" | command grep -nE '\(?[0-9]{3}[[:space:]]*W\b' || true)"
eq "no wattage literal in executed strings" "$wlits" ""

# ===========================================================================
echo "test-gpu-mode-power-cap: ${PASS} passed, ${FAIL} failed"
[[ "$FAIL" -eq 0 ]] || exit 1
