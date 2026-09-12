#!/usr/bin/env bash
# test-report-power-state — a shared report must say what power envelope it was
# taken under, and must never let "power was not read" look like "power is fine".
#
# Why this test exists. A run on this stack measured ~40% below its own recorded
# baseline with a signature that read as an upstream speculative-decoding
# regression: spec_n=1 and spec_n=2 produced identical throughput while draft
# acceptance stayed healthy. An upstream bug report was nearly filed. The cause
# was a persistent 230 W cap against 370/420 W factory defaults, applied at boot
# by a systemd unit, and found only by manually reading `power.limit`.
#
# report.sh has emitted a per-GPU wattage since its first commit, so "is the
# number present" was never the gap. The gap was interpretability:
#   1. a cap was a soft parenthetical nested two levels into a mid-report
#      section, in a paste that gets truncated from the bottom;
#   2. it was stated in absolute watts, and whoever triages someone else's rig
#      does not know that card's stock TDP, so "limit=230.00 W" reads as normal;
#   3. when nvidia-smi was installed but its query FAILED, the GPU section
#      rendered as a heading with nothing under it — so the absence of a cap
#      warning was indistinguishable from a cap nobody ever looked for.
#
# Contract asserted here:
#   - a cap below default surfaces UP FRONT, per GPU, as watts-of-watts AND a
#     percentage of that card's own default;
#   - an uncapped rig emits no cap warning anywhere (the negative control — a
#     banner that always fired would assert nothing);
#   - enforced.power.limit is surfaced when, and only when, it undercuts the
#     set limit;
#   - a present-but-unreadable query says "NOT READ" explicitly;
#   - a machine with no nvidia-smi at all still produces a report.
set -uo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0
DUMP_DIR="$(mktemp -d)"
DUMP_IDX=0
CURRENT_DUMP="(none)"

ok() { PASS=$((PASS+1)); }
no() { echo "FAIL: $1" >&2; FAIL=$((FAIL+1)); }
eq() { if [[ "$2" == "$3" ]]; then ok; else no "$1: expected '$3', got '$2'"; fi; }
# Full report output is CAPTURED to a file and only its path is printed: nine
# failing assertions each dumping 80 lines of report is unreadable, and the
# diagnostic still has to survive.
has()   { if [[ "$2" == *"$3"* ]]; then ok; else no "$1: missing '$3' — report: $CURRENT_DUMP"; fi; }
hasnt() { if [[ "$2" != *"$3"* ]]; then ok; else no "$1: must NOT contain '$3' — report: $CURRENT_DUMP"; fi; }

# ===========================================================================
# Layer 1 — gpu_power_caps() as a unit. Extracted with sed (the repo pattern)
# so report.sh's main body does not run.
# ===========================================================================
HELPERS="$(mktemp --suffix=.sh)"
tmp="$(mktemp -d)"
cleanup_unit() { rm -rf "$tmp" "$HELPERS"; }
trap cleanup_unit EXIT

sed -n '/^gpu_power_caps()/,/^}/p' scripts/report.sh > "$HELPERS"
if [[ ! -s "$HELPERS" ]]; then
  # Recorded as a failure rather than exiting, so a run against a report.sh
  # WITHOUT the capture still exercises the end-to-end assertions below and
  # reports every gap rather than only the first.
  no "gpu_power_caps() not found in scripts/report.sh — the power-state capture is absent"
  printf 'gpu_power_caps() { return 0; }\n' > "$HELPERS"
fi
printf 'have() { command -v "$1" >/dev/null 2>&1; }\n' >> "$HELPERS"

# shellcheck source=/dev/null
source "$HELPERS"

# A scripted nvidia-smi whose rows come from the environment, so one stub serves
# every arm below.
cat > "$tmp/nvidia-smi" <<'EOS'
#!/usr/bin/env bash
if [[ "$*" == *"index,power.limit,power.default_limit"* ]]; then
  printf '%s\n' "${MOCK_ROWS:-}"
fi
exit 0
EOS
chmod +x "$tmp/nvidia-smi"
export PATH="$tmp:$PATH"

# 1a. both cards capped — the incident's exact numbers
export MOCK_ROWS="$(printf '0, 230.00, 370.00\n1, 230.00, 420.00')"
eq "capped: both cards reported with percentages" "$(gpu_power_caps)" "$(printf '0|230|370|62\n1|230|420|54')"

# 1b. NEGATIVE CONTROL — nothing capped means no rows at all
export MOCK_ROWS="$(printf '0, 370.00, 370.00\n1, 420.00, 420.00')"
eq "uncapped: no rows" "$(gpu_power_caps)" ""

# 1c. one capped, one not — only the capped card is listed
export MOCK_ROWS="$(printf '0, 230.00, 370.00\n1, 420.00, 420.00')"
eq "mixed: only the capped card" "$(gpu_power_caps)" "0|230|370|62"

# 1d. an overclocked card is not a cap
export MOCK_ROWS="0, 390.00, 370.00"
eq "above default is not a cap" "$(gpu_power_caps)" ""

# 1e. [N/A] / [Not Supported] must not be scored as a cap (would be a false alarm)
export MOCK_ROWS="$(printf '0, [N/A], [N/A]\n1, [Not Supported], 420.00')"
eq "non-numeric fields yield no rows" "$(gpu_power_caps)" ""

# 1f. a machine with no nvidia-smi must not error. gpu_power_caps uses only
# builtins plus nvidia-smi, so an empty PATH is a faithful "no GPU" rig.
out="$(PATH="$tmp/definitely-absent" gpu_power_caps 2>&1)"; rc=$?
eq "no nvidia-smi: empty output" "$out" ""
eq "no nvidia-smi: exit 0" "$rc" "0"
unset MOCK_ROWS

# ===========================================================================
# Layer 2 — the REAL report.sh end to end, under a scripted nvidia-smi.
# No GPU, no engine, no container.
# ===========================================================================
trap - EXIT
cleanup_unit

# shellcheck source=fixtures/report-harness/report-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/report-harness/report-env.sh"
report_env_init "$ROOT_DIR"
cleanup_all() { report_env_cleanup; }
trap cleanup_all EXIT

# report_run re-enables `set -e` on exit; this test counts failures instead of
# dying on the first one, so -e is turned back off and each report is saved.
run_report() {
  report_run "$@"
  set +e
  DUMP_IDX=$((DUMP_IDX+1))
  CURRENT_DUMP="${DUMP_DIR}/report-${DUMP_IDX}.md"
  printf '%s\n' "$REPORT_OUT" > "$CURRENT_DUMP"
}

# Answers every nvidia-smi call the GPU section makes. Limits come from the
# environment so the capped and uncapped arms share one stub. Dispatch order
# matters: the big identity query also contains the substring "power.limit".
install_nvsmi() {
  report_nvsmi_stub <<'EOS'
#!/usr/bin/env bash
args="$*"
L0="${MOCK_LIM0:-370.00}"; L1="${MOCK_LIM1:-420.00}"
D0="${MOCK_DEF0:-370.00}"; D1="${MOCK_DEF1:-420.00}"
E0="${MOCK_ENF0:-$L0}";    E1="${MOCK_ENF1:-$L1}"
case "$args" in
  *enforced.power.limit*)
    # big per-GPU identity + envelope query (csv,noheader — units included)
    [[ "${MOCK_QUERY_FAILS:-0}" == "1" ]] && exit 1
    if [[ "${MOCK_NO_ENFORCED_FIELD:-0}" == "1" ]]; then
      # Verbatim shape of a real nvidia-smi meeting an unknown field: the error
      # goes to STDOUT and the exit code is non-zero.
      echo 'Field "enforced.power.limit" is not a valid field to query.'
      exit 2
    fi
    echo "0, MockForce RTX 3090, 24576 MiB, 610.57.04, 94.02.42.80.88, Enabled, ${L0} W, ${D0} W, 390.00 W, 41.00 W, 00000000:01:00.0, 1, 4, 16, 16, ${E0} W"
    echo "1, MockForce RTX 3090, 24576 MiB, 610.57.04, 94.02.42.C0.05, Enabled, ${L1} W, ${D1} W, 450.00 W, 28.00 W, 00000000:02:00.0, 1, 4, 8, 16, ${E1} W"
    exit 0 ;;
  *index,power.limit,power.default_limit*)
    # gpu_power_caps (csv,noheader,nounits)
    [[ "${MOCK_QUERY_FAILS:-0}" == "1" ]] && exit 1
    echo "0, ${L0}, ${D0}"
    echo "1, ${L1}, ${D1}"
    exit 0 ;;
  *pcie.link.width.max*)
    # the fallback field set, without enforced.power.limit (older driver)
    [[ "${MOCK_QUERY_FAILS:-0}" == "1" ]] && exit 1
    echo "0, MockForce RTX 3090, 24576 MiB, 470.00, 94.02.42.80.88, Enabled, ${L0} W, ${D0} W, 390.00 W, 41.00 W, 00000000:01:00.0, 1, 4, 16, 16"
    echo "1, MockForce RTX 3090, 24576 MiB, 470.00, 94.02.42.C0.05, Enabled, ${L1} W, ${D1} W, 450.00 W, 28.00 W, 00000000:02:00.0, 1, 4, 8, 16"
    exit 0 ;;
  -L) echo "GPU 0: MockForce RTX 3090 (UUID: GPU-0)"
      echo "GPU 1: MockForce RTX 3090 (UUID: GPU-1)"; exit 0 ;;
  *ecc.mode.current*)  echo "[N/A]"; exit 0 ;;
  *index,memory.used*) echo "0, 0"; echo "1, 0"; exit 0 ;;
  *nvlink*)            exit 1 ;;
  *topo*-p2p*)         exit 1 ;;
  *topo*)              echo "     GPU0 GPU1"; echo "GPU0  X   SYS"; echo "GPU1 SYS   X"; exit 0 ;;
  "")                  echo "CUDA Version: 13.2"; exit 0 ;;
  *)                   exit 0 ;;
esac
EOS
}
install_nvsmi

# --- 2a. CAPPED: the incident state — 230 W on cards defaulting to 370/420 ---
export MOCK_LIM0=230.00 MOCK_LIM1=230.00
run_report --no-redact
CAPPED_OUT="$REPORT_OUT"; CAPPED_DUMP="$CURRENT_DUMP"
eq "capped report still exits 0" "$REPORT_RC" "0"

has "capped: up-front banner"         "$CAPPED_OUT" "GPU power cap active"
has "capped: banner warns on numbers" "$CAPPED_OUT" "NOT at stock power"
has "capped: GPU0 watts-of-watts + %" "$CAPPED_OUT" "GPU 0 capped to 230 W of 370 W default (62%)"
has "capped: GPU1 watts-of-watts + %" "$CAPPED_OUT" "GPU 1 capped to 230 W of 420 W default (54%)"
has "capped: per-GPU line states % of that card's own default" \
    "$CAPPED_OUT" "limit=230.00 W (62% of 370.00 W default, max=390.00 W)"
has "capped: per-GPU anomaly marker"  "$CAPPED_OUT" "CAPPED BELOW DEFAULT"
has "capped: names the reproducing command" "$CAPPED_OUT" \
    "nvidia-smi --query-gpu=index,power.limit,power.default_limit,enforced.power.limit --format=csv"

# The banner must PRECEDE the GPU section: a paste truncated from the bottom has
# to still carry the warning. `|| true` — grep exiting 1 under pipefail would
# otherwise abort the run instead of failing this assertion.
banner_line="$(command grep -n 'GPU power cap active' "$CAPPED_DUMP" 2>/dev/null | head -1 | cut -d: -f1 || true)"
gpusec_line="$(command grep -n '^## GPU hardware' "$CAPPED_DUMP" 2>/dev/null | head -1 | cut -d: -f1 || true)"
if [[ -n "$banner_line" && -n "$gpusec_line" && "$banner_line" -lt "$gpusec_line" ]]; then ok
else no "banner must appear before '## GPU hardware' (banner=${banner_line:-absent} gpu=${gpusec_line:-absent}) — report: $CAPPED_DUMP"; fi
unset MOCK_LIM0 MOCK_LIM1

# --- 2b. NEGATIVE CONTROL: an uncapped rig warns about nothing ---------------
run_report --no-redact
eq "uncapped report exits 0" "$REPORT_RC" "0"
hasnt "uncapped: no banner"        "$REPORT_OUT" "GPU power cap active"
hasnt "uncapped: no cap marker"    "$REPORT_OUT" "CAPPED BELOW DEFAULT"
hasnt "uncapped: no bogus percent" "$REPORT_OUT" "% of 370.00 W default"
# ...but the envelope is still stated, so a reader can see it WAS read.
has "uncapped: envelope still stated" "$REPORT_OUT" "limit=370.00 W (default=370.00 W, max=390.00 W)"

# --- 2c. enforced.power.limit undercutting the set limit is surfaced ---------
export MOCK_ENF0=200.00
run_report --no-redact
has "enforced below set limit is flagged"  "$REPORT_OUT" "enforced=200.00 W"
has "enforced flag explains the mechanism" "$REPORT_OUT" "thermal / HW slowdown"
unset MOCK_ENF0
# ...and stays silent when enforced == limit, so the common case costs nothing.
run_report --no-redact
hasnt "enforced == limit prints nothing" "$REPORT_OUT" "enforced="

# --- 2d. THE FALSE-CLEAN: nvidia-smi installed, query fails ------------------
# Pre-fix this rendered '## GPU hardware' followed by NOTHING, so "no cap
# warning" was indistinguishable from "power was never read".
export MOCK_QUERY_FAILS=1
run_report --no-redact
eq "unreadable query still exits 0" "$REPORT_RC" "0"
has "unreadable: says the fields could not be read"   "$REPORT_OUT" "GPU fields unreadable"
has "unreadable: says POWER specifically was not read" "$REPORT_OUT" "Power limit: NOT READ"
has "unreadable: denies the clean reading" "$REPORT_OUT" \
    "does *not* mean these cards are at their default power limit"
hasnt "unreadable: must not claim a cap" "$REPORT_OUT" "GPU power cap active"
# ...and the section must not be an empty block.
gpu_body="$(sed -n '/^## GPU hardware/,/^### /p' "$CURRENT_DUMP" | sed '1d;$d' | tr -d '[:space:]' || true)"
if [[ -n "$gpu_body" ]]; then ok
else no "GPU hardware section is an EMPTY block when the query fails — report: $CURRENT_DUMP"; fi
unset MOCK_QUERY_FAILS

# --- 2d-bis. a driver that does not know enforced.power.limit ----------------
# nvidia-smi prints "Field ... is not a valid field to query." to STDOUT and
# exits non-zero, so an emptiness check would have parsed that error text as a
# GPU row. The section must fall back to the universal field set and still
# report the cap — losing one optional column, not the whole section.
export MOCK_NO_ENFORCED_FIELD=1 MOCK_LIM0=230.00 MOCK_LIM1=230.00
run_report --no-redact
eq "old driver still exits 0" "$REPORT_RC" "0"
has   "old driver: GPU rows survive the fallback" "$REPORT_OUT" "**GPU 0:**"
# The discriminator: the fallback field set reports driver 470.00, the primary
# one 610.57.04. Without this, every other assertion in this arm would also
# pass if the fallback had never been taken.
has   "old driver: the FALLBACK query is what answered" "$REPORT_OUT" "driver 470.00"
has   "old driver: cap still reported"            "$REPORT_OUT" "62% of 370.00 W default"
has   "old driver: banner still fires"            "$REPORT_OUT" "GPU power cap active"
hasnt "old driver: error text never parsed as a GPU row" "$REPORT_OUT" "is not a valid field to query"
hasnt "old driver: no bogus enforced value" "$REPORT_OUT" "enforced="
unset MOCK_NO_ENFORCED_FIELD MOCK_LIM0 MOCK_LIM1

# --- 2e. no nvidia-smi at all: a GPU-less machine still reports --------------
printf '#!/usr/bin/env bash\nexit 1\n' > "${REPORT_FAKE_BIN}/nvidia-smi"
run_report --no-redact
eq "no-GPU machine exits 0" "$REPORT_RC" "0"
has   "no-GPU machine still emits the GPU section" "$REPORT_OUT" "## GPU hardware"
hasnt "no-GPU machine claims no cap"               "$REPORT_OUT" "GPU power cap active"

# --- 2f. the warning survives redaction (it carries no host/user/path) ------
install_nvsmi
export MOCK_LIM0=230.00 MOCK_LIM1=230.00
run_report
has "redacted run still warns about the cap" "$REPORT_OUT" "GPU power cap active"
has "redacted run keeps the percentage"      "$REPORT_OUT" "of 370 W default (62%)"
unset MOCK_LIM0 MOCK_LIM1

echo "----------------------------------------"
echo "PASS: $PASS  FAIL: $FAIL"
if [[ "$FAIL" -ne 0 ]]; then
  echo "Saved reports: $DUMP_DIR" >&2
  exit 1
fi
rm -rf "$DUMP_DIR"
echo "OK: report.sh power-state capture"
