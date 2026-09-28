#!/usr/bin/env bash
# Gate: setup.sh's weight download must survive ONE transient network stall
# (club-3090#1305).
#
# THE DEFECT
# ----------
# `_hf_download_repo` called `hf download` exactly once. Under `set -euo
# pipefail` a single mid-transfer stall — reproduced in the wild as
# `httpx.ConnectTimeout: _ssl.c:1015: The handshake operation timed out`
# against a hub that was otherwise answering in 0.65s — aborted the whole setup
# run with a ~40-line unhandled Python traceback, 1.1 GB into a 17 GB download.
#
# The recovery was trivial and entirely undiscoverable: re-running the identical
# command resumed from the `.incomplete` markers the hub client had already
# written. Nothing in the output said so, so a first-time user reads the
# traceback as "this is broken" rather than "run it again".
#
# ASSERTED HERE (hermetic — no network, no weights, no GPU)
# ---------------------------------------------------------
#   1. a clean download still takes exactly one attempt (no gratuitous retries);
#   2. two transient failures followed by success => the download SUCCEEDS, and
#      the retry is announced rather than silent;
#   3. a permanent failure is BOUNDED (it does not retry forever) and exits
#      non-zero;
#   4. that final failure message says the transfer is resumable and that
#      re-running the same command continues it — the sentence whose absence
#      was the actual user-visible bug.
#
# The function is mid-script in a setup.sh that would otherwise run a whole
# installer, so extract it and drive it directly (the shape
# test-setup-verify-count.sh uses for _verify_downloaded_files).
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SETUP="$ROOT_DIR/scripts/setup.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

awk '/^_hf_download_repo\(\) \{/{f=1} f{print} f&&/^\}$/{exit}' "$SETUP" > "$TMP/fn.sh"
command grep -q '_hf_download_repo()' "$TMP/fn.sh" \
  || fail "could not extract _hf_download_repo from setup.sh (did its shape change?)"
command grep -q '^}$' "$TMP/fn.sh" || fail "extraction did not capture the function's closing brace"

mkdir -p "$TMP/bin"
# Fake hf CLI: fails the first $(cat failuntil) attempts the way the real one
# does — non-zero exit with a traceback tail on stderr — then succeeds.
cat > "$TMP/bin/hf" <<SH
#!/usr/bin/env bash
n=\$(cat "$TMP/attempts" 2>/dev/null || echo 0)
n=\$((n + 1)); printf '%s' "\$n" > "$TMP/attempts"
if [ "\$n" -le "\$(cat "$TMP/failuntil")" ]; then
  echo "Traceback (most recent call last):" >&2
  echo "httpx.ConnectTimeout: _ssl.c:1015: The handshake operation timed out" >&2
  exit 1
fi
exit 0
SH
chmod +x "$TMP/bin/hf"

# run_case <fail-this-many-attempts> — sets OUT, RC and ATTEMPTS as globals.
# NOT called in $( ): command substitution runs it in a subshell, and the rc it
# captured would never reach the caller (it reads as "unbound variable", which is
# at least loud — a defaulted one would have silently scored every case as pass).
run_case() {
  printf '%s' "$1" > "$TMP/failuntil"
  printf '0' > "$TMP/attempts"
  set +e
  OUT="$(PATH="$TMP/bin:$PATH" \
    MODEL_DIR="$TMP/models" \
    HF_DOWNLOAD_RETRY_SLEEP=0 \
    bash -c '
      set -uo pipefail
      source "'"$TMP"'/fn.sh"
      ensure_hf_cli() { return 0; }
      _hf_download_repo some/repo subdir
    ' 2>&1)"
  RC=$?
  set -e
  ATTEMPTS="$(cat "$TMP/attempts")"
}

# --- 1. clean download: exactly one attempt --------------------------------
run_case 0; out="$OUT"
[[ "$RC" -eq 0 ]] || fail "case 1: a clean download failed (rc=$RC): $out"
[[ "$ATTEMPTS" == "1" ]] || fail "case 1: clean download took $ATTEMPTS attempts, expected 1"
echo "  ✓ a clean download still takes exactly one attempt"

# --- 2. two transient stalls, then success ---------------------------------
run_case 2; out="$OUT"
[[ "$RC" -eq 0 ]] || fail "case 2: two transient stalls were not survived (rc=$RC): $out"
[[ "$ATTEMPTS" == "3" ]] || fail "case 2: expected 3 attempts (2 stalls + 1 success), got $ATTEMPTS"
command grep -qi 'retry' <<<"$out" || fail "case 2: the retry was silent — nothing in the output says it retried: $out"
echo "  ✓ two transient stalls are retried and the download completes"

# --- 3. permanent failure is bounded and non-zero --------------------------
run_case 999; out="$OUT"
[[ "$RC" -ne 0 ]] || fail "case 3: a permanently failing download reported success"
(( ATTEMPTS >= 2 )) || fail "case 3: a permanent failure was never retried at all ($ATTEMPTS attempt)"
(( ATTEMPTS <= 8 )) || fail "case 3: unbounded retry — $ATTEMPTS attempts"
echo "  ✓ a permanent failure retries a bounded number of times ($ATTEMPTS) and exits non-zero"

# --- 4. the give-up message tells the user what to do ----------------------
command grep -qi 'resum' <<<"$out" \
  || fail "case 4: the give-up message never says the transfer is resumable: $out"
command grep -qiE 're-?run|run (it|the same)' <<<"$out" \
  || fail "case 4: the give-up message never says re-running continues the download: $out"
echo "  ✓ the give-up message says the transfer is resumable and re-running continues it"

# --- 5. a mistyped knob errors clearly, it does not crash ------------------
# `(( attempt >= max ))` resolves a non-numeric `max` as a VARIABLE name, so
# under `set -u` the pre-fix shape died with "foo: unbound variable" — a crash
# inside the code that exists to handle crashes.
printf '1' > "$TMP/failuntil"; printf '0' > "$TMP/attempts"
set +e
bad="$(PATH="$TMP/bin:$PATH" MODEL_DIR="$TMP/models" HF_DOWNLOAD_RETRIES=foo \
  bash -c '
    set -uo pipefail
    source "'"$TMP"'/fn.sh"
    ensure_hf_cli() { return 0; }
    _hf_download_repo some/repo subdir
  ' 2>&1)"
bad_rc=$?
set -e
[[ "$bad_rc" -ne 0 ]] || fail "case 5: a non-numeric HF_DOWNLOAD_RETRIES was accepted"
command grep -q 'unbound variable' <<<"$bad" \
  && fail "case 5: a mistyped knob crashes on an unbound variable instead of erroring: $bad"
command grep -q 'HF_DOWNLOAD_RETRIES' <<<"$bad" \
  || fail "case 5: the error does not name the offending variable: $bad"
echo "  ✓ a mistyped retry knob gets a clear error, not an unbound-variable crash"

echo "test-setup-download-retry: ok (club-3090#1305)"
