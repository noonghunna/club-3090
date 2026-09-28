#!/usr/bin/env bash
# test-setup-verify-count.sh — #857: setup.sh must never count an unverifiable
# file as SHA-verified.
#
# The defect: `_verify_downloaded_files` printed `SKIP (no etag)` when the etag
# lookup came back empty, did NOT count it as a failure, and then printed
# `[done] N file(s) SHA-verified` with N including every skipped file. So the
# reassuring line was emitted over files that nothing had checked.
#
# Compounding it, the lookup used a redirect-FOLLOWING `curl -sfI`. On a
# Xet-backed repo the 302 lands on the CAS bridge, whose response carries the
# CAS blob ETag and no `x-linked-etag` — the canonical sha256 lives on the FIRST
# hop. So on modern repos the SKIP was the NORMAL outcome, not a rare edge, and
# `downloader.py` hard-failed the identical case. Two surfaces, two meanings of
# "verified", and the weaker one printed the comforting message. Same failure
# shape as the historical "DONE (hash-verified)" silent-corruption incident.
#
# Asserted here (hermetic — no network, no downloads, no GPU):
#   1. the redirect-following curl etag lookup is GONE from setup.sh, and the
#      lookup delegates to hf_fetch.resolve_head (#855's no-redirect opener)
#      rather than carrying a second implementation;
#   2. an all-hashed pull reports every file verified;
#   3. a file with NO published hash is reported UNVERIFIED and is EXCLUDED
#      from the verified count — this is the defect itself;
#   4. the word "verified" NEVER attaches to a size-only observation (the same
#      rule test-pull-verify-in-place.sh asserts for the downloader surfaces);
#   5. a hash MISMATCH still hard-fails;
#   6. a SIZE mismatch on a hashless file hard-fails too — it is the only
#      cross-check left, so a failure there is real evidence of corruption.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SETUP="$ROOT_DIR/scripts/setup.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# ===========================================================================
# 1. the lookup itself: no-redirect, and not re-implemented here
# ===========================================================================
# The old line was:
#   expected="$(curl -sfI "https://huggingface.co/${repo}/resolve/..." | grep -i '^x-linked-etag:' ...)"
# `curl -I` follows redirects only with -L, but the historic call chain resolved
# through the bridge regardless; either way the fix is to stop parsing headers
# in bash and use the one implementation that knows which hop carries the hash.
if command grep -nE "curl -[a-zA-Z]*I[a-zA-Z]* .*resolve/" "$SETUP" | command grep -q 'x-linked-etag' \
   || command grep -q "grep -i '\^x-linked-etag:'" "$SETUP"; then
  command grep -n 'x-linked-etag' "$SETUP" >&2
  fail "setup.sh still parses x-linked-etag out of a bash curl HEAD — that is the lookup #857 replaced"
fi
command grep -q 'hf_fetch' "$SETUP" \
  || fail "setup.sh must delegate the hash lookup to hf_fetch (#855's no-redirect resolve HEAD)"
command grep -q 'resolve_head' "$SETUP" \
  || fail "setup.sh must call hf_fetch.resolve_head, not re-implement the resolve hop"
echo "  ✓ the etag lookup delegates to hf_fetch.resolve_head — no second bash implementation"

# ===========================================================================
# 2-6. the counting logic, against a stubbed metadata source
# ===========================================================================
# `_verify_downloaded_files` is mid-script in a setup.sh that would otherwise
# run a whole installer, so extract just the function and drive it directly.
# The extraction is anchored on a column-0 `}` — the function's own closing
# brace — which is the file's style throughout.
awk '/^_verify_downloaded_files\(\) \{/{f=1} f{print} f&&/^\}$/{exit}' "$SETUP" > "$TMP/fn.sh"
command grep -q '_verify_downloaded_files()' "$TMP/fn.sh" \
  || fail "could not extract _verify_downloaded_files from setup.sh (did its shape change?)"
command grep -q '^}$' "$TMP/fn.sh" || fail "extraction did not capture the function's closing brace"

# Drive one scenario. $1=name, then a list of "<file>:<sha-or-EMPTY>:<size-or-EMPTY>"
# describing what the HUB publishes; the on-disk bytes are written by the caller.
run_verify() {   # $1=scenario dir  -> prints combined output, returns the rc
  local dir="$1"
  (
    set +e
    cat > "$TMP/harness.sh" <<HARNESS
set -uo pipefail
MODEL_DIR="$dir"
ROOT_DIR="$ROOT_DIR"
# Stub the metadata source: the real one is network-bound and its no-redirect
# behaviour is hf_fetch's own contract (asserted in section 1 + its own suite).
# What is under test HERE is what the counter does with each answer. Mirrors the
# real contract exactly: "<sha|-> <size|->", non-zero only when the hub said
# nothing at all.
_hf_remote_meta() {
  local key="\$3"
  local line
  line="\$(command grep -m1 "^\${key}|" "$dir/.hubmeta" 2>/dev/null || true)"
  [[ -n "\$line" ]] || return 1
  echo "\$(cut -d'|' -f2 <<<"\$line") \$(cut -d'|' -f3 <<<"\$line")"
}
source "$TMP/fn.sh"
_verify_downloaded_files "org/Repo" "weights" "*.safetensors" "main"
HARNESS
    bash "$TMP/harness.sh" 2>&1
  )
}

new_case() {   # $1=name -> echoes the scenario dir
  local d="$TMP/$1/weights"
  mkdir -p "$d"
  : > "$TMP/$1/.hubmeta"
  echo "$TMP/$1"
}

# --- 2. every file hashed → every file verified -----------------------------
D="$(new_case allhashed)"
for n in a b c; do
  printf 'weights-%s' "$n" > "$D/weights/$n.safetensors"
  sha="$(sha256sum "$D/weights/$n.safetensors" | awk '{print $1}')"
  sz="$(stat -c '%s' "$D/weights/$n.safetensors")"
  echo "$n.safetensors|$sha|$sz" >> "$D/.hubmeta"
done
out="$(run_verify "$D")" || fail "an all-hashed, all-matching pull must succeed:\n$out"
command grep -q '3/3 file(s) SHA-verified' <<<"$out" \
  || { echo "$out" >&2; fail "an all-hashed pull must report 3/3 verified"; }
command grep -q 'UNVERIFIED' <<<"$out" \
  && { echo "$out" >&2; fail "an all-hashed pull must not report anything UNVERIFIED"; }
echo "  ✓ all-hashed pull: 3/3 verified, nothing flagged"

# --- 3+4. the #857 defect: hashless files must NOT reach the verified count --
D="$(new_case mixed)"
for n in a b c; do printf 'weights-%s' "$n" > "$D/weights/$n.safetensors"; done
# only `a` has a published hash; b/c are the Xet-backed shape — a correct SIZE
# and no hash, which is precisely the combination that used to read as verified.
sha="$(sha256sum "$D/weights/a.safetensors" | awk '{print $1}')"
echo "a.safetensors|$sha|$(stat -c '%s' "$D/weights/a.safetensors")" >> "$D/.hubmeta"
for n in b c; do
  echo "$n.safetensors|-|$(stat -c '%s' "$D/weights/$n.safetensors")" >> "$D/.hubmeta"
done
out="$(run_verify "$D")" || fail "hashless files are UNVERIFIED, not a hard failure:\n$out"
command grep -q '1/3 file(s) SHA-verified' <<<"$out" \
  || { echo "$out" >&2; fail "THE DEFECT: only the hash-checked file may count as SHA-verified (got the line above)"; }
command grep -q '3/3 file(s) SHA-verified' <<<"$out" \
  && { echo "$out" >&2; fail "THE DEFECT: unverifiable files were counted as SHA-verified"; }
command grep -q '2/3 file(s) UNVERIFIED' <<<"$out" \
  || { echo "$out" >&2; fail "the unverifiable files must be reported, and counted, as UNVERIFIED"; }
command grep -qE 'b\.safetensors +UNVERIFIED' <<<"$out" \
  || { echo "$out" >&2; fail "each unverifiable file must be named on its own line"; }
# A matching size must be labelled as what it is, not quietly upgraded.
command grep -q 'NOT a verification' <<<"$out" \
  || { echo "$out" >&2; fail "a size match must say out loud that it is not a verification"; }
# The rule, stated as a test: "verified" never appears on a line that only saw a
# size. A wrapper on this rig once printed "DONE (hash-verified)" over silent
# corruption; this makes that phrasing a failure, not a style note.
while IFS= read -r line; do
  command grep -qiE '\bverified\b' <<<"${line/UNVERIFIED/}" \
    && { echo "$line" >&2; fail "the word 'verified' attached to a size-only observation"; }
done < <(command grep 'UNVERIFIED' <<<"$out")
echo "  ✓ hashless files: reported UNVERIFIED, excluded from the verified count, never called verified"

# --- 5. hash mismatch still hard-fails --------------------------------------
D="$(new_case badhash)"
printf 'weights-a' > "$D/weights/a.safetensors"
echo "a.safetensors|$(printf 'deadbeef%.0s' 1 2 3 4 5 6 7 8)|9" >> "$D/.hubmeta"
if out="$(run_verify "$D")"; then
  echo "$out" >&2; fail "a sha256 mismatch must hard-fail"
fi
command grep -q 'FAIL' <<<"$out" || { echo "$out" >&2; fail "the mismatch must be named per-file"; }
command grep -q 'failed their integrity check' <<<"$out" \
  || { echo "$out" >&2; fail "the summary must say the run failed"; }
echo "  ✓ sha256 mismatch hard-fails"

# --- 6. size mismatch on a hashless file also hard-fails --------------------
# Size is not a verification, but a size DISAGREEMENT is real evidence: the hub
# says N bytes and the disk says something else. Passing that as "unverified but
# fine" would be the truncated-download case going quiet.
D="$(new_case badsize)"
printf 'weights-a' > "$D/weights/a.safetensors"
echo "a.safetensors|-|999999" >> "$D/.hubmeta"     # no sha, and the size disagrees
if out="$(run_verify "$D")"; then
  echo "$out" >&2; fail "a size disagreement on a hashless file must hard-fail (truncated download)"
fi
command grep -q 'size mismatch' <<<"$out" \
  || { echo "$out" >&2; fail "the size failure must name itself as a size mismatch"; }
echo "  ✓ size mismatch on a hashless file hard-fails"

# --- 7. no metadata at all → still UNVERIFIED, still counted honestly -------
D="$(new_case nometa)"
printf 'weights-a' > "$D/weights/a.safetensors"
: > "$D/.hubmeta"                                  # the lookup itself fails
out="$(run_verify "$D")" || fail "a lookup failure is UNVERIFIED, not fatal:\n$out"
command grep -q 'hub published neither hash nor size' <<<"$out" \
  || { echo "$out" >&2; fail "with no metadata at all the file must still be reported UNVERIFIED with the reason"; }
command grep -q '0/1 file(s) SHA-verified' <<<"$out" \
  || { echo "$out" >&2; fail "a pull where nothing could be hashed must report 0 verified"; }
command grep -q '1/1 file(s) UNVERIFIED' <<<"$out" \
  || { echo "$out" >&2; fail "the UNVERIFIED tally must cover it"; }
echo "  ✓ no-metadata files: 0 verified, reported with the reason, not silently passed"

echo "test-setup-verify-count: ok"
