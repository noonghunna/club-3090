#!/usr/bin/env bash
# test-slug-settings-store — the per-slug launch-settings store, slugs.json (#1465 phase 3c).
#
# scripts/lib/slug_settings.py is the only reader and writer of <config dir>/slugs.json.
# What it promises, and what this proves against the REAL module and its CLI:
#   * round trip: saved values read back exactly; an emptied slug entry disappears;
#   * it refuses what it can't hold — credential-looking keys (secrets stay global, in
#     secrets.env), values bash / docker compose / systemd would read differently, the
#     empty string, bad key or slug names — and a refused call saves NOTHING;
#   * schema: "version" must be 1. A newer version, bad JSON or a wrong shape is refused
#     by readers AND writers, and the file is left byte-for-byte as it was; an unknown
#     top-level key is reported and kept;
#   * atomic: the file is replaced (new inode), no temp file is left, an existing mode
#     is kept, a failed write leaves the old file intact;
#   * locked: a writer waits for <config dir>/.lock (the loader's lock file), and 12
#     concurrent writers to one slug lose nothing.
# Every call points CLUB3090_CONFIG_DIR at a temp dir: never your real settings.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
# Credentials exported in the calling shell never reach anything this test prints.
while IFS= read -r _v; do unset "$_v"; done < <(compgen -e | command grep -E '(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|MASTER_KEY|_KEY)$')
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
STORE="$ROOT/scripts/lib/slug_settings.py"
[[ -f "$STORE" ]] || { echo "  ✗ $STORE is missing" >&2; echo "test-slug-settings-store: FAIL"; exit 1; }
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }
T="$(mktemp -d)"; trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT

st() { local d="$1"; shift; CLUB3090_CONFIG_DIR="$d" python3 "$STORE" "$@"; }
sum() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
show() { st "$1" show 2>/dev/null | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), sort_keys=True))'; }

# ── 1. round trip ───────────────────────────────────────────────────────────────
echo "[1] round trip"
D="$T/rt"
st "$D" set sgl/qwen38-27b-dual-fast KV_OFFLOAD_GB=64 REASONING_EFFORT=medium 2>/dev/null || bad "set failed"
st "$D" set vllm/dual SPEC_N=0 2>/dev/null || bad "second slug: set failed"
got="$(show "$D")"
want='{"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64", "REASONING_EFFORT": "medium"}, "vllm/dual": {"SPEC_N": "0"}}'
[[ "$got" == "$want" ]] && ok "two slugs saved and read back exactly" || bad "read back '$got', want '$want'"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["version"] == 1 and set(d) == {"version","slugs"}' "$D/slugs.json" \
  && ok "the file is {\"version\": 1, \"slugs\": …}" || bad "file shape: $(cat "$D/slugs.json")"
st "$D" unset vllm/dual SPEC_N 2>/dev/null
got="$(show "$D")"
[[ "$got" == '{"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64", "REASONING_EFFORT": "medium"}}' ]] \
  && ok "unsetting a slug's last key removes the slug's entry" || bad "after unset: '$got'"
out="$(st "$D" unset sgl/qwen38-27b-dual-fast SPEC_N 2>&1)"
[[ "$out" == *"nothing (not set)"* ]] && ok "unsetting a key that isn't stored says so" || bad "unset of a missing key: '$out'"
st "$T/never" unset vllm/dual SPEC_N >/dev/null 2>&1
[[ ! -e "$T/never/slugs.json" ]] && ok "unset never creates the file" || bad "unset created $T/never/slugs.json"

# ── 2. refusals save nothing ────────────────────────────────────────────────────
echo "[2] refusals"
before="$(sum "$D/slugs.json")"
refuse() {  # <label> <want-in-message> <args…>
  local label="$1" want="$2"; shift 2
  local out rc
  out="$(st "$D" "$@" 2>&1)"; rc=$?
  if (( rc == 0 )); then bad "$label: accepted"
  elif [[ "$out" != *"$want"* ]]; then bad "$label: refused, but the message lacks '$want': $out"
  elif [[ "$(sum "$D/slugs.json")" != "$before" ]]; then bad "$label: refused, but the file changed"
  else ok "$label: refused, nothing saved"; fi
}
refuse "credential-looking key (*_TOKEN)"  "secrets.env" set vllm/dual HF_TOKEN=hf_abc
refuse "credential-looking key (*_API_KEY)" "secrets.env" set vllm/dual OPENAI_API_KEY=sk-x
refuse "a quote in the value"               "can't be stored" set vllm/dual SPEC_N='3"'
refuse "a \$ in the value"                  "can't be stored" set vllm/dual SPEC_N='$HOME'
refuse "an empty value"                     "empty value" set vllm/dual SPEC_N=
refuse "leading whitespace"                 "whitespace" set vllm/dual 'SPEC_N= 3'
refuse "a bad key name"                     "not a valid setting name" set vllm/dual 3SPEC=1
refuse "a bad slug name"                    "not a valid slug" set 'bad slug' SPEC_N=3
refuse "one bad pair among good ones"       "secrets.env" set vllm/dual SPEC_N=3 HF_TOKEN=x
got="$(CLUB3090_CONFIG_DIR="$D" python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from scripts.lib import slug_settings as s
try:
    s.set_values("vllm/dual", {"SPEC_N": "3\n4"}); print("accepted")
except s.StoreError as e:
    print("refused")' "$ROOT")"
[[ "$got" == refused && "$(sum "$D/slugs.json")" == "$before" ]] && ok "a newline in a value: refused (module API), nothing saved" || bad "newline value: $got"

# ── 3. schema ───────────────────────────────────────────────────────────────────
echo "[3] schema"
schema_case() {  # <label> <content> <want-in-message>
  local label="$1" content="$2" want="$3" S="$T/schema-$RANDOM" out rc h
  mkdir -p "$S"; printf '%s' "$content" > "$S/slugs.json"; h="$(sum "$S/slugs.json")"
  out="$(st "$S" show 2>&1)"; rc=$?
  (( rc != 0 )) && [[ "$out" == *"$want"* ]] && ok "$label: readers refuse it ('$want')" || bad "$label: show rc=$rc: $out"
  out="$(st "$S" set vllm/dual SPEC_N=3 2>&1)"; rc=$?
  if (( rc != 0 )) && [[ "$(sum "$S/slugs.json")" == "$h" ]]; then ok "$label: writers refuse it and leave it byte-for-byte"
  else bad "$label: set rc=$rc, file now: $(cat "$S/slugs.json")"; fi
}
schema_case "a newer version"        '{"version": 2, "slugs": {}}'                   "newer club-3090 (version 2)"
schema_case "no version"             '{"slugs": {}}'                                 '"version" must be 1'
schema_case "not JSON"               '{"version": 1, "slugs": {},}'                  "not valid JSON"
schema_case "slugs is a list"        '{"version": 1, "slugs": []}'                   '"slugs" must be an object'
schema_case "a number, not a string" '{"version": 1, "slugs": {"vllm/dual": {"SPEC_N": 3}}}' "must be a string"
S="$T/extra"; mkdir -p "$S"; printf '{"version": 1, "slugs": {}, "note": "mine"}' > "$S/slugs.json"
out="$(st "$S" show 2>&1 >/dev/null)"
[[ "$out" == *"unknown top-level key 'note'"* ]] && ok "an unknown top-level key is reported" || bad "no unknown-key warning: '$out'"
st "$S" set vllm/dual SPEC_N=3 2>/dev/null
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["note"] == "mine" and d["slugs"]["vllm/dual"]["SPEC_N"] == "3"' "$S/slugs.json" \
  && ok "… and kept when the file is rewritten" || bad "unknown key lost on rewrite: $(cat "$S/slugs.json")"

# ── 4. atomic replace ───────────────────────────────────────────────────────────
echo "[4] atomic replace"
A="$T/atomic"
st "$A" set vllm/dual SPEC_N=3 2>/dev/null
[[ "$(stat -c %a "$A/slugs.json")" == 644 ]] && ok "a new file is created 0644 (it never holds secrets)" || bad "new file mode $(stat -c %a "$A/slugs.json")"
chmod 600 "$A/slugs.json"
ino="$(stat -c %i "$A/slugs.json")"
st "$A" set vllm/dual SPEC_N=4 2>/dev/null
[[ "$(stat -c %i "$A/slugs.json")" != "$ino" ]] && ok "a write replaces the file (new inode: temp file + rename, never an in-place rewrite)" \
  || bad "same inode after a write — rewritten in place"
[[ "$(stat -c %a "$A/slugs.json")" == 600 ]] && ok "an existing file keeps its mode" || bad "mode now $(stat -c %a "$A/slugs.json")"
left="$(find "$A" -name '.slugs.json.*' | wc -l)"
[[ "$left" == 0 ]] && ok "no temp file left behind" || bad "$left temp file(s) left in $A"
h="$(sum "$A/slugs.json")"; chmod 500 "$A"
out="$(st "$A" set vllm/dual SPEC_N=5 2>&1)"; rc=$?; chmod 700 "$A"
(( rc != 0 )) && [[ "$(sum "$A/slugs.json")" == "$h" ]] && ok "a write that can't complete (read-only dir) fails and leaves the old file intact" \
  || bad "failed write: rc=$rc, file $(cat "$A/slugs.json")"
[[ "$out" != *Traceback* ]] && ok "… with a message, not a traceback" || bad "traceback on a failed write: $out"

# ── 5. the lock ─────────────────────────────────────────────────────────────────
echo "[5] lock"
L="$T/lock"; st "$L" set vllm/dual SPEC_N=1 2>/dev/null
python3 - "$L/.lock" > "$T/holder.out" <<'PY' &
import fcntl, sys, time
fh = open(sys.argv[1], "a")
fcntl.flock(fh, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(3)
PY
holder=$!
for _ in $(seq 1 50); do command grep -q held "$T/holder.out" 2>/dev/null && break; sleep 0.1; done
st "$L" set vllm/dual SPEC_N=2 2>/dev/null &
writer=$!
sleep 1.5
mid="$(show "$L")"
[[ "$mid" == '{"vllm/dual": {"SPEC_N": "1"}}' ]] && ok "a writer waits while <config dir>/.lock is held" || bad "the writer did not wait for the lock: $mid"
wait "$holder"; wait "$writer"
[[ "$(show "$L")" == '{"vllm/dual": {"SPEC_N": "2"}}' ]] && ok "… and writes once it is released" || bad "after release: $(show "$L")"

C="$T/concurrent"
for i in $(seq 1 12); do st "$C" set vllm/dual "K$i=v$i" 2>/dev/null & done
wait
n="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["slugs"]["vllm/dual"]))' "$C/slugs.json" 2>/dev/null)"
[[ "$n" == 12 ]] && ok "12 concurrent writers to one slug: all 12 keys kept" || bad "concurrent writers: $n of 12 keys kept"

[[ $fail -eq 0 ]] && echo "test-slug-settings-store: ok" || echo "test-slug-settings-store: FAIL"
exit $fail
