#!/usr/bin/env bash
# test-registry-emit-no-yaml.sh — the switch.sh/launch.sh table derivation must
# be python-STDLIB-ONLY and locale-proof (#584, ryan's Proxmox VM):
#
#   1. PyYAML absent  → the regex container_name fallback must produce output
#      BYTE-IDENTICAL to the yaml parse (CLUB3090_EMIT_NO_YAML=1 forces the
#      fallback on rigs where yaml IS installed — i.e. CI and this rig).
#   2. Non-UTF-8 locale (LC_ALL=C + PYTHONUTF8=0 + PYTHONCOERCECLOCALE=0 — the
#      #599 repro recipe; modern python silently coerces C→C.UTF-8 without the
#      extra vars) → reads AND writes must still work: compose headers carry
#      unicode, and a piped stdout under the C locale would otherwise
#      UnicodeEncodeError printing status notes.
#   3. Both at once — the actual community-rig configuration that broke.
#
# The --json contract path (c3 / baselines join) legitimately requires PyYAML
# and must fail with an actionable Fix: line, not a bare traceback.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/registry-emit.sh
source "$ROOT/scripts/lib/registry-emit.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

base="$(registry_variant_rows "$ROOT")" || fail "baseline emit errored"
[[ -n "$base" ]] || fail "baseline emit produced no rows"

# 1 — yaml-less parity
noyaml="$(CLUB3090_EMIT_NO_YAML=1 registry_variant_rows "$ROOT")" \
  || fail "yaml-less emit errored (regex container_name fallback broken)"
diff <(printf '%s\n' "$base") <(printf '%s\n' "$noyaml") >/dev/null \
  || fail "yaml-less output drifts from the yaml parse (container_name regex fallback wrong for some compose)"

# 2 — C-locale (reads #599 + writes)
clocale="$(LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 registry_variant_rows "$ROOT")" \
  || fail "C-locale emit errored (encoding regression — reads need encoding=utf-8, writes need stdout reconfigure)"
diff <(printf '%s\n' "$base") <(printf '%s\n' "$clocale") >/dev/null \
  || fail "C-locale output drifts from the UTF-8 run"

# 3 — the ryan-rig combination: no PyYAML + ASCII locale
ryan="$(CLUB3090_EMIT_NO_YAML=1 LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 registry_variant_rows "$ROOT")" \
  || fail "no-yaml + C-locale emit errored (the #584 community-rig configuration)"
diff <(printf '%s\n' "$base") <(printf '%s\n' "$ryan") >/dev/null \
  || fail "no-yaml + C-locale output drifts"

# 4 — the --json path must refuse WITHOUT yaml via an actionable message.
#     Simulate by hiding yaml with a poisoned import stub on PYTHONPATH.
stubdir="$(mktemp -d)"
trap 'rm -rf "$stubdir"' EXIT
printf 'raise ImportError("PyYAML hidden by test-registry-emit-no-yaml")\n' > "$stubdir/yaml.py"
set +e
json_err="$(PYTHONPATH="$stubdir" bash "$ROOT/scripts/lib/registry-emit.sh" --json 2>&1 >/dev/null)"
json_rc=$?
set -e
[[ $json_rc -ne 0 ]] || fail "--json without PyYAML should exit non-zero"
grep -q "requires PyYAML" <<<"$json_err" || fail "--json no-yaml error lacks the actionable message (got: ${json_err:0:200})"
grep -q "Fix:" <<<"$json_err" || fail "--json no-yaml error lacks a Fix: hint"

echo "PASS: launcher table path is stdlib-only + locale-proof; --json fails actionably without PyYAML"
