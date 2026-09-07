#!/usr/bin/env bash
# Guard: compose-fact derivation has exactly ONE implementation, reachable from
# BOTH the CLI (scripts/) and the cockpit (tools/serve-cockpit/).
#
# Why this exists (#1202 P1): derive_compose_facts used to live inside
# club3090_cockpit.data. Nothing outside the cockpit could import it, which is
# *why* the local model layer was write-only from the UI (#1153) -- the UI owned
# the only implementation of compose->spec derivation. It now lives in
# scripts/lib/profiles/compose_facts.py and the cockpit delegates.
#
# The failure this pins is a silent re-fork: someone "fixes" a parsing bug in one
# consumer and the two drift. The test derives the SAME compose through BOTH
# import paths and requires byte-identical results.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
rc=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/c.yml" <<'YML'
# Status: 🧪 Experimental
services:
  my-local-thing:
    image: ghcr.io/someuser/my-llamacpp:v9.9-custom
    ports:
      - "${PORT:-8199}:8199"
    command: >
      /app/llama-server -m /models/m.gguf -a my-local-model
      -c 65536 -ctk q8_0 -ts 1,1 --port 8199
YML

FIELDS="ok engine image model_path served_name port max_ctx kv_dtype tp service status_header"
EXTRACT="
import json, sys
f = derive_compose_facts(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])
print(json.dumps({k: getattr(f, k) for k in '$FIELDS'.split()}, sort_keys=True))
"

# A) the CLI path -- plain python from the repo root, no venv
cli="$(cd "$ROOT" && python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.lib.profiles.compose_facts import derive_compose_facts
$EXTRACT" "$TMP/c.yml" 2>&1)" || { echo "  FAIL CLI path could not derive: $cli"; exit 1; }

# B) the cockpit path -- through data.py, in the cockpit venv if present.
VENV="$ROOT/tools/serve-cockpit/.venv/bin/python"
if [[ -x "$VENV" ]]; then
  cockpit="$(cd "$ROOT/tools/serve-cockpit" && "$VENV" -c "
from club3090_cockpit.data import derive_compose_facts
$EXTRACT" "$TMP/c.yml" 2>&1)" || { echo "  FAIL cockpit path could not derive: $cockpit"; exit 1; }
  if [[ "$cli" == "$cockpit" ]]; then
    echo "  ok   both import paths agree"
  else
    echo "  FAIL the two consumers DIVERGED"
    echo "    cli:     $cli"
    echo "    cockpit: $cockpit"
    rc=1
  fi
  # ⛔ IDENTITY, not just equal output. Output parity cannot tell "cockpit
  # delegates to the shared module" from "the old implementation is still inline
  # and the shared module is a COPY of it" -- both produce identical facts. That
  # is not hypothetical: during this change a stray `git checkout --` reverted
  # data.py to its inline version and the output-parity check still passed, so
  # the duplication would have shipped. Same class object, or it is a re-fork.
  if "$VENV" -c "
import sys
from pathlib import Path
from club3090_cockpit.data import ComposeFacts as FromCockpit
sys.path.insert(0, str(Path('$ROOT').resolve()))
from scripts.lib.profiles.compose_facts import ComposeFacts as FromShared
raise SystemExit(0 if FromCockpit is FromShared else 1)
" 2>/dev/null; then
    echo "  ok   cockpit ComposeFacts IS the shared class (delegating, not duplicated)"
  else
    echo "  FAIL cockpit ComposeFacts is a DIFFERENT class object — the implementation"
    echo "       has been re-inlined or copied; the move has been undone"
    rc=1
  fi

  # The delegation must not have orphaned the type.
  if "$VENV" -c "from club3090_cockpit.data import ComposeFacts" 2>/dev/null; then
    echo "  ok   ComposeFacts still importable from club3090_cockpit.data"
  else
    echo "  FAIL ComposeFacts no longer importable from club3090_cockpit.data"; rc=1
  fi
else
  echo "  skip cockpit venv absent — CLI path only"
fi

# The shared module must stay stdlib-only: it is imported on the launcher's
# no-PyYAML path, and a repo import would re-create the coupling this removed.
if command grep -qE "^(import|from) (yaml|club3090)" scripts/lib/profiles/compose_facts.py; then
  echo "  FAIL compose_facts.py grew a yaml/cockpit import — it must stay stdlib-only"; rc=1
else
  echo "  ok   compose_facts.py is stdlib-only"
fi

# Source-level duplication check, independent of imports.
if command grep -qE "^class ComposeFacts" tools/serve-cockpit/club3090_cockpit/data.py; then
  echo "  FAIL data.py defines its own ComposeFacts again — implementation re-inlined"; rc=1
else
  echo "  ok   data.py carries no inline ComposeFacts"
fi

# Refuse a vacuous pass: the extraction must actually have produced fields.
case "$cli" in
  *'"served_name": "my-local-model"'*) echo "  ok   derivation produced real fields" ;;
  *) echo "  FAIL derivation returned nothing usable: $cli"; rc=1 ;;
esac

[[ "$rc" == "0" ]] && echo "PASS: one implementation, both consumers agree" || echo "FAIL: compose-facts sharing regression"
exit "$rc"
