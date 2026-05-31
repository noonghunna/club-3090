#!/usr/bin/env bash
# Test: vLLM composes use the reboot-surviving restart knob.
#
# Contract:
#   1. Every shipped vLLM compose declares
#        restart: ${CLUB3090_RESTART:-unless-stopped}
#      — none ships restart: "no" (which would NOT come back after a host
#      reboot, defeating launch.sh-as-a-service usage).
#   2. The pull/derived emitter (generate_compose.py generate_from_profile)
#      emits the same knob, so newly-derived composes don't reintroduce "no".
#
# Why unless-stopped: it is the only Docker policy that reliably restarts on
# daemon boot (host VM/bare-metal reboot) yet honors a manual stop
# (switch.sh --down / docker stop). The ${CLUB3090_RESTART:-...} form lets a
# user opt out with CLUB3090_RESTART=no. The ik-llama/llama-cpp/beellama
# composes already use a literal unless-stopped and are out of scope here.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

KNOB='${CLUB3090_RESTART:-unless-stopped}'
fail=0

# 1. Shipped vLLM composes all carry the knob.
bad="$(python3 - "$KNOB" <<'PY'
import glob, sys, yaml
knob = sys.argv[1]
for f in sorted(glob.glob("models/*/vllm/compose/**/*.yml", recursive=True)):
    try:
        d = yaml.safe_load(open(f)) or {}
    except Exception as e:
        print(f"{f}: YAML parse error: {e}"); continue
    for name, body in (d.get("services") or {}).items():
        if not isinstance(body, dict):
            continue
        r = body.get("restart")
        if r is None:
            print(f"{f}: service '{name}' has no restart: key")
        elif str(r) != knob:
            print(f"{f}: service '{name}' restart={r!r} (want {knob!r})")
PY
)"
if [[ -n "$bad" ]]; then
  echo "FAIL: vLLM composes not on the reboot-surviving restart knob:" >&2
  echo "$bad" | sed 's/^/  /' >&2
  fail=1
fi

# 2. Derived emitter (generate_compose.py) emits the knob, not "no".
if grep -q 'restart: "no"' scripts/lib/generate_compose.py; then
  echo "FAIL: generate_compose.py still emits restart: \"no\" for derived composes" >&2
  fail=1
fi
if ! grep -qF 'restart: ${CLUB3090_RESTART:-unless-stopped}' scripts/lib/generate_compose.py; then
  echo "FAIL: generate_compose.py does not emit the CLUB3090_RESTART knob" >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "[compose-restart-policy] FAIL" >&2
  exit 1
fi
echo "[compose-restart-policy] PASS: all vLLM composes + derived emitter use \${CLUB3090_RESTART:-unless-stopped}"
