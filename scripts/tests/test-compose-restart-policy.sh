#!/usr/bin/env bash
# Test: vLLM composes use the reboot-surviving restart knob.
#
# Contract:
#   1. Every serving service in a shipped vLLM compose declares
#        restart: ${CLUB3090_RESTART:-unless-stopped}
#      — none ships restart: "no" (which would NOT come back after a host
#      reboot, defeating launch.sh-as-a-service usage).
#      Init services required via service_completed_successfully use "no";
#      restarting a completed init service would repeatedly rerun its job.
#   2. The pull/derived emitter (generate_compose.py generate_from_profile)
#      emits the same knob, so newly-derived composes don't reintroduce "no".
#
# Why unless-stopped: it is the only Docker policy that reliably restarts on
# daemon boot (host VM/bare-metal reboot) yet honors a manual stop
# (switch.sh --down / docker stop). The ${CLUB3090_RESTART:-...} form lets a
# user opt out with CLUB3090_RESTART=no. The ik-llama/llama-cpp/beellama
# composes already use a literal unless-stopped and are out of scope here.
set -uo pipefail
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
    services = d.get("services") or {}
    completed = {name for service in services.values() if isinstance(service, dict)
                 for name, dependency in (service.get("depends_on") or {}).items()
                 if isinstance(dependency, dict)
                 and dependency.get("condition") == "service_completed_successfully"}
    for name, body in services.items():
        if not isinstance(body, dict):
            continue
        r = body.get("restart")
        if name in completed and not body.get("ports"):
            if str(r) != "no":
                print(f"{f}: completed init service '{name}' restart={r!r} (want 'no')")
            continue
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
