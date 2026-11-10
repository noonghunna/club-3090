#!/usr/bin/env bash
# test-qengine-metrics-port — QE_METRICS_PORT publishes QEngine's metrics port on
# 127.0.0.1, and only when it is set.
#
# QEngine serves its dashboard, /metrics and /api/stats on 9393 inside the container,
# with no auth and a write endpoint (POST /api/reset); the composes don't publish it.
# Compose can't publish a port conditionally, so switch.sh adds an override written by
# scripts/lib/qengine_metrics.py when QE_METRICS_PORT is set.
#
#   1. qengine_metrics.py: the override publishes 127.0.0.1:<port>:<port> for the
#      compose's service; unset and 0 write none; a non-port is refused with a Fix.
#   2. docker compose merges it (render only): the loopback mapping beside the API
#      port, and the compose forwards QE_METRICS_PORT to the engine.
#   3. switch.sh end to end (docker shimmed): the override reaches `compose up` from a
#      shell prefix AND from a saved setting (the two ways to set a QEngine name), no
#      override without it or on another engine's slug, and a bad value is refused
#      before the running slug is torn down.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export CLUB3090_DATA_DIR="$T/data" CLUB3090_CACHE_DIR="$T/cache"   # never your real ones
SLUG=qengine/qwen38-27b-dual
C27="models/qwen3.8-27b/qengine/compose/dual/redhat-int4/mtp.yml"
CFN="models/qwen3.8-flash-next/qengine/compose/dual/qengine-int4/mtp.yml"
OVR="$T/data/compose-overrides/qengine-metrics-qengine-qwen38-27b-dual.yml"
qm() { python3 scripts/lib/qengine_metrics.py "$@"; }

echo "1. qengine_metrics.py"
o="$(qm override --slug "$SLUG" --compose "$C27" --port 9393)"
[[ "$o" == "$OVR" ]] && ok "the override goes in the data dir" || bad "override path: '$o'"
command grep -qx '      - "127.0.0.1:9393:9393"' "$OVR" 2>/dev/null \
  && command grep -qx '  qengine-qwen38-27b-dual:' "$OVR" \
  && ok "it publishes 127.0.0.1:9393:9393 for the compose's service" || bad "override content: $(cat "$OVR" 2>/dev/null)"
[[ -z "$(qm override --slug "$SLUG" --compose "$C27" --port '')" ]] && ok "unset: no override" || bad "unset wrote an override"
[[ -z "$(qm override --slug "$SLUG" --compose "$C27" --port 0)" ]] && ok "0 (QEngine's off): no override" || bad "0 wrote an override"
for v in abc 70000 09393 -1 9393x; do
  out="$(qm override --slug "$SLUG" --compose "$C27" --port "$v" 2>&1)"; rc=$?
  [[ $rc -eq 2 && "$out" == *"is not a port number"* && "$out" == *"Fix:"* && "$out" != */* ]] \
    && ok "'$v' refused with a Fix line" || bad "'$v': rc=$rc out=$out"
done
qm check --port 8443 && qm check --port '' && ok "check passes a port and an unset value" || bad "check refused a valid value"
# every QEngine compose, the multi-card ones too (2026-11-10), not just the two named above
mapfile -t QE_COMPOSES < <(ls models/*/qengine/compose/*/*/*.yml)
[[ ${#QE_COMPOSES[@]} -ge 6 && " ${QE_COMPOSES[*]} " == *" $C27 "* && " ${QE_COMPOSES[*]} " == *" $CFN "* ]] \
  && ok "${#QE_COMPOSES[@]} QEngine composes found" || bad "QEngine compose glob found ${#QE_COMPOSES[@]}: ${QE_COMPOSES[*]}"
for c in "${QE_COMPOSES[@]}"; do
  command grep -qxE '      - QE_METRICS_PORT' "$c" && ok "${c#models/}: the compose forwards QE_METRICS_PORT (bare)" \
    || bad "$c doesn't forward QE_METRICS_PORT as a bare passthrough"
  command grep -qE '^\s+- "[^"]*9393' "$c" && bad "$c publishes 9393 itself" || ok "$(basename "$c"): publishes no metrics port of its own"
done

echo "2. docker compose merges it"
if docker compose version >/dev/null 2>&1; then
  render() { (cd "$(dirname "$C27")" && env -u HF_TOKEN -u LITELLM_MASTER_KEY MODEL_DIR=/nonexistent "$@" docker compose -f mtp.yml ${OVR_ARG:+-f "$OVR_ARG"} config --format json 2>/dev/null); }
  got="$(OVR_ARG="$OVR" render QE_METRICS_PORT=9393 | python3 -c '
import json, sys
s = next(iter(json.load(sys.stdin)["services"].values()))
print(sorted((p.get("host_ip", ""), str(p["published"]), p["target"]) for p in s["ports"]), s["environment"].get("QE_METRICS_PORT"))')"
  [[ "$got" == "[('0.0.0.0', '8197', 8080), ('127.0.0.1', '9393', 9393)] 9393" ]] \
    && ok "rendered: the API on 0.0.0.0:8197, metrics on 127.0.0.1:9393, QE_METRICS_PORT=9393 to the engine" || bad "render with the override: $got"
  got="$(OVR_ARG="" render | python3 -c '
import json, sys
s = next(iter(json.load(sys.stdin)["services"].values()))
print(len(s["ports"]), (s.get("environment") or {}).get("QE_METRICS_PORT"))')"
  # an unset bare passthrough renders as null, which docker doesn't set in the container
  [[ "$got" == "1 None" ]] && ok "rendered without it: the API port only, no value forwarded" || bad "render without: $got"
else
  echo "  - skipped: no docker compose"
fi

echo "3. switch.sh (docker shimmed)"
mkdir -p "$T/bin" "$T/running" "$T/config"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
sub="${1:-}"; shift || true
case "$sub" in
  ps) printf '%s\n' "$MOCK_RUNNING" ;;
  inspect)
    fmt=""
    while [[ $# -gt 0 ]]; do case "$1" in --format) fmt="$2"; shift 2 ;; *) shift ;; esac; done
    if [[ "$fmt" == *working_dir* ]]; then printf '%s\n' "$MOCK_WORKDIR"
    elif [[ "$fmt" == *config_files* ]]; then printf 'running.yml\n'; fi ;;
  stop|rm|kill) printf 'TEARDOWN %s %s\n' "$sub" "$*" >> "$MOCK_LOG" ;;
  compose)
    case " $* " in
      *" down "*) printf 'TEARDOWN compose %s\n' "$*" >> "$MOCK_LOG" ;;
      *" up "*)   printf 'UP compose %s\n' "$*" >> "$MOCK_LOG" ;;
    esac ;;
esac
exit 0
EOF
chmod +x "$T/bin/docker"
export MOCK_LOG="$T/calls.log" MOCK_WORKDIR="$T/running" MOCK_RUNNING="vllm-qwen38-27b-dual-fast"
export PREFLIGHT_NO_FETCH=1 PREFLIGHT_NO_COMPOSE_DEPS=1 CLUB3090_SHM_CLEANUP=0
unset FORCE QE_METRICS_PORT
launch() { : > "$MOCK_LOG"; env -u QE_METRICS_PORT PATH="$T/bin:$PATH" "$@" bash scripts/switch.sh --no-wait --no-owui --force "${LAUNCH_SLUG:-$SLUG}" 2>&1; }
up_line() { command grep '^UP' "$MOCK_LOG" | head -1; }

out="$(launch QE_METRICS_PORT=9393)"; rc=$?
[[ $rc -eq 0 && "$(up_line)" == *"-f $OVR "* ]] && ok "shell prefix: compose up gets -f …/qengine-metrics-qengine-qwen38-27b-dual.yml" \
  || bad "shell prefix: rc=$rc up='$(up_line)' $(printf '%s' "$out" | tail -3)"
[[ "$out" == *"QEngine metrics and dashboard: http://127.0.0.1:9393/"* ]] && ok "switch.sh says where" || bad "no metrics line in: $(printf '%s' "$out" | tail -3)"

printf 'QE_METRICS_PORT=9395\n' > "$T/config/club3090.env"
out="$(launch CLUB3090_CONFIG_DIR="$T/config")"; rc=$?
[[ $rc -eq 0 && "$(up_line)" == *"-f $OVR "* ]] && command grep -q '"127.0.0.1:9395:9395"' "$OVR" \
  && ok "saved setting (settings.sh set QE_METRICS_PORT=9395): the override carries 9395" \
  || bad "saved setting: rc=$rc up='$(up_line)' override=$(command grep 127.0.0.1 "$OVR" 2>/dev/null) $(printf '%s' "$out" | tail -3)"

out="$(launch)"; rc=$?
[[ $rc -eq 0 && -n "$(up_line)" && "$(up_line)" != *compose-overrides* ]] && ok "unset: compose up gets no override" \
  || bad "unset: rc=$rc up='$(up_line)'"
out="$(launch QE_METRICS_PORT=0)"; rc=$?
[[ $rc -eq 0 && -n "$(up_line)" && "$(up_line)" != *compose-overrides* ]] && ok "0: no override (QEngine's metrics off)" \
  || bad "0: rc=$rc up='$(up_line)'"
out="$(LAUNCH_SLUG=sgl/qwen38-27b-dual-fast launch QE_METRICS_PORT=9393)"; rc=$?
[[ $rc -eq 0 && -n "$(up_line)" && "$(up_line)" != *compose-overrides* ]] && ok "another engine's slug: no override" \
  || bad "sgl slug: rc=$rc up='$(up_line)'"

out="$(launch QE_METRICS_PORT=abc)"; rc=$?
if (( rc == 0 )); then bad "QE_METRICS_PORT=abc: launch not refused"
elif command grep -q TEARDOWN "$MOCK_LOG"; then bad "QE_METRICS_PORT=abc: refused after tearing the running slug down"
elif [[ "$out" != *"is not a port number"* ]]; then bad "QE_METRICS_PORT=abc: refused for another reason: $(printf '%s' "$out" | tail -3)"
else ok "QE_METRICS_PORT=abc: refused before the teardown, running slug left up"; fi

[ "$fail" -eq 0 ] && echo "test-qengine-metrics-port: ok" || { echo "test-qengine-metrics-port: FAIL"; exit 1; }
