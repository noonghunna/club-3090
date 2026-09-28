#!/usr/bin/env bash
# test-litellm-log — scripts/litellm-log.sh turns gateway request logging on and
# off by recreating the RUNNING gateway from its own compose project, and never
# drops what that gateway already had.
#
# WHY THIS TEST EXISTS
# --------------------
# The switch recreates a live service, so the ways it can go wrong are silent:
# recreate from the wrong checkout (the gateway re-mounts a stale runtime view —
# the worktree desync #1438 fixed for litellm-sync), lose the keys this rig's own
# routes need (they come from secrets.env through CLUB3090_LITELLM_ROUTE_KEYS, or
# an older local.env the compose loads on every recreate), or leave the level set
# when asked for "off". Offline: `docker` and `curl` are shims in $T/bin,
# prepended INLINE on each call; the shims log what they were asked to do.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/state"
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

# docker shim: `inspect` answers from $T/state; `compose … up` records the call and
# "recreates" the container with the LITELLM_LOG it was given.
cat > "$T/bin/docker" <<'SH'
#!/usr/bin/env bash
S="$(dirname "$0")/../state"
case "$1" in
  inspect)
    [[ -f "$S/running" ]] || exit 1
    fmt="${*: -1}"
    case "$fmt" in
      *Config.Env*)   cat "$S/env" ;;
      *working_dir*)  echo "/srv/main-checkout/services/litellm" ;;
      *config_files*) echo "/srv/main-checkout/services/litellm/docker-compose.yml" ;;
      *project\"*)    echo "litellm" ;;
      *)              echo "{}" ;;
    esac ;;
  compose)
    echo "PWD=$PWD ARGS=$* LITELLM_LOG=${LITELLM_LOG-<unset>} ROUTE_KEYS=${CLUB3090_LITELLM_ROUTE_KEYS-<unset>}" >> "$S/calls"
    # keep a copy of any --env-file, which the script deletes on exit
    prev=""; for a in "$@"; do [[ "$prev" == --env-file ]] && cp "$a" "$S/envfile"; prev="$a"; done
    # …and of the route keys file (#1466, 4c)
    [[ -n "${CLUB3090_LITELLM_ROUTE_KEYS:-}" ]] && cp "$CLUB3090_LITELLM_ROUTE_KEYS" "$S/routekeys"
    { echo "LITELLM_MASTER_KEY=k"
      if [[ -n "${LITELLM_LOG+x}" ]]; then echo "LITELLM_LOG=$LITELLM_LOG"; fi; } > "$S/env" ;;
  *) exit 0 ;;
esac
SH
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/curl"
chmod +x "$T/bin/docker" "$T/bin/curl"
# The shims must be what the script runs — a real docker here would recreate the live gateway.
[[ "$(PATH="$T/bin:$PATH" command -v docker)" == "$T/bin/docker" ]] || { echo "shim not first on PATH — refusing to run" >&2; exit 1; }
run() { PATH="$T/bin:$PATH" HOME="$T" bash "$ROOT/scripts/litellm-log.sh" "$@"; }
# `cd "$workdir"` needs the recorded compose dir to exist; point it into $T.
sed -i "s#/srv/main-checkout#$T/srv/main-checkout#" "$T/bin/docker"
mkdir -p "$T/srv/main-checkout/services/litellm"

# gateway not running → clear failure, no compose call
out="$(run on 2>&1)"; rc=$?
[[ $rc -ne 0 && "$out" == *"no 'litellm' container"* && ! -f "$T/state/calls" ]] || bad "no gateway must refuse without recreating (rc=$rc): $out"

# running, logging off
touch "$T/state/running"; printf 'LITELLM_MASTER_KEY=k\n' > "$T/state/env"
out="$(env -u LITELLM_LOG PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" status 2>&1)"
[[ "$out" == *"off (default)"* ]] || bad "status must report off by default: $out"

out="$(env -u LITELLM_LOG PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on 2>&1)" || bad "on failed: $out"
last="$(tail -1 "$T/state/calls")"
[[ "$last" == *"LITELLM_LOG=DEBUG"* ]] || bad "on must recreate with LITELLM_LOG=DEBUG: $last"
[[ "$last" == *"PWD=$T/srv/main-checkout/services/litellm"* && "$last" == *"-f $T/srv/main-checkout/services/litellm/docker-compose.yml"* && "$last" == *"-p litellm"* ]] \
  || bad "must recreate from the RUNNING gateway's compose project, not this checkout: $last"
[[ "$last" == *"up -d --force-recreate litellm"* ]] || bad "must force-recreate only the litellm service: $last"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" status 2>&1)"
[[ "$out" == *"ON (LITELLM_LOG=DEBUG)"* ]] || bad "status must report ON: $out"

# on again → no-op
n=$(wc -l < "$T/state/calls")
PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on >/dev/null 2>&1
[[ $(wc -l < "$T/state/calls") == "$n" ]] || bad "on when already on must not recreate"

# off → recreated with LITELLM_LOG UNSET (not empty)
out="$(PATH="$T/bin:$PATH" LITELLM_LOG=DEBUG bash "$ROOT/scripts/litellm-log.sh" off 2>&1)" || bad "off failed: $out"
last="$(tail -1 "$T/state/calls")"
[[ "$last" == *"LITELLM_LOG=<unset>"* ]] || bad "off must recreate with LITELLM_LOG unset, even if this shell exports it: $last"
[[ "$out" == *"off (default)"* ]] || bad "off must confirm: $out"

# A per-install gateway key (secrets.env, #1467) must survive the recreate: it goes in
# through --env-file, not the shell. Without it the compose falls back to the public
# default and every client carrying the stored key is refused.
KC="$T/kcfg"; mkdir -p "$KC"; printf 'LITELLM_MASTER_KEY=sk-club-test0000\n' > "$KC/secrets.env"
rm -f "$T/state/envfile"
out="$(env -u LITELLM_LOG -u LITELLM_MASTER_KEY PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$KC" bash "$ROOT/scripts/litellm-log.sh" on 2>&1)" || bad "on (with a stored key) failed: $out"
last="$(tail -1 "$T/state/calls")"
[[ "$last" == *"--env-file "* ]] || bad "a recreate must pass the settings with --env-file: $last"
command grep -qxF "LITELLM_MASTER_KEY='sk-club-test0000'" "$T/state/envfile" 2>/dev/null \
  || bad "the recreate's env file must carry the stored gateway key: $(cat "$T/state/envfile" 2>/dev/null || echo '<none>')"
envpath="$(sed -n 's/.*--env-file \([^ ]*\).*/\1/p' <<<"$last")"
[[ -n "$envpath" && ! -e "$envpath" ]] || bad "the temp env file must be removed after the recreate: $envpath"
PATH="$T/bin:$PATH" LITELLM_LOG=DEBUG bash "$ROOT/scripts/litellm-log.sh" off >/dev/null 2>&1 || true
# nothing configured → no --env-file (today's call shape)
out="$(env -u LITELLM_LOG PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on 2>&1)" || bad "on failed: $out"
[[ "$(tail -1 "$T/state/calls")" != *"--env-file"* ]] || bad "with nothing configured, no --env-file: $(tail -1 "$T/state/calls")"
PATH="$T/bin:$PATH" LITELLM_LOG=DEBUG bash "$ROOT/scripts/litellm-log.sh" off >/dev/null 2>&1 || true

# The keys this rig's own routes use (secrets.env, #1466 4c) must survive the recreate
# too: a 0600 file of just those keys, its path in CLUB3090_LITELLM_ROUTE_KEYS (which
# the compose loads as an env_file), removed afterwards. Not the HF token.
mkdir -p "$KC/litellm"
printf 'model_list:\n  - model_name: c\n    litellm_params:\n      model: openai/c\n      api_key: os.environ/RIG_CLOUD_KEY\n' > "$KC/litellm/config.local.yaml"
printf 'HF_TOKEN=hf_test_not_for_gateway\nRIG_CLOUD_KEY=sk-route-test0000\n' >> "$KC/secrets.env"
rm -f "$T/state/routekeys"
out="$(env -u LITELLM_LOG -u LITELLM_MASTER_KEY -u HF_TOKEN PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$KC" TMPDIR="$T" bash "$ROOT/scripts/litellm-log.sh" on 2>&1)" || bad "on (with a route key) failed: $out"
last="$(tail -1 "$T/state/calls")"
[[ "$(cat "$T/state/routekeys" 2>/dev/null)" == "RIG_CLOUD_KEY='sk-route-test0000'" ]] \
  || bad "the recreate must get the route's key (only it) through CLUB3090_LITELLM_ROUTE_KEYS: $last"
kpath="$(sed -n 's/.* ROUTE_KEYS=\([^ ]*\)$/\1/p' <<<"$last")"
[[ -n "$kpath" && "$kpath" != "<unset>" && ! -e "$kpath" ]] || bad "the route keys file must be removed after the recreate: $kpath"
[[ "$last$out" != *sk-route-test0000* ]] || bad "a route key reached a command line or the output"
PATH="$T/bin:$PATH" LITELLM_LOG=DEBUG CLUB3090_CONFIG_DIR="$KC" TMPDIR="$T" bash "$ROOT/scripts/litellm-log.sh" off >/dev/null 2>&1 || true
out="$(env -u LITELLM_LOG PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on 2>&1)" || bad "on failed: $out"
[[ "$(tail -1 "$T/state/calls")" == *"ROUTE_KEYS=<unset>" ]] || bad "no route key saved → CLUB3090_LITELLM_ROUTE_KEYS stays unset: $(tail -1 "$T/state/calls")"
PATH="$T/bin:$PATH" LITELLM_LOG=DEBUG bash "$ROOT/scripts/litellm-log.sh" off >/dev/null 2>&1 || true

# bad level → refused
PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on TRACE >/dev/null 2>&1 && bad "an unknown level must be refused"

# the compose really forwards it, and only when set (bare pass-through, not `=${X:-}`)
command grep -qE '^\s*-\s*LITELLM_LOG\s*$' "$ROOT/services/litellm/docker-compose.yml" \
  || bad "services/litellm/docker-compose.yml must pass LITELLM_LOG through bare (- LITELLM_LOG)"
# keys for this rig's own routes reach every (re)created gateway: the older ./local.env,
# then the keys file from the settings (so a saved key wins), both optional
python3 - "$ROOT/services/litellm/docker-compose.yml" <<'PY' || bad "the gateway compose must load ./local.env, then \${CLUB3090_LITELLM_ROUTE_KEYS:-/dev/null}, as OPTIONAL env_files"
import io, sys, yaml
svc = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))["services"]["litellm"]
got = [(e.get("path"), e.get("required")) for e in svc.get("env_file") or [] if isinstance(e, dict)]
sys.exit(0 if got == [("./local.env", False), ("${CLUB3090_LITELLM_ROUTE_KEYS:-/dev/null}", False)] else 1)
PY

[[ $fail -eq 0 ]] && echo "test-litellm-log: ok (refuses without a gateway, status, on/off recreate from the running project, keys via optional local.env and the route keys file, unset-not-empty, idempotent, bad level)"
exit $fail
