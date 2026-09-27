#!/usr/bin/env bash
# test-litellm-log — scripts/litellm-log.sh turns gateway request logging on and
# off by recreating the RUNNING gateway from its own compose project, and never
# drops what that gateway already had.
#
# WHY THIS TEST EXISTS
# --------------------
# The switch recreates a live service, so the ways it can go wrong are silent:
# recreate from the wrong checkout (the gateway re-mounts a stale runtime view —
# the worktree desync #1438 fixed for litellm-sync), lose the cloud key the
# gateway was started with (the DashScope routes then 401), or leave the level
# set when asked for "off". Offline: `docker` and `curl` are shims in $T/bin,
# prepended INLINE on each call; the shims log what they were asked to do.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/state"
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

# docker shim: `inspect` answers from $T/state; `compose … up` records the call and
# "recreates" the container with the LITELLM_LOG / DASHSCOPE_API_KEY it was given.
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
    echo "PWD=$PWD ARGS=$* LITELLM_LOG=${LITELLM_LOG-<unset>} DASHSCOPE=${DASHSCOPE_API_KEY-<unset>}" >> "$S/calls"
    { echo "LITELLM_MASTER_KEY=k"; echo "DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-}"
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

# running, started by another shell WITH the cloud key; this shell has none
touch "$T/state/running"; printf 'LITELLM_MASTER_KEY=k\nDASHSCOPE_API_KEY=sk-cloud\n' > "$T/state/env"
out="$(env -u DASHSCOPE_API_KEY -u LITELLM_LOG PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" status 2>&1)"
[[ "$out" == *"off (default)"* ]] || bad "status must report off by default: $out"

out="$(env -u DASHSCOPE_API_KEY -u LITELLM_LOG PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on 2>&1)" || bad "on failed: $out"
last="$(tail -1 "$T/state/calls")"
[[ "$last" == *"LITELLM_LOG=DEBUG"* ]] || bad "on must recreate with LITELLM_LOG=DEBUG: $last"
[[ "$last" == *"DASHSCOPE=sk-cloud"* ]] || bad "on must carry the running gateway's cloud key over: $last"
[[ "$last" == *"PWD=$T/srv/main-checkout/services/litellm"* && "$last" == *"-f $T/srv/main-checkout/services/litellm/docker-compose.yml"* && "$last" == *"-p litellm"* ]] \
  || bad "must recreate from the RUNNING gateway's compose project, not this checkout: $last"
[[ "$last" == *"up -d --force-recreate litellm"* ]] || bad "must force-recreate only the litellm service: $last"
out="$(PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" status 2>&1)"
[[ "$out" == *"ON (LITELLM_LOG=DEBUG)"* ]] || bad "status must report ON: $out"

# on again → no-op
n=$(wc -l < "$T/state/calls")
PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on >/dev/null 2>&1
[[ $(wc -l < "$T/state/calls") == "$n" ]] || bad "on when already on must not recreate"

# off → recreated with LITELLM_LOG UNSET (not empty), key still carried
out="$(env -u DASHSCOPE_API_KEY PATH="$T/bin:$PATH" LITELLM_LOG=DEBUG bash "$ROOT/scripts/litellm-log.sh" off 2>&1)" || bad "off failed: $out"
last="$(tail -1 "$T/state/calls")"
[[ "$last" == *"LITELLM_LOG=<unset>"* ]] || bad "off must recreate with LITELLM_LOG unset, even if this shell exports it: $last"
[[ "$last" == *"DASHSCOPE=sk-cloud"* ]] || bad "off must keep the cloud key: $last"
[[ "$out" == *"off (default)"* ]] || bad "off must confirm: $out"

# bad level → refused
PATH="$T/bin:$PATH" bash "$ROOT/scripts/litellm-log.sh" on TRACE >/dev/null 2>&1 && bad "an unknown level must be refused"

# the compose really forwards it, and only when set (bare pass-through, not `=${X:-}`)
command grep -qE '^\s*-\s*LITELLM_LOG\s*$' "$ROOT/services/litellm/docker-compose.yml" \
  || bad "services/litellm/docker-compose.yml must pass LITELLM_LOG through bare (- LITELLM_LOG)"

[[ $fail -eq 0 ]] && echo "test-litellm-log: ok (refuses without a gateway, status, on/off recreate from the running project, cloud key kept, unset-not-empty, idempotent, bad level)"
exit $fail
