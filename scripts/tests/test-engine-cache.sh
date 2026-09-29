#!/usr/bin/env bash
# test-engine-cache — compile caches and the KV disk tier live in per-user directories,
# keyed by engine image, on every launch path (club-3090#1466, phases 4a/4b).
#
# Contract (scripts/lib/engine_cache.py, called by switch.sh, gpu-mode.sh, the estate
# planner and c3's generated-compose serve):
#   key        <image ref, sanitised>-<first 12 hex of the image ID>. Same image → same
#              directory, from any checkout; another image, or the same tag re-pulled
#              to a new ID, → another directory.
#   image      whatever `docker compose config` renders in the launch's own environment:
#              a VLLM_IMAGE the shell or the settings set wins over the compose default,
#              and on gpu-mode's path only what `sudo docker compose` will see counts.
#   dirs       created as you BEFORE compose runs (docker would create them root:root),
#              under CLUB3090_CACHE_DIR / CLUB3090_DATA_DIR (defaults: XDG, then HOME).
#   fallback   a raw `docker compose up`, a missing image, COMPOSE_BIN=: → the old
#              in-repo paths; the launch is never failed.
#   caches     settings.sh caches reports; --remove-legacy asks, removes what you own,
#              and prints the sudo command for what docker created as root.
#
# Offline and read-only: `docker` and `sudo` are shims in $T/bin (prepended INLINE on
# every launcher call); `docker compose config` is the real one (it renders, starts
# nothing, and needs no daemon). No container is started or stopped.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
while IFS= read -r _v; do unset "$_v"; done < <(compgen -e | command grep -E '(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|MASTER_KEY|_KEY)$')
while IFS= read -r _v; do unset "$_v"; done < <(compgen -e | command grep -E '^(CLUB3090_(CACHE|DATA|ENGINE_CACHE)_DIR|XDG_(CACHE|DATA)_HOME|KV_OFFLOAD_.*|VLLM_IMAGE|MODEL_DIR|FORCE|COMPOSE_BIN)$')
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/scripts/lib/engine_cache.py"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }
T="$(mktemp -d)"
trap 'chmod -R u+rwX "$T" 2>/dev/null; rm -rf "$T"' EXIT
[[ -f "$PY" ]] || { bad "scripts/lib/engine_cache.py is missing"; echo "test-engine-cache: FAIL"; exit 1; }

REAL_DOCKER="$(command -v docker || true)"
HAVE_COMPOSE=0
[[ -n "$REAL_DOCKER" ]] && "$REAL_DOCKER" compose version >/dev/null 2>&1 && HAVE_COMPOSE=1

ID_030=sha256:30030030030030030030030030030030030030030030030030030030030030
ID_029=sha256:29029029029029029029029029029029029029029029029029029029029029
ID_NEW=sha256:77777777777777777777777777777777777777777777777777777777777777
ID_022=sha256:22022022022022022022022022022022022022022022022022022022022022
KEY_030=vllm-vllm-openai-v0.30.0-300300300300
VCOMPOSE=models/qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml    # cache + KV disk tier
VSLUG=vllm/qwen38-27b-dual-fast                                          # the slug for that compose

# ── 1. the key ────────────────────────────────────────────────────────────────
echo "[1] the key"
key() { python3 "$PY" key --image "$1" --id "$2" 2>/dev/null; }
k_ok=1
while IFS='|' read -r ref id want; do
  got="$(key "$ref" "$id")"
  [[ "$got" == "$want" ]] || { bad "key($ref, $id) = '$got', want '$want'"; k_ok=0; }
done <<EOF
vllm/vllm-openai:v0.30.0|sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90|vllm-vllm-openai-v0.30.0-8a69ffad015f
docker.io/library/ubuntu:24.04|sha256:ABCDEF0123456789abcdef|ubuntu-24.04-abcdef012345
ghcr.io/noonghunna/llamacpp-club3090:moecachev1.5-rc0|sha256:0123456789abcdef00|ghcr.io-noonghunna-llamacpp-club3090-moecachev1.5-rc0-0123456789ab
lmsysorg/sglang@sha256:06e4f2ed21af06e4f2ed21af|sha256:06e4f2ed21af06e4f2ed21af|lmsysorg-sglang-06e4f2ed21af
vllm/vllm-openai:gemma4-unified|sha256:111111111111aaaa|vllm-vllm-openai-gemma4-unified-111111111111
EOF
[[ $k_ok -eq 1 ]] && ok "keys: <ref, sanitised>-<12 hex of the ID>; docker.io/library/ and digest suffixes dropped"
[[ "$(key vllm/vllm-openai:gemma4-unified sha256:222222222222bbbb)" != "$(key vllm/vllm-openai:gemma4-unified sha256:111111111111aaaa)" ]] \
  && ok "the same tag re-pulled to a new ID gets a NEW key (a moved tag is different engine code)" || bad "a re-pulled tag kept its key"
python3 "$PY" key --image x:1 --id "not-an-id" >/dev/null 2>&1 && bad "a malformed image ID was accepted" || ok "a malformed image ID is refused"

# ── shims ─────────────────────────────────────────────────────────────────────
mkdir -p "$T/bin" "$T/images" "$T/running"
img_file() { printf '%s/%s' "$T/images" "$(printf '%s' "$1" | tr '/:@' '___')"; }
set_image() { printf '%s\n' "$2" > "$(img_file "$1")"; }
set_image vllm/vllm-openai:v0.30.0 "$ID_030"
set_image vllm/vllm-openai:v0.29.0 "$ID_029"
set_image vllm/vllm-openai:v0.22.0 "$ID_022"
cat > "$T/bin/docker" <<EOF
#!/usr/bin/env bash
# docker shim: image IDs from \$STUB_IMAGES, pulls logged, compose 'config' forwarded to the
# REAL docker compose (render only), compose 'up' recorded + rendered, everything else a no-op.
sub="\${1:-}"
case "\$sub" in
  image)
    [[ "\${2:-}" == inspect ]] || exit 1
    f="$T/images/\$(printf '%s' "\${@: -1}" | tr '/:@' '___')"
    [[ -f "\$f" ]] && { cat "\$f"; exit 0; }
    exit 1 ;;
  pull)
    echo "PULL \${2:-}" >> "\$MOCK_LOG"
    [[ -n "\${STUB_PULL_ID:-}" ]] || exit 1
    printf '%s\n' "\$STUB_PULL_ID" > "$T/images/\$(printf '%s' "\${2:-}" | tr '/:@' '___')"; exit 0 ;;
  ps) printf '%s\n' "\${MOCK_RUNNING:-}" ;;
  inspect)
    fmt=""; while [[ \$# -gt 0 ]]; do case "\$1" in --format|-f) fmt="\$2"; shift 2 ;; *) shift ;; esac; done
    if [[ "\$fmt" == *working_dir* ]]; then printf '%s\n' "$T/running"
    elif [[ "\$fmt" == *config_files* ]]; then printf 'running.yml\n'; fi ;;
  stop|rm|kill|restart) echo "TEARDOWN \$*" >> "\$MOCK_LOG" ;;
  compose)
    shift
    case " \$* " in
      *" config "*) exec "$REAL_DOCKER" compose "\$@" ;;
      *" up "*)
        printf 'CACHE=%s\nDATA=%s\n' "\${CLUB3090_ENGINE_CACHE_DIR-<unset>}" "\${CLUB3090_DATA_DIR-<unset>}" > "\$MOCK_ENV_OUT"
        pre=(); for a in "\$@"; do [[ "\$a" == up ]] && break; pre+=("\$a"); done
        "$REAL_DOCKER" compose "\${pre[@]}" config --format json > "\$MOCK_RENDER" 2>>"\$MOCK_LOG" || echo RENDER_FAILED >> "\$MOCK_LOG"
        echo "UP \$*" >> "\$MOCK_LOG" ;;
      *" down "*) echo "TEARDOWN compose \$*" >> "\$MOCK_LOG" ;;
    esac ;;
esac
exit 0
EOF
cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
# sudo shim: log (and apply) the VAR=val assignments a real sudo would pass, then run.
while [[ "${1:-}" == *=* ]]; do echo "SUDO_ENV $1" >> "$MOCK_LOG"; export "$1"; shift; done
exec "$@"
EOF
chmod +x "$T/bin/docker" "$T/bin/sudo"
[[ "$(PATH="$T/bin:$PATH" command -v docker)" == "$T/bin/docker" && "$(PATH="$T/bin:$PATH" command -v sudo)" == "$T/bin/sudo" ]] \
  || { bad "shims not first on PATH — refusing to run"; echo "test-engine-cache: FAIL"; exit 1; }
export MOCK_LOG="$T/calls.log" MOCK_ENV_OUT="$T/env.out" MOCK_RENDER="$T/render.json"
export PREFLIGHT_NO_FETCH=1 PREFLIGHT_NO_COMPOSE_DEPS=1 CLUB3090_SHM_CLEANUP=0 C3_LITELLM_FAKE_LIVE=
reset_mock() { : > "$MOCK_LOG"; rm -f "$MOCK_ENV_OUT" "$MOCK_RENDER"; }
envout() { command grep -E "^$1=" "$MOCK_ENV_OUT" 2>/dev/null | head -1 | cut -d= -f2-; }
# mounted <container path> → the host source the rendered compose mounts there
mounted() {
  python3 - "$MOCK_RENDER" "$1" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(next((v["source"] for s in d["services"].values() for v in s.get("volumes", []) if v["target"] == sys.argv[2]), "<none>"))
PY
}
mine() { [[ -d "$1" && "$(stat -c %u "$1")" == "$(id -u)" ]]; }

# Fixture checkouts: the repo's code and composes, two "worktrees" of one repo.
for w in wt1 wt2; do
  mkdir -p "$T/$w"; for d in scripts models tools; do ln -s "$ROOT/$d" "$T/$w/$d"; done; : > "$T/$w/.env"
done
C="$T/cfg"; mkdir -p "$C"

if [[ "$HAVE_COMPOSE" != 1 ]]; then
  echo "  - SKIP [2]-[5]: no docker compose here (every leg renders with it)"
else
# ── 2. switch.sh: the real launch path ────────────────────────────────────────
echo "[2] switch.sh → docker compose up"
sw() {  # <worktree> [-u VAR …] [VAR=val …] — the REAL switch.sh of that fixture, docker shimmed; $SLUG or $VSLUG
  local w="$1"; shift
  local -a un=() as=() base=("CLUB3090_CACHE_DIR=$T/cache" "CLUB3090_DATA_DIR=$T/data") keep=()
  local b
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == -u ]]; then
      un+=(-u "$2"); keep=(); for b in "${base[@]}"; do [[ "$b" == "$2="* ]] || keep+=("$b"); done; base=("${keep[@]}"); shift 2
    else as+=("$1"); shift; fi
  done
  reset_mock
  env "${un[@]}" PATH="$T/bin:$PATH" COMPOSE_BIN="$T/bin/docker compose" CLUB3090_CONFIG_DIR="$C" HOME="$T/home" \
      MODEL_DIR=/nonexistent "${base[@]}" "${as[@]}" timeout 240 bash "$T/$w/scripts/switch.sh" --no-wait --no-owui --force "${SLUG:-$VSLUG}" 2>&1
}
out="$(sw wt1)"; rc=$?
if (( rc != 0 )) || ! command grep -q '^UP' "$MOCK_LOG"; then
  bad "switch.sh did not reach compose up (rc=$rc): $(tail -4 <<<"$out")"
else
  c1="$(envout CACHE)"; d1="$(envout DATA)"
  [[ "$c1" == "$T/cache/$KEY_030" && "$d1" == "$T/data" ]] \
    && ok "compose up gets CLUB3090_ENGINE_CACHE_DIR=<cache>/$KEY_030 and CLUB3090_DATA_DIR=<data>" \
    || bad "environment at compose up: cache='$c1' data='$d1'"
  t="$(mounted /root/.triton/cache)|$(mounted /root/.cache/vllm/torch_compile_cache)|$(mounted /kv-offload)"
  [[ "$t" == "$T/cache/$KEY_030/triton|$T/cache/$KEY_030/torch_compile|$T/data/kv-offload" ]] \
    && ok "… and the rendered compose mounts exactly those: <key>/triton, <key>/torch_compile, <data>/kv-offload" \
    || bad "rendered mounts: $t"
  mine "$T/cache/$KEY_030/triton" && mine "$T/cache/$KEY_030/torch_compile" && mine "$T/data/kv-offload" \
    && ok "all three existed, owned by you, before compose up (docker would have made them root:root)" \
    || bad "a mounted dir is missing or not yours: $(ls -ld "$T/cache/$KEY_030"/* "$T/data/kv-offload" 2>&1 | tr '\n' ' ')"
  command grep -qF "[cache] compile cache: $T/cache/$KEY_030" <<<"$out" && ok "the launch log names the compile-cache dir" \
    || bad "no '[cache] compile cache:' line in the launch log"
fi
out="$(sw wt2)"
[[ "$(envout CACHE)" == "$T/cache/$KEY_030" ]] \
  && ok "a second checkout (fresh worktree) of the same slug gets the SAME cache dir — it reuses the warm cache" \
  || bad "second worktree: cache='$(envout CACHE)'"
# A slug on another engine pin (vllm-gemma-stable → v0.22.0), whose compose keeps its own
# variant subdirs (triton_int8 / torch_compile_int8, 160e8fce: a different overlay set).
out="$(SLUG=vllm/gemma-26ba4b-single sw wt1)"
K22="$T/cache/vllm-vllm-openai-v0.22.0-220220220220"
[[ "$(envout CACHE)" == "$K22" && "$(mounted /root/.triton/cache)" == "$K22/triton_int8" \
   && "$(mounted /root/.cache/vllm/torch_compile_cache)" == "$K22/torch_compile_int8" ]] && mine "$K22/triton_int8" \
  && ok "a slug on another image (the engine profile's v0.22.0) gets its own dir, with the compose's own variant subdirs" \
  || bad "gemma v0.22.0 slug: cache='$(envout CACHE)' mounts='$(mounted /root/.triton/cache) $(mounted /root/.cache/vllm/torch_compile_cache)'"
set_image vllm/vllm-openai:v0.30.0 "$ID_NEW"
out="$(sw wt1)"
[[ "$(envout CACHE)" == "$T/cache/vllm-vllm-openai-v0.30.0-777777777777" ]] \
  && ok "the same tag re-pulled to a new image ID gets a new dir" || bad "re-pulled tag: cache='$(envout CACHE)'"
set_image vllm/vllm-openai:v0.30.0 "$ID_030"
out="$(sw wt1 -u CLUB3090_DATA_DIR XDG_DATA_HOME="$T/xd-explicit" KV_OFFLOAD_DIR=/srv/kv-explicit)"
# (the data dir itself may exist: switch.sh keeps the slug-label override in
# <data dir>/compose-labels — what must not appear is a KV-offload dir)
[[ "$(envout DATA)" == "<unset>" && "$(mounted /kv-offload)" == /srv/kv-explicit && ! -e "$T/xd-explicit/club-3090/kv-offload" \
   && "$(envout CACHE)" == "$T/cache/$KEY_030" ]] \
  && ok "an explicit KV_OFFLOAD_DIR still wins (no data dir handed over, no KV-offload dir created)" \
  || bad "explicit KV_OFFLOAD_DIR: data='$(envout DATA)' mount='$(mounted /kv-offload)' created=$( [[ -e "$T/xd-explicit/club-3090/kv-offload" ]] && echo yes || echo no)"
rm -rf "$T/xdg"
out="$(sw wt1 -u CLUB3090_CACHE_DIR -u CLUB3090_DATA_DIR XDG_CACHE_HOME="$T/xdg/c" XDG_DATA_HOME="$T/xdg/d")"
[[ "$(envout CACHE)" == "$T/xdg/c/club-3090/$KEY_030" && "$(envout DATA)" == "$T/xdg/d/club-3090" ]] \
  && ok "no override: \$XDG_CACHE_HOME/club-3090 and \$XDG_DATA_HOME/club-3090" \
  || bad "XDG defaults: cache='$(envout CACHE)' data='$(envout DATA)'"
printf 'CLUB3090_CACHE_DIR=%s\n' "$T/saved cache" > "$C/club3090.env"
out="$(sw wt1 -u CLUB3090_CACHE_DIR)"
[[ "$(envout CACHE)" == "$T/saved cache/$KEY_030" && "$(mounted /root/.triton/cache)" == "$T/saved cache/$KEY_030/triton" ]] \
  && ok "CLUB3090_CACHE_DIR saved in club3090.env is honoured (a path with a space renders intact)" \
  || bad "saved CLUB3090_CACHE_DIR: cache='$(envout CACHE)' mount='$(mounted /root/.triton/cache)'"
: > "$C/club3090.env"
out="$(sw wt1 CLUB3090_ENGINE_CACHE_DIR=/stale/inherited)"
[[ "$(envout CACHE)" == "$T/cache/$KEY_030" ]] && ok "an inherited CLUB3090_ENGINE_CACHE_DIR is recomputed, not trusted" \
  || bad "inherited CLUB3090_ENGINE_CACHE_DIR survived: '$(envout CACHE)'"
out="$(sw wt1 CLUB3090_CACHE_DIR=relative/cache)"
[[ "$(envout CACHE)" == "<unset>" && "$(realpath -m "$(mounted /root/.triton/cache)")" == "$ROOT/models/qwen3.8-27b/vllm/cache/triton" ]] \
  && command grep -qF 'CLUB3090_CACHE_DIR must be an absolute path' <<<"$out" \
  && ok "a relative CLUB3090_CACHE_DIR is refused with a reason; the launch goes ahead on the in-repo cache" \
  || bad "relative CLUB3090_CACHE_DIR: cache='$(envout CACHE)' mount='$(mounted /root/.triton/cache)'"
rm -f "$(img_file vllm/vllm-openai:v0.30.0)"
out="$(sw wt1)"
[[ "$(envout CACHE)" == "<unset>" && "$(realpath -m "$(mounted /root/.triton/cache)")" == "$ROOT/models/qwen3.8-27b/vllm/cache/triton" ]] \
  && command grep -q '^PULL vllm/vllm-openai:v0.30.0' "$MOCK_LOG" \
  && ok "an image that can't be pulled: the launch still happens, on the in-repo cache (and says why)" \
  || bad "unpullable image: cache='$(envout CACHE)' pulls=$(command grep -c PULL "$MOCK_LOG")"
out="$(sw wt1 STUB_PULL_ID="$ID_030")"
[[ "$(envout CACHE)" == "$T/cache/$KEY_030" ]] && command grep -q '^PULL vllm/vllm-openai:v0.30.0' "$MOCK_LOG" \
  && ok "a missing image is pulled first, then keyed by its ID" || bad "pull-then-key: cache='$(envout CACHE)'"

# ── 3. gpu-mode: `sudo docker compose`, which drops HOME and the environment ───
echo "[3] gpu-mode compose_at_env → sudo docker compose up"
# The REAL compose_at_env, lifted out of gpu-mode.sh (running gpu-mode would run a whole mode).
sed -n '/^compose_at_env()/,/^}/p' "$ROOT/scripts/gpu-mode.sh" > "$T/cae.sh"
[[ -s "$T/cae.sh" ]] || bad "compose_at_env not found in gpu-mode.sh"
gm() {  # [VAR=val …] — compose_at_env "up -d" on $VCOMPOSE, sudo + docker shimmed
  reset_mock
  env PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$C" HOME="$T/home" CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" \
      MODEL_DIR=/nonexistent FIX="$T/wt1" LIB="$ROOT/scripts/lib" CAE="$T/cae.sh" D="$T/wt1/$(dirname "$VCOMPOSE")" F="$(basename "$VCOMPOSE")" "$@" \
      bash -c '. "$LIB/club-config.sh"; . "$LIB/engine-cache.sh"; . "$CAE"; GPU_MODE_SHELL_ENV=(); CLUB3090_DIR="$FIX"; compose_at_env "$D" "up -d" "$F"' 2>&1
}
out="$(gm)"
if ! command grep -q '^UP' "$MOCK_LOG"; then
  bad "compose_at_env did not reach compose up: $(tail -3 <<<"$out")"
else
  command grep -qxF "SUDO_ENV CLUB3090_ENGINE_CACHE_DIR=$T/cache/$KEY_030" "$MOCK_LOG" \
    && command grep -qxF "SUDO_ENV CLUB3090_DATA_DIR=$T/data" "$MOCK_LOG" \
    && ok "both dirs cross sudo as VAR=val arguments (sudo drops the environment and HOME)" \
    || bad "sudo got: $(command grep SUDO_ENV "$MOCK_LOG" | tr '\n' ' ')"
  [[ "$(mounted /root/.triton/cache)" == "$T/cache/$KEY_030/triton" && "$(mounted /kv-offload)" == "$T/data/kv-offload" ]] \
    && ok "the same dirs as switch.sh for the same image" || bad "gpu-mode render: $(mounted /root/.triton/cache) $(mounted /kv-offload)"
fi
out="$(gm VLLM_IMAGE=vllm/vllm-openai:v0.29.0)"
command grep -qxF "SUDO_ENV CLUB3090_ENGINE_CACHE_DIR=$T/cache/$KEY_030" "$MOCK_LOG" \
  && ok "a VLLM_IMAGE only in gpu-mode's shell is ignored — sudo won't pass it, so compose runs the default image" \
  || bad "shell-only VLLM_IMAGE leaked into the key: $(command grep ENGINE_CACHE "$MOCK_LOG")"
printf 'VLLM_IMAGE=vllm/vllm-openai:v0.29.0\n' > "$C/club3090.env"
out="$(gm)"
command grep -qxF "SUDO_ENV CLUB3090_ENGINE_CACHE_DIR=$T/cache/vllm-vllm-openai-v0.29.0-290290290290" "$MOCK_LOG" \
  && [[ "$(mounted /root/.triton/cache)" == "$T/cache/vllm-vllm-openai-v0.29.0-290290290290/triton" ]] \
  && ok "a VLLM_IMAGE saved in the settings reaches compose through --env-file, and keys the cache" \
  || bad "saved VLLM_IMAGE: $(command grep ENGINE_CACHE "$MOCK_LOG") mount=$(mounted /root/.triton/cache)"
: > "$C/club3090.env"
reset_mock
env PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$C" LIB="$ROOT/scripts/lib" CAE="$T/cae.sh" D="$ROOT/services/searxng" \
  bash -c '. "$LIB/club-config.sh"; . "$LIB/engine-cache.sh"; . "$CAE"; GPU_MODE_SHELL_ENV=(); CLUB3090_DIR=/nonexistent; compose_at_env "$D" "up -d" docker-compose.yml' >/dev/null 2>&1
! command grep -q 'SUDO_ENV CLUB3090_' "$MOCK_LOG" && ok "a service compose that mounts neither gets nothing (and no python call)" \
  || bad "a service compose got: $(command grep SUDO_ENV "$MOCK_LOG")"

# ── 4. the estate planner (launch.sh --estate) ────────────────────────────────
echo "[4] estate run_compose"
if python3 -c 'import yaml' 2>/dev/null; then
  reset_mock
  out="$(env PATH="$T/bin:$PATH" COMPOSE_BIN="$T/bin/docker compose" CLUB3090_CONFIG_DIR="$C" HOME="$T/home" \
         CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" MODEL_DIR=/nonexistent CLUB3090_ESTATE_BOOT_LOG_DIR="$T/boot" \
         python3 - "$ROOT" "$VSLUG" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[1])
from scripts.lib.profiles import estate_cli as e
e.resolve_gpu_uuids = lambda idx: None                 # no nvidia-smi in a test
e.run_compose(e.InstanceSpec(name="t", compose_name=sys.argv[2], gpu_indices=(0, 1), port=8999), "up")
PY
)"
  [[ "$(envout CACHE)" == "$T/cache/$KEY_030" && "$(mounted /root/.triton/cache)" == "$T/cache/$KEY_030/triton" \
     && "$(mounted /kv-offload)" == "$T/data/kv-offload" ]] \
    && ok "an estate instance gets the same dirs as switch.sh for the same image" \
    || bad "estate: cache='$(envout CACHE)' mounts=$(mounted /root/.triton/cache) $(mounted /kv-offload) — $(tail -2 <<<"$out")"
else
  echo "  - SKIP estate leg: PyYAML not installed (the estate planner needs it)"
fi

# ── 5. fallbacks and notices ──────────────────────────────────────────────────
echo "[5] fallbacks"
prep() { env CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" MODEL_DIR=/nonexistent PATH="$T/bin:$PATH" "$@"; }
out="$(prep python3 "$PY" prepare --compose "$ROOT/$VCOMPOSE" --compose-bin : 2>&1)"
[[ -z "$out" ]] && ok "COMPOSE_BIN=: (the tests' never-launch switch): nothing printed, nothing prepared" || bad "COMPOSE_BIN=: printed: $out"
out="$(prep python3 "$PY" prepare --compose "$ROOT/services/searxng/docker-compose.yml" 2>&1)"
[[ -z "$out" ]] && ok "a compose that mounts neither variable: no output" || bad "non-cache compose printed: $out"
# The legacy notice: the compose's old in-repo cache dir still holds a cache.
LG="$T/legacy"; mkdir -p "$LG/models/m/vllm/compose/single/q" "$LG/models/m/vllm/cache/triton/abc"
printf 'services:\n  s:\n    image: vllm/vllm-openai:v0.30.0\n    volumes:\n      - ${CLUB3090_ENGINE_CACHE_DIR:-../../../cache}/triton:/root/.triton/cache\n' > "$LG/models/m/vllm/compose/single/q/base.yml"
out="$(prep python3 "$PY" prepare --compose "$LG/models/m/vllm/compose/single/q/base.yml" --root "$LG" 2>&1)"
command grep -qF "models/m/vllm/cache still holds this checkout's old compile cache" <<<"$out" \
  && command grep -qF 'settings.sh caches' <<<"$out" \
  && ok "a checkout whose old in-repo cache still holds files is told, with the command that reclaims it" \
  || bad "no legacy notice: $out"
fi

# ── 6. settings.sh caches ─────────────────────────────────────────────────────
echo "[6] settings.sh caches"
R="$T/repo"; mkdir -p "$R/models/a/vllm/cache/triton/x" "$R/models/a/vllm/cache/torch_compile/y" "$R/models/b/vllm/cache/triton/z" "$R/kv-offload/blocks"
printf 'x\n' > "$R/models/a/vllm/cache/.gitignore"; printf 'r\n' > "$R/models/a/vllm/cache/README.md"; printf 'k\n' > "$R/kv-offload/.gitignore"
head -c 300000 /dev/zero > "$R/models/a/vllm/cache/torch_compile/y/graph.bin"
printf 'k\n' > "$R/kv-offload/blocks/b0.bin"; printf 't\n' > "$R/models/b/vllm/cache/triton/z/k.bin"
chmod 0555 "$R/models/b/vllm/cache/triton/z"     # stands in for a root-owned folder: you can't delete inside it
mkdir -p "$T/cache/vllm-vllm-openai-v0.30.0-300300300300/triton"
cs() { env CLUB3090_CONFIG_DIR="$C" CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" python3 "$PY" caches --root "$R" "$@" 2>&1; }
out="$(cs)"
command grep -q 'vllm-vllm-openai-v0.30.0-300300300300' <<<"$out" && command grep -qE 'models/a/vllm/cache/torch_compile +[0-9.]+ KiB' <<<"$out" \
  && command grep -qE 'models/b/vllm/cache/triton .*root-owned: needs sudo' <<<"$out" && command grep -q 'kv-offload/blocks' <<<"$out" \
  && ! command grep -qE 'README.md|\.gitignore' <<<"$out" \
  && ok "report: per-image dirs, the old in-repo caches with sizes (tracked README/.gitignore not listed), the ones needing sudo flagged" \
  || bad "report: $out"
out="$(cs --remove-legacy </dev/null)"; rc=$?
[[ $rc -eq 2 && -d "$R/models/a/vllm/cache/torch_compile" ]] && ok "--remove-legacy with no terminal and no --yes refuses, removes nothing" \
  || bad "non-interactive --remove-legacy: rc=$rc, torch_compile $( [[ -d "$R/models/a/vllm/cache/torch_compile" ]] && echo kept || echo GONE)"
out="$(cs --remove-legacy --yes)"
[[ ! -e "$R/models/a/vllm/cache/torch_compile" && ! -e "$R/models/a/vllm/cache/triton" && ! -e "$R/kv-offload/blocks" \
   && -f "$R/models/a/vllm/cache/.gitignore" && -f "$R/models/a/vllm/cache/README.md" && -f "$R/kv-offload/.gitignore" \
   && -d "$R/models/b/vllm/cache/triton/z" ]] \
  && command grep -qF "sudo rm -rf -- $R/models/b/vllm/cache/triton" <<<"$out" \
  && ok "--remove-legacy --yes removes what you own, keeps the tracked files, and prints the sudo command for the rest" \
  || bad "--remove-legacy --yes: $out"
[[ -d "$T/cache/vllm-vllm-openai-v0.30.0-300300300300" ]] && ok "… and never touches the shared per-image caches" || bad "a shared cache dir was removed"
chmod 0755 "$R/models/b/vllm/cache/triton/z"
mkdir -p "$R/kv-offload/live"; out="$(env KV_OFFLOAD_DIR="$R/kv-offload" CLUB3090_CONFIG_DIR="$C" CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" python3 "$PY" caches --root "$R" 2>&1)"
! command grep -q 'kv-offload/live' <<<"$out" && command grep -q 'is your KV_OFFLOAD_DIR, so it is in use' <<<"$out" \
  && ok "the repo kv-offload/ is not offered for removal when KV_OFFLOAD_DIR points at it" || bad "in-use kv-offload listed: $out"
out="$(env CLUB3090_CONFIG_DIR="$C" CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" bash "$ROOT/scripts/settings.sh" caches 2>&1)"
command grep -q '^Compile caches' <<<"$out" && ok "bash scripts/settings.sh caches reaches it" || bad "settings.sh caches: $out"
out="$(env CLUB3090_CONFIG_DIR="$C" CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" bash "$ROOT/scripts/settings.sh" path 2>&1)"
command grep -qE "^cache dir: +$T/cache " <<<"$out" && command grep -qE "^data dir: +$T/data " <<<"$out" \
  && ok "settings.sh path lists the cache and data dirs" || bad "settings.sh path: $out"

# ── 7. every launch path goes through the helper ──────────────────────────────
echo "[7] launch paths"
up_fn="$(sed -n '/^up_variant()/,/^}/p' scripts/switch.sh)"
[[ "$up_fn" == *'club_engine_cache_export "${full_dir}" "${file}"'* ]] \
  && awk '/club_engine_cache_export/{e=NR} /up -d --remove-orphans/{u=NR} END{exit !(e && u && e<u)}' <<<"$up_fn" \
  && ok "switch.sh up_variant prepares the dirs before its compose up" || bad "switch.sh up_variant does not call club_engine_cache_export before up"
command grep -q 'club_engine_cache_env "$dir" "$file"' "$T/cae.sh" 2>/dev/null \
  && ok "gpu-mode compose_at_env prepares them (as you, before sudo)" || bad "gpu-mode compose_at_env does not call club_engine_cache_env"
command grep -q 'engine_cache_prepare(\[compose_path, override\]' scripts/lib/profiles/estate_cli.py \
  && ok "estate run_compose prepares them" || bad "estate_cli.run_compose does not call engine_cache.prepare"
sg="$(awk '/def serve_generated/{f=1} f{print} f&&/return ActionPlan/{exit}' tools/serve-cockpit/club3090_cockpit/services.py)"
[[ "$sg" == *'self._engine_cache_env(compose_path, env)'* ]] \
  && ok "c3's generated/brought-compose serve prepares them (catalog serves go through switch.sh)" \
  || bad "c3 serve_generated does not call _engine_cache_env"

if [[ $fail -ne 0 ]]; then echo "test-engine-cache: FAIL" >&2; exit 1; fi
echo "test-engine-cache: ok"
