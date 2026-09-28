#!/usr/bin/env bash
# test-gateway-key-init — a fresh install gets a gateway key of its own; an existing
# install never gets one it didn't ask for (#1467).
#
# WHY THIS TEST EXISTS
# --------------------
# `gateway-key.sh init` (run by setup.sh) stores a random LITELLM_MASTER_KEY so a new
# install doesn't start its gateway on the public default every club-3090 install
# shares. The dangerous direction is the other one: a key generated on an install
# whose clients (omp, pi, Hermes, Claude Code, Open WebUI) already hold the current
# key breaks every one of them at the next gateway start (`400 No connected db.`), and
# nothing says why. So each sign of an existing install is tested ALONE against an
# otherwise fresh fixture — the fresh fixture itself must generate (positive control),
# or a check that never generates at all would pass every negative case.
#   1. fresh → a key sk-club-<32 hex>, secrets.env 0600, never printed; a second run
#      keeps it; Docker not installed at all still counts as fresh;
#   2. every signal alone → no key written, the reason said;
#   3. setup.sh runs it after preflight and before the model step (so SKIP_MODEL=1
#      runs it too), reports a stored key as stored (not as "set in this shell",
#      though setup exported it), and carries on when init fails.
# Offline: temp HOME / config dirs / checkouts, `docker` and `curl` PATH shims
# prepended INLINE on every call (docker answers from fixture files; curl answers
# nothing unless told to), a stub `hermes`. No container is touched, no real
# settings or agent config is read.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; fail=1; }
GK="$ROOT/scripts/gateway-key.sh"
unset LITELLM_MASTER_KEY HF_TOKEN PI_CODING_AGENT_DIR HERMES_HOME CLUB3090_GATEWAY_URL CLUB3090_DIR XDG_CONFIG_HOME   # a real token in this shell stays out of anything resolved here

mkdir -p "$T/bin"
# docker: `info` fails when $D/down exists; `ps -a` / `volume ls` list $D/containers / $D/volumes.
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "docker $*" >> "$D/calls"
case "$1" in
  info)   [[ -e "$D/down" ]] && exit 1; exit 0 ;;
  ps)     [[ " $* " == *" -a "* ]] || { echo "ps without -a" >> "$D/calls"; exit 0; }
          cat "$D/containers" 2>/dev/null; exit 0 ;;
  volume) cat "$D/volumes" 2>/dev/null; exit 0 ;;
  *) exit 0 ;;
esac
EOF
# curl: "nothing answers" (000) unless $D/answer exists.
cat > "$T/bin/curl" <<'EOF'
#!/usr/bin/env bash
echo "curl $*" >> "$D/calls"
if [[ -e "$D/answer" ]]; then printf 200; exit 0; fi
printf 000; exit 7
EOF
cat > "$T/bin/hermes" <<'EOF'
#!/usr/bin/env bash
# only `config get providers.club.api` is asked by the detection
echo "http://gpu-box.invalid:4000/v1"
EOF
chmod +x "$T/bin/"*
for b in docker curl; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run" >&2; exit 1; }
done
# A PATH with the tools gateway-key.sh uses and NO docker, for "Docker not installed".
mkdir -p "$T/nodocker"
for tool in bash sh env python3 awk sed sort grep dirname cat head tr mktemp rm chmod stat md5sum cut ls; do
  p="$(command -v "$tool")" && ln -sf "$p" "$T/nodocker/$tool"
done
cp "$T/bin/curl" "$T/nodocker/curl"
PATH="$T/nodocker" command -v docker >/dev/null 2>&1 && { echo "✗ fixture: docker still on the no-docker PATH" >&2; exit 1; }

n=0
# fresh <name> — a fresh fixture: empty config dir, checkout, HOME and Docker; sets
# CFG CHK HM D. Callers then add ONE signal.
fresh() {
  n=$((n + 1)); local b="$T/case$n-$1"
  CFG="$b/cfg"; CHK="$b/checkout"; HM="$b/home"; D="$b/docker"
  mkdir -p "$CFG" "$CHK/services/litellm" "$HM" "$D"; : > "$D/calls"
}
# run_init [VAR=val ...] — gateway-key.sh init against the fixture.
run_init() {
  env -u LITELLM_MASTER_KEY CLUB3090_CONFIG_DIR="$CFG" CLUB3090_DIR="$CHK" HOME="$HM" D="$D" \
    CLUB3090_GATEWAY_URL=http://127.0.0.1:9 HERMES_BIN="$T/bin/hermes" "$@" PATH="$T/bin:$PATH" bash "$GK" init 2>&1
}
stored() { sed -n 's/^LITELLM_MASTER_KEY=//p' "$CFG/secrets.env" 2>/dev/null; }

# ── 1. fresh ────────────────────────────────────────────────────────────────
echo "1. a fresh install"
fresh fresh
out="$(run_init)"; rc=$?; k="$(stored)"
[[ $rc -eq 0 && "$k" =~ ^sk-club-[0-9a-f]{32}$ ]] && ok "stores a random sk-club-<32 hex> key (positive control)" || bad "fresh: rc=$rc out=$out"
[[ "$(stat -c %a "$CFG/secrets.env" 2>/dev/null)" == 600 ]] && ok "secrets.env is 0600" || bad "secrets.env mode $(stat -c %a "$CFG/secrets.env" 2>/dev/null)"
[[ -n "$k" && "$out" != *"$k"* && "$out" == *"Fresh install"* ]] && ok "says so, and never prints the key" || bad "fresh output: printed the key, or no 'Fresh install'"
command grep -q '^docker ps -a' "$D/calls" && command grep -q '^docker volume ls' "$D/calls" \
  && ok "it asked Docker for containers in any state (ps -a) and for volumes" || bad "docker calls: $(tr '\n' '|' < "$D/calls")"
out="$(run_init)"; rc=$?
[[ $rc -eq 0 && "$(stored)" == "$k" && "$out" == *"already stored (secrets.env)"* ]] && ok "a second run keeps the key" || bad "second init: rc=$rc changed=$([[ "$(stored)" == "$k" ]] && echo no || echo yes) $out"

fresh nodocker
out="$(env -u LITELLM_MASTER_KEY CLUB3090_CONFIG_DIR="$CFG" CLUB3090_DIR="$CHK" HOME="$HM" D="$D" CLUB3090_GATEWAY_URL=http://127.0.0.1:9 \
       PATH="$T/nodocker" bash "$GK" init 2>&1)"; rc=$?
[[ $rc -eq 0 && "$(stored)" =~ ^sk-club- ]] && ok "Docker not installed at all: no gateway can exist, so a key is stored" || bad "no docker: rc=$rc $out"

# ── 2. every sign of an existing install, alone ─────────────────────────────
echo "2. an existing install keeps its key"
# expect_kept <label> <expected reason substring> [VAR=val ...]
expect_kept() {
  local label="$1" want="$2"; shift 2
  local before out rc
  before="$(cat "$CFG/secrets.env" 2>/dev/null)"
  out="$(run_init "$@")"; rc=$?
  if [[ $rc -eq 0 && "$(cat "$CFG/secrets.env" 2>/dev/null)" == "$before" && "$out" == *"$want"* ]] \
     && ! command grep -q '^LITELLM_MASTER_KEY=sk-club-' "$CFG/secrets.env" 2>/dev/null; then
    ok "$label → no key generated ('$want')"
  else
    bad "$label: rc=$rc, secrets.env $([[ "$(cat "$CFG/secrets.env" 2>/dev/null)" == "$before" ]] && echo unchanged || echo CHANGED), out: $out"
  fi
}
fresh shell;       expect_kept "LITELLM_MASTER_KEY in the shell" "set in this shell" LITELLM_MASTER_KEY=mine
fresh secrets;     printf 'LITELLM_MASTER_KEY=my-own-key\n' > "$CFG/secrets.env"
                   expect_kept "a key in secrets.env" "already stored (secrets.env)"
fresh global;      printf 'LITELLM_MASTER_KEY=my-own-key\n' > "$CFG/club3090.env"
                   expect_kept "a key in club3090.env" "already stored (club3090.env)"
fresh repoenv;     printf 'LITELLM_MASTER_KEY=my-own-key\n' > "$CHK/.env"
                   expect_kept "a key in the repo .env" "already stored (repo .env)"
fresh empty;       printf 'LITELLM_MASTER_KEY=\n' > "$CFG/secrets.env"
                   expect_kept "LITELLM_MASTER_KEY set but empty" "to the public default"
fresh default;     printf 'LITELLM_MASTER_KEY=sk-litellm-master-key\n' > "$CFG/club3090.env"
                   expect_kept "the public default stored explicitly" "to the public default"
fresh runtime;     : > "$CHK/services/litellm/config.runtime.yaml"
                   expect_kept "this checkout's rendered gateway config" "config.runtime.yaml exists"
fresh c-litellm;   printf 'searxng\nlitellm\n' > "$D/containers"
                   expect_kept "a litellm container (any state)" "a 'litellm' container exists"
fresh c-owui;      printf 'open-webui\n' > "$D/containers"
                   expect_kept "an open-webui container" "a 'open-webui' container exists"
fresh c-near;      printf 'litellm-old\nmy-open-webui\n' > "$D/containers"
                   out="$(run_init)"; [[ "$(stored)" =~ ^sk-club- ]] && ok "…names are matched exactly (litellm-old / my-open-webui don't count)" \
                     || bad "a near-miss container name blocked the key: $out"
fresh volume;      printf 'qdrant_qdrant-data\nopenwebui_open-webui-data\n' > "$D/volumes"
                   expect_kept "Open WebUI's data volume" "openwebui_open-webui-data"
fresh docker-down; : > "$D/down"
                   expect_kept "Docker installed but can't be asked" "Docker can't be asked"
fresh answering;   : > "$D/answer"
                   expect_kept "something answers at the gateway's address" "something already answers"
fresh omp;         mkdir -p "$HM/.omp/agent"
                   printf 'providers:\n  # >>> club-3090 local models (generated by club-3090 scripts/omp-setup.sh; re-run to refresh) >>>\n  club:\n    baseUrl: http://127.0.0.1:4000/v1\n  # <<< club-3090 local models <<<\n' > "$HM/.omp/agent/models.yml"
                   expect_kept "an omp setup" "omp's config"
fresh pi;          mkdir -p "$HM/.pi/agent"
                   printf '{"providers":{"club":{"name":"club-3090 local models (generated by club-3090 scripts/pi-setup.sh; re-run to refresh)","baseUrl":"http://gpu-box.invalid:4000/v1"}}}\n' > "$HM/.pi/agent/models.json"
                   expect_kept "a pi setup (even on another box's gateway)" "pi's config"
fresh hermes;      mkdir -p "$HM/.hermes"
                   printf 'providers:\n  club:\n    name: club-3090 local models (scripts/hermes-setup.sh)\n' > "$HM/.hermes/config.yaml"
                   expect_kept "a Hermes setup" "hermes's config"

# ── 3. setup.sh ──────────────────────────────────────────────────────────────
echo "3. setup.sh"
# A checkout whose scripts/ is this tree (setup.sh derives its root with a logical
# cd, so the fixture is the root), as in test-setup-saved-settings.
FX="$T/fx"; mkdir -p "$FX"; ln -s "$ROOT/scripts" "$FX/scripts"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  -L) echo "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-test)" ;;
  *compute_cap*) echo "8.6" ;;
  *) echo "0, NVIDIA GeForce RTX 3090, 24576, 8.6" ;;
esac
EOF
chmod +x "$T/bin/nvidia-smi"
setup() {  # setup [VAR=val ...] — setup.sh qwen3.6-27b, stopped at SKIP_MODEL
  env -u MODEL_DIR -u HF_TOKEN -u PYTORCH_CUDA_ALLOC_CONF -u LITELLM_MASTER_KEY CLUB3090_CONFIG_DIR="$CFG" HOME="$HM" D="$D" \
    CLUB3090_GATEWAY_URL=http://127.0.0.1:9 MODEL_DIR="$T/weights" PREFLIGHT_DISK_GB=0 SKIP_MODEL=1 "$@" PATH="$T/bin:$PATH" \
    timeout 120 bash "$FX/scripts/setup.sh" qwen3.6-27b 2>&1
}
line_of() { command grep -nF -m1 "$1" <<<"$2" | cut -d: -f1; }
fresh setup
out="$(setup)"; rc=$?; k="$(stored)"
pre="$(line_of '[preflight] ok.' "$out")"; gk="$(line_of '[gateway-key] Fresh install' "$out")"; sk="$(line_of '[model]   SKIP_MODEL=1' "$out")"
[[ $rc -eq 0 && "$k" =~ ^sk-club-[0-9a-f]{32}$ ]] && ok "a fresh setup.sh run stores a gateway key" || bad "setup fresh: rc=$rc $(tail -5 <<<"$out")"
[[ -n "$pre" && -n "$gk" && -n "$sk" && "$pre" -lt "$gk" && "$gk" -lt "$sk" ]] \
  && ok "…after preflight, before the model step (SKIP_MODEL=1 runs it too)" || bad "order: preflight=$pre gateway-key=$gk skip=$sk"
[[ -n "$k" && "$out" != *"$k"* ]] && ok "…and setup never prints it" || bad "setup printed the key"
out="$(setup)"
[[ "$(stored)" == "$k" && "$out" == *"[gateway-key] a gateway key is already stored (secrets.env)"* ]] \
  && ok "a re-run keeps it, and reports it as stored in secrets.env (setup's own export of it doesn't read as a shell key)" \
  || bad "setup re-run: $(command grep -m2 'gateway-key' <<<"$out")"
fresh setup-existing; printf 'litellm\n' > "$D/containers"
out="$(setup)"; rc=$?
[[ $rc -eq 0 && ! -e "$CFG/secrets.env" && "$out" == *"a 'litellm' container exists"* && "$out" == *"[model]   SKIP_MODEL=1"* ]] \
  && ok "an existing install: no key, says why, setup carries on" || bad "setup existing: rc=$rc $(command grep -m2 'gateway-key' <<<"$out")"
fresh setup-shell
out="$(setup LITELLM_MASTER_KEY=exported-by-me)"
[[ ! -e "$CFG/secrets.env" && "$out" == *"set in this shell"* ]] && ok "a key the user exported counts as chosen" || bad "setup shell key: $(command grep -m2 'gateway-key' <<<"$out")"
fresh setup-unwritable; : > "$T/not-a-dir"
out="$(CFG="$T/not-a-dir/cfg"; setup CLUB3090_CONFIG_DIR="$T/not-a-dir/cfg")"; rc=$?
[[ $rc -eq 0 && "$out" == *"could not set up a gateway key"* && "$out" == *"[model]   SKIP_MODEL=1"* && "$out" != *"Traceback"* ]] \
  && ok "init failing (config dir unwritable) warns in one line and setup carries on" || bad "unwritable: rc=$rc $(command grep -m4 -iE 'gateway-key|traceback|error' <<<"$out")"

[ "$fail" -eq 0 ] && echo "test-gateway-key-init: ok (fresh → a 0600 key, never printed; each existing-install sign alone keeps the key; setup.sh runs it after preflight, before the model step, and survives its failure)"
exit "$fail"
