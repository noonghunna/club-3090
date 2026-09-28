#!/usr/bin/env bash
# test-litellm-local-config — this rig's own gateway routes and their keys live in the
# settings dir, and an older install keeps working (club-3090#1466, phase 4c).
#
# WHY THIS TEST EXISTS
# --------------------
# The gateway's per-rig files used to sit in the checkout: services/litellm/
# config.local.yaml (routes) and services/litellm/local.env (their keys, loaded by the
# compose as an env_file). They move to the settings dir: routes to
# <config dir>/litellm/config.local.yaml, keys to secrets.env. Each way this can go
# wrong is silent — a cloud route that answers 401, a gateway handed the HF token, a
# key on a command line — so every reader is checked through its real entry point:
#   1. litellm-sync reads the config-dir routes, else the checkout's (legacy), and
#      says so once;
#   2. the gateway gets ONLY the keys a route names, from a 0600 temp file whose
#      path (never a value) is handed over, and docker compose loads it after the
#      legacy local.env, so a saved key wins and a legacy-only rig is unchanged;
#   3. gpu-mode (a mode start and `gpu-mode gateway`) hands that file over through
#      sudo and removes it;
#   4. `settings.sh migrate` copies both files, never changes the repo ones, changes
#      no key the gateway gets, and prints no secret.
# Offline: temp config dirs and fixture checkouts; `docker compose config` only
# renders (a throwaway project, a scratch copy of the compose, so the checkout's
# own local.env is never read); gpu-mode runs as a scratch clone with docker / sudo
# / curl PATH shims prepended INLINE. No container is started or touched.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$ROOT/scripts/lib"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; fail=1; }
unset HF_TOKEN LITELLM_MASTER_KEY RIG_CLOUD_KEY C3_LITELLM_LOCAL_CONFIG C3_LITELLM_FAKE_LIVE CLUB3090_LITELLM_ROUTE_KEYS
export HOME="$T/home"; mkdir -p "$HOME" "$T/tmp"
nosecret() {   # <label> <text>: no fixture secret (zz-…) anywhere in it
  if command grep -qoE 'zz-[A-Za-z0-9-]+' <<<"$2"; then bad "$1 — printed a secret: $(command grep -oE 'zz-[A-Za-z0-9-]+' <<<"$2" | sort -u | tr '\n' ' ')"
  else ok "$1"; fi
}

# A fixture checkout: the tracked catalog, plus an older install's own files.
R="$T/repo"; mkdir -p "$R/services/litellm"
cp "$ROOT/services/litellm/config.yaml" "$ROOT/services/litellm/docker-compose.yml" "$R/services/litellm/"
cat > "$R/services/litellm/config.local.yaml" <<'YAML'
model_list:
  - model_name: legacy-cloud
    litellm_params:
      model: openai/legacy-cloud
      api_base: https://cloud.example.invalid/v1
      api_key: os.environ/RIG_CLOUD_KEY
      # api_key: os.environ/COMMENTED_KEY   (a comment: not a reference)
  - model_name: other-cloud
    litellm_params:
      model: openai/other-cloud
      api_base: https://other.example.invalid/v1
      api_key: os.environ/ODD_VALUE_KEY
YAML
printf 'RIG_CLOUD_KEY=zz-legacy-route\nODD_VALUE_KEY=has$dollar\nUNREF_KEY=zz-legacy-unref\n' > "$R/services/litellm/local.env"
mk_cfg() {   # <dir>: a config dir holding the HF token and other secrets, 0600
  mkdir -p "$1"; chmod 700 "$1"
  printf 'HF_TOKEN=zz-hf\nLITELLM_MASTER_KEY=zz-master\nCOMMENTED_KEY=zz-commented\n' > "$1/secrets.env"; chmod 600 "$1/secrets.env"
}
py() { env -u HF_TOKEN -u LITELLM_MASTER_KEY python3 - "$LIB" "$@"; }

# ── 1. where litellm-sync reads the routes from ─────────────────────────────
echo "1. the routes litellm-sync serves"
CFG="$HOME/.config/club-3090"; mk_cfg "$CFG"    # where it lands by default
got="$(CLUB3090_CONFIG_DIR="$CFG" py "$R" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import litellm_sync as s
print(s.local_routes(sys.argv[2]))
PY
)"
[[ "$got" == *"model_name: legacy-cloud"* && "$got" == *"THIS RIG'S OWN ROUTES — services/litellm/config.local.yaml"* ]] \
  && ok "no routes file in the settings dir: the checkout's (older install) is served, and named as the source" \
  || bad "legacy routes not served: ${got:0:300}"
mkdir -p "$CFG/litellm"
printf 'model_list:\n  - model_name: cfg-cloud\n    litellm_params:\n      model: openai/cfg-cloud\n      api_base: https://cfg.example.invalid/v1\n      api_key: os.environ/RIG_CLOUD_KEY\n' > "$CFG/litellm/config.local.yaml"
got="$(CLUB3090_CONFIG_DIR="$CFG" py "$R" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import litellm_sync as s
print(s.local_routes(sys.argv[2]))
PY
)"
[[ "$got" == *"model_name: cfg-cloud"* && "$got" != *legacy-cloud* && "$got" == *"— ~/.config/club-3090/litellm/config.local.yaml (not tracked)"* ]] \
  && ok "the settings dir's routes file wins; the checkout's copy is not read" \
  || bad "config-dir routes: ${got:0:300}"
got="$(CLUB3090_CONFIG_DIR="$CFG" C3_LITELLM_LOCAL_CONFIG="$T/none.yaml" py "$R" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import litellm_sync as s
print(repr(s.local_routes(sys.argv[2])))
PY
)"
[[ "$got" == "''" ]] && ok "C3_LITELLM_LOCAL_CONFIG (the test seam) still overrides both" || bad "override: $got"

# "Still read from the checkout": a plain sync says it every time. A quiet sync (switch.sh,
# gpu-mode) leaves it to their own one-time migrate notice (club_config.migrate_notice,
# test-migrate-notice.sh), which counts these files too — so it isn't said twice.
CFG="$T/cfg-notice"; mk_cfg "$CFG"
notice() { CLUB3090_CONFIG_DIR="$CFG" py "$R" "$1" <<'PY' 2>&1
import sys; sys.path.insert(0, sys.argv[1]); import litellm_local as l
l.notice(sys.argv[2], quiet=sys.argv[3] == "quiet")
PY
}
n1="$(notice quiet)"; n3="$(notice loud)"; n5="$(notice loud)"
[[ -z "$n1" && "$n3" == *"still come from the checkout"* && "$n3" == *"settings.sh migrate"* && "$n5" == "$n3" ]] \
  && ok "legacy files: a plain sync notes it every time; a quiet one leaves it to the launchers' one-time notice" \
  || bad "notice: quiet=[$n1] loud=[$n3] loud-again=[$n5]"
got="$(CLUB3090_CONFIG_DIR="$CFG" py "$R" <<'PY' 2>&1
import sys; sys.path.insert(0, sys.argv[1]); import club_config as c
print("; ".join(c.migrate_pending(sys.argv[2])[0]))
PY
)"
[[ "$got" == *"config.local.yaml"* && "$got" == *"route key(s)"* ]] \
  && ok "the launchers' migrate notice counts the gateway files" || bad "migrate_pending: $got"
nosecret "the notice names files and counts keys, never a value" "$n1$n3$got"
n4="$(CLUB3090_CONFIG_DIR="$CFG" C3_LITELLM_LOCAL_CONFIG="$T/x.yaml" py "$R" <<'PY' 2>&1
import sys; sys.path.insert(0, sys.argv[1]); import litellm_local as l
l.notice(sys.argv[2], quiet=False)
PY
)"
[[ -z "$n4" ]] && ok "no notice while the test seam supplies the routes" || bad "notice under the seam: $n4"

# ── 2. the keys the gateway gets ─────────────────────────────────────────────
echo "2. the route keys file"
CFG="$T/cfg2"; mk_cfg "$CFG"; mkdir -p "$CFG/litellm"
cat > "$CFG/litellm/config.local.yaml" <<'YAML'
model_list:
  - model_name: cfg-cloud
    litellm_params:
      model: openai/cfg-cloud
      api_base: https://cfg.example.invalid/v1
      api_key: os.environ/RIG_CLOUD_KEY   # the key this route needs
      # api_base: os.environ/COMMENTED_KEY
  - model_name: master-ref
    litellm_params:
      model: openai/x
      api_key: os.environ/LITELLM_MASTER_KEY
  - model_name: unsaved
    litellm_params:
      model: openai/y
      api_key: os.environ/NEVER_SAVED_KEY
YAML
printf 'RIG_CLOUD_KEY=zz-route#x$y\n' >> "$CFG/secrets.env"   # hand-written: $ and # must survive compose
keys_file() { env -u HF_TOKEN -u LITELLM_MASTER_KEY CLUB3090_CONFIG_DIR="$CFG" TMPDIR="$T/tmp" "$@" python3 "$LIB/litellm_local.py" route-keys-file --root "$R" 2>"$T/kerr"; }
out="$(keys_file)"; rc=$?
if [[ $rc -eq 0 && "$(wc -l <<<"$out")" == 1 && -f "$out" ]]; then
  ok "route-keys-file prints one line: the path of the file it wrote"
  [[ "$(stat -c %a "$out")" == 600 ]] && ok "…mode 0600" || bad "keys file mode $(stat -c %a "$out")"
  [[ "$(cat "$out")" == "RIG_CLOUD_KEY='zz-route#x\$y'" ]] \
    && ok "…holding ONLY the key a route names: no HF token, no gateway key, no commented or unsaved reference" \
    || bad "keys file holds: $(sed 's/=.*/=…/' "$out" | tr '\n' ' ')"
  rm -f "$out"
else
  bad "route-keys-file rc=$rc stdout=[$out] stderr=[$(cat "$T/kerr")]"
fi
nosecret "route-keys-file prints no key (stdout and stderr)" "$out$(cat "$T/kerr")"
out="$(keys_file env RIG_CLOUD_KEY=zz-from-shell)"
[[ -f "$out" && "$(cat "$out")" == "RIG_CLOUD_KEY='zz-from-shell'" ]] && ok "the shell wins over the saved key, as in every launch" \
  || bad "shell precedence: $(cat "$out" 2>/dev/null | sed 's/=.*/=…/')"
rm -f "$out"
EMPTY="$T/cfg-empty"; mkdir -p "$EMPTY"
R0="$T/repo-bare"; mkdir -p "$R0/services/litellm"; cp "$ROOT/services/litellm/config.yaml" "$R0/services/litellm/"
out="$(env -u HF_TOKEN CLUB3090_CONFIG_DIR="$EMPTY" TMPDIR="$T/tmp" python3 "$LIB/litellm_local.py" route-keys-file --root "$R0")"; rc=$?
[[ $rc -eq 0 && -z "$out" && -z "$(ls -A "$T/tmp")" ]] && ok "no route needs a saved key: prints nothing, writes no file" \
  || bad "empty case rc=$rc out=[$out] tmp=[$(ls -A "$T/tmp")]"

# A route whose saved key the running gateway lacks: litellm-sync can only restart it,
# which keeps the old environment, so it must say to recreate (names only, no values).
got="$(CLUB3090_CONFIG_DIR="$CFG" py "$R" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import litellm_local as l
print(l.keys_missing_from_gateway(sys.argv[2], {"LITELLM_MASTER_KEY", "PATH"}),
      l.keys_missing_from_gateway(sys.argv[2], {"RIG_CLOUD_KEY"}))
PY
)"
[[ "$got" == "['RIG_CLOUD_KEY'] []" ]] && ok "a route's saved key missing from the gateway's environment is reported (and not once it is there)" \
  || bad "keys_missing_from_gateway: $got"

# ── 3. what docker compose hands the container ───────────────────────────────
echo "3. docker compose (config only, a scratch copy of the compose)"
if docker compose version >/dev/null 2>&1; then
  S="$T/scratch/services/litellm"; mkdir -p "$S"; cp "$ROOT/services/litellm/docker-compose.yml" "$S/"
  printf 'RIG_CLOUD_KEY=zz-legacy-route\nLEGACY_ONLY_KEY=zz-legacy-only\n' > "$S/local.env"
  PROJECT="c3test-litellm-local-$$-$RANDOM"
  # `config` prints a literal $ as $$ (so its output can be read back); undo that to
  # get what the container receives (checked once against a real container: a
  # single-quoted 'a#x$y' in an env_file arrives as a#x$y).
  cenv() {  # [VAR=val…] → the gateway's container environment, as JSON
    (cd "$S" && env -u HF_TOKEN -u LITELLM_MASTER_KEY -u CLUB3090_LITELLM_ROUTE_KEYS "$@" docker compose -p "$PROJECT" -f docker-compose.yml config --format json 2>"$T/cerr") \
      | python3 -c 'import json,sys; e=json.load(sys.stdin)["services"]["litellm"]["environment"]; print(json.dumps({k: v if v is None else v.replace("$$", "$") for k, v in e.items()}, sort_keys=True))' 2>/dev/null
  }
  base="$(cenv)"
  want_base="$(python3 -c 'import json; print(json.dumps({"LEGACY_ONLY_KEY":"zz-legacy-only","LITELLM_LOG":None,"LITELLM_MASTER_KEY":"sk-litellm-master-key","RIG_CLOUD_KEY":"zz-legacy-route"}, sort_keys=True))')"
  [[ "$base" == "$want_base" ]] && ok "nothing passed (an older install, or a plain docker compose up): exactly the keys in local.env, as before" \
    || bad "legacy env: $base $(cat "$T/cerr")"
  kf="$(keys_file)"
  got="$(cenv CLUB3090_LITELLM_ROUTE_KEYS="$kf")"
  python3 - "$got" <<'PY' 2>/dev/null && ok "with the keys file: the saved key wins over local.env, byte-exact (\$ and # kept); local.env's other keys stay; no HF token" || bad "with keys file: $got"
import json, sys
e = json.loads(sys.argv[1])
assert e.get("RIG_CLOUD_KEY") == "zz-route#x$y", e
assert e.get("LEGACY_ONLY_KEY") == "zz-legacy-only", e
assert "HF_TOKEN" not in e and "COMMENTED_KEY" not in e, e
PY
  cp "$kf" "$T/kf2"
  h1="$(cd "$S" && env -u HF_TOKEN CLUB3090_LITELLM_ROUTE_KEYS="$kf" docker compose -p "$PROJECT" config --hash litellm 2>/dev/null)"
  h2="$(cd "$S" && env -u HF_TOKEN CLUB3090_LITELLM_ROUTE_KEYS="$T/kf2" docker compose -p "$PROJECT" config --hash litellm 2>/dev/null)"
  [[ -n "$h1" && "$h1" == "$h2" ]] && ok "the container's config hash depends on the keys, not the temp file's path (no needless recreate)" \
    || bad "hash differs by path: [$h1] [$h2]"
  rm -f "$kf" "$T/kf2"
else
  echo "  - SKIP: docker compose not available"
fi

# ── 4. gpu-mode hands the file over ─────────────────────────────────────────
echo "4. gpu-mode"
C="$T/club"; mkdir -p "$C/scripts/lib" "$C/services/litellm"
cp "$ROOT/scripts/gpu-mode.sh" "$C/scripts/"
cp "$LIB/club-config.sh" "$LIB/club_config.py" "$LIB/litellm_local.py" "$C/scripts/lib/"
cp "$ROOT/services/litellm/docker-compose.yml" "$ROOT/services/litellm/config.yaml" "$C/services/litellm/"
# The real sync would re-render (and probe for) this checkout's routes: a stub.
printf '#!/usr/bin/env bash\nexit 0\n' > "$C/scripts/lib/litellm-sync.sh"
chmod +x "$C/scripts/gpu-mode.sh" "$C/scripts/lib/litellm-sync.sh"
mkdir -p "$T/bin" "$T/stub"
# sudo passes VAR=val through to the command, as the real one does; logs its argv.
cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
echo "SUDO $*" >> "$STUB_LOG"
exec env "$@"
EOF
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "[$(basename "$PWD")] docker $*" >> "$STUB_LOG"
case "$1" in
  info) exit 0 ;;
  compose)
    if [[ " $* " == *" config "* ]]; then
      printf '{"name":"litellm","services":{"litellm":{"image":"ghcr.io/berriai/litellm:new","container_name":"litellm"}}}\n'; exit 0
    fi
    if [[ " $* " == *" up "* ]]; then
      n=$(( $(cat "$STUB_DIR/ups" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STUB_DIR/ups"
      echo "UP $n KEYSVAR=${CLUB3090_LITELLM_ROUTE_KEYS-<unset>}" >> "$STUB_LOG"
      if [[ -n "${CLUB3090_LITELLM_ROUTE_KEYS:-}" ]]; then
        cp "$CLUB3090_LITELLM_ROUTE_KEYS" "$STUB_DIR/keys.$n"; stat -c %a "$CLUB3090_LITELLM_ROUTE_KEYS" > "$STUB_DIR/keys.$n.mode"
      fi
    fi
    exit 0 ;;
  inspect)
    case "$4" in *Config.Image*) echo "ghcr.io/berriai/litellm:old" ;; *State.Running*) echo true ;; *) echo "" ;; esac ;;
  *) exit 0 ;;
esac
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/curl"          # the liveliness wait succeeds; nothing is reached
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/nvidia-smi"
chmod +x "$T/bin/"*
for b in docker sudo curl; do
  [[ "$(PATH="$T/bin:$PATH" command -v "$b")" == "$T/bin/$b" ]] || { echo "✗ $b shim not first on PATH — refusing to run gpu-mode" >&2; exit 1; }
done
export STUB_DIR="$T/stub" STUB_LOG="$T/stub.log"
CFG="$T/cfg4"; mk_cfg "$CFG"; mkdir -p "$CFG/litellm"
printf 'model_list:\n  - model_name: c\n    litellm_params:\n      model: openai/c\n      api_key: os.environ/RIG_CLOUD_KEY\n' > "$CFG/litellm/config.local.yaml"
printf 'RIG_CLOUD_KEY=zz-gpumode-route\n' >> "$CFG/secrets.env"
gm() { : > "$STUB_LOG"; rm -f "$STUB_DIR/"{ups,keys.*}; rm -rf "$T/tmp"; mkdir -p "$T/tmp"
       env -u HF_TOKEN -u LITELLM_MASTER_KEY CLUB3090_CONFIG_DIR="$1" CLUB3090_GATEWAY_URL=http://127.0.0.1:9 GPU_MODE_GATEWAY_WAIT_S=4 \
         TMPDIR="$T/tmp" PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" "${@:2}" > "$T/gm.out" 2>&1; }
for how in gateway upgrade; do
  if [[ "$how" == upgrade ]]; then gm "$CFG" upgrade --no-backup; else gm "$CFG" gateway; fi
  up="$(command grep -m1 '^UP 1 ' "$STUB_LOG" || true)"
  if [[ -f "$STUB_DIR/keys.1" ]]; then
    [[ "$(cat "$STUB_DIR/keys.1")" == "RIG_CLOUD_KEY='zz-gpumode-route'" && "$(cat "$STUB_DIR/keys.1.mode")" == 600 ]] \
      && ok "gpu-mode $how: compose up gets the route's key (only it) in a 0600 file via CLUB3090_LITELLM_ROUTE_KEYS" \
      || bad "gpu-mode $how: keys file holds $(sed 's/=.*/=…/' "$STUB_DIR/keys.1") mode $(cat "$STUB_DIR/keys.1.mode")"
  else
    bad "gpu-mode $how: compose up got no route keys file ($up); output: $(tail -5 "$T/gm.out")"
  fi
  command grep -q '^SUDO CLUB3090_LITELLM_ROUTE_KEYS=.* docker compose ' "$STUB_LOG" \
    && ok "gpu-mode $how: …its path rides sudo's argv (sudo strips the environment)" || bad "gpu-mode $how: not through sudo: $(command grep '^SUDO' "$STUB_LOG")"
  nosecret "gpu-mode $how: no key on any sudo/docker command line or in its output" "$(cat "$STUB_LOG" "$T/gm.out")"
  left="$(ls -A "$T/tmp" | command grep -E 'club3090-litellm-' || true)"
  [[ -z "$left" ]] && ok "gpu-mode $how: the keys file is removed after compose" || bad "gpu-mode $how: left behind: $left"
done
gm "$T/cfg-empty" gateway
command grep -q '^UP 1 KEYSVAR=<unset>$' "$STUB_LOG" && ! command grep -q 'CLUB3090_LITELLM_ROUTE_KEYS=' "$STUB_LOG" \
  && ok "gpu-mode gateway with no saved route key: nothing passed (compose loads only a legacy local.env)" \
  || bad "no-keys case: $(command grep -E '^(UP|SUDO)' "$STUB_LOG")"

# ── 5. settings.sh migrate ───────────────────────────────────────────────────
echo "5. settings.sh migrate"
mkdir -p "$R/scripts"; cp "$ROOT/scripts/settings.sh" "$R/scripts/"; ln -s "$LIB" "$R/scripts/lib"
st() { env -i PATH="$PATH" HOME="$HOME" CLUB3090_CONFIG_DIR="$CFG" bash "$R/scripts/settings.sh" "$@" > "$T/out" 2> "$T/err"; RC=$?; OUT="$(cat "$T/out")"; ERR="$(cat "$T/err")"; }
# What the gateway ends up with: local.env as compose loads it, then the keys file over it.
gateway_keys() { env -i PATH="$PATH" HOME="$HOME" CLUB3090_CONFIG_DIR="$CFG" TMPDIR="$T/tmp" python3 - "$LIB" "$R" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import club_config as c, litellm_local as l
from pathlib import Path
env = dict(c.parse_env_file(Path(sys.argv[2]) / "services/litellm/local.env"))
env.update(l.route_keys(sys.argv[2]))
print("\n".join(f"{k}={v}" for k, v in sorted(env.items())))
PY
}
stamp() { stat -c '%s %a %Y' "$1"; }
touch -d '2020-01-02 03:04:05' "$R/services/litellm/config.local.yaml" "$R/services/litellm/local.env"
chmod 640 "$R/services/litellm/local.env"
rs="$(stamp "$R/services/litellm/config.local.yaml")"; ks="$(stamp "$R/services/litellm/local.env")"
cp -p "$R/services/litellm/config.local.yaml" "$T/routes.orig"; cp -p "$R/services/litellm/local.env" "$T/keys.orig"
CFG="$T/m-none"
st migrate --dry-run
[[ $RC -eq 0 && ! -e "$CFG" ]] && ok "migrate --dry-run creates nothing, not even the config dir" || bad "dry run: rc=$RC created=$(ls -A "$CFG" 2>/dev/null)"
[[ "$OUT" == *"would copy $R/services/litellm/config.local.yaml to $CFG/litellm/config.local.yaml"* \
   && "$OUT" == *"would copy route keys from $R/services/litellm/local.env to secrets.env: RIG_CLOUD_KEY"* ]] \
  && ok "migrate --dry-run: the plan names both files and the key to copy" || bad "dry-run plan: $OUT"
nosecret "migrate --dry-run: no secret value" "$OUT$ERR"

CFG="$T/m"; mk_cfg "$CFG"
before="$(gateway_keys)"
st show
[[ "$OUT" == *"still come from the checkout"* ]] && ok "settings.sh show says the gateway files still live in the checkout" || bad "show: no gateway note: $OUT"
st migrate
[[ $RC -eq 0 ]] || bad "migrate rc=$RC: $ERR"
cmp -s "$CFG/litellm/config.local.yaml" "$T/routes.orig" && [[ "$(stat -c %a "$CFG/litellm/config.local.yaml")" == 600 ]] \
  && ok "migrate: the routes file is copied byte for byte into the config dir (0600)" || bad "routes copy wrong"
command grep -qxF 'RIG_CLOUD_KEY=zz-legacy-route' "$CFG/secrets.env" && [[ "$(stat -c %a "$CFG/secrets.env")" == 600 ]] \
  && ok "migrate: the key a route uses goes to secrets.env (0600)" || bad "secrets.env: $(sed 's/=.*/=…/' "$CFG/secrets.env" | tr '\n' ' ')"
! command grep -qE '^(UNREF_KEY|ODD_VALUE_KEY)=' "$CFG/secrets.env" \
  && ok "migrate: a key no route uses, and a value the writer refuses, stay in local.env" || bad "copied an unreferenced or refused key"
[[ "$OUT" == *"ODD_VALUE_KEY: can't be stored"* && "$OUT" == *"no route uses UNREF_KEY"* && "$OUT" == *"Keep it for ODD_VALUE_KEY, UNREF_KEY"* ]] \
  && ok "migrate: says which keys stay, why, and that local.env must be kept for them" || bad "migrate output: $OUT"
[[ "$OUT" == *"config.local.yaml was not changed and is no longer read: you can delete it."* ]] \
  && ok "migrate: says the checkout's routes file can be deleted" || bad "migrate removal hint: $OUT"
nosecret "migrate: no secret value anywhere in its output" "$OUT$ERR"
cmp -s "$R/services/litellm/config.local.yaml" "$T/routes.orig" && cmp -s "$R/services/litellm/local.env" "$T/keys.orig" \
  && [[ "$(stamp "$R/services/litellm/config.local.yaml")" == "$rs" && "$(stamp "$R/services/litellm/local.env")" == "$ks" ]] \
  && ok "migrate: both repo files byte-identical, same mode and mtime" || bad "migrate MODIFIED a repo file"
after="$(gateway_keys)"
[[ "$before" == "$after" && -n "$after" ]] && ok "migrate: every key the gateway gets is unchanged (only where it comes from)" \
  || bad "gateway keys changed: $(diff <(sed 's/=.*/=…/' <<<"$before") <(sed 's/=.*/=…/' <<<"$after") | tr '\n' ' ')"
cp "$CFG/secrets.env" "$T/s1"
st migrate
cmp -s "$CFG/secrets.env" "$T/s1" && [[ "$OUT" == *"already in $CFG/litellm/config.local.yaml, same content"* && "$OUT" == *"already saved, same value: RIG_CLOUD_KEY"* ]] \
  && ok "migrate is idempotent: the second run copies nothing" || bad "second migrate: $OUT"
st show
[[ "$OUT" != *"still come from the checkout"* ]] && ok "settings.sh show: no gateway note once migrated" || bad "show after migrate: $OUT"
printf '# edited after the move\n' >> "$R/services/litellm/config.local.yaml"
st show
[[ "$OUT" == *"is no longer read"*"and the two differ"* ]] && ok "an edit to the checkout's copy after the move is called out (that copy is no longer read)" \
  || bad "no stale-copy note: $OUT"
st path
[[ "$OUT" == *"gateway routes:"*"$CFG/litellm/config.local.yaml"*"exists"* && "$OUT" == *"gateway keys (old):"*"local.env"* ]] \
  && ok "settings.sh path lists the gateway routes file and the old local.env" || bad "path: $OUT"

[[ $fail -eq 0 ]] && echo "test-litellm-local-config: ok" || echo "test-litellm-local-config: FAIL"
exit $fail
