#!/usr/bin/env bash
# test-launch-settings — per-slug launch settings: the resolver, validation before the
# teardown, delivery to the container, and switch.sh --set/--unset/--explain (#1465, 3b/3c).
#
# Contract (scripts/lib/launch_settings.py, wired into scripts/switch.sh and launch.sh):
#   precedence   shell > this slug (slugs.json) > model pin (CLUB3090_THINKING_<MODEL>,
#                ENABLE_THINKING only) > club3090.env > secrets.env > repo .env > compose
#                default. A key the settings loader exported is NOT the shell.
#   validation   before the running slug is taken down: a value outside the slug's
#                catalogued domain, an unmet dependency, a RAM tier bigger than the host
#                can hold, an unreadable slugs.json. Never a compose default. Saved values
#                the slug doesn't read are warned about, not refused.
#   delivery     the resolved values reach `docker compose` on the real switch.sh path,
#                and through launch.sh, which delegates to switch.sh.
#
# Drives the REAL switch.sh / launch.sh from a fixture checkout (symlinks to this repo's
# scripts/ models/ tools/, plus its own .env) against a `docker` shim that logs every
# teardown and `compose up`. At `compose up` the COMPOSE_BIN shim records the environment
# and renders the compose with the real `docker compose config` (read-only) — the value
# must show up in what the container would get. Nothing is started or stopped.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
# Credentials exported in the calling shell never reach a render or anything printed.
while IFS= read -r _v; do unset "$_v"; done < <(compgen -e | command grep -E '(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|MASTER_KEY|_KEY)$')
# The knobs, pins and switches under test come only from each case.
while IFS= read -r _v; do unset "$_v"; done < <(compgen -e | command grep -E '^(KV_OFFLOAD_GB|KV_OFFLOAD_DISK|KV_OFFLOAD_DISK_GB|ENABLE_THINKING|REASONING_EFFORT|SPEC_N|SPEC|FORCE|MODEL_DIR|CLUB3090_THINKING_.*|CLUB3090_MEMINFO_FILE)$')
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
[[ -f scripts/lib/launch_settings.py ]] || { echo "  ✗ scripts/lib/launch_settings.py is missing" >&2; echo "test-launch-settings: FAIL"; exit 1; }
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
# a launch writes the slug-label override (data dir) and cache dirs: never into your real ones
export CLUB3090_DATA_DIR="$T/data" CLUB3090_CACHE_DIR="$T/cache"

# Fixture checkout: the repo's code, its own legacy .env. No services/ — so the gateway
# sync switch.sh runs after a teardown finds no template and touches nothing.
FIX="$T/root"; mkdir -p "$FIX"
for d in scripts models tools; do ln -s "$ROOT/$d" "$FIX/$d"; done
: > "$FIX/.env"

REAL_DOCKER="$(command -v docker || true)"
HAVE_COMPOSE=0
[[ -n "$REAL_DOCKER" ]] && "$REAL_DOCKER" compose version >/dev/null 2>&1 && HAVE_COMPOSE=1

mkdir -p "$T/bin" "$T/running"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
sub="${1:-}"; shift || true
case "$sub" in
  ps) printf '%s\n' "${MOCK_RUNNING:-}" ;;
  inspect)
    fmt=""
    while [[ $# -gt 0 ]]; do case "$1" in --format|-f) fmt="$2"; shift 2 ;; *) shift ;; esac; done
    if [[ "$fmt" == *working_dir* ]]; then printf '%s\n' "$MOCK_WORKDIR"
    elif [[ "$fmt" == *config_files* ]]; then printf 'running.yml\n'; fi ;;
  stop|rm|kill|restart) printf 'TEARDOWN %s %s\n' "$sub" "$*" >> "$MOCK_LOG" ;;
  compose)
    case " $* " in
      *" down "*) printf 'TEARDOWN compose %s\n' "$*" >> "$MOCK_LOG" ;;
      *" up "*)   printf 'UP compose %s\n' "$*" >> "$MOCK_LOG" ;;
    esac ;;
esac
exit 0
EOF
cat > "$T/bin/compose-capture" <<EOF
#!/usr/bin/env bash
# COMPOSE_BIN: at 'up', record the launch knobs in the environment and render the compose
# with it (docker compose config — read-only); log the call like the docker shim does.
args=("\$@"); fargs=()
while [[ \$# -gt 0 ]]; do case "\$1" in -f) fargs+=(-f "\$2"); shift 2 ;; *) shift ;; esac; done
case " \${args[*]} " in
  *" up "*)
    env | command grep -E '^(KV_OFFLOAD_GB|KV_OFFLOAD_DISK|KV_OFFLOAD_DISK_GB|ENABLE_THINKING|REASONING_EFFORT|SPEC_N)=' | sort > "\$MOCK_ENV_OUT"
    if [[ "$HAVE_COMPOSE" == 1 ]]; then
      "$REAL_DOCKER" compose "\${fargs[@]}" config --format json > "\$MOCK_RENDER" 2>/dev/null || echo RENDER_FAILED >> "\$MOCK_LOG"
    fi
    printf 'UP compose %s\n' "\${args[*]}" >> "\$MOCK_LOG" ;;
  *" down "*) printf 'TEARDOWN compose %s\n' "\${args[*]}" >> "\$MOCK_LOG" ;;
esac
exit 0
EOF
# launch.sh delegates to \$SWITCH: here the REAL switch.sh of the fixture, non-interactive.
cat > "$T/bin/switch-real" <<EOF
#!/usr/bin/env bash
exec bash "$FIX/scripts/switch.sh" --no-wait --no-owui --force "\$@"
EOF
chmod +x "$T/bin/docker" "$T/bin/compose-capture" "$T/bin/switch-real"
export MOCK_LOG="$T/calls.log" MOCK_WORKDIR="$T/running" MOCK_ENV_OUT="$T/env.out" MOCK_RENDER="$T/render.json"
export PREFLIGHT_NO_FETCH=1 PREFLIGHT_NO_COMPOSE_DEPS=1 CLUB3090_SHM_CLEANUP=0 C3_LITELLM_FAKE_LIVE=

cfg() {  # <name> → a fresh settings dir; FILE=CONTENT pairs follow
  local d="$T/cfg-$1"; shift
  rm -rf "$d"; mkdir -p "$d"
  while [[ $# -gt 0 ]]; do printf '%s' "${1#*=}" > "$d/${1%%=*}"; shift; done
  printf '%s' "$d"
}
reset_mock() { : > "$MOCK_LOG"; rm -f "$MOCK_ENV_OUT" "$MOCK_RENDER"; }
# rendered <service-env-key> → its value in the rendered compose ("<absent>" when missing)
rendered() {
  python3 - "$MOCK_RENDER" "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
vals = {svc.get("environment", {}).get(sys.argv[2], "<absent>") for svc in d["services"].values()}
print(sorted(vals)[0] if len(vals) == 1 else "|".join(sorted(v or "" for v in vals)))
PY
}
envout() { command grep -E "^$1=" "$MOCK_ENV_OUT" 2>/dev/null | head -1 | cut -d= -f2-; }
has_env() { command grep -qE "^$1=" "$MOCK_ENV_OUT" 2>/dev/null; }

MEM64="$T/meminfo-64g"; printf 'MemTotal:       67108864 kB\nMemFree:        1 kB\n' > "$MEM64"
SGL=sgl/qwen38-27b-dual-fast      # reads all six knobs
VLL=vllm/qwen38-27b-dual-fast     # reads all but KV_OFFLOAD_DISK_GB
Q36=vllm/dual                     # qwen3.6: reads SPEC_N only

# ── 1 + 2. resolver: precedence and validation (the Python API, many layer combos) ──
echo "[1] precedence  ·  [2] validation (resolver API)"
python3 - "$ROOT" "$FIX" "$T" "$MEM64" <<'PY' || fail=1
import json, sys
from pathlib import Path
repo, fix, T, mem64 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(repo))
from scripts.lib import launch_settings as ls

bad = 0
def check(cond, msg, extra=""):
    global bad
    if cond:
        print(f"  ✓ {msg}")
    else:
        bad += 1
        print(f"  ✗ {msg}{(': ' + extra) if extra else ''}", file=sys.stderr)

SGL, VLL, Q36, GLM = ("sgl/qwen38-27b-dual-fast", "vllm/qwen38-27b-dual-fast", "vllm/dual",
                      "llamacpp-club3090/glm53-flash-dual-iq3xxs-moecache")
cfg = T / "api"; cfg.mkdir()
dotenv = fix / ".env"
def files(club="", secrets="", legacy="", slugs=None, raw_slugs=None):
    (cfg / "club3090.env").write_text(club)
    (cfg / "secrets.env").write_text(secrets)
    dotenv.write_text(legacy)
    p = cfg / "slugs.json"
    if raw_slugs is not None:
        p.write_text(raw_slugs)
    elif slugs is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(json.dumps({"version": 1, "slugs": slugs}))
def res(slug, shell=None, loaded=None, force=False, mem=mem64):
    env = {"CLUB3090_CONFIG_DIR": str(cfg), "CLUB3090_MEMINFO_FILE": mem, **(shell or {})}
    return ls.resolve(slug, root=fix, environ=env, loaded=loaded, force=force)
def eff(slug, knob, **kw):
    s = res(slug, **kw).settings[knob]
    return (s.source, s.value)

# Every layer set, each to a different value; peel them off from the top.
R = "REASONING_EFFORT"
layers = dict(shell={R: "max"}, slug={VLL: {R: "medium"}}, club=f"{R}=xhigh\n", secrets=f"{R}=high\n", legacy=f"{R}=minimal\n")
want = [("shell", "max"), ("this slug", "medium"), ("club3090.env", "xhigh"), ("secrets.env", "high"),
        ("repo .env", "minimal"), ("compose default", "low")]
got = []
for step in range(6):
    files(club=layers["club"], secrets=layers["secrets"], legacy=layers["legacy"], slugs=layers["slug"])
    got.append(eff(VLL, R, shell=layers["shell"]))
    for key, empty in (("shell", {}), ("slug", {}), ("club", ""), ("secrets", ""), ("legacy", "")):
        if layers[key] != empty:
            layers[key] = empty
            break
check(got == want, "REASONING_EFFORT: shell > this slug > club3090.env > secrets.env > repo .env > compose default",
      f"got {got}")

# The pin layer (ENABLE_THINKING only): between this slug and the global files.
E, PIN = "ENABLE_THINKING", "CLUB3090_THINKING_QWEN3_8_27B"
steps = [
    (dict(shell={E: "true"}, slugs={VLL: {E: "false"}}, club=f"{PIN}=on\n{E}=false\n"), ("shell", "true")),
    (dict(shell={}, slugs={VLL: {E: "false"}}, club=f"{PIN}=on\n{E}=false\n"), ("this slug", "false")),
    (dict(shell={}, slugs={}, club=f"{PIN}=on\n{E}=false\n"), ("model pin", "true")),
    (dict(shell={}, slugs={}, club=f"{PIN}=inherit\n{E}=false\n"), ("club3090.env", "false")),
    (dict(shell={}, slugs={}, club=""), ("compose default", "true")),
]
got = []
for kw, _ in steps:
    files(club=kw["club"], slugs=kw["slugs"])
    got.append(eff(VLL, E, shell=kw["shell"]))
check(got == [w for _, w in steps],
      "ENABLE_THINKING: shell > this slug > model pin (on→true) > club3090.env > compose default; inherit adds no layer",
      f"got {got}")
files(club=f"{PIN}=OFF\n{E}=true\n")
check(eff(VLL, E) == ("model pin", "false"), "a thinking pin beats a GLOBAL ENABLE_THINKING (pin OFF → false, case-insensitive)")

# A key the loader exported is the file's, not the shell's.
files(club=f"{R}=xhigh\n", slugs={VLL: {R: "medium"}})
check(eff(VLL, R, shell={R: "xhigh"}, loaded={R: "club3090.env"}) == ("this slug", "medium"),
      "a global value the loader exported does NOT count as shell: the slug's own value still wins")
check(eff(VLL, R, shell={R: "xhigh"}) == ("shell", "xhigh"),
      "… the same value NOT listed as loaded is the shell's, and wins")
check(eff(VLL, R, shell={R: "max"}, loaded={R: "club3090.env"}) == ("shell", "max"),
      "… and a loaded key changed after loading counts as the process's own (shell)")

# An empty shell value = unset for this launch: masks saved layers, is never validated.
files(slugs={SGL: {"KV_OFFLOAD_GB": "64"}})
r = res(SGL, shell={"KV_OFFLOAD_GB": ""})
s = r.settings["KV_OFFLOAD_GB"]
check(s.source == "shell" and s.is_unset and not r.errors,
      "KV_OFFLOAD_GB= in the shell turns a saved tier off for one launch, with no refusal", f"{s.source} {s.value!r} {r.errors}")

# ── validation ──
files()
r = res(SGL)
check(not r.errors and all(s.source == "compose default" for s in r.settings.values()),
      "nothing saved: every knob at its compose default, nothing refused", str(r.errors))
files(slugs={SGL: {R: "high"}})
e = " ".join(res(SGL).errors)
check("REASONING_EFFORT=high (from this slug)" in e and "low | medium | xhigh" in e,
      "an out-of-domain per-slug value is refused, naming the knob, its source and the allowed values", e)
files(club=f"{R}=extreme\n")
e = " ".join(res(VLL).errors)
check("(from club3090.env)" in e and "low | medium | xhigh | high | minimal | max" in e and "fail every request" in e,
      "an out-of-domain GLOBAL value is refused for a slug that reads it (vLLM: request-time failure named)", e)
files()
e = " ".join(res(VLL, shell={E: "on"}).errors)
check("ENABLE_THINKING=on (from shell)" in e and "true | false" in e, "an out-of-domain SHELL value is refused too", e)
files(club="KV_OFFLOAD_DISK=1\n")
e = " ".join(res(SGL).errors)
check("KV_OFFLOAD_DISK=1 needs KV_OFFLOAD_GB set" in e and "KV_OFFLOAD_DISK=1 from club3090.env" in e,
      "dependency: KV_OFFLOAD_DISK=1 without KV_OFFLOAD_GB is refused, with where each side came from", e)
files(slugs={SGL: {"KV_OFFLOAD_DISK_GB": "3"}})
e = " ".join(res(SGL).errors)
check("KV_OFFLOAD_DISK_GB=3 needs KV_OFFLOAD_DISK=1" in e and "KV_OFFLOAD_DISK=0 from compose default" in e,
      "dependency: a disk cap without the disk tier is refused", e)
files(slugs={SGL: {"KV_OFFLOAD_DISK_GB": "3", "KV_OFFLOAD_DISK": "1", "KV_OFFLOAD_GB": "16"}})
check(not res(SGL).errors, "dependency met (RAM tier + disk tier + cap): nothing refused", str(res(SGL).errors))

# Host RAM: tier × factor + 28 GiB must fit in MemTotal (64 GiB here). vLLM factor 1.0, SGLang 74/64.
for slug, gb, refused in ((VLL, "36", False), (VLL, "37", True), (SGL, "31", False), (SGL, "32", True)):
    files(slugs={slug: {"KV_OFFLOAD_GB": gb}})
    e = " ".join(res(slug).errors)
    check(("GiB in total" in e) == refused,
          f"RAM rule on a 64 GiB host: {slug.split('/')[0]} KV_OFFLOAD_GB={gb} {'refused' if refused else 'allowed'}", e)
files(slugs={VLL: {"KV_OFFLOAD_GB": "37"}})
check(any("at most 36 GiB" in x for x in res(VLL, force=True).errors), "the RAM rule applies even with --force, and says what fits")
r = res(VLL, mem=str(T / "no-such-meminfo"))
check(not r.errors and any("cannot read MemTotal" in w for w in r.warnings), "an unreadable meminfo: warned, not refused", str(r.errors))

# --force: only domains nothing enforces become warnings.
files(slugs={GLM: {R: "medium"}})
r1, r2 = res(GLM), res(GLM, force=True)
check(r1.errors and not r2.errors and any("Launching anyway (--force)" in w for w in r2.warnings),
      "GLM (enforced: none): out-of-domain effort refused; --force turns it into a warning", f"{r1.errors} / {r2.errors}")
files(slugs={SGL: {R: "high"}})
check(res(SGL, force=True).errors, "SGLang (enforced: boot): --force does not wave an out-of-domain value through")

# An unreadable store refuses (its values can't be known); nothing is guessed.
files(raw_slugs='{"version": 1, "slugs": {')
check(any("per-slug settings can't be read" in x for x in res(SGL).errors), "a corrupt slugs.json is refused")

# Saved values this slug doesn't read: warned, never refused.
files(club="KV_OFFLOAD_GB=32\nCLUB3090_THINKING_QWEN3_6_27B=on\n",
      slugs={Q36: {"KV_OFFLOAD_DISK": "1", "NOT_A_KNOB": "x"}})
r = res(Q36)
unread = {(k, src.split(" ")[0] if src.startswith("model pin") else src) for k, src, _ in r.unread}
check(not r.errors, "unread saved values never refuse a launch", str(r.errors))
check(("KV_OFFLOAD_GB", "club3090.env") in unread, "a GLOBAL value the slug doesn't read is flagged", str(r.unread))
check(("KV_OFFLOAD_DISK", "this slug") in unread, "a PER-SLUG value the slug doesn't read is flagged", str(r.unread))
check(("ENABLE_THINKING", "model") in unread, "a thinking pin the slug can't use is flagged", str(r.unread))
check(any("NOT_A_KNOB" in w and "not a catalogued launch setting" in w for w in r.warnings),
      "a per-slug key that isn't a catalogued knob is flagged", str(r.warnings))

dotenv.write_text("")
sys.exit(1 if bad else 0)
PY

# ── 3. delivery on the real switch.sh path ──────────────────────────────────────
echo "[3] delivery: switch.sh → docker compose"
sw() {  # <config dir> [VAR=val …] -- <switch.sh args…>  (docker shimmed; compose only renders)
  local c="$1"; shift
  local -a extra=()
  while [[ $# -gt 0 && "$1" != -- ]]; do extra+=("$1"); shift; done
  shift
  env PATH="$T/bin:$PATH" COMPOSE_BIN="$T/bin/compose-capture" CLUB3090_CONFIG_DIR="$c" "${extra[@]}" timeout 240 bash "$FIX/scripts/switch.sh" "$@" 2>&1
}
C="$(cfg deliver club3090.env=$'KV_OFFLOAD_GB=32\nREASONING_EFFORT=xhigh\nCLUB3090_THINKING_QWEN3_8_27B=off\n' \
               slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64"}}}')"
reset_mock
out="$(sw "$C" -- --no-wait --no-owui --force "$SGL")"; rc=$?
if (( rc != 0 )) || ! command grep -q '^UP' "$MOCK_LOG"; then
  bad "the launch did not reach compose up (rc=$rc): $(tail -3 <<<"$out")"
else
  got="$(envout KV_OFFLOAD_GB)|$(envout REASONING_EFFORT)|$(envout ENABLE_THINKING)"
  [[ "$got" == "64|xhigh|false" ]] && ok "compose up gets this slug's KV_OFFLOAD_GB=64 (not the global 32), the global effort, the pin's thinking=false" \
    || bad "environment at compose up: '$got' (want 64|xhigh|false)"
  if [[ "$HAVE_COMPOSE" == 1 ]]; then
    got="$(rendered KV_OFFLOAD_GB)|$(rendered REASONING_EFFORT)|$(rendered ENABLE_THINKING)"
    [[ "$got" == "64|xhigh|false" ]] && ok "… and docker compose config puts exactly those into the container's environment" \
      || bad "rendered container environment: '$got' (want 64|xhigh|false)"
  else
    echo "  - SKIP render leg: no docker compose here"
  fi
  if [[ "$HAVE_COMPOSE" == 1 ]]; then
    got="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("|".join(sorted({(s.get("labels") or {}).get("club3090.slug","<absent>") for s in d["services"].values()})))' "$MOCK_RENDER" 2>/dev/null)"
    [[ "$got" == "$SGL" ]] && ok "… and the container is labelled club3090.slug=$SGL (the slug launched, not just its compose)" \
      || bad "club3090.slug label at compose up: '$got' (want $SGL)"
  fi
  command grep -qF "launch setting KV_OFFLOAD_GB=64  (this slug; overrides club3090.env=32)" <<<"$out" \
    && ok "the launch log names each value's source and what it overrode" || bad "no source line in the launch log: $(command grep -m2 'launch setting' <<<"$out")"
fi
reset_mock
sw "$C" KV_OFFLOAD_GB=16 -- --no-wait --no-owui --force "$SGL" >/dev/null
[[ "$(envout KV_OFFLOAD_GB)" == 16 ]] && ok "a shell export beats the slug's saved value on the real path" || bad "shell vs this slug: '$(envout KV_OFFLOAD_GB)'"
[[ "$HAVE_COMPOSE" != 1 || "$(rendered KV_OFFLOAD_GB)" == 16 ]] || bad "rendered with a shell export: '$(rendered KV_OFFLOAD_GB)'"
reset_mock
out="$(sw "$C" KV_OFFLOAD_GB= -- --no-wait --no-owui --force "$SGL")"
if command grep -q '^UP' "$MOCK_LOG" && has_env KV_OFFLOAD_GB && [[ -z "$(envout KV_OFFLOAD_GB)" ]]; then
  ok "KV_OFFLOAD_GB= in the shell launches with the tier off, over the saved 64"
else
  bad "empty shell value: calls=$(tr '\n' ' ' < "$MOCK_LOG") env='$(envout KV_OFFLOAD_GB)'"
fi
C2="$(cfg unread club3090.env=$'KV_OFFLOAD_DISK_GB=5\n')"
reset_mock
out="$(sw "$C2" -- --no-wait --no-owui --force "$VLL")"
if command grep -q '^UP' "$MOCK_LOG" && command grep -qF "saved KV_OFFLOAD_DISK_GB=5 (club3090.env) is not read by $VLL" <<<"$out"; then
  ok "a saved value the slug doesn't read: warned, and the launch goes ahead"
else
  bad "unread-knob warning: calls=$(tr '\n' ' ' < "$MOCK_LOG") $(command grep -m1 -i 'not read' <<<"$out")"
fi

# ── 4. refusals happen BEFORE the running slug is taken down ────────────────────
echo "[4] refused before teardown"
refused() {  # <label> <config dir> <want…> -- [VAR=val …]
  local label="$1" c="$2"; shift 2
  local -a want=() extra=()
  while [[ $# -gt 0 && "$1" != -- ]]; do want+=("$1"); shift; done
  shift
  extra=("$@")
  reset_mock
  local out rc w
  out="$(env MOCK_RUNNING=vllm-qwen38-27b-dual-fast PATH="$T/bin:$PATH" COMPOSE_BIN="$T/bin/compose-capture" CLUB3090_CONFIG_DIR="$c" \
         "${extra[@]}" timeout 240 bash "$FIX/scripts/switch.sh" --no-wait --no-owui --force "$SGL" 2>&1)"; rc=$?
  if (( rc == 0 )); then bad "$label: NOT refused"; return; fi
  if command grep -qE '^(TEARDOWN|UP)' "$MOCK_LOG"; then bad "$label: refused, but only after: $(tr '\n' ' ' < "$MOCK_LOG")"; return; fi
  for w in "${want[@]}"; do
    [[ "$out" == *"$w"* ]] || { bad "$label: refused, but the message lacks '$w': $(command grep -m3 -E 'ERROR|  - ' <<<"$out")"; return; }
  done
  ok "$label: refused, running slug left up"
}
refused "out-of-domain per-slug value" \
  "$(cfg r1 slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"REASONING_EFFORT": "high"}}}')" \
  "REASONING_EFFORT=high (from this slug)" "low | medium | xhigh" "nothing was torn down" --
refused "out-of-domain shell value" "$(cfg r2)" "ENABLE_THINKING=on (from shell)" "true | false" -- ENABLE_THINKING=on
refused "unmet dependency (global disk tier, no RAM tier)" "$(cfg r3 club3090.env=$'KV_OFFLOAD_DISK=1\n')" \
  "KV_OFFLOAD_DISK=1 needs KV_OFFLOAD_GB set" --
refused "RAM tier bigger than the host (64 GiB host, --force)" \
  "$(cfg r4 slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "40"}}}')" \
  "KV_OFFLOAD_GB=40 (from this slug)" "at most 31 GiB fits" -- CLUB3090_MEMINFO_FILE="$MEM64"
refused "unreadable slugs.json" "$(cfg r5 slugs.json='{"version": 1,')" "per-slug settings can't be read" --
# Positive control: the same shim, a valid saved value → the old slug goes down, then the new one comes up.
reset_mock
out="$(env MOCK_RUNNING=vllm-qwen38-27b-dual-fast PATH="$T/bin:$PATH" COMPOSE_BIN="$T/bin/compose-capture" \
       CLUB3090_CONFIG_DIR="$(cfg ok slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"REASONING_EFFORT": "medium"}}}')" \
       timeout 240 bash "$FIX/scripts/switch.sh" --no-wait --no-owui --force "$SGL" 2>&1)"; rc=$?
down="$(command grep -n '^TEARDOWN' "$MOCK_LOG" | head -1 | cut -d: -f1)"; up="$(command grep -n '^UP' "$MOCK_LOG" | head -1 | cut -d: -f1)"
if (( rc == 0 )) && [[ -n "$down" && -n "$up" ]] && (( down < up )) && [[ "$(envout REASONING_EFFORT)" == medium ]]; then
  ok "positive control: valid settings tear the running slug down, then bring the new one up with them"
else
  bad "positive control: rc=$rc calls=$(tr '\n' ' ' < "$MOCK_LOG") effort='$(envout REASONING_EFFORT)' $(tail -2 <<<"$out")"
fi

# ── 5. launch.sh → switch.sh ────────────────────────────────────────────────────
echo "[5] delivery through launch.sh"
lsh() {  # <config dir> [VAR=val …]
  local c="$1"; shift
  reset_mock
  env PATH="$T/bin:$PATH" SWITCH="$T/bin/switch-real" COMPOSE_BIN="$T/bin/compose-capture" CLUB3090_CONFIG_DIR="$c" "$@" \
    timeout 300 bash "$FIX/scripts/launch.sh" --no-preflight --no-verify --no-projection --variant "$SGL" </dev/null >"$T/launch.out" 2>&1
}
C="$(cfg viaLaunch club3090.env=$'KV_OFFLOAD_GB=32\n' \
               slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64"}}}')"
lsh "$C"
if command grep -q '^UP' "$MOCK_LOG" && [[ "$(envout KV_OFFLOAD_GB)" == 64 ]]; then
  ok "launch.sh: the slug's saved KV_OFFLOAD_GB=64 reaches compose, although launch.sh itself loaded the global 32"
else
  bad "launch.sh: env='$(envout KV_OFFLOAD_GB)' calls=$(tr '\n' ' ' < "$MOCK_LOG") $(command grep -m2 -E 'ERROR|launch setting' "$T/launch.out")"
fi
[[ "$HAVE_COMPOSE" != 1 || "$(rendered KV_OFFLOAD_GB)" == 64 ]] || bad "launch.sh: rendered '$(rendered KV_OFFLOAD_GB)'"
lsh "$C" KV_OFFLOAD_GB=16
[[ "$(envout KV_OFFLOAD_GB)" == 16 ]] && ok "launch.sh: a shell export still wins" || bad "launch.sh shell: '$(envout KV_OFFLOAD_GB)'"
lsh "$(cfg viaLaunch2 club3090.env=$'KV_OFFLOAD_GB=32\n')"
[[ "$(envout KV_OFFLOAD_GB)" == 32 ]] && ok "launch.sh: with no per-slug value the global one is delivered" || bad "launch.sh global: '$(envout KV_OFFLOAD_GB)'"

# ── 6. the CLI: --set / --unset / --explain / --help ────────────────────────────
echo "[6] switch.sh --set / --unset / --explain"
cli() { local c="$1"; shift; env PATH="$T/bin:$PATH" COMPOSE_BIN=: CLUB3090_CONFIG_DIR="$c" timeout 120 bash "$FIX/scripts/switch.sh" "$@" 2>&1; }
stored() { python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["slugs"], sort_keys=True))' "$1/slugs.json" 2>/dev/null || echo "<none>"; }
C="$(cfg cli)"
out="$(cli "$C" --set "$SGL" KV_OFFLOAD_GB=64 REASONING_EFFORT=medium)"; rc=$?
if (( rc == 0 )) && [[ "$(stored "$C")" == '{"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64", "REASONING_EFFORT": "medium"}}' ]] \
   && [[ "$out" == *"applies from the next launch"* && "$out" == *"KV_OFFLOAD_GB on the next launch of $SGL: 64 (this slug)"* ]]; then
  ok "--set saves the values and says they apply at the next launch"
else
  bad "--set: rc=$rc stored=$(stored "$C") $(tail -2 <<<"$out")"
fi
set_refused() {  # <label> <want> <args…>
  local label="$1" want="$2"; shift 2
  local before out rc
  before="$(stored "$C")"
  out="$(cli "$C" --set "$@")"; rc=$?
  if (( rc != 0 )) && [[ "$out" == *"$want"* && "$(stored "$C")" == "$before" ]]; then ok "--set refuses $label"
  else bad "--set $label: rc=$rc stored=$(stored "$C") $(tail -2 <<<"$out")"; fi
}
set_refused "an out-of-domain value (allowed values named)" "'high' is not one of: low | medium | xhigh" "$SGL" REASONING_EFFORT=high
set_refused "a knob the slug doesn't read" "$Q36 doesn't read KV_OFFLOAD_GB" "$Q36" KV_OFFLOAD_GB=64
set_refused "a name that isn't a launch knob" "not a launch setting club-3090 knows" "$SGL" KV_OFFLOAD_G=64
set_refused "a credential (secrets stay global)" "not a launch setting" "$SGL" HF_TOKEN=hf_x
set_refused "an unknown slug" "unknown slug" vllm/no-such-slug KV_OFFLOAD_GB=64
set_refused "the whole call when one pair is bad" "is not one of" "$SGL" SPEC_N=2 REASONING_EFFORT=high
set_refused "a flag after the slug" "takes only KEY=VALUE" "$SGL" SPEC_N=2 --force
out="$(env CLUB3090_MEMINFO_FILE="$MEM64" PATH="$T/bin:$PATH" COMPOSE_BIN=: CLUB3090_CONFIG_DIR="$C" timeout 120 bash "$FIX/scripts/switch.sh" --set "$SGL" KV_OFFLOAD_GB=48 2>&1)"; rc=$?
(( rc != 0 )) && [[ "$out" == *"at most 31 GiB fits"* ]] && ok "--set refuses a RAM tier this host can't hold" || bad "--set RAM: rc=$rc $(tail -1 <<<"$out")"

python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["slugs"]["vllm/dual"]={"KV_OFFLOAD_GB": "8"}; json.dump(d, open(p, "w"))' "$C/slugs.json"
out="$(cli "$C" --unset "$Q36" KV_OFFLOAD_GB)"; rc=$?
(( rc == 0 )) && [[ "$(stored "$C")" != *vllm/dual* ]] && ok "--unset removes a stale key the slug doesn't read (the cleanup path for the warning)" \
  || bad "--unset stale: rc=$rc stored=$(stored "$C") $(tail -1 <<<"$out")"
out="$(cli "$C" --unset "$SGL" KV_OFFLOAD_G)"; rc=$?
(( rc != 0 )) && [[ "$out" == *"not a launch setting"* ]] && ok "--unset refuses a typo" || bad "--unset typo: rc=$rc $(tail -1 <<<"$out")"
out="$(cli "$C" --unset "$SGL" REASONING_EFFORT)"; rc=$?
(( rc == 0 )) && [[ "$(stored "$C")" == '{"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64"}}' && "$out" == *"REASONING_EFFORT on the next launch of $SGL: low (compose default)"* ]] \
  && ok "--unset removes the key and shows what the knob falls back to" || bad "--unset: rc=$rc stored=$(stored "$C") $(tail -1 <<<"$out")"

printf 'REASONING_EFFORT=xhigh\nCLUB3090_THINKING_QWEN3_8_27B=off\nKV_OFFLOAD_DISK_GB=5\n' > "$C/club3090.env"
printf 'SPEC_N=909091\n' > "$C/secrets.env"
out="$(env KV_OFFLOAD_DISK=0 PATH="$T/bin:$PATH" COMPOSE_BIN=: CLUB3090_CONFIG_DIR="$C" timeout 120 bash "$FIX/scripts/switch.sh" --explain "$VLL" 2>&1)"
block="$(sed -n '/Launch settings/,$p' <<<"$out")"
miss=""
for want in "KV_OFFLOAD_GB        (unset)        compose default" \
            "ENABLE_THINKING      false          model pin — CLUB3090_THINKING_QWEN3_8_27B=off from club3090.env" \
            "REASONING_EFFORT     xhigh          club3090.env" \
            "KV_OFFLOAD_DISK      0              shell" \
            "SPEC_N               <set, hidden>  secrets.env" \
            "KV_OFFLOAD_DISK_GB=5   (club3090.env)"; do
  [[ "$block" == *"$want"* ]] || miss+=" [$want]"
done
[[ -z "$miss" ]] && ok "--explain shows each knob's value and source (shell · model pin · club3090.env · secrets.env · compose default) and what it doesn't read" \
  || bad "--explain block lacks:$miss"$'\n'"$block"
command grep -qF 909091 <<<"$out" && bad "--explain printed a secrets.env value" || ok "--explain hides a secrets.env value"
C3="$(cfg explain2 slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_GB": "64"}}}' club3090.env=$'KV_OFFLOAD_GB=32\n')"
got="$(env PATH="$T/bin:$PATH" COMPOSE_BIN=: CLUB3090_CONFIG_DIR="$C3" timeout 120 bash "$FIX/scripts/switch.sh" --explain "$SGL" --json 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)["launch_settings"]; k={x["knob"]: x for x in d["knobs"]}["KV_OFFLOAD_GB"]; print(k["value"], k["source"], k["overrides"][0]["source"])')"
[[ "$got" == "64 this slug club3090.env" ]] && ok "--explain --json carries launch_settings (value, source, what it overrides)" || bad "--explain --json: '$got'"
# c3's Launch settings form (3d) shows these two per-knob fields next to the value.
C4="$(cfg explain3 slugs.json='{"version": 1, "slugs": {"sgl/qwen38-27b-dual-fast": {"KV_OFFLOAD_DISK": "1"}}}')"
got="$(env PATH="$T/bin:$PATH" COMPOSE_BIN=: CLUB3090_CONFIG_DIR="$C4" timeout 120 bash "$FIX/scripts/switch.sh" --explain "$SGL" --json 2>/dev/null \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)["launch_settings"]
k = {x["knob"]: x for x in d["knobs"]}
dep = [e for e in d["errors"] if "needs KV_OFFLOAD_GB" in e]
print(k["REASONING_EFFORT"]["allowed"], "/", k["KV_OFFLOAD_GB"]["allowed"], "/",
      bool(dep) and k["KV_OFFLOAD_DISK"]["errors"] == dep == k["KV_OFFLOAD_GB"]["errors"], "/", k["SPEC_N"]["errors"])')"
[[ "$got" == "low | medium | xhigh / a number of GiB, e.g. 64 or 0.5; at least 1.86265 / True / []" ]] \
  && ok "--explain --json: each knob carries its allowed values and the refusals about it (c3's form shows both)" \
  || bad "--explain --json allowed/errors: '$got'"
out="$(cli "$C" --help)"
[[ "$out" == *"--set <slug> KEY=VALUE"* && "$out" == *"--unset <slug> KEY"* ]] && ok "--help lists --set and --unset" || bad "--help lacks --set/--unset"

[[ $fail -eq 0 ]] && echo "test-launch-settings: ok" || echo "test-launch-settings: FAIL"
exit $fail
