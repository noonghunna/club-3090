#!/usr/bin/env bash
# test-settings-cli — `bash scripts/settings.sh`, the one user-facing way to see and
# change club-3090 settings, and report.sh's "Settings" section (club-3090#1466).
#
# Every command is driven through the REAL entry point: scripts/settings.sh is
# copied into a fixture checkout whose scripts/lib links to this one, so it runs
# exactly as shipped while its repo root — whose legacy .env it reads, and which
# `unset` edits — is a temporary directory, never a real checkout. Every call runs
# under `env -i`, so nothing from the caller's shell (a real HF_TOKEN, MODEL_DIR…)
# reaches the assertions or the output.
#
# What must hold, and why:
#   * secret values never print without --show-secrets — not from show, show
#     --json, set, migrate or report.sh (even with --no-redact). Every secret in
#     the fixtures is a unique zz-… string, and --show-secrets proving they DO
#     resolve is the positive control: an absent secret proves nothing otherwise;
#   * migrate never modifies the repo .env (byte-identical, same mtime and mode),
#     changes no effective value, and is a no-op the second time;
#   * sources are right: shell > club3090.env > secrets.env > repo .env. report.sh
#     must take its snapshot BEFORE club_config_load, or every key reads "shell".
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/scripts/lib/club_config.py"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"
cleanup() { [[ "$(type -t report_env_cleanup)" == function ]] && report_env_cleanup; rm -rf "$T"; }
trap cleanup EXIT

REPO="$T/repo"; mkdir -p "$REPO/scripts"
if [[ -f "$ROOT/scripts/settings.sh" ]]; then cp "$ROOT/scripts/settings.sh" "$REPO/scripts/settings.sh"
else bad "scripts/settings.sh does not exist"; fi
ln -s "$ROOT/scripts/lib" "$REPO/scripts/lib"

# st [VAR=value ...] -- <settings.sh args>: one call in a clean environment. Sets
# OUT (stdout), ERR (stderr) and RC. CFG is the config dir of the current arm.
st() {
  local -a extra=()
  while [[ $# -gt 0 && "$1" != -- ]]; do extra+=("$1"); shift; done
  shift
  env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$CFG" "${extra[@]}" \
    bash "$REPO/scripts/settings.sh" "$@" > "$T/out" 2> "$T/err"
  RC=$?; OUT="$(cat "$T/out")"; ERR="$(cat "$T/err")"
}
# resolved values only (KEY<TAB>VALUE), ignoring sources: what a launch would see.
values() { env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$CFG" python3 "$PY" resolve --show-secrets --root "$REPO" | cut -f1,3; }
has()   { [[ "$2" == *"$3"* ]] && ok "$1" || bad "$1 — missing [$3] in: ${2:0:600}"; }
hasnt() { [[ "$2" != *"$3"* ]] && ok "$1" || bad "$1 — must not contain [$3]"; }
nosecret() {   # <label> <text>: no fixture secret (zz-…) anywhere in it
  if command grep -qoE 'zz-[a-z0-9-]+' <<<"$2"; then bad "$1 — printed a secret: $(command grep -oE 'zz-[a-z0-9-]+' <<<"$2" | sort -u | tr '\n' ' ')"
  else ok "$1"; fi
}
row() { command grep -E "^$1 " <<<"$2" | tr -s ' '; }   # a `show` table row, spaces squeezed

# ── help and usage ──────────────────────────────────────────────────────────
CFG="$T/help"
st -- --help
help_ok=1
for w in show get set unset migrate path compose-env-file --json --show-secrets --dry-run --out club3090.env secrets.env "repo .env"; do
  [[ "$OUT" == *"$w"* ]] || { bad "--help does not mention '$w'"; help_ok=0; }
done
[[ $RC -eq 0 && $help_ok -eq 1 ]] && ok "--help exits 0 and covers every command, flag and settings file"
st -- ; [[ $RC -eq 2 && "$ERR" == *usage* ]] && ok "no command: usage on stderr, exit 2" || bad "no command: rc=$RC"
st -- frobnicate; [[ $RC -eq 2 ]] && ok "an unknown command exits 2" || bad "unknown command: rc=$RC"

# ── path ────────────────────────────────────────────────────────────────────
CFG="$T/p/club-3090"
st -- path
# (the cache and data dir rows, #1466 phase 4, are counted apart: they depend on XDG/HOME)
[[ $RC -eq 0 && "$OUT" == *"$CFG"* && "$(command grep -vE '^(cache|data) dir:' <<<"$OUT" | command grep -c 'not created yet')" == 3 ]] \
  && ok "path: the config dir and both files, none created yet" || bad "path (empty): rc=$RC out: $OUT"
[[ "$(command grep -cE '^(cache|data) dir: ' <<<"$OUT")" == 2 ]] && ok "path: also lists the cache and data dirs" \
  || bad "path: no cache/data dir rows: $OUT"
[[ "$OUT" == *"$REPO/.env"*"none"* ]] && ok "path: names the repo .env, absent here" || bad "path: repo .env line: $OUT"
[[ ! -e "$CFG" ]] && ok "path creates nothing" || bad "path created $CFG"

# ── set ─────────────────────────────────────────────────────────────────────
CFG="$T/s"
st -- set MODEL_DIR=/fixture/models HF_TOKEN=zz-set-hf THREADS=24
[[ $RC -eq 0 ]] || bad "set: rc=$RC err: $ERR"
[[ "$(cat "$CFG/club3090.env" 2>/dev/null)" == $'MODEL_DIR=/fixture/models\nTHREADS=24' ]] \
  && ok "set: ordinary keys go to club3090.env" || bad "set: club3090.env is: $(cat "$CFG/club3090.env" 2>/dev/null)"
[[ "$(cat "$CFG/secrets.env" 2>/dev/null)" == "HF_TOKEN=zz-set-hf" && "$(stat -c %a "$CFG/secrets.env" 2>/dev/null)" == 600 ]] \
  && ok "set: a credential-looking key goes to secrets.env, created 0600" || bad "set: secrets.env mode $(stat -c %a "$CFG/secrets.env" 2>/dev/null)"
has "set: reports both files it wrote" "$OUT" "$CFG/secrets.env"
nosecret "set: the saved token is not echoed" "$OUT$ERR"
st -- path
[[ "$OUT" == *"club3090.env:"*"(exists)"* && "$OUT" == *"secrets.env:"*"exists, mode 0600"* ]] \
  && ok "path: both files now exist, secrets.env 0600" || bad "path after set: $OUT"

# A key already in secrets.env stays there whatever its name; a credential sitting
# in club3090.env (which is read first) moves to secrets.env, or the old value wins.
printf 'CLOUD_ROUTE=zz-old-route\n' >> "$CFG/secrets.env"
printf 'OPENAI_API_KEY=zz-old-openai\n' >> "$CFG/club3090.env"
st -- set CLOUD_ROUTE=zz-new-route OPENAI_API_KEY=zz-new-openai
command grep -qx 'CLOUD_ROUTE=zz-new-route' "$CFG/secrets.env" && ! command grep -q CLOUD_ROUTE "$CFG/club3090.env" \
  && ok "set: a key already in secrets.env stays in secrets.env" || bad "set: CLOUD_ROUTE left secrets.env"
command grep -qx 'OPENAI_API_KEY=zz-new-openai' "$CFG/secrets.env" && ! command grep -q OPENAI_API_KEY "$CFG/club3090.env" \
  && ok "set: a credential found in club3090.env is moved out of it (it would shadow secrets.env)" || bad "set: OPENAI_API_KEY placement wrong"
st -- get OPENAI_API_KEY; [[ "$OUT" == zz-new-openai ]] && ok "set: the new value is the effective one" || bad "get after set: [$OUT]"
st -- set CLOUD_ROUTE=x ; nosecret "set: no old or new secret in the messages" "$OUT$ERR"

# Refusals: the writer's reason, never the value; all or nothing; exit 2.
cp "$CFG/club3090.env" "$T/g.before"; cp "$CFG/secrets.env" "$T/s.before"
st -- set GOOD=1 'BAD_ONE=zz$refused'
[[ $RC -eq 2 ]] && ok "set: a refused value exits 2" || bad "set refused: rc=$RC"
has "set: the refusal says why" "$ERR" "'\$'"
nosecret "set: the refused value is not printed" "$OUT$ERR"
cmp -s "$CFG/club3090.env" "$T/g.before" && cmp -s "$CFG/secrets.env" "$T/s.before" \
  && ok "set: one refused value saves nothing (GOOD=1 was not written either)" || bad "set: files changed after a refusal"
st -- set zz-no-equals-token; [[ $RC -eq 2 ]] && nosecret "set: an argument without '=' exits 2 and is not echoed" "$OUT$ERR" || bad "set without '=': rc=$RC"
st -- set 1BAD=x; [[ $RC -eq 2 ]] && ok "set: an invalid name exits 2" || bad "set 1BAD: rc=$RC"
st MODEL_DIR=/from/shell -- set MODEL_DIR=/fixture/models2
has "set: warns when the shell still overrides the saved value" "$ERR" "MODEL_DIR is also set in your shell"

# ── get ─────────────────────────────────────────────────────────────────────
printf 'ONLY_REPO=legacy\n' > "$REPO/.env"
st -- get MODEL_DIR;  [[ $RC -eq 0 && "$OUT" == /fixture/models2 ]] && ok "get: a saved value" || bad "get MODEL_DIR: rc=$RC [$OUT]"
st -- get ONLY_REPO;  [[ $RC -eq 0 && "$OUT" == legacy ]] && ok "get: a value only in the repo .env" || bad "get ONLY_REPO: rc=$RC [$OUT]"
st MODEL_DIR=/from/shell -- get MODEL_DIR; [[ "$OUT" == /from/shell ]] && ok "get: the shell wins" || bad "get shell: [$OUT]"
st -- get NOT_SET_ANYWHERE; [[ $RC -eq 1 && -z "$OUT" ]] && ok "get: exit 1 and no output when unset" || bad "get unset: rc=$RC [$OUT]"
rm -f "$REPO/.env"

# ── show ────────────────────────────────────────────────────────────────────
CFG="$T/show"; mkdir -p "$CFG"
printf 'HF_TOKEN=zz-show-hf\nCLOUD_ROUTE=zz-show-route\n' > "$CFG/secrets.env"
printf 'MODEL_DIR=/fixture/models\nOVERRIDE=from-global\nSHELLWINS=from-file\nEMPTYV=\n' > "$CFG/club3090.env"
printf '# legacy\nLEGACY_ONLY=legacy\nMY_API_KEY=zz-show-apikey\nOVERRIDE=from-repo\nNOT_STORABLE=a$b\n' > "$REPO/.env"
st SHELLWINS=from-shell -- show
[[ $RC -eq 0 ]] || bad "show: rc=$RC err: $ERR"
show_ok=1
for want in "CLOUD_ROUTE <set, hidden> secrets.env" "HF_TOKEN <set, hidden> secrets.env" \
            "LEGACY_ONLY legacy repo .env" "MODEL_DIR /fixture/models club3090.env" \
            "MY_API_KEY <set, hidden> repo .env" "OVERRIDE from-global club3090.env" \
            "SHELLWINS from-shell shell" "EMPTYV <empty> club3090.env" "NOT_STORABLE a\$b repo .env"; do
  k="${want%% *}"
  [[ "$(row "$k" "$OUT")" == "$want" ]] || { bad "show row $k: got [$(row "$k" "$OUT")], want [$want]"; show_ok=0; }
done
[[ $show_ok -eq 1 ]] && ok "show: every key with its effective value and source (shell, club3090.env, secrets.env, repo .env)"
nosecret "show: no secret value without --show-secrets (by secrets.env and by name)" "$OUT$ERR"
want_hint="2 setting(s) still live in $REPO/.env — bash scripts/settings.sh migrate moves them to $CFG/."
[[ "$(tail -n 1 <<<"$OUT")" == "$want_hint" ]] && ok "show: ends with the migrate line, counting only what migrate can move (LEGACY_ONLY, MY_API_KEY)" \
  || bad "show: last line [$(tail -n 1 <<<"$OUT")], want [$want_hint]"
st CLOUD_ROUTE=zz-show-shell -- show
[[ "$(row CLOUD_ROUTE "$OUT")" == "CLOUD_ROUTE <set, hidden> shell" ]] && nosecret "show: a secrets.env key stays hidden when the shell overrides it" "$OUT" \
  || bad "show: shell-overridden secrets.env key: [$(row CLOUD_ROUTE "$OUT")]"
st -- show --json
if python3 - "$T/out" "$CFG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = d["settings"]
assert d["config_dir"] == sys.argv[2], d["config_dir"]
assert s["MODEL_DIR"] == {"value": "/fixture/models", "source": "club3090.env", "secret": False}, s["MODEL_DIR"]
assert s["HF_TOKEN"] == {"value": "<set, hidden>", "source": "secrets.env", "secret": True}, s["HF_TOKEN"]
assert s["MY_API_KEY"]["source"] == "repo .env" and s["MY_API_KEY"]["secret"], s["MY_API_KEY"]
assert d["still_in_repo_env"] == ["LEGACY_ONLY", "MY_API_KEY"], d["still_in_repo_env"]
PY
then ok "show --json: valid, with sources, secret flags and the keys still to migrate"; else bad "show --json: $(head -c 400 "$T/out")"; fi
nosecret "show --json: no secret value" "$OUT"
# Positive control: the secrets DO resolve, so their absence above means hidden.
st -- show --show-secrets
[[ "$OUT" == *zz-show-hf* && "$OUT" == *zz-show-route* && "$OUT" == *zz-show-apikey* ]] \
  && ok "show --show-secrets prints them (positive control for every 'no secret' check)" || bad "show --show-secrets: $OUT"
st -- show --json --show-secrets; [[ "$OUT" == *zz-show-hf* ]] && ok "show --json --show-secrets prints them too" || bad "show --json --show-secrets hid them"
printf 'OVERRIDE=from-repo\n' > "$REPO/.env"
st -- show; [[ "$OUT" != *"still live in"* ]] && ok "show: no migrate line once the store has every key" || bad "show: stray migrate line: $(tail -n 1 <<<"$OUT")"
CFG="$T/empty"; rm -f "$REPO/.env"
st -- show; [[ $RC -eq 0 && "$OUT" == *"No settings saved yet"* ]] && ok "show: an empty store says so" || bad "show empty: rc=$RC $OUT"

# ── compose-env-file ────────────────────────────────────────────────────────
# For running `docker compose` by hand. The file holds secrets, so its contents are
# compared, never printed; TMPDIR keeps it inside this test's directory.
CFG="$T/ce"; mkdir -p "$CFG" "$T/tmpdir"
printf 'CE_TOKEN=zz-ce-token\n' > "$CFG/secrets.env"
printf 'CE_DIR=/fixture/models\nCE_SHELL=from-file\n' > "$CFG/club3090.env"
printf 'CE_LEGACY=legacy value\n' > "$REPO/.env"
st TMPDIR="$T/tmpdir" CE_SHELL=from-shell -- compose-env-file
f="$OUT"
if [[ $RC -eq 0 && -f "$f" && "$f" == "$T/tmpdir/"* ]]; then
  [[ "$(stat -c %a "$f")" == 600 ]] && ok "compose-env-file: prints the path of a new 0600 file" || bad "compose-env-file: mode $(stat -c %a "$f")"
  printf "%s\n" "CE_DIR='/fixture/models'" "CE_LEGACY='legacy value'" "CE_SHELL='from-shell'" "CE_TOKEN='zz-ce-token'" > "$T/ce.want"
  cmp -s "$f" "$T/ce.want" && ok "compose-env-file: every resolved setting (store, repo .env, shell winning), compose-quoted" \
    || bad "compose-env-file: contents differ from the expected four lines (not printed: they hold a secret)"
else bad "compose-env-file: rc=$RC out=[$OUT] err=[$ERR]"; fi
nosecret "compose-env-file: prints only the path, never a value" "$OUT$ERR"
st -- compose-env-file --out "$T/ce.env"
[[ $RC -eq 0 && "$OUT" == "$T/ce.env" && "$(stat -c %a "$T/ce.env" 2>/dev/null)" == 600 ]] \
  && ok "compose-env-file --out PATH: writes that file, 0600" || bad "compose-env-file --out: rc=$RC [$OUT]"
rm -f "$REPO/.env"

# ── unset ───────────────────────────────────────────────────────────────────
CFG="$T/u"; mkdir -p "$CFG"
printf 'PIN=store\nKEEP=1\n' > "$CFG/club3090.env"
printf 'PIN_TOKEN=zz-unset-token\n' > "$CFG/secrets.env"
printf '# my legacy settings\nexport PIN=legacy\nPIN_TOKEN=zz-unset-legacy\n\n# tail\nOTHER=x\n' > "$REPO/.env"
st -- unset PIN PIN_TOKEN NEVER_SET
[[ $RC -eq 0 ]] || bad "unset: rc=$RC err: $ERR"
[[ "$(cat "$CFG/club3090.env")" == "KEEP=1" && ! -s "$CFG/secrets.env" && "$(cat "$REPO/.env")" == $'# my legacy settings\n\n# tail\nOTHER=x' ]] \
  && ok "unset: removed from club3090.env, secrets.env and the repo .env (its comments kept)" \
  || bad "unset: files now: [$(cat "$CFG/club3090.env")] [$(cat "$CFG/secrets.env")] [$(cat "$REPO/.env")]"
[[ "$OUT" == *"$CFG/club3090.env"* && "$OUT" == *"$CFG/secrets.env"* && "$OUT" == *"$REPO/.env"* ]] \
  && ok "unset: reports every file it changed" || bad "unset output: $OUT"
has "unset: says which keys were saved nowhere" "$OUT" "NEVER_SET: not saved in any settings file"
nosecret "unset: no secret value printed" "$OUT$ERR"
st -- get PIN; [[ $RC -eq 1 ]] && ok "unset: the key resolves nowhere afterwards" || bad "PIN still resolves: $OUT"
st PIN=shell -- unset PIN; has "unset: warns that a shell value stays in effect" "$ERR" "PIN is also set in your shell"
rm -f "$REPO/.env"

# ── migrate ─────────────────────────────────────────────────────────────────
CFG="$T/m"; mkdir -p "$CFG"
cat > "$REPO/.env" <<'EOF'
# my old settings — comments, export, quotes, duplicates
export MODEL_DIR="/data/models"
THREADS=24
THREADS=28
QUOTED='single quoted'
EMPTY=
HF_TOKEN=zz-mig-hf
LITELLM_MASTER_KEY=zz-mig-master
BAD=has$dollar
SAME=same-value
DIFF=repo-value
DIFF_API_KEY=zz-mig-diff-repo
EOF
chmod 640 "$REPO/.env"; touch -d '2020-01-02 03:04:05' "$REPO/.env"
printf '# mine\nSAME=same-value\nDIFF=store-value\n' > "$CFG/club3090.env"
printf 'DIFF_API_KEY=zz-mig-diff-store\n' > "$CFG/secrets.env"; chmod 600 "$CFG/secrets.env"
cp -p "$REPO/.env" "$T/repo.env.orig"; stamp() { stat -c '%s %a %Y' "$1"; }; repo_stamp="$(stamp "$REPO/.env")"
cp "$CFG/club3090.env" "$T/m.g0"; cp "$CFG/secrets.env" "$T/m.s0"
values > "$T/values.before"

st -- migrate --dry-run
[[ $RC -eq 0 ]] || bad "migrate --dry-run: rc=$RC err: $ERR"
cmp -s "$CFG/club3090.env" "$T/m.g0" && cmp -s "$CFG/secrets.env" "$T/m.s0" && [[ ! -e "$CFG/.lock" ]] \
  && ok "migrate --dry-run writes nothing (not even the lock file)" || bad "migrate --dry-run changed the store"
has "migrate --dry-run: prints the plan" "$OUT" "would copy to club3090.env: MODEL_DIR, THREADS, QUOTED, EMPTY"
has "migrate --dry-run: credentials to secrets.env" "$OUT" "would copy to secrets.env: HF_TOKEN, LITELLM_MASTER_KEY"
nosecret "migrate --dry-run: no secret value" "$OUT$ERR"
CFG="$T/m-none"
st -- migrate --dry-run; [[ $RC -eq 0 && ! -e "$CFG" ]] && ok "migrate --dry-run does not even create the config dir" || bad "dry run created $CFG"
CFG="$T/m"

st -- migrate
[[ $RC -eq 0 ]] || bad "migrate: rc=$RC err: $ERR"
want_g=$'# mine\nSAME=same-value\nDIFF=store-value\nMODEL_DIR=/data/models\nTHREADS=28\nQUOTED=single quoted\nEMPTY='
[[ "$(cat "$CFG/club3090.env")" == "$want_g" ]] && ok "migrate: club3090.env gains the ordinary keys (export and quotes stripped, last duplicate wins)" \
  || bad "migrate: club3090.env is: $(cat "$CFG/club3090.env")"
[[ "$(cat "$CFG/secrets.env")" == $'DIFF_API_KEY=zz-mig-diff-store\nHF_TOKEN=zz-mig-hf\nLITELLM_MASTER_KEY=zz-mig-master' && "$(stat -c %a "$CFG/secrets.env")" == 600 ]] \
  && ok "migrate: credentials go to secrets.env (0600); a key already saved keeps its saved value" || bad "migrate: secrets.env wrong"
has "migrate: a refused value is listed with the writer's reason" "$OUT" "BAD: value contains '\$'"
hasnt "migrate: … never with the value" "$OUT" 'has$dollar'
has "migrate: a differing non-secret shows both values" "$OUT" "DIFF: club3090.env has 'store-value', the repo .env has 'repo-value'"
has "migrate: a differing secret is named, values hidden" "$OUT" "DIFF_API_KEY: in secrets.env (a secret — values not shown)"
nosecret "migrate: no secret value anywhere in its output" "$OUT$ERR"
cmp -s "$REPO/.env" "$T/repo.env.orig" && [[ "$(stamp "$REPO/.env")" == "$repo_stamp" ]] \
  && ok "migrate: the repo .env is byte-identical, same mode and mtime" || bad "migrate MODIFIED the repo .env"
values > "$T/values.after"
cmp -s "$T/values.before" "$T/values.after" && ok "migrate: no effective value changed (only where it comes from)" \
  || { bad "migrate changed effective values:"; diff "$T/values.before" "$T/values.after" | command grep -v zz- | sed 's/^/      /' >&2; }
cp "$CFG/club3090.env" "$T/m.g1"; cp "$CFG/secrets.env" "$T/m.s1"
st -- migrate
cmp -s "$CFG/club3090.env" "$T/m.g1" && cmp -s "$CFG/secrets.env" "$T/m.s1" && [[ $RC -eq 0 && "$OUT" == *"Nothing to copy"* ]] \
  && ok "migrate is idempotent: the second run copies nothing" || bad "migrate second run: rc=$RC $OUT"
cmp -s "$REPO/.env" "$T/repo.env.orig" && [[ "$(stamp "$REPO/.env")" == "$repo_stamp" ]] \
  && ok "migrate (again): the repo .env is still untouched" || bad "second migrate MODIFIED the repo .env"
st -- show; [[ "$OUT" != *"still live in"* ]] && ok "show after migrate: no migrate line (the refused value can't move)" || bad "show after migrate: $(tail -n 1 <<<"$OUT")"
rm -f "$REPO/.env"
st -- migrate; [[ $RC -eq 0 && "$OUT" == *"nothing to migrate"* ]] && ok "migrate without a repo .env: nothing to do" || bad "migrate, no .env: rc=$RC $OUT"
# The core is a plain function for c3: a JSON-serialisable report.
if python3 - "$ROOT/scripts/lib" "$T/m" <<'PY'
import json, sys, tempfile, pathlib
sys.path.insert(0, sys.argv[1]); import club_config as c
with tempfile.TemporaryDirectory() as repo:
    pathlib.Path(repo, ".env").write_text("A=1\nX_TOKEN=zz-c3\n", encoding="utf-8")
    r = c.migrate(repo, dry_run=True, environ={"CLUB3090_CONFIG_DIR": sys.argv[2] + "-c3"})
    assert r["copied"] == {"club3090.env": ["A"], "secrets.env": ["X_TOKEN"]} and r["dry_run"], r
    assert "zz-c3" not in json.dumps(r)
PY
then ok "migrate(): a plain, JSON-serialisable report for c3 to reuse"; else bad "migrate() as a function"; fi

# ── report.sh: the Settings section ─────────────────────────────────────────
# The real report.sh in a throwaway root (the shared harness: docker and
# nvidia-smi fail, so nothing reaches a container or a GPU; sudo is stubbed too).
# The config dir's path carries the user name, so redact() must turn it into <USER>.
# shellcheck source=fixtures/report-harness/report-env.sh
source "$ROOT/scripts/tests/fixtures/report-harness/report-env.sh"
report_env_init "$ROOT"
set +e                                     # the harness's report_run re-enables -e
printf '#!/usr/bin/env bash\nexit 1\n' > "$REPORT_FAKE_BIN/sudo"; chmod +x "$REPORT_FAKE_BIN/sudo"
U="${USER:-$(whoami)}"
RCFG="$REPORT_ENV_DIR/cfg-$U"; mkdir -p "$RCFG"
printf 'HF_TOKEN=zz-report-hf\nCLOUD_ROUTE=zz-report-route\n' > "$RCFG/secrets.env"
printf 'THREADS=28\nSHELLWINS=from-file\nWEIRD=a|b\n' > "$RCFG/club3090.env"
printf 'LEGACY_ONLY=legacy\nMY_API_KEY=zz-report-apikey\n' > "$REPORT_FAKE_ROOT/.env"
section_of() { awk '/^## Settings$/ {on=1; next} on && /^## / {exit} on' <<<"$1"; }
for mode in default --no-redact; do
  args=(); [[ "$mode" == --no-redact ]] && args=(--no-redact)
  ( unset HF_TOKEN CLOUD_ROUTE MY_API_KEY THREADS LEGACY_ONLY
    export CLUB3090_CONFIG_DIR="$RCFG" SHELLWINS=from-shell
    report_run "${args[@]}"; printf '%s\n' "$REPORT_OUT" ) > "$T/report.$mode" 2>&1
  R="$(cat "$T/report.$mode")"; S="$(section_of "$R")"
  nosecret "report.sh ($mode): no secret value anywhere in the report" "$R"
  if [[ -z "$S" ]]; then bad "report.sh ($mode): no '## Settings' section"; continue; fi
  has "report.sh ($mode): a club3090.env setting with its source" "$S" '| `THREADS` | `28` | club3090.env |'
  has "report.sh ($mode): the shell wins, and says so" "$S" '| `SHELLWINS` | `from-shell` | shell |'
  has "report.sh ($mode): secrets.env values hidden" "$S" '| `CLOUD_ROUTE` | `<set, hidden>` | secrets.env |'
  has "report.sh ($mode): a credential in the repo .env hidden" "$S" '| `MY_API_KEY` | `<set, hidden>` | repo .env |'
  has "report.sh ($mode): a '|' in a value can't break the table" "$S" '| `WEIRD` | `a\|b` | club3090.env |'
  has "report.sh ($mode): the settings still in the repo .env" "$S" "2 setting(s) still live in"
done
S="$(section_of "$(cat "$T/report.default")")"
has "report.sh: paths scrubbed by report.sh's own redact() (user name → <USER>)" "$S" "cfg-<USER>"
hasnt "report.sh: … the raw config path is gone" "$S" "$RCFG"
S="$(section_of "$(cat "$T/report.--no-redact")")"
has "report.sh --no-redact: paths kept, secrets still hidden" "$S" "$RCFG"

[[ $fail -eq 0 ]] && echo "test-settings-cli: ok" || echo "test-settings-cli: FAIL"
exit $fail
