#!/usr/bin/env bash
# test-club-config — the one config loader, in bash and Python, and the one writer
# (club-3090#1466).
#
# The two loaders must agree BYTE FOR BYTE: the old readers disagreed on duplicate
# keys, quotes and whitespace, and that disagreement is the defect #1466 removes.
# Agreement alone could mean both are wrong the same way, so the output is also
# pinned to a golden list, and a self-test proves the parity check can fail.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SH="$ROOT/scripts/lib/club-config.sh"
PY="$ROOT/scripts/lib/club_config.py"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
CFG="$T/cfg"; REPO="$T/repo"; mkdir -p "$CFG" "$REPO"

# ── fixtures: every edge case the old parsers disagreed on ──────────────────
{
  printf '# legacy repo .env\n'
  printf 'MODEL_DIR=/mnt/models/huggingface\n'
  printf 'export THREADS=24\n'
  printf 'DUP=first\nDUP=second\n'
  printf 'QUOTED="a b"\n'
  printf "SQUOTED='c d'\n"
  printf 'UNMATCHED="open\n'
  printf 'INNER="x"y"\n'
  printf '  SPACES  =  padded value  \n'
  printf 'EMPTY=\n'
  printf 'EQ=a=b=c\n'
  printf 'HASHVAL=v # not stripped\n'
  printf 'CRLF=win\r\n'
  printf 'UTF=naïve — ok\n'
  printf '1BAD=x\nBAD-KEY=y\nnoequals\n=novalue\n'
  printf 'SHARED=from-repo\n'
  printf 'SHELLWINS=from-file\nEMPTYSHELL=from-file\n'
  printf 'export  TWOSPACE=z'                     # no trailing newline
} > "$REPO/.env"
printf 'HF_TOKEN=hf_secret\nSHARED=from-secrets\nDUAL=secrets\n' > "$CFG/secrets.env"
printf '# globals\nSHARED=from-global\nDUAL=global\nTHREADS=32\n' > "$CFG/club3090.env"

cat > "$T/golden" <<EOF
DUAL	club3090.env	global
DUP	repo .env	second
EQ	repo .env	a=b=c
HASHVAL	repo .env	v # not stripped
HF_TOKEN	secrets.env	hf_secret
INNER	repo .env	x"y
MODEL_DIR	repo .env	/mnt/models/huggingface
QUOTED	repo .env	a b
SHARED	club3090.env	from-global
SHELLWINS	shell	shell
SPACES	repo .env	padded value
SQUOTED	repo .env	c d
THREADS	club3090.env	32
TWOSPACE	repo .env	z
UNMATCHED	repo .env	"open
UTF	repo .env	naïve — ok
EOF
# Lines whose value is empty (or came from a CRLF line) are written with printf:
# a heredoc keeps no trailing tab.
printf 'CRLF\trepo .env\twin\nEMPTY\trepo .env\t\nEMPTYSHELL\tshell\t\n' >> "$T/golden"
LC_ALL=C sort -o "$T/golden" "$T/golden"

# A clean environment: only what the loaders need, plus two keys "set in the shell"
# (one of them empty — an empty shell value still wins).
run_env() { env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$CFG" SHELLWINS=shell EMPTYSHELL= "$@"; }
run_env bash -c '. "$1"; club_config_resolve "$2"' _ "$SH" "$REPO" > "$T/bash.out"
run_env python3 "$PY" resolve --show-secrets --root "$REPO" > "$T/py.out"

if cmp -s "$T/bash.out" "$T/py.out"; then ok "bash and Python loaders agree byte for byte ($(wc -l < "$T/py.out") keys)"
else bad "bash and Python loaders disagree:"; diff "$T/bash.out" "$T/py.out" | sed 's/^/      /' >&2; fi
if cmp -s "$T/py.out" "$T/golden"; then ok "output matches the golden list (precedence, duplicates, quotes, CRLF, UTF-8, invalid keys)"
else bad "output differs from the golden list:"; diff "$T/golden" "$T/py.out" | sed 's/^/      /' >&2; fi

# Self-test: the parity check must be able to fail. A first-assignment-wins parser
# (switch.sh's old rule) has to produce a visible difference.
run_env python3 - "$ROOT/scripts/lib" "$REPO" > "$T/firstwins.out" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import club_config as c
orig = c.parse_env_file
def first_wins(path):                      # stands in for switch.sh's old first-assignment rule
    d = orig(path)
    if "DUP" in d:
        d["DUP"] = "first"
    return d
c.parse_env_file = first_wins
sys.stdout.write(c.format_resolved(c.resolve(sys.argv[2])))
PY
cmp -s "$T/firstwins.out" "$T/bash.out" && bad "self-test: a first-wins parser was NOT caught by the parity check" \
                                       || ok "self-test: the parity check catches a first-wins parser"

# Secrets stay hidden unless asked for: by source (secrets.env) and by name (a token
# still sitting in the legacy .env).
printf '\nMY_API_KEY=legacy-leak\n' >> "$REPO/.env"   # the fixture ends without a newline
hid="$(run_env python3 "$PY" resolve --root "$REPO")"
if command grep -qE 'hf_secret|legacy-leak' <<<"$hid"; then bad "resolve printed a secret without --show-secrets"
elif command grep -qxF $'HF_TOKEN\tsecrets.env\t<set, hidden>' <<<"$hid" && command grep -qxF $'MY_API_KEY\trepo .env\t<set, hidden>' <<<"$hid"; then
  ok "resolve hides secret values by default (secrets.env source and credential-looking names)"
else bad "resolve redaction output unexpected: $(command grep -E 'HF_TOKEN|MY_API_KEY' <<<"$hid")"; fi
run_env python3 "$PY" resolve --json --root "$REPO" | command grep -qE 'hf_secret|legacy-leak' && bad "resolve --json printed a secret" \
  || ok "resolve --json hides them too"
sed -i '/^MY_API_KEY=/d' "$REPO/.env"

# ── load: exports only what the shell doesn't set, and records the source ──
out="$(run_env bash -c '. "$1"; club_config_load "$2"
  printf "%s|%s|%s|%s|%s\n" "$THREADS" "$SHELLWINS" "${EMPTYSHELL-UNSET}" "${CLUB3090_CONFIG_SOURCE[THREADS]}" "${CLUB3090_CONFIG_SOURCE[SHELLWINS]-none}"
  env | command grep -c "^HF_TOKEN=hf_secret$"' _ "$SH" "$REPO")"
[[ "$out" == $'32|shell||club3090.env|none\n1' ]] && ok "bash load: exports file values, leaves shell values (even empty) alone, records sources" \
                                                 || bad "bash load: got '$out'"
out="$(run_env python3 - "$ROOT/scripts/lib" "$REPO" <<'PY'
import os, sys; sys.path.insert(0, sys.argv[1]); import club_config as c
inj = c.load(sys.argv[2])
print(os.environ["THREADS"], os.environ["SHELLWINS"], repr(os.environ["EMPTYSHELL"]), inj.get("THREADS"), "SHELLWINS" in inj)
PY
)"
[[ "$out" == "32 shell '' club3090.env False" ]] && ok "Python load: same result" || bad "Python load: got '$out'"

# ── literal values: warn when one looks like it expected shell expansion ────
# launch.sh/report.sh/setup.sh used to `source` the repo .env; now nothing expands.
X="$T/x"; XR="$T/xr"; mkdir -p "$X" "$XR"
printf 'W_HOME=$HOME/models\nW_TILDE=~/models\nW_BRACE=${ROOT}/x\nQ_DIGIT=cost$5\nQ_TILDEUSER=~bob/x\nMY_TOKEN=$tok\n' > "$XR/.env"
printf 'PASSWORD_ISH=$ecret\n' > "$X/secrets.env"
xenv() { env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$X" "$@"; }
bw="$(xenv bash -c '. "$1"; club_config_load "$2"' _ "$SH" "$XR" 2>&1 >/dev/null | command grep -oE 'WARN: [A-Z_]+' | sort)"
pw="$(xenv python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import club_config as c; c.load(sys.argv[2])' "$ROOT/scripts/lib" "$XR" 2>&1 >/dev/null | command grep -oE 'WARN: [A-Z_]+' | sort)"
want=$'WARN: W_BRACE\nWARN: W_HOME\nWARN: W_TILDE'
[[ "$bw" == "$want" && "$pw" == "$want" ]] && ok "expansion warnings: \$VAR, \${VAR} and ~/ flagged; secrets, \$5 and ~user left alone; bash = Python" \
  || bad "expansion warnings: bash [$bw] python [$pw] want [$want]"
xenv bash -c '. "$1"; club_config_load "$2"' _ "$SH" "$XR" 2>&1 >/dev/null | command grep -q 'models' \
  && bad "a warning printed the value" || ok "warnings name the key and file, never the value"

# ── get: one setting's effective value, bash = Python ──────────────────────
G="$T/g"; GR="$T/gr"; mkdir -p "$G" "$GR"
printf 'DUAL=global\nSHADOW=x DUAL\tlooks like a key\n' > "$G/club3090.env"
printf 'ONLY_REPO=legacy\n' > "$GR/.env"
genv() { env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$G" SHELLKEY=from-shell "$@"; }
gb() { genv bash -c '. "$1"; club_config_get "$2" "$3"; echo "rc=$?"' _ "$SH" "$1" "$GR"; }
gp() { genv bash -c 'python3 "$1" get "$2" --root "$3"; echo "rc=$?"' _ "$PY" "$1" "$GR"; }
get_ok=1
for case in "DUAL|global" "ONLY_REPO|legacy" "SHELLKEY|from-shell" "NOPE|"; do
  k="${case%%|*}"; v="${case#*|}"
  want="$v"$'\nrc=0'; [[ -z "$v" ]] && want="rc=1"
  [[ "$(gb "$k")" == "$want" && "$(gp "$k")" == "$want" ]] || { bad "get $k: bash [$(gb "$k")] python [$(gp "$k")] want [$want]"; get_ok=0; }
done
[[ $get_ok -eq 1 ]] && ok "get: file, legacy .env, shell and absent (exit 1) agree in bash and Python; another value containing 'DUAL<TAB>' can't shadow DUAL"

# ── compose env file: what docker compose reads back is exactly what we resolved ──
E="$T/e"; ER="$T/er"; mkdir -p "$E" "$ER"
{ printf 'Q_APOS=it'"'"'s\n'; printf 'Q_DOLLAR=has $HOME and ${X}\n'; printf 'Q_HASH=a # b\n'
  printf 'Q_DQ=dq"inside\n'; printf 'Q_BS=back\\slash\\n\n'; printf 'Q_EMPTY=\n'
  printf 'Q_ALL=it'"'"'s "both" \\ $X\n'; printf 'Q_SHELL=from-file\n'; } > "$ER/.env"
envf="$(env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$E" Q_SHELL=from-shell bash -c '. "$1"; club_config_compose_env_file "$2"' _ "$SH" "$ER")"
[[ -f "$envf" && "$(stat -c %a "$envf")" == 600 ]] && ok "compose env file is written 0600 (it can hold secrets)" || bad "compose env file: '$envf' mode $(stat -c %a "$envf" 2>/dev/null)"
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  keys="Q_APOS Q_DOLLAR Q_HASH Q_DQ Q_BS Q_EMPTY Q_ALL Q_SHELL"
  { echo "services:"; echo "  x:"; echo "    image: busybox"; echo "    environment:"; for k in $keys; do echo "      $k: \"\${$k}\""; done; } > "$T/ce.yml"
  got="$(env -i PATH="$PATH" HOME="$T/home" docker compose --env-file "$envf" -f "$T/ce.yml" config --format json 2>&1)"
  want="$(env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$E" Q_SHELL=from-shell python3 "$PY" resolve --show-secrets --root "$ER")"
  ce_ok=1
  for k in $keys; do
    w="$(awk -F'\t' -v k="$k" '$1 == k { sub(/^[^\t]*\t[^\t]*\t/, ""); print; exit }' <<<"$want")"
    # `docker compose config` RENDERS a literal $ as $$ (so its output re-parses
    # safely); undo that. Verified 2026-09-28 in a container: `docker compose run`
    # sees `has $HOME and ${X}` and `it's "both" \ $X` exactly.
    v="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["services"]["x"]["environment"][sys.argv[1]].replace("$$", "$"), end="")' "$k" <<<"$got" 2>/dev/null)" \
      || { bad "docker compose did not parse the env file: ${got:0:200}"; ce_ok=0; break; }
    [[ "$v" == "$w" ]] || { bad "compose read $k as [$v], resolved [$w]"; ce_ok=0; }
  done
  [[ $ce_ok -eq 1 ]] && ok "docker compose reads back every resolved value exactly (', \$HOME, #, \", backslash, empty, shell-wins)"
else
  ok "compose env file round trip skipped (no docker compose here)"
fi
rm -f "$envf"

# ── config dir: override > XDG > HOME, identical in both ────────────────────
for case in "CLUB3090_CONFIG_DIR=/x/y|/x/y" "XDG_CONFIG_HOME=/xdg|/xdg/club-3090" "XDG_CONFIG_HOME=|/h/.config/club-3090" "|/h/.config/club-3090"; do
  assign="${case%%|*}"; want="${case##*|}"
  b="$(env -i PATH="$PATH" HOME=/h ${assign:+"$assign"} bash -c '. "$1"; club_config_dir' _ "$SH")"
  p="$(env -i PATH="$PATH" HOME=/h ${assign:+"$assign"} python3 "$PY" dir)"
  [[ "$b" == "$want" && "$p" == "$want" ]] || bad "config dir with '${assign:-nothing}': bash '$b', python '$p', want '$want'"
done
[[ $fail -eq 0 ]] && ok "config dir: CLUB3090_CONFIG_DIR > XDG_CONFIG_HOME > HOME/.config, both loaders"

# ── cache and data dirs (#1466 phase 4): the same rule, identical in both ─────
# Each case also sets the OTHER two variables of its family to decoys, so a twin that
# read the wrong one (XDG_CONFIG_HOME for the cache dir, say) can't pass by accident.
cd_fail=0
for case in "cache|CLUB3090_CACHE_DIR=/c/d|/c/d" "cache|XDG_CACHE_HOME=/xc|/xc/club-3090" "cache|XDG_CACHE_HOME=|/h/.cache/club-3090" "cache||/h/.cache/club-3090" \
            "data|CLUB3090_DATA_DIR=/d/e|/d/e" "data|XDG_DATA_HOME=/xd|/xd/club-3090" "data|XDG_DATA_HOME=|/h/.local/share/club-3090" "data||/h/.local/share/club-3090"; do
  kind="${case%%|*}"; rest="${case#*|}"; assign="${rest%%|*}"; want="${rest##*|}"
  decoys=(XDG_CONFIG_HOME=/decoy-config CLUB3090_CONFIG_DIR=/decoy-cfgdir)
  [[ "$kind" == cache ]] && decoys+=(XDG_DATA_HOME=/decoy-data CLUB3090_DATA_DIR=/decoy-datadir) \
                         || decoys+=(XDG_CACHE_HOME=/decoy-cache CLUB3090_CACHE_DIR=/decoy-cachedir)
  b="$(env -i PATH="$PATH" HOME=/h "${decoys[@]}" ${assign:+"$assign"} bash -c '. "$1"; "club_config_$2_dir"' _ "$SH" "$kind")"
  p="$(env -i PATH="$PATH" HOME=/h "${decoys[@]}" ${assign:+"$assign"} python3 "$PY" "$kind-dir")"
  [[ "$b" == "$want" && "$p" == "$want" ]] || { bad "$kind dir with '${assign:-nothing}': bash '$b', python '$p', want '$want'"; cd_fail=1; }
done
[[ $cd_fail -eq 0 ]] && ok "cache dir (CLUB3090_CACHE_DIR > XDG_CACHE_HOME > HOME/.cache) and data dir (CLUB3090_DATA_DIR > XDG_DATA_HOME > HOME/.local/share): bash = Python"

# ── writer ──────────────────────────────────────────────────────────────────
W="$T/w"
wr() { env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$W" python3 "$PY" "$@" 2>"$T/wr.err"; }
wr set --file secrets HF_TOKEN=abc && wr set MODEL_DIR=/m
m_dir="$(stat -c %a "$W")"; m_sec="$(stat -c %a "$W/secrets.env")"; m_glob="$(stat -c %a "$W/club3090.env")"
[[ "$m_dir" == 700 && "$m_sec" == 600 && "$m_glob" == 644 ]] && ok "new files: config dir 0700, secrets.env 0600, club3090.env 0644" \
  || bad "modes: dir $m_dir secrets $m_sec global $m_glob"
printf '# header comment\nA=1\n\nexport B=2\nB=dup\n# tail comment\nC=3\n' > "$W/club3090.env"
chmod 640 "$W/club3090.env"
wr set B=new D=4 && wr unset C
want=$'# header comment\nA=1\n\nB=new\n# tail comment\nD=4'
[[ "$(cat "$W/club3090.env")" == "$want" ]] && ok "set/unset keep comments and order, collapse duplicates, append new keys" \
  || { bad "rewrite result:"; cat "$W/club3090.env" | sed 's/^/      /' >&2; }
[[ "$(stat -c %a "$W/club3090.env")" == 640 ]] && ok "rewrite keeps an existing file's mode" || bad "mode changed to $(stat -c %a "$W/club3090.env")"
ls -A "$W" | command grep -qE '^\.(club3090|secrets)\.env\.' && bad "a temporary file was left behind" || ok "no temporary files left behind"

before="$(cat "$W/club3090.env")"; refused=0
for v in 'has"quote' "has'quote" 'has$dollar' 'has\backslash' 'has`tick' ' lead' 'trail ' 'a #comment' '#start' $'a\tb #x'; do
  if wr set "K=$v"; then bad "writer accepted unsafe value [$v]"; else refused=$((refused+1)); fi
done
wr set "1BAD=x" && bad "writer accepted an invalid key" || refused=$((refused+1))
wr set "NOEQUALS" && bad "writer accepted an argument without '='" || refused=$((refused+1))
[[ "$(cat "$W/club3090.env")" == "$before" ]] && ok "writer refuses $refused unsafe values/keys and leaves the file untouched" \
  || bad "file changed after refused writes"
command grep -q 'docker compose would read as a comment' "$T/wr.err" 2>/dev/null || wr set 'K=a #c'; command grep -q "comment" "$T/wr.err" \
  && ok "refusals explain themselves" || bad "refusal message unclear: $(cat "$T/wr.err")"

# Round trip: what the writer accepts, every reader reads back unchanged —
# including docker compose --env-file, the delivery path for containers.
R="$T/rt"; declare -A RT=([RT_PLAIN]=plain [RT_SPACE]='/mnt/models/hugging face' [RT_HASH]='a#b' [RT_EMPTY]= [RT_EQ]='x=y=z' [RT_UTF]='naïve—ok' [RT_PATH]='/opt/x_y-z.1')
for k in "${!RT[@]}"; do env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$R" python3 "$PY" set "$k=${RT[$k]}" 2>/dev/null || bad "writer refused safe value [$k=${RT[$k]}]"; done
b="$(env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$R" bash -c '. "$1"; club_config_resolve' _ "$SH")"
p="$(env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$R" python3 "$PY" resolve --show-secrets)"
rt_ok=1
for k in "${!RT[@]}"; do
  line="$k"$'\tclub3090.env\t'"${RT[$k]}"
  command grep -qxF -- "$line" <<<"$b" || { bad "bash read back $k wrong"; rt_ok=0; }
  command grep -qxF -- "$line" <<<"$p" || { bad "python read back $k wrong"; rt_ok=0; }
done
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  { echo "services:"; echo "  x:"; echo "    image: busybox"; echo "    environment:"; for k in "${!RT[@]}"; do echo "      $k: \"\${$k}\""; done; } > "$T/c.yml"
  got="$(env -i PATH="$PATH" HOME="$T/home" docker compose --env-file "$R/club3090.env" -f "$T/c.yml" config --format json 2>&1)"
  for k in "${!RT[@]}"; do
    v="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["services"]["x"]["environment"][sys.argv[1]], end="")' "$k" <<<"$got" 2>/dev/null)" \
      || { bad "docker compose did not parse the written file: ${got:0:200}"; rt_ok=0; break; }
    [[ "$v" == "${RT[$k]}" ]] || { bad "docker compose read $k as [$v], want [${RT[$k]}]"; rt_ok=0; }
  done
  [[ $rt_ok -eq 1 ]] && ok "round trip: ${#RT[@]} written values read back unchanged by bash, Python and docker compose --env-file"
else
  [[ $rt_ok -eq 1 ]] && ok "round trip: ${#RT[@]} written values read back unchanged by bash and Python (docker compose not available here)"
fi

# unset --root: clearing a setting also clears the checkout's legacy .env, which is
# read last — otherwise an old copy there (a cleared model-default pin) comes back.
U="$T/u"; UR="$T/ur"; mkdir -p "$U" "$UR"
printf 'PIN=store\nKEEP=1\n' > "$U/club3090.env"
printf '# my legacy settings\nMODEL_DIR=/m\nexport PIN=legacy\n\n# tail\nOTHER=x\n' > "$UR/.env"; chmod 640 "$UR/.env"
ur() { env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$U" python3 "$PY" "$@"; }
msg="$(ur unset --root "$UR" PIN 2>&1)"
[[ "$(cat "$U/club3090.env")" == "KEEP=1" ]] || bad "unset --root left the store: $(cat "$U/club3090.env")"
[[ "$(cat "$UR/.env")" == $'# my legacy settings\nMODEL_DIR=/m\n\n# tail\nOTHER=x' ]] \
  && ok "unset --root clears the key from the store AND the legacy .env, keeping its comments and order" \
  || { bad "legacy .env after unset --root:"; sed 's/^/      /' "$UR/.env" >&2; }
[[ "$(stat -c %a "$UR/.env")" == 640 ]] && ok "the legacy .env keeps its mode" || bad "legacy .env mode $(stat -c %a "$UR/.env")"
[[ -z "$(ls -A "$UR" | command grep -vx '.env')" ]] && ok "nothing added to the checkout (lock and temp files live elsewhere)" \
  || bad "unset --root left files in the checkout: $(ls -A "$UR" | command grep -vx '.env')"
command grep -qF "$UR/.env" <<<"$msg" && command grep -qF "$U/club3090.env" <<<"$msg" \
  && ok "unset reports every file it changed" || bad "unset message: $msg"
b="$(env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$U" bash -c '. "$1"; club_config_resolve "$2"' _ "$SH" "$UR" | command grep -c '^PIN	' || true)"
[[ "$b" == 0 ]] && ok "after unset --root the key resolves nowhere" || bad "PIN still resolves after unset --root"
NR="$T/nr"; mkdir -p "$NR"
ur unset --root "$NR" KEEP 2>/dev/null
[[ ! -e "$NR/.env" ]] && ok "unset --root never creates a legacy .env" || bad "unset --root created $NR/.env"
printf 'ONLYREPO=1\n' > "$NR/.env"; before="$(stat -c %Y "$NR/.env")"; sleep 1
ur unset --root "$NR" ABSENT 2>/dev/null
[[ "$(stat -c %Y "$NR/.env")" == "$before" ]] && ok "a legacy .env without the key is not rewritten" || bad "legacy .env rewritten for an absent key"

# Concurrent writers: the lock must not lose an update.
C="$T/conc"
for i in $(seq 1 20); do env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$C" python3 "$PY" set "K$i=v$i" 2>/dev/null & done; wait
n="$(command grep -cE '^K[0-9]+=v[0-9]+$' "$C/club3090.env")"
[[ "$n" == 20 ]] && ok "20 concurrent writers: all 20 keys present" || bad "concurrent writers: $n of 20 keys present"

[[ $fail -eq 0 ]] && echo "test-club-config: ok" || echo "test-club-config: FAIL"
exit $fail
