#!/usr/bin/env bash
# test-club-config — the one config loader, in bash and Python, and the one writer
# (club-3090#1466).
#
# The two loaders must agree BYTE FOR BYTE: the old readers disagreed on duplicate
# keys, quotes and whitespace, and that disagreement is the defect #1466 removes.
# Agreement alone could mean both are wrong the same way, so the output is also
# pinned to a golden list, and a self-test proves the parity check can fail.
set -uo pipefail
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

# ── config dir: override > XDG > HOME, identical in both ────────────────────
for case in "CLUB3090_CONFIG_DIR=/x/y|/x/y" "XDG_CONFIG_HOME=/xdg|/xdg/club-3090" "XDG_CONFIG_HOME=|/h/.config/club-3090" "|/h/.config/club-3090"; do
  assign="${case%%|*}"; want="${case##*|}"
  b="$(env -i PATH="$PATH" HOME=/h ${assign:+"$assign"} bash -c '. "$1"; club_config_dir' _ "$SH")"
  p="$(env -i PATH="$PATH" HOME=/h ${assign:+"$assign"} python3 "$PY" dir)"
  [[ "$b" == "$want" && "$p" == "$want" ]] || bad "config dir with '${assign:-nothing}': bash '$b', python '$p', want '$want'"
done
[[ $fail -eq 0 ]] && ok "config dir: CLUB3090_CONFIG_DIR > XDG_CONFIG_HOME > HOME/.config, both loaders"

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

# Concurrent writers: the lock must not lose an update.
C="$T/conc"
for i in $(seq 1 20); do env -i PATH="$PATH" HOME="$T/home" CLUB3090_CONFIG_DIR="$C" python3 "$PY" set "K$i=v$i" 2>/dev/null & done; wait
n="$(command grep -cE '^K[0-9]+=v[0-9]+$' "$C/club3090.env")"
[[ "$n" == 20 ]] && ok "20 concurrent writers: all 20 keys present" || bad "concurrent writers: $n of 20 keys present"

[[ $fail -eq 0 ]] && echo "test-club-config: ok" || echo "test-club-config: FAIL"
exit $fail
