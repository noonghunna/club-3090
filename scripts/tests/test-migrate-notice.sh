#!/usr/bin/env bash
# test-migrate-notice — the one-time "your settings still live in this checkout" notice
# (club-3090#1466). An install updated with `git pull` keeps its settings in the repo .env
# (and its gateway routes/keys in services/litellm); they keep working, and the launchers
# say ONCE how to move them: `settings.sh migrate`.
#
#   1. club_config.migrate_notice: shown once per set of pending items (a hash of key NAMES
#      in the config dir), shown again when a new key lands in the checkout, silent after
#      migrate, silent when nothing is pending or the stamp can't be written — and it never
#      prints a value.
#   2. The real entry points, run from a scratch copy of the tracked tree with a planted
#      .env (this test never writes the checkout's own .env): switch.sh (before its
#      refusals), launch.sh, gpu-mode (a mode command, not status/listings — docker and
#      sudo are shims), setup.sh's non-interactive path. One stamp shared by all of them.
#   3. setup.sh offers the copy on a terminal (driven through `script` for a pty).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
SECRET="not-a-real-token-migrate-notice-0001"
VALUE="/value/that/must/not/print"
nosecret() {   # <label> <text>
  if [[ "$2" == *"$SECRET"* || "$2" == *"$VALUE"* ]]; then bad "$1: a value was printed"; else ok "$1"; fi
}
cli() { env -u HF_TOKEN -u LITELLM_MASTER_KEY -u MODEL_DIR -u THREADS -u NEWKEY python3 scripts/lib/club_config.py "$@"; }

# ── 1. the notice itself ──────────────────────────────────────────────────────
echo "1. club_config.migrate_notice"
R="$T/repo"; mkdir -p "$R"
printf 'MODEL_DIR=%s\nTHREADS=4\nHF_TOKEN=%s\n' "$VALUE" "$SECRET" > "$R/.env"
C="$T/cfg1"
n1="$(CLUB3090_CONFIG_DIR="$C" cli migrate-notice --root "$R" --prefix "[t]" 2>&1)"; rc=$?
n2="$(CLUB3090_CONFIG_DIR="$C" cli migrate-notice --root "$R" --prefix "[t]" 2>&1)"
[[ $rc -eq 0 && "$n1" == "[t] NOTE: Your settings still live in this checkout (3 setting(s) in $R/.env)"* \
   && "$n1" == *"settings.sh migrate --dry-run"* && -z "$n2" ]] \
  && ok "shown once (3 settings), then quiet" || bad "once: rc=$rc first=[$n1] second=[$n2]"
[[ "$(stat -c %a "$C")" == 700 ]] && ok "the config dir it creates for the stamp is 0700" || bad "config dir mode $(stat -c %a "$C")"
nosecret "the notice counts settings and names files, never a value" "$n1"
command grep -qE "$SECRET|THREADS|MODEL_DIR" "$C/.notice-migrate" && bad "the stamp holds names or values in clear" \
  || ok "the stamp holds only a hash"
printf 'NEWKEY=x\n' >> "$R/.env"
n3="$(CLUB3090_CONFIG_DIR="$C" cli migrate-notice --root "$R" 2>&1)"
[[ "$n3" == *"(4 setting(s) in "* ]] && ok "shown again when a new key lands in the checkout" || bad "new key: [$n3]"
out="$(CLUB3090_CONFIG_DIR="$C" cli settings --root "$R" migrate 2>&1)"
n4="$(CLUB3090_CONFIG_DIR="$C" cli migrate-notice --root "$R" 2>&1)"
p4="$(CLUB3090_CONFIG_DIR="$C" cli migrate-pending --root "$R" 2>&1)"
[[ -z "$n4" && -z "$p4" ]] && ok "quiet after settings.sh migrate (nothing pending)" || bad "after migrate: notice=[$n4] pending=[$p4]"
nosecret "migrate's own output" "$out"
[[ -f "$R/.env" ]] && command grep -q "^HF_TOKEN=$SECRET$" "$R/.env" && ok "migrate left the repo .env as it was" || bad "repo .env changed"

R2="$T/repo2"; mkdir -p "$R2"; printf 'THREADS=4\n' > "$R2/.env"
n5="$(cli migrate-notice --root "$R2" 2>&1)"; rc=$?   # CLUB3090_CONFIG_DIR=/nonexistent/… can't be created
[[ $rc -eq 0 && -z "$n5" ]] && ok "no stamp possible → no notice (it would repeat every launch), exit 0" || bad "unwritable: rc=$rc [$n5]"
R3="$T/repo3"; mkdir -p "$R3"; C3="$T/cfg3"
n6="$(CLUB3090_CONFIG_DIR="$C3" cli migrate-notice --root "$R3" 2>&1)"
[[ -z "$n6" && ! -e "$C3" ]] && ok "nothing pending → no notice, and no config dir created" || bad "nothing pending: [$n6] dir=$(ls -d "$C3" 2>/dev/null)"
n7="$(CLUB3090_CONFIG_DIR="$T/cfg7" bash -c ". scripts/lib/club-config.sh; club_config_migrate_notice '$T/no/such/root' '[x]'; echo rc=\$?" 2>&1)"
[[ "$n7" == "rc=0" ]] && ok "the bash wrapper never fails its caller" || bad "wrapper: [$n7]"
R4="$T/repo4"; mkdir -p "$R4/services/litellm"
printf 'model_list:\n  - model_name: cloud\n    litellm_params:\n      api_key: os.environ/CLOUD_KEY\n' > "$R4/services/litellm/config.local.yaml"
printf 'CLOUD_KEY=%s\n' "$SECRET" > "$R4/services/litellm/local.env"
n8="$(CLUB3090_CONFIG_DIR="$T/cfg8" cli migrate-notice --root "$R4" 2>&1)"
[[ "$n8" == *"services/litellm/config.local.yaml"* && "$n8" == *"1 route key(s)"* ]] \
  && ok "the gateway's own routes and keys count too" || bad "gateway files: [$n8]"
nosecret "the gateway part names files, never a key" "$n8"

# ── 2. the real entry points, on a scratch copy of the tree ──────────────────
echo "2. switch.sh / launch.sh / gpu-mode / setup.sh"
S="$T/tree"; mkdir -p "$S"
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$S"
printf 'MODEL_DIR=%s\nTHREADS=4\nHF_TOKEN=%s\n' "$VALUE" "$SECRET" > "$S/.env"
mkdir -p "$T/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/docker"
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/nvidia-smi"
printf '#!/usr/bin/env bash\necho "$*" >> "%s"\nexit 0\n' "$T/sudo-called" > "$T/bin/sudo"
printf '#!/usr/bin/env bash\necho "$*" >> "%s"\nexit 1\n' "$T/switch-called" > "$T/bin/switch-mock"
chmod +x "$T/bin/"*
for b in docker sudo nvidia-smi; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run the launchers" >&2; exit 1; }
done
run() { env -u HF_TOKEN -u LITELLM_MASTER_KEY -u MODEL_DIR -u THREADS -u CLUB3090_DIR PATH="$T/bin:$PATH" "$@"; }

C="$T/cfg-switch"
o1="$(run CLUB3090_CONFIG_DIR="$C" COMPOSE_BIN=: timeout 120 bash "$S/scripts/switch.sh" zzz/no-such-slug 2>&1)"
o2="$(run CLUB3090_CONFIG_DIR="$C" COMPOSE_BIN=: timeout 120 bash "$S/scripts/switch.sh" zzz/no-such-slug 2>&1)"
nl="$(command grep -n '\[switch\] NOTE: Your settings still live' <<<"$o1" | cut -d: -f1)"
rl="$(command grep -n "unknown variant 'zzz/no-such-slug'" <<<"$o1" | cut -d: -f1)"
[[ -n "$nl" && -n "$rl" && "$nl" -lt "$rl" && "$o2" != *NOTE:*"still live"* ]] \
  && ok "switch.sh: the notice once, before its refusals; quiet the second time" || bad "switch.sh: first=[$o1] second=[$o2]"
# switch.sh echoes MODEL_DIR itself (not a secret); the notice lines must carry no value,
# and the token must appear nowhere.
nosecret "switch.sh's notice lines" "$(command grep 'NOTE:' <<<"$o1")"
[[ "$o1" != *"$SECRET"* ]] && ok "switch.sh printed no token" || bad "switch.sh printed the token"
o3="$(run CLUB3090_CONFIG_DIR="$C" SWITCH="$T/bin/switch-mock" COMPOSE_BIN=: timeout 120 bash "$S/scripts/launch.sh" --no-preflight --model zzz-no-such-model 2>&1)"
[[ "$o3" != *NOTE:*"still live"* ]] && ok "one stamp for every launcher: launch.sh stays quiet after switch.sh said it" || bad "launch.sh repeated it: [$o3]"
C="$T/cfg-launch"
o4="$(run CLUB3090_CONFIG_DIR="$C" SWITCH="$T/bin/switch-mock" COMPOSE_BIN=: timeout 120 bash "$S/scripts/launch.sh" --no-preflight --model zzz-no-such-model 2>&1)"
[[ "$o4" == *"[launch] NOTE: Your settings still live in this checkout (3 setting(s)"* ]] && ok "launch.sh says it (fresh config dir)" || bad "launch.sh: [$o4]"
[[ -e "$T/switch-called" ]] && bad "launch.sh went past model selection: $(cat "$T/switch-called")" || ok "launch.sh never reached switch (mock not invoked)"

C="$T/cfg-gpu"
o5="$(run CLUB3090_CONFIG_DIR="$C" timeout 120 bash "$S/scripts/gpu-mode.sh" --list-modes 2>&1)"
[[ "$o5" != *"still live"* ]] && ok "gpu-mode listings stay quiet (no stamp spent)" || bad "gpu-mode --list-modes printed it"
# PATH="$T/bin:…" spelled out on the call itself: that (docker + sudo shims, checked above)
# is what keeps gpu-mode from starting anything, and test-tests-never-launch looks for it here.
o6="$(env -u HF_TOKEN -u LITELLM_MASTER_KEY -u MODEL_DIR -u THREADS -u CLUB3090_DIR PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$C" timeout 120 bash "$S/scripts/gpu-mode.sh" upgrade 2>&1)"
[[ "$o6" == *"[gpu-mode] NOTE: Your settings still live in this checkout (3 setting(s)"* ]] && ok "gpu-mode says it on a mode command (upgrade)" || bad "gpu-mode upgrade: [$o6]"
nosecret "gpu-mode's notice lines" "$(command grep 'NOTE:' <<<"$o6")"
[[ "$o6" != *"$SECRET"* ]] && ok "gpu-mode printed no token" || bad "gpu-mode printed the token"

C="$T/cfg-setup"
o7="$(run CLUB3090_CONFIG_DIR="$C" timeout 120 bash "$S/scripts/setup.sh" --help </dev/null 2>&1; run CLUB3090_CONFIG_DIR="$C" SETUP_DUMP_KEYS=1 timeout 120 bash "$S/scripts/setup.sh" qwen3.6-27b </dev/null 2>&1)"
[[ "$o7" != *"still live"* ]] && ok "setup.sh --help / the key-dump surface stay quiet" || bad "setup.sh early exits printed it"

# ── 3. setup.sh offers the copy on a terminal ────────────────────────────────
echo "3. setup.sh on a terminal"
if command -v script >/dev/null 2>&1; then
  C="$T/cfg-tty"
  # Answer the offer with Enter (= yes); stop right after it, before MODEL_DIR / preflight:
  # MODEL_DIR is then set from the copied settings, and the rest of setup fails fast on the
  # docker shim / missing weights, which is fine — only the offer is under test.
  printf '\n' | run CLUB3090_CONFIG_DIR="$C" timeout 120 script -qec "bash '$S/scripts/setup.sh' qwen3.6-27b" "$T/tty.log" >/dev/null 2>&1
  o8="$(cat "$T/tty.log" 2>/dev/null)"
  [[ "$o8" == *"Your settings still live in this checkout (3 setting(s)"* && "$o8" == *"Copy them now? [Y/n]"* ]] \
    && ok "setup.sh offers the copy" || bad "setup.sh offer: $(tail -c 600 <<<"$o8")"
  [[ -f "$C/club3090.env" ]] && command grep -q '^THREADS=4$' "$C/club3090.env" && [[ "$(stat -c %a "$C/secrets.env" 2>/dev/null)" == 600 ]] \
    && ok "accepting it copies the settings (secrets.env 0600)" || bad "setup.sh did not migrate: $(ls -la "$C" 2>&1)"
  nosecret "setup.sh's offer" "$o8"
  command grep -q "^HF_TOKEN=$SECRET$" "$S/.env" && ok "the repo .env is left as it was" || bad "setup.sh changed the repo .env"
  # End of input at the offer (Ctrl-D, a scripted pty) is a "no", not a setup failure:
  # under set -e a bare `read` at EOF used to end setup.sh right there (test-setup-picker).
  C="$T/cfg-eof"
  run CLUB3090_CONFIG_DIR="$C" timeout 120 script -qec "bash '$S/scripts/setup.sh' qwen3.6-27b" "$T/eof.log" </dev/null >/dev/null 2>&1
  o9="$(cat "$T/eof.log" 2>/dev/null)"
  [[ "$o9" == *"Copy them now? [Y/n]"* && "$o9" == *"left as they are"* && ! -e "$C/club3090.env" ]] \
    && ok "end of input at the offer counts as no; setup carries on" || bad "EOF at the offer: $(tail -c 600 <<<"$o9")"
else
  echo "  - skipped: no \`script\` for a pty"
fi

[ "$fail" -eq 0 ] && echo "test-migrate-notice: ok" || { echo "test-migrate-notice: FAIL"; exit 1; }
