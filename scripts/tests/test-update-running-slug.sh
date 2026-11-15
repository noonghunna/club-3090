#!/usr/bin/env bash
# Gate for #2050: with no argument, update.sh re-checks the weights the rig is
# actually serving, never a model it may not run.
#
# update.sh used to re-run `setup.sh qwen3.6-27b` after every pull (a hard-coded
# MODEL default). On a rig serving anything else that verified — and on a rig
# without it would have started downloading — weights nobody uses, and its hint
# said `switch.sh <variant>` even though it had found the running container.
#
# Validates, against a docker shim and a throwaway clone (this checkout is never
# pulled or touched):
#   1. a running container's club3090.slug label is what setup.sh checks, and the
#      hint names the exact `switch.sh <slug>`; a non-club container is ignored
#   2. two containers of one slug are checked once; two slugs, once each
#   3. nothing running → setup.sh is skipped (exit 0) and says how to check a slug
#   4. a club container with no label → not guessed: skipped and named
#   5. labelled + unlabelled → the labelled slug is checked, the other named
#   6. an argument, or MODEL=, still wins over what is running
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export CLUB3090_DATA_DIR="$T/data" CLUB3090_CACHE_DIR="$T/cache"

fail=0
ok()  { echo "  ok   — $*"; }
bad() { echo "  FAIL — $*" >&2; fail=1; }
G() { git -c user.name=test -c user.email=test@example.invalid -c init.defaultBranch=master "$@"; }

# --- fixture: origin + an up-to-date clone on master -------------------------
# Only update.sh and the container helper are real. setup.sh and gpu-mode.sh are
# stubs that record their arguments. Without the rest of the tree the registry
# can't be read, so club-containers.sh degrades to its prefix match — the shim's
# names below use those prefixes.
src="$T/src"; mkdir -p "$src/scripts/lib"
cp "$ROOT/scripts/update.sh" "$src/scripts/update.sh"
cp "$ROOT/scripts/lib/club-containers.sh" "$src/scripts/lib/club-containers.sh"
cat > "$src/scripts/setup.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$UPDATE_TEST_CALLS"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$src/scripts/gpu-mode.sh"
chmod +x "$src/scripts/"*.sh
G init -q "$src" && G -C "$src" add -A && G -C "$src" commit -q -m fixture
G init -q --bare "$T/origin.git"
G -C "$src" push -q "$T/origin.git" HEAD:refs/heads/master
git -C "$T/origin.git" symbolic-ref HEAD refs/heads/master
G clone -q "$T/origin.git" "$T/clone"

# docker shim: `ps` prints $UPDATE_TEST_PS (name<TAB>slug rows) whatever the format;
# anything else fails, as for a container that isn't there.
mkdir -p "$T/bin"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  ps) [[ -n "${UPDATE_TEST_PS:-}" ]] && printf '%b\n' "$UPDATE_TEST_PS"; exit 0 ;;
  *)  exit 1 ;;
esac
EOF
chmod +x "$T/bin/docker"

export UPDATE_TEST_CALLS="$T/calls"
# run_update <ps rows> [update.sh args...]: sets $out, $rc and $calls
run_update() {
  local rows="$1"; shift
  : > "$UPDATE_TEST_CALLS"
  out="$(cd "$T/clone" && env -u MODEL PATH="$T/bin:$PATH" UPDATE_TEST_PS="$rows" \
         bash scripts/update.sh --force "$@" 2>&1)"; rc=$?
  calls="$(cat "$UPDATE_TEST_CALLS")"
}

echo "1. a labelled container: its slug is checked, and the hint names it"
run_update 'qengine-qwen38-27b-dual\tqengine/qwen38-27b-dual\nopen-webui\tnot/a-club-slug'
[[ $rc -eq 0 ]] || bad "exit $rc: $out"
[[ "$calls" == "qengine/qwen38-27b-dual" ]] && ok "setup.sh checks the running slug's weights" \
  || bad "setup.sh was called with '${calls}', want the running slug only"
# The expected hint is built from a variable: written out whole, the comparison reads
# to test-tests-never-launch as a switch.sh invocation.
hint="switch.sh qengine/qwen38-27b-dual"
[[ "$out" == *"bash scripts/$hint"* ]] && ok "the hint names the exact switch.sh <slug>" \
  || bad "no exact restart hint: $(printf '%s' "$out" | tail -5)"
[[ "$calls" != *"qwen3.6-27b"* && "$out" != *"not/a-club-slug"* ]] && ok "no model default, and a non-club container is ignored" \
  || bad "a default model or a non-club container leaked in: calls='${calls}'"

echo "2. one slug twice, then two slugs"
run_update 'vllm-a\tvllm/dual\nvllm-b\tvllm/dual'
[[ "$calls" == "vllm/dual" ]] && ok "two containers of one slug: checked once" || bad "want one check of vllm/dual, got '${calls}'"
run_update 'vllm-a\tvllm/dual\nqengine-b\tqengine/qwen38-27b-dual'
[[ "$calls" == $'qengine/qwen38-27b-dual\nvllm/dual' ]] && ok "two slugs: each checked once" \
  || bad "want both slugs once each, got '$(tr '\n' ';' <<<"$calls")'"

echo "3. nothing running"
run_update ''
[[ $rc -eq 0 && -z "$calls" ]] && ok "setup.sh is skipped, exit 0" || bad "exit $rc, setup.sh called with '${calls}'"
[[ "$out" == *"nothing is serving"* && "$out" == *"bash scripts/update.sh <slug>"* ]] && ok "it says so, and how to check a slug" \
  || bad "no skip message: $(printf '%s' "$out" | tail -6)"

echo "4. a club container without a label"
run_update 'vllm-handmade\t'
[[ -z "$calls" ]] && ok "not guessed: setup.sh is skipped" || bad "setup.sh was called with '${calls}'"
[[ "$out" == *"not checked: vllm-handmade"* ]] && ok "the container is named" || bad "unlabelled container not named: $(printf '%s' "$out" | tail -6)"

echo "5. labelled and unlabelled together"
run_update 'vllm-a\tvllm/dual\nvllm-handmade\t'
[[ "$calls" == "vllm/dual" && "$out" == *"not checked: vllm-handmade"* ]] && ok "the labelled slug is checked, the other named" \
  || bad "calls='${calls}'; $(printf '%s' "$out" | command grep -F 'not checked' || echo 'no not-checked line')"

echo "6. an explicit choice still wins"
run_update 'vllm-a\tvllm/dual' qwen3.6-27b
[[ "$calls" == "qwen3.6-27b" ]] && ok "an argument wins over the running slug" || bad "want qwen3.6-27b, got '${calls}'"
: > "$UPDATE_TEST_CALLS"
out="$(cd "$T/clone" && env PATH="$T/bin:$PATH" UPDATE_TEST_PS='vllm-a\tvllm/dual' MODEL=gemma-4-31b \
       bash scripts/update.sh --force 2>&1)"; calls="$(cat "$UPDATE_TEST_CALLS")"
[[ "$calls" == "gemma-4-31b" ]] && ok "MODEL= wins over the running slug" || bad "want gemma-4-31b, got '${calls}'"

[[ $fail -eq 0 ]] && echo "test-update-running-slug: ok" || echo "test-update-running-slug: FAIL"
exit $fail
