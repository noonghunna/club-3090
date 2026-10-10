#!/usr/bin/env bash
# install.sh — apply patch sglang-effort-thinking-budget inside the SGLang container.
#
# Run from the compose entrypoint AFTER `effort_budget.py shell-env --engine sglang` has exported
# CLUB3090_REASONING_EFFORT_BUDGETS (+ SGLANG_MAX_THINK_TOKENS), BEFORE `sglang.launch_server`:
#   1. copies /etc/club3090/effort_budget.py -> site-packages/club3090_effort_budget.py and this
#      dir's club3090_sglang_effort_budget.py next to it (re-copied whenever the content differs,
#      so a restart after `git pull` never runs stale logic);
#   2. parses the exported map with the copied module — a malformed map stops the boot here
#      instead of failing every request;
#   3. patches the three SGLang call sites (patch_sglang.py: strict anchors, all-or-nothing,
#      AST-verified) unless the marker is already there.
# Idempotent: a container restart re-runs it and changes nothing. Prints ONE line starting
# `[sglang-effort-thinking-budget] applied` on success; any failure exits non-zero.
#
# Usage:  install.sh            apply (or confirm already applied)
#         install.sh --verify   check modules + hooks, change nothing
# Overrides (selftest only): CLUB3090_SGLANG_PKG_DIR, CLUB3090_SITE_DIR, CLUB3090_EFFORT_BUDGET_PY
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

TAG="[sglang-effort-thinking-budget]"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODULE_SRC="${CLUB3090_EFFORT_BUDGET_PY:-/etc/club3090/effort_budget.py}"
GLUE_SRC="$HERE/club3090_sglang_effort_budget.py"
PATCHER="$HERE/patch_sglang.py"

die() {
  echo "$TAG ERROR: $*" >&2
  echo "$TAG see models/qwen3.8-27b/sglang/patches/sglang-effort-thinking-budget/README.md" >&2
  exit 1
}

[ -f "$MODULE_SRC" ] || die "$MODULE_SRC is missing: the compose mounts scripts/lib/effort_budget.py there"
[ -f "$GLUE_SRC" ] && [ -f "$PATCHER" ] || die "patch files missing from $HERE (is the patch dir mounted?)"

# Locate the package WITHOUT importing it: `import sglang` costs ~30 s on every boot.
SGL_PKG="${CLUB3090_SGLANG_PKG_DIR:-$(python3 -c '
import importlib.util, sys
s = importlib.util.find_spec("sglang")
if s is None or not s.submodule_search_locations:
    sys.exit("sglang is not importable here")
print(list(s.submodule_search_locations)[0])')}" || die "cannot locate the sglang package"
SITE="${CLUB3090_SITE_DIR:-$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}" \
  || die "cannot locate site-packages"
[ -d "$SGL_PKG" ] && [ -d "$SITE" ] || die "sglang package ($SGL_PKG) or site-packages ($SITE) is not a directory"

pairs=("$MODULE_SRC:$SITE/club3090_effort_budget.py" "$GLUE_SRC:$SITE/club3090_sglang_effort_budget.py")

# The modules as they will be imported, the map as the server will read it.
check_modules() {
  PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
import club3090_effort_budget as m, club3090_sglang_effort_budget as g
b = m.budgets_from_env()
print(",".join(f"{k}={b[k]}" for k in ("low", "medium", "xhigh") if k in b) or "off")' 2>&1
}

if [ "${1:-}" = "--verify" ]; then
  for p in "${pairs[@]}"; do
    cmp -s "${p%%:*}" "${p#*:}" || die "verify: ${p#*:} is missing or differs from ${p%%:*}"
  done
  map="$(check_modules)" || die "verify: the budget map does not load: $map"
  python3 "$PATCHER" verify --root "$SGL_PKG" || die "verify: hooks not in place"
  echo "$TAG verify OK (map: $map)"
  exit 0
fi

copied=0
for p in "${pairs[@]}"; do
  src="${p%%:*}" dst="${p#*:}"
  cmp -s "$src" "$dst" && continue
  cp -f "$src" "$dst.club3090.tmp" && chmod 0644 "$dst.club3090.tmp" && mv -f "$dst.club3090.tmp" "$dst" \
    || die "cannot write $dst"
  copied=$((copied + 1))
done

map="$(check_modules)" || die "the budget map does not load (CLUB3090_REASONING_EFFORT_BUDGETS): $map"
out="$(python3 "$PATCHER" apply --root "$SGL_PKG" 2>&1)" || die "${out#"$TAG "}"
case "$out" in
  *"already applied"*) hooks="already in place" ;;
  *) hooks="patched 3/3" ;;
esac
echo "$TAG applied (hooks: $hooks; modules: $([ "$copied" -gt 0 ] && echo "$copied refreshed" || echo unchanged); map: $map)"
