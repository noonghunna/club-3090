#!/usr/bin/env bash
# Guard (#2028): `catalog.sh register --compose <path>` files the copy under the
# compose's OWN topology and serving name, says which values are placeholders, and
# `update --set max_ctx=…` re-applies the workload rule registration uses.
#
# Before the fix every copy was filed as `<topology from TP>/<quant>/base.yml`. A
# QEngine compose sets TP at boot, so it read as TP 1 and a four-card
# `multi4/qengine-int4/mtp.yml` was recorded as `single/qengine-int4/base.yml`;
# the dry-run printed "(derived)" beside max_ctx 4096 / kv f16 / tp 1, which the
# compose states none of; and correcting max_ctx left workload at fast-chat.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
rc=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
ok()   { echo "  ok   $*"; }
fail() { echo "  FAIL $*"; rc=1; }

CM=qwen3.8-flash-next
QC="models/$CM/qengine/compose"
LCOMP="scripts/lib/profiles-local/composes/$CM/qengine/compose"

R="$TMP/root"
mkdir -p "$R/scripts" "$R/tools" "$R/models" "$R/.github"
cp -r scripts/lib "$R/scripts/"
rm -rf "$R/scripts/lib/profiles-local/"{models.d,composes,engines.d,extends.d,registry.local.json}
cp -r tools/tui-core "$R/tools/"
cp scripts/detect_nvlink.sh "$R/scripts/"
cp -r "models/$CM" "$R/models/"
cp -r .github/PULL_REQUEST_TEMPLATE "$R/.github/"

reg() {   # reg <compose> <slug> [extra…]
  local c="$1" s="$2"; shift 2
  bash scripts/catalog.sh register --compose "$c" --engine qengine --model "$CM" \
    --weights-variant qengine-int4 --slug "$s" --root "$R" -y "$@" 2>&1
}
upd() { bash scripts/catalog.sh update --root "$R" -y "$@" 2>&1; }
entry() {   # entry <slug> -> "compose_path tp max_ctx workload"
  (cd "$R" && python3 -c "
import json
e = json.load(open('scripts/lib/profiles-local/registry.local.json'))['$1']
print(e['compose_path'], e['tp'], e['max_ctx'], e['workload'])")
}

# 1. dry-run: the plan names the compose's own path, and placeholders say so
out="$(reg "$R/$QC/multi4/qengine-int4/mtp.yml" qengine/p-multi4 --dry-run)"; code=$?
row() { printf '%s\n' "$out" | command grep -E "^\[catalog\]   $1 " | head -1; }
if [[ $code -eq 0 && "$(row compose)" == *"compose/multi4/qengine-int4/mtp.yml"*"(given)" ]]; then
  ok "dry-run: compose row is multi4/qengine-int4/mtp.yml (given)"
else fail "dry-run compose row (rc=$code): $(row compose)"; fi
[[ "$(row tp)" == *" 4 "*"(from its directory)" ]] \
  && ok "dry-run: tp 4, read from the multi4/ directory (the compose states none)" \
  || fail "dry-run tp row: $(row tp)"
[[ "$(row max_ctx)" == *"(defaulted)" && "$(row kv_format)" == *"(defaulted)" \
   && "$(row workload)" == *"(defaulted)" ]] \
  && ok "dry-run: max_ctx, kv_format and workload say (defaulted), not (derived)" \
  || fail "dry-run defaulted rows: $(row max_ctx) / $(row kv_format) / $(row workload)"
[[ "$out" == *"defaulted: max_ctx, kv_format"*"catalog.sh update --slug qengine/p-multi4 --set max_ctx="*"--set kv_format="* ]] \
  && ok "dry-run: names the placeholders and the update command that replaces them" \
  || fail "dry-run defaulted hint missing: $(printf '%s\n' "$out" | command grep defaulted:)"
[[ ! -e "$R/scripts/lib/profiles-local/registry.local.json" && ! -d "$R/scripts/lib/profiles-local/composes" ]] \
  && ok "dry-run wrote nothing" || fail "dry-run wrote files"

# 2. the real run: the entry's compose_path is the compose that was read
out="$(reg "$R/$QC/multi4/qengine-int4/mtp.yml" qengine/p-multi4)"; code=$?
want="$LCOMP/multi4/qengine-int4/mtp.yml 4 4096 fast-chat"
got="$(entry qengine/p-multi4 2>&1)"
if [[ $code -eq 0 && "$out" == *"PROMOTE_OK qengine/p-multi4"* && "$got" == "$want" ]]; then
  ok "multi4 compose -> $LCOMP/multi4/qengine-int4/mtp.yml, tp 4"
else fail "multi4 register (rc=$code): got '$got', want '$want' / $(printf '%s\n' "$out" | tail -3)"; fi
command grep -q 'qengine-qwen38-flash-next-multi4' "$R/$LCOMP/multi4/qengine-int4/mtp.yml" \
  && ok "the file at that path is the multi4 compose" || fail "copied compose is not multi4's"

out="$(reg "$R/$QC/dual/qengine-int4/mtp.yml" qengine/p-dual)"; code=$?
got="$(entry qengine/p-dual 2>&1)"
[[ $code -eq 0 && "$got" == "$LCOMP/dual/qengine-int4/mtp.yml 2 4096 fast-chat" ]] \
  && ok "dual compose -> dual/qengine-int4/mtp.yml, tp 2" \
  || fail "dual register (rc=$code): $got / $(printf '%s\n' "$out" | tail -3)"

# 3. the same compose under a second slug keeps its directory, takes the slug's name
out="$(reg "$R/$QC/multi4/qengine-int4/mtp.yml" qengine/p-multi4-b)"; code=$?
got="$(entry qengine/p-multi4-b 2>&1)"
[[ $code -eq 0 && "$got" == "$LCOMP/multi4/qengine-int4/p-multi4-b.yml 4 4096 fast-chat" \
   && "$(entry qengine/p-multi4)" == "$LCOMP/multi4/qengine-int4/mtp.yml "* \
   && -f "$R/$LCOMP/multi4/qengine-int4/mtp.yml" ]] \
  && ok "a second slug of the same compose: multi4/qengine-int4/p-multi4-b.yml, the first is kept" \
  || fail "second slug (rc=$code): $got / $(printf '%s\n' "$out" | tail -3)"

# 4. a compose outside the layout states no topology: TP decides, file is base.yml
cp "$R/$QC/multi4/qengine-int4/mtp.yml" "$R/loose.yml"
out="$(reg "$R/loose.yml" qengine/p-loose)"; code=$?
got="$(entry qengine/p-loose 2>&1)"
[[ $code -eq 0 && "$got" == "$LCOMP/single/qengine-int4/base.yml 1 4096 fast-chat" ]] \
  && ok "a compose outside compose/<topology>/<quant>/ is still filed by TP as base.yml" \
  || fail "loose compose (rc=$code): $got / $(printf '%s\n' "$out" | tail -3)"

# 5. update --set max_ctx re-applies the workload rule, and only to a rule-made value
out="$(upd --slug qengine/p-multi4 --set max_ctx=262144 --dry-run)"; code=$?
[[ $code -eq 0 && "$out" == *"workload: 'fast-chat' -> 'long-ctx-single'"* \
   && "$(entry qengine/p-multi4)" == *" 4096 fast-chat" ]] \
  && ok "update --dry-run shows workload following max_ctx, and writes nothing" \
  || fail "update dry-run (rc=$code): $out"
out="$(upd --slug qengine/p-multi4 --set max_ctx=262144)"; code=$?
[[ $code -eq 0 && "$out" == *"workload follows max_ctx"* \
   && "$(entry qengine/p-multi4)" == *" 262144 long-ctx-single" ]] \
  && ok "max_ctx 4096 -> 262144: workload fast-chat -> long-ctx-single" \
  || fail "update max_ctx up (rc=$code): $(entry qengine/p-multi4) / $out"
out="$(upd --slug qengine/p-multi4 --set max_ctx=8192)"; code=$?
[[ $code -eq 0 && "$(entry qengine/p-multi4)" == *" 8192 fast-chat" ]] \
  && ok "max_ctx 262144 -> 8192: workload back to fast-chat" \
  || fail "update max_ctx down (rc=$code): $(entry qengine/p-multi4) / $out"
out="$(upd --slug qengine/p-dual --set max_ctx=262144 --set workload=tool-heavy)"; code=$?
[[ $code -eq 0 && "$(entry qengine/p-dual)" == *" 262144 tool-heavy" && "$out" != *"workload follows"* ]] \
  && ok "a workload set in the same call wins" \
  || fail "explicit workload (rc=$code): $(entry qengine/p-dual) / $out"
out="$(upd --slug qengine/p-dual --set max_ctx=4096)"; code=$?
[[ $code -eq 0 && "$(entry qengine/p-dual)" == *" 4096 tool-heavy" ]] \
  && ok "a workload the rule did not make is left alone when max_ctx changes" \
  || fail "chosen workload kept (rc=$code): $(entry qengine/p-dual) / $out"

# 6. a copy that now carries a curated compose's own path is refused at export, and
#    the curated file is untouched (export_pr.py refuses any path core already has)
before="$(sha256sum < "$R/$QC/multi4/qengine-int4/mtp.yml")"
echo "{\"model_id\": \"$CM\"}" > "$TMP/spec.json"
out="$(cd "$R" && python3 scripts/lib/profiles/export_pr.py --spec-file "$TMP/spec.json" --check 2>&1)"; code=$?
[[ $code -eq 3 && "$out" == *"$QC/multi4/qengine-int4/mtp.yml already exists in the curated catalog"* \
   && "$(sha256sum < "$R/$QC/multi4/qengine-int4/mtp.yml")" == "$before" ]] \
  && ok "export of a copy at a curated compose's path is refused; the curated file is untouched" \
  || fail "export of a curated-path copy (rc=$code): $(printf '%s\n' "$out" | tail -4)"

[[ $rc -eq 0 ]] && echo "PASS: catalog.sh register keeps the compose's own path (#2028)" \
  || echo "FAIL: test-catalog-register-compose-path"
exit $rc
