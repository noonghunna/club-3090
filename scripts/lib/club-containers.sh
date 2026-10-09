#!/usr/bin/env bash
#
# club-containers.sh — THE single source of truth for "is this container ours?"
#
# Companion to engine-kind.sh. That file answers *which engine family* a
# container belongs to; this one answers *which containers exist at all*. They
# are different questions and were failing in different places.
#
# WHY THIS FILE EXISTS
# --------------------
# "Find the running inference container" was re-implemented per script as a
# hardcoded name-prefix list, and every new engine silently fell out of every
# copy. This is the SAME defect that #281 fixed in switch.sh — its teardown used
# a fixed `^(vllm-|llama-cpp-)` regex, missed beellama-/ik-llama-/sglang-
# containers and leaked their VRAM across switches — and it was fixed there by
# deriving the set from the registry (VARIANT_CONTAINER). The other copies were
# never converted:
#
#   report.sh  --filter 'name=vllm-' 'name=llama-cpp-' 'name=beellama-'
#              'name=club3090-' 'name=ik-llama-'      → no sglang-, no exl3
#   health.sh  ^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)
#                                                     → no exl3
#
# So on a rig serving exl3, `report.sh` and `health.sh` reported "no engine
# container running" over a perfectly healthy server — silently, because "none
# found" is indistinguishable from "none there".
#
# ⭐ REGISTRY-DERIVED, so a new engine is covered the moment its slug lands. No
# list to update, and no way to forget one.
#
# Degrades OPEN: if the registry cannot be read (no python3, broken checkout) it
# falls back to the union of the legacy prefixes, so callers are never WORSE off
# than the hardcoded version they replaced.

# ⚠️ PICK THE RIGHT FORM. club_container_re anchors registry names at BOTH ends;
# club_container_re_loose anchors only at the start. A caller that greps a
# COMPOSITE line — `{{.Names}}|{{.Ports}}`, `{{.Names}}\t{{.Image}}…` — must use
# _loose, because a `$`-anchored name can never match there. Getting this wrong is
# silent: the exact arms simply never fire and the matcher quietly degrades to the
# legacy prefix list, i.e. back to the bug. preflight.sh shipped that way for one
# commit. Audit with: for each call site, look at the `docker ps --format` feeding it.
#
# COST: club_container_names shells out to registry-emit.sh --json (~2.5 s cold)
# because the container name is NOT in registry.yaml — it is derived per slug from
# the compose `container_name:`. Cached for the process, and every caller reaches it
# only after cheaper checks have already found something serving, so no launch pays
# it on the empty-rig path.
#
# Repo convention (#779): UTF-8 mode before any python3, so a rig on a real
# non-UTF-8 locale cannot mangle reads/stdout/argv. Defaulted, not forced, so a
# user who deliberately sets PYTHONUTF8=0 keeps control. Guarded by
# test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

# Cache: the registry is static for a process, and callers poll in loops
# (health.sh --watch re-resolves every WATCH_INTERVAL seconds).
_CLUB_CONTAINERS_CACHE=""

# club_container_names — every container name the registry knows, one per line.
club_container_names() {
  if [[ -n "$_CLUB_CONTAINERS_CACHE" ]]; then
    printf '%s\n' "$_CLUB_CONTAINERS_CACHE"
    return 0
  fi
  local root names
  root="${CLUB_ROOT:-${REPO_ROOT:-${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}}"
  names="$(bash "${root}/scripts/lib/registry-emit.sh" --json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit(1)
seen=[]
for v in d.get("variants",[]):
    c=(v.get("container") or "").strip()
    if c and c not in seen: seen.append(c)
print("\n".join(sorted(seen)))
' 2>/dev/null || true)"
  if [[ -z "$names" ]]; then
    return 1                      # caller falls back; do NOT cache a failure
  fi
  _CLUB_CONTAINERS_CACHE="$names"
  printf '%s\n' "$names"
}

# ⚠️ LEGACY FALLBACK ONLY — never the primary path. Kept because a rig without
# python3 must still get the old behaviour rather than nothing. `tabbyapi-` and
# `exl3-` are present here so the fallback is not itself missing an engine, which
# was the original sin.
CLUB_CONTAINER_PREFIX_RE_FALLBACK='^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-|club3090-|tabbyapi-|exl3-|exllamav3-)'

# club_container_re — an anchored alternation matching any known container.
# Registry names are EXACT, so they are anchored at both ends; the legacy
# prefixes stay prefix-anchored. An estate instance renames its container
# (ESTATE_CONTAINER), so the prefix arm is what still catches those.
club_container_re() {
  local names re=""
  if names="$(club_container_names)"; then
    local n
    while IFS= read -r n; do
      [[ -z "$n" ]] && continue
      re+="|^$(sed 's/[][\.*^$(){}?+|/]/\\&/g' <<<"$n")\$"
    done <<<"$names"
  fi
  # Always include the prefix arm: estate//ad-hoc containers are not in the
  # registry by exact name but are still ours.
  printf '%s' "(${CLUB_CONTAINER_PREFIX_RE_FALLBACK#^}${re})"
}

# club_container_re_loose — like club_container_re but anchored only at the
# START. For callers that grep a composite line (soak-test.sh matches against
# `{{.Names}}|{{.Ports}}`, so a `$`-anchored name can never match).
club_container_re_loose() {
  local names re=""
  if names="$(club_container_names)"; then
    local n
    while IFS= read -r n; do
      [[ -z "$n" ]] && continue
      re+="|^$(sed 's/[][\.*^$(){}?+|/]/\\&/g' <<<"$n")"
    done <<<"$names"
  fi
  printf '%s' "(${CLUB_CONTAINER_PREFIX_RE_FALLBACK#^}${re})"
}

# Engine-INTERNAL ports, used to find a serving container by its `docker ps`
# port mapping: vLLM 8000 / llama.cpp 8080 / SGLang 30000 / TabbyAPI (exllamav3)
# 5000. ⚠️ Mirrored in tools/tui-core/club3090_tui_core/detect.py and
# scripts/catalog.sh; test-engine-port-set-drift.sh asserts every copy matches.
CLUB_ENGINE_PORTS_ANY='8000|8080|30000|5000'
# Ports that identify an engine on their own. 5000 and 8080 are not enough by
# themselves — they are also the default ports of unrelated apps (Flask; SearXNG,
# Open WebUI, …) — so a line publishing only those needs EVIDENCE: the container
# is ours by name, or its image names an engine (engine_kind_from_image).
CLUB_ENGINE_PORTS_SELF_EVIDENT='8000|30000'
CLUB_ENGINE_PORTS_NEED_EVIDENCE='8080|5000'

# club_engine_port_lines — filter `name|ports[|image]` lines (docker ps --format
# '{{.Names}}|{{.Ports}}|{{.Image}}') on stdin down to the ones publishing an
# engine port. 8000/30000 qualify on the port alone; 8080/5000 need the evidence
# above. #1360: TabbyAPI listens on 5000, and without it every exl3 server was
# invisible to endpoint autodetection. #1584: 8080 used to qualify on the port
# alone, so with no engine up every serving script on a rig running the stack's
# own SearXNG (8088→8080) picked it as "the inference container". The image arm
# keeps a BYO llama.cpp container (any name, a llama.cpp image) visible. Callers
# that still send two fields get the name arm only.
club_engine_port_lines() {
  declare -F engine_kind_from_image >/dev/null 2>&1 \
    || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/engine-kind.sh"
  local line re name rest ports image
  re="$(club_container_re_loose)"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    name="${line%%|*}"; rest="${line#*|}"; image=""
    if [[ "$rest" == *"|"* ]]; then ports="${rest%%|*}"; image="${rest#*|}"; else ports="$rest"; fi
    if command grep -qE -- "->(${CLUB_ENGINE_PORTS_SELF_EVIDENT})/tcp" <<<"$ports"; then
      printf '%s\n' "$line"
    elif command grep -qE -- "->(${CLUB_ENGINE_PORTS_NEED_EVIDENCE})/tcp" <<<"$ports" \
        && { command grep -qE -- "$re" <<<"$name" \
             || [[ -n "$image" && "$(engine_kind_from_image "$image")" != unknown ]]; }; then
      printf '%s\n' "$line"
    fi
  done
  return 0
}

# club_running_container [fallback_re] — the first RUNNING container that is
# ours, or empty. Ordering is docker's; callers that need a specific one pass
# CONTAINER= explicitly (every caller already supports that).
club_running_container() {
  local re="${1:-}"
  [[ -z "$re" ]] && re="$(club_container_re)"
  docker ps --format '{{.Names}}' 2>/dev/null | command grep -E "$re" | head -1
}
