#!/usr/bin/env bash
# Guard: every copy of the engine-internal port set, and of the engine name
# prefixes, must agree — and ports 5000 / 8080 must only ever admit evidenced
# engines: OUR containers by name, or an engine image (#1584).
#
# #1360: TabbyAPI (exllamav3) listens on 5000 inside its container and our
# composes name it `tabbyapi-…`. Six hand-copied lists of "engine ports"
# (8000/8080/30000) and the c3 prefix list all predated exllamav3, so an exl3
# server was invisible to endpoint autodetection, the verify scripts, report.sh,
# soak-test.sh and c3. The shell helpers had already learned `tabbyapi-`; the
# TUI's copy had not. Drift between copies is the defect this guard pins.
#
# Checks:
#   1. the canonical shell set (CLUB_ENGINE_PORTS_ANY) == detect.py's
#      ENGINE_INTERNAL_PORTS == both detect.py port regexes == catalog.sh's
#      _DETECT_PORTS == the endpoint loops in report.sh and soak-test.sh;
#   2. detect.py's ENGINE_PREFIXES covers every engine prefix in the shell
#      fallback (CLUB_CONTAINER_PREFIX_RE_FALLBACK minus the estate `club3090-`);
#   3. club_engine_port_lines: an exl3 container on 5000 is admitted, an
#      unrelated app on 5000 is NOT (the negative control), 8000 lines behave
#      exactly as before. #1584: 8080 took the 5000 rule — SearXNG / Open WebUI
#      on 8080 are NOT admitted, our llama-cpp- containers and a BYO llama.cpp
#      image are. The evidence rule also admits a BYO TabbyAPI image on 5000,
#      deliberately: #1360 was name-only, and an engine image is the same
#      strength of evidence the 8080 rule needs.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
rc=0
ok()  { printf '  ok   %s\n' "$1"; }
bad() { printf '  FAIL %s\n' "$1"; rc=1; }

# Every digit run in the input, sorted and de-duplicated (tolerates quotes,
# commas, parens, pipes and a loop's trailing `;`).
norm() { command grep -oE '[0-9]+' | sort -n -u | tr '\n' ' ' | sed 's/ $//'; }

# shellcheck source=../lib/club-containers.sh
source scripts/lib/club-containers.sh
canonical="$(printf '%s' "$CLUB_ENGINE_PORTS_ANY" | norm)"
[[ -n "$canonical" ]] && ok "canonical shell set: {$canonical}" || bad "CLUB_ENGINE_PORTS_ANY is empty"
[[ " $canonical " == *" 5000 "* ]] && ok "canonical set includes TabbyAPI's 5000" || bad "canonical set lacks 5000"

check_same() {  # check_same <label> <ports-string>
  local got; got="$(printf '%s' "$2" | norm)"
  if [[ "$got" == "$canonical" ]]; then ok "$1 matches: {$got}"; else bad "$1 is {$got}, expected {$canonical}"; fi
}

py_detect="$(python3 - <<'PY'
import re, pathlib
s = pathlib.Path("tools/tui-core/club3090_tui_core/detect.py").read_text()
ports = re.search(r'ENGINE_INTERNAL_PORTS\s*=\s*\{([^}]*)\}', s).group(1)
print("SET", " ".join(re.findall(r'"(\d+)"', ports)))
for name in ("PORT_MAP_RE", "PORT_MAP_BROAD_RE"):
    block = re.search(name + r'\s*=\s*re\.compile\(\s*r"([^"]*)"', s).group(1)
    print(name, block.split("->(")[1].split(")")[0])
cls = re.search(r'def _classify_engine\(.*?return \{(.*?)\}', s, re.S).group(1)
print("CLASSIFY", " ".join(re.findall(r'"(\d+)"\s*:', cls)))
PY
)"
check_same "detect.py ENGINE_INTERNAL_PORTS" "$(sed -n 's/^SET //p' <<<"$py_detect")"
check_same "detect.py PORT_MAP_RE" "$(sed -n 's/^PORT_MAP_RE //p' <<<"$py_detect")"
check_same "detect.py PORT_MAP_BROAD_RE" "$(sed -n 's/^PORT_MAP_BROAD_RE //p' <<<"$py_detect")"
check_same "detect.py _classify_engine keys" "$(sed -n 's/^CLASSIFY //p' <<<"$py_detect")"
check_same "catalog.sh _DETECT_PORTS" "$(command grep -m1 '^_DETECT_PORTS' scripts/catalog.sh)"
check_same "report.sh endpoint loop" "$(command grep -m1 -E '^\s*for internal in' scripts/report.sh | sed 's/#.*//')"
check_same "soak-test.sh endpoint loop" "$(command grep -m1 -E '^\s*for internal in' scripts/soak-test.sh | sed 's/#.*//')"

# 2. prefixes
missing="$(python3 - <<'PY'
import re, pathlib
s = pathlib.Path("tools/tui-core/club3090_tui_core/detect.py").read_text()
tui = set(re.search(r'ENGINE_PREFIXES\s*=\s*re\.compile\(r"\^\(([^)]*)\)"', s).group(1).split("|"))
sh = pathlib.Path("scripts/lib/club-containers.sh").read_text()
fb = re.search(r"CLUB_CONTAINER_PREFIX_RE_FALLBACK='\^\(([^)]*)\)'", sh).group(1).split("|")
engine_prefixes = {p for p in fb if p != "club3090-"}   # estate, classified separately in c3
print(" ".join(sorted(engine_prefixes - tui)))
PY
)"
[[ -z "$missing" ]] && ok "detect.py ENGINE_PREFIXES covers every shell engine prefix" \
  || bad "detect.py ENGINE_PREFIXES lacks: $missing"

# 3. the 5000 guard, exercised on synthetic docker ps lines
out="$(printf '%s\n' \
  "vllm-qwen36-27b|0.0.0.0:8020->8000/tcp" \
  "tabbyapi-qwen38-flash-next-exl3-405|0.0.0.0:8181->5000/tcp" \
  "exl3-custom|127.0.0.1:9999->5000/tcp" \
  "some-flask-app|0.0.0.0:5000->5000/tcp" \
  "qdrant|0.0.0.0:6333->6333/tcp" \
  "searxng|0.0.0.0:8088->8080/tcp, [::]:8088->8080/tcp|searxng/searxng:latest" \
  "open-webui|0.0.0.0:8080->8080/tcp|ghcr.io/open-webui/open-webui:main" \
  "my-llm|0.0.0.0:8099->8080/tcp|ghcr.io/ggml-org/llama.cpp:server-cuda" \
  "llama-cpp-qwen38|0.0.0.0:8030->8080/tcp|example/custom:1" \
  "my-tabby|0.0.0.0:8182->5000/tcp|ghcr.io/theroyallab/tabbyapi:latest" | club_engine_port_lines)"
command grep -q '^tabbyapi-qwen38-flash-next-exl3-405|' <<<"$out" && ok "exl3 container on 5000 admitted" || bad "exl3 container on 5000 NOT admitted"
command grep -q '^exl3-custom|' <<<"$out" && ok "exl3- prefixed container on 5000 admitted" || bad "exl3- container NOT admitted"
command grep -q '^vllm-qwen36-27b|' <<<"$out" && ok "vLLM on 8000 still admitted" || bad "vLLM on 8000 dropped"
! command grep -q '^some-flask-app|' <<<"$out" && ok "unrelated app on 5000 NOT admitted (negative control)" || bad "unrelated app on 5000 was admitted"
! command grep -q '^qdrant|' <<<"$out" && ok "non-engine port never admitted" || bad "qdrant admitted"
# #1584: 8080 needs the same evidence as 5000 — ours by name, or an engine image.
! command grep -q '^searxng|' <<<"$out" && ok "SearXNG on 8080 NOT admitted (negative control)" || bad "SearXNG on 8080 was admitted"
! command grep -q '^open-webui|' <<<"$out" && ok "Open WebUI on 8080 NOT admitted (negative control)" || bad "Open WebUI on 8080 was admitted"
command grep -q '^my-llm|' <<<"$out" && ok "BYO llama.cpp image on 8080 admitted" || bad "BYO llama.cpp image on 8080 NOT admitted"
command grep -q '^llama-cpp-qwen38|' <<<"$out" && ok "our llama-cpp- container on 8080 admitted, any image" || bad "our llama-cpp- container on 8080 NOT admitted"
command grep -q '^my-tabby|' <<<"$out" && ok "BYO TabbyAPI image on 5000 admitted" || bad "BYO TabbyAPI image on 5000 NOT admitted"

if [[ $rc -eq 0 ]]; then echo "PASS: engine port/prefix sets agree and 5000/8080 admit only evidenced engines"; else echo "FAIL: engine port/prefix drift (see above)"; fi
exit $rc
