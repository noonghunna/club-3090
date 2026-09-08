#!/usr/bin/env bash
# Guard: registering a slug c3 cannot SEE must say so at registration time.
#
# The estate layer classifies engine containers by NAME PREFIX and by the
# CONTAINER-SIDE port (club3090_tui_core.detect). Those are heuristics we own,
# and they predate the local layer: a user's own engine cannot be expected to
# satisfy them. When it does not, everything looks fine — the slug registers, the
# weights resolve, the model serves — and c3 reports "○ not reachable" while the
# GPU bars show the model plainly loaded. That happened for real on 2026-09-08.
#
# So registration WARNS (never refuses — the registration is correct; our
# detector is the narrow part), and this guard pins three things:
#   1. an undetectable container NAME warns;
#   2. an undetectable container-side PORT warns;
#   3. a conforming compose does NOT warn (the negative control — without it the
#      warning could fire always and still "pass");
# plus an ANTI-DRIFT check: catalog.sh duplicates detect.py's prefixes/ports
# because it must run on the launcher's no-extra-deps path, so if detect.py ever
# changes, this guard fails rather than leaving the warning quietly wrong.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
rc=0

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/tools" "$TMP/w"
cp -r scripts/lib "$TMP/scripts/"; cp -r tools/tui-core "$TMP/tools/"
cat > "$TMP/w/config.json" <<'JSON'
{"model_type":"qwen","hidden_size":64,"num_hidden_layers":2,
 "num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":4096}
JSON

mk() {  # mk <file> <container_name> <internal_port>
  cat > "$1" <<YML
services:
  probe:
    image: ghcr.io/someone/their-runtime:locked
    container_name: $2
    ports:
      - "\${BIND_HOST:-0.0.0.0}:\${ESTATE_PORT:-\${PORT:-20272}}:$3"
    environment:
      MAX_MODEL_LEN: "\${MAX_MODEL_LEN:-32768}"
    command: ["/model"]
YML
}

reg() {  # reg <compose> <model> -> stdout+stderr
  scripts/catalog.sh register --compose "$1" --engine their-vllm --engine-type vllm \
    --model "$2" --weights "$TMP/w" --workload fast-chat --root "$TMP" --dry-run -y 2>&1
}

mk "$TMP/bad-name.yml" "qwen38-flash-next-ple" 8000
out="$(reg "$TMP/bad-name.yml" p-badname)"
command grep -q "WILL NOT SEE IT SERVING" <<<"$out" && command grep -q "container_name" <<<"$out" \
  && echo "  ok   an unprefixed container name warns" \
  || { echo "  FAIL an unprefixed container name did NOT warn"; rc=1; }

mk "$TMP/bad-port.yml" "vllm-probe" 20272
out="$(reg "$TMP/bad-port.yml" p-badport)"
command grep -q "container-side port" <<<"$out" \
  && echo "  ok   an undetectable container-side port warns" \
  || { echo "  FAIL an undetectable container-side port did NOT warn"; rc=1; }

# NEGATIVE CONTROL — a conforming compose must stay silent, or the checks above
# would pass for every input and prove nothing.
mk "$TMP/good.yml" "vllm-probe" 8000
out="$(reg "$TMP/good.yml" p-good)"
if command grep -q "WILL NOT SEE IT SERVING" <<<"$out"; then
  echo "  FAIL a CONFORMING compose warned — the check fires unconditionally"; rc=1
else
  echo "  ok   a conforming compose does not warn"
fi

# ANTI-DRIFT — catalog.sh's copy must still match detect.py.
python3 - <<'PY' || rc=1
import re, sys, pathlib
det = pathlib.Path("tools/tui-core/club3090_tui_core/detect.py").read_text(encoding="utf-8")
cat = pathlib.Path("scripts/catalog.sh").read_text(encoding="utf-8")
m = re.search(r"ENGINE_PREFIXES\s*=\s*re\.compile\(r\"\^\(([^)]*)\)\"\)", det)
n = re.search(r'_DETECT_PREFIXES\s*=\s*\(([^)]*)\)', cat)
if not m or not n:
    print("  FAIL could not read the prefix lists from both files"); sys.exit(1)
det_p = set(x for x in m.group(1).split("|") if x)
cat_p = set(re.findall(r'"([^"]+)"', n.group(1)))
if det_p != cat_p:
    print(f"  FAIL prefix DRIFT — detect.py={sorted(det_p)} catalog.sh={sorted(cat_p)}")
    sys.exit(1)
mp = re.search(r"PORT_MAP_RE\s*=\s*re\.compile\(\s*\n?\s*r\"[^\"]*\((\d+(?:\|\d+)*)\)", det)
np_ = re.search(r'_DETECT_PORTS\s*=\s*\(([^)]*)\)', cat)
if not mp or not np_:
    print("  FAIL could not read the port lists from both files"); sys.exit(1)
det_ports = set(mp.group(1).split("|"))
cat_ports = set(re.findall(r'"(\d+)"', np_.group(1)))
if det_ports != cat_ports:
    print(f"  FAIL port DRIFT — detect.py={sorted(det_ports)} catalog.sh={sorted(cat_ports)}")
    sys.exit(1)
print("  ok   catalog.sh's copy still matches detect.py (no drift)")
PY

[[ "$rc" == "0" ]] && echo "PASS: registration warns when c3 could not see the slug" \
                   || echo "FAIL: detectability warning regression"
exit "$rc"
