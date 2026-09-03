#!/usr/bin/env bash
# Gate: every LOCAL litellm gateway entry's api_base port resolves to a real
# registry default_port.
#
# Catches the stale/typo'd/removed-slug port class — the reverse of the #1062
# missing-entry gap. When a slug is renamed, repointed, or dropped, this fails if
# its litellm route still points at the old port. Cloud entries (external https
# api_base — e.g. DashScope) are exempt; they don't map to a local slug.
#
# ⚠️ Scope, deliberately: this does NOT assert every model HAS a gateway entry.
# That's the harder half (#1062) — the canonical port is scene-dependent and not
# every model belongs on the LAN gateway, so a "every model -> entry" gate would
# be a false-positive magnet. The real fix is to DERIVE the litellm config from
# the registry (kills the drift class); this gate guards the config we hand-write
# until then (see #1078). Inactive-scene routes (ports currently closed) are fine — they must
# still map to a real registry default_port, which is exactly what this checks.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CFG="${LITELLM_CONFIG:-services/litellm/config.yaml}"

python3 - "$CFG" <<'PY'
import re, sys
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY as R

cfg_path = sys.argv[1]
valid = {e.get("default_port") for e in R.values() if e.get("default_port")}

model = None
local = []          # (model_name, port) for entries pointing at the local host
for ln in open(cfg_path, encoding="utf-8"):
    m = re.match(r"\s*-\s*model_name:\s*(\S+)", ln)
    if m:
        model = m.group(1)
        continue
    a = re.search(
        r"api_base:\s*https?://(?:host\.docker\.internal|localhost|127\.0\.0\.1):(\d+)",
        ln,
    )
    if a and model:
        local.append((model, int(a.group(1))))

bad = [(m, p) for m, p in local if p not in valid]
if bad:
    print("FAIL: litellm LOCAL entries whose port matches no registry default_port:")
    for m, p in bad:
        print(f"    {m} -> :{p}")
    print("Fix: repoint api_base to the model's registry default_port, or remove the")
    print("     stale entry. (A *missing* entry is the other half — see #1078.)")
    sys.exit(1)

print(f"OK: all {len(local)} local litellm entries resolve to real registry "
      f"default_ports ({len(valid)} ports in the registry).")
PY
