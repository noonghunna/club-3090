#!/usr/bin/env bash
# END-TO-END lifecycle of a LOCAL-layer model: promote it, prove it is really
# registered, remove it, prove it is really gone.
#
# Why this exists: promote.py could add a local model, but nothing could remove
# one. Its docstring said the rollback was "delete the file(s)" — three of them,
# by hand, and MISSING THE REGISTRY ENTRY is the failure that bites: the files
# are gone while the merged registry still advertises the slug, so every
# launcher offers a config whose compose no longer exists. demote.py closes
# that, and this guard proves BOTH directions actually work rather than just
# exiting 0.
#
# Everything happens in a THROWAWAY root (mktemp -d). The real profiles-local/
# layer is never read or written — a test that promotes into the developer's own
# checkout would leave a slug behind on failure.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail=0
pass=0
ok()   { pass=$((pass+1)); printf 'ok   %s\n' "$1"; }
bad()  { fail=$((fail+1)); printf 'FAIL %s\n' "$1"; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# A throwaway root needs the whole scripts/lib tree: after writing, promote.py
# re-checks itself via `scripts/lib/registry-emit.sh --json`, so a partial copy
# makes it exit 2 AFTER a successful write — which reads as "promote is broken"
# when it is the fixture that is incomplete.
mkdir -p "$TMP/scripts" "$TMP/tools"
cp -r scripts/lib "$TMP/scripts/"
# registry-emit.sh imports the shared VariantRow from tools/tui-core.
cp -r tools/tui-core "$TMP/tools/"

SLUG="local/lifecycle-probe"
MID="lifecycle-probe"
SPEC="$TMP/spec.json"

cat > "$SPEC" <<JSON
{
  "model_id": "$MID",
  "display_name": "Lifecycle Probe",
  "family": "dense",
  "arch": {
    "hidden_size": 2048,
    "num_hidden_layers": 8,
    "num_attn_heads": 16,
    "num_kv_heads": 2,
    "head_dim_attn": 128,
    "max_ctx_supported": 4096,
    "attention_k_eq_v": false
  },
  "weights": {"q4": {"path": "Probe-Q4", "local_subdir": "Probe-Q4", "size_gb": 1.0,
                     "format": "gguf", "status": "incubating",
                     "hf_repo": "org/Probe", "engine": "llama-cpp", "kind": "main",
                     "verify_glob": "*.gguf"}},
  "default_weight_variant": "q4",
  "vision_capable": false,
  "compose": {
    "path": "scripts/lib/profiles-local/composes/$MID/llama-cpp/compose/single/q4/base.yml",
    "content": "# Profile (at-a-glance):\n#   Status:    🐣 Incubating\n#   Caveats:   throwaway test fixture\n# ---\nservices:\n  probe:\n    image: busybox\n"
  },
  "registry_entry": {
    "slug": "$SLUG",
    "kwargs": {
      "model": "$MID",
      "weights_variant": "q4",
      "workload": "fast-chat",
      "engine": "llama-cpp-local",
      "drafter": null,
      "kv_format": "q8_0",
      "tp": 1,
      "max_ctx": 4096,
      "max_num_seqs": 1,
      "mem_util": 0.9,
      "compose_path": "scripts/lib/profiles-local/composes/$MID/llama-cpp/compose/single/q4/base.yml",
      "default_port": 20255,
      "kvcalc_key": "SKIP"
    }
  }
}
JSON

REG="$TMP/scripts/lib/profiles-local/registry.local.json"
PROFILE="$TMP/scripts/lib/profiles-local/models.d/$MID.yml"
COMPOSES="$TMP/scripts/lib/profiles-local/composes/$MID"

# ── 1. PROMOTE ───────────────────────────────────────────────────────────────
if python3 scripts/lib/profiles/promote.py --spec-file "$SPEC" --root "$TMP" \
        --layer local --yes >"$TMP/promote.log" 2>&1; then
  ok "promote.py exits 0"
else
  bad "promote.py failed: $(tail -3 "$TMP/promote.log")"
fi

command grep -q "PROMOTE_OK $SLUG" "$TMP/promote.log" \
  && ok "promote emits PROMOTE_OK" || bad "no PROMOTE_OK marker"

# All THREE writes must land — exiting 0 is not evidence.
[ -f "$PROFILE" ]   && ok "wrote the model profile"   || bad "no profile at $PROFILE"
[ -d "$COMPOSES" ]  && ok "wrote the compose tree"    || bad "no composes at $COMPOSES"
[ -f "$REG" ]       && ok "wrote registry.local.json" || bad "no $REG"

python3 - "$REG" "$SLUG" <<'PY' && ok "slug present in registry.local.json" || bad "slug missing from registry"
import json,sys
raw=json.load(open(sys.argv[1],encoding="utf-8"))
sys.exit(0 if sys.argv[2] in raw else 1)
PY

# The slug must be visible through the MERGED view every launcher reads — the
# thing that actually decides whether the model is usable.
python3 - "$TMP" "$SLUG" <<'PY' && ok "slug visible via get_registry() merged view" || bad "slug NOT in the merged registry"
import sys
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root/"scripts"/"lib"/"profiles"))
import compose_registry as cr
try: reg=cr.get_registry(root)
except TypeError: reg=cr.get_registry()
sys.exit(0 if sys.argv[2] in reg else 1)
PY

# ── 2. DEMOTE --dry-run must change NOTHING ─────────────────────────────────
python3 scripts/lib/profiles/demote.py --slug "$SLUG" --root "$TMP" --dry-run \
  >"$TMP/dry.log" 2>&1 && ok "demote --dry-run exits 0" || bad "demote --dry-run failed"
[ -f "$PROFILE" ] && [ -f "$REG" ] \
  && ok "--dry-run removed nothing" || bad "--dry-run DELETED something"
command grep -q "DEMOTE_OK" "$TMP/dry.log" \
  && bad "--dry-run wrongly emitted DEMOTE_OK" || ok "--dry-run emits no success marker"

# ── 3. DEMOTE ───────────────────────────────────────────────────────────────
if python3 scripts/lib/profiles/demote.py --slug "$SLUG" --root "$TMP" --yes \
        >"$TMP/demote.log" 2>&1; then
  ok "demote.py exits 0"
else
  bad "demote.py failed: $(tail -3 "$TMP/demote.log")"
fi
command grep -q "DEMOTE_OK $SLUG" "$TMP/demote.log" \
  && ok "demote emits DEMOTE_OK" || bad "no DEMOTE_OK marker"

# All three must be gone — and the registry entry is the one that matters most.
[ ! -f "$PROFILE" ]  && ok "profile removed"       || bad "profile still at $PROFILE"
[ ! -d "$COMPOSES" ] && ok "compose tree removed"  || bad "composes still at $COMPOSES"

python3 - "$TMP" "$SLUG" <<'PY' && ok "slug gone from the merged registry" || bad "DANGLING slug still in the merged registry"
import sys
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root/"scripts"/"lib"/"profiles"))
import compose_registry as cr
try: reg=cr.get_registry(root)
except TypeError: reg=cr.get_registry()
sys.exit(1 if sys.argv[2] in reg else 0)
PY

# ── 4. Refusals — nothing removed, non-zero exit ────────────────────────────
python3 scripts/lib/profiles/demote.py --slug "$SLUG" --root "$TMP" --yes >/dev/null 2>&1 \
  && bad "removing an absent slug should refuse" || ok "absent slug refused"

python3 scripts/lib/profiles/demote.py --slug "vllm/dual" --root "$TMP" --yes >"$TMP/core.log" 2>&1 \
  && bad "a CORE slug must never be removable" || ok "core slug refused"
command grep -qi "not a local slug" "$TMP/core.log" \
  && ok "core refusal names the reason" || bad "core refusal message unclear"

# ── 5. Round-trip: promote again after a removal ────────────────────────────
python3 scripts/lib/profiles/promote.py --spec-file "$SPEC" --root "$TMP" \
  --layer local --yes >/dev/null 2>&1 \
  && ok "re-promote after demote succeeds (removal was complete)" \
  || bad "re-promote FAILED — demote left a collision behind"

printf '\n%s: %d passed, %d failed\n' "$(basename "$0")" "$pass" "$fail"
[ "$fail" -eq 0 ]
