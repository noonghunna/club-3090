#!/usr/bin/env bash
# test-envelopes — guards for scripts/lib/profiles/envelopes.yml + its launcher
# injection (#246 Phase 2, concurrency-only first pass).
#
# REDs on: schema violations · unknown slugs · unknown card-class ids · a row
# missing its `validated` block (born-from-measurement discipline) · a broken
# injection contract. An EMPTY envelopes file is valid (rows land from probes).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
HELPER="scripts/lib/profiles/launch_compat.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

# --- 1. schema (pure python) --------------------------------------------------
python3 - <<'PY'
import sys
from pathlib import Path
import yaml
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402
from scripts.lib.profiles.compat import load_profiles  # noqa: E402

doc = yaml.safe_load(Path("scripts/lib/profiles/envelopes.yml").read_text())
assert doc.get("schema_version") == 1, "schema_version must be 1"
rows = doc.get("envelopes") or {}
cards = set(load_profiles().hardware)
errors = []
for slug, byc in rows.items():
    if slug not in COMPOSE_REGISTRY:
        errors.append(f"envelopes[{slug}]: unknown registry slug"); continue
    if not isinstance(byc, dict):
        errors.append(f"envelopes[{slug}]: must be a card-class map"); continue
    for card, row in byc.items():
        w = f"envelopes[{slug}][{card}]"
        if card not in cards:
            errors.append(f"{w}: unknown hardware-profile id"); continue
        if not isinstance(row.get("max_num_seqs"), int):
            errors.append(f"{w}.max_num_seqs: required int")
        if "compose_default" in row and not isinstance(row["compose_default"], int):
            errors.append(f"{w}.compose_default: int")
        # born-from-measurement: a value MUST cite a validated soak
        if not isinstance(row.get("validated"), dict) or not row["validated"]:
            errors.append(f"{w}.validated: required {{concurrency_soak, ...}} — no guessed rows")
if errors:
    print("test-envelopes: FAIL", file=sys.stderr)
    for e in errors: print(f"  ✗ {e}", file=sys.stderr)
    sys.exit(1)
print(f"  ✓ schema ({len(rows)} slug rows; empty is valid)")
PY

# --- 2. injection contract (fixture: temp envelopes with a 5090 row) ----------
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
REAL="scripts/lib/profiles/envelopes.yml"
BACKUP="$TMP/backup.yml"; cp "$REAL" "$BACKUP"
REAL_SUM="$(sha256sum "$REAL" | cut -d' ' -f1)"

cat > "$REAL" <<'EOF'
schema_version: 1
envelopes:
  vllm/dual:
    rtx-5090:
      max_num_seqs: 4
      compose_default: 2
      validated: { concurrency_soak: "4 @262K, 0-growth" }
EOF

pin() { python3 "$HELPER" resolve-variant-pin --variant "$1" --format shell --gpu-spec "$2" 2>/dev/null; }
S5090="0|RTX 5090|32607|12.0;1|RTX 5090|32607|12.0"
S3090="0|RTX 3090|24576|8.6;1|RTX 3090|24576|8.6"
HET="0|RTX 5090|32607|12.0;1|RTX 3090|24576|8.6"

grep -q "MAX_NUM_SEQS=4" <(pin vllm/dual "$S5090") || fail "5090 with a row must inject MAX_NUM_SEQS=4"
grep -q "MAX_NUM_SEQS" <(pin vllm/dual "$S3090") && fail "3090 (no row) must NOT inject"
grep -q "MAX_NUM_SEQS" <(pin vllm/dual "$HET") && fail "heterogeneous rig must NOT inject"
grep -q "MAX_NUM_SEQS" <(MAX_NUM_SEQS=9 pin vllm/dual "$S5090" | grep "MAX_NUM_SEQS=4") && fail "user env must win"
# value at/below compose_default must not fire
cat > "$REAL" <<'EOF'
schema_version: 1
envelopes:
  vllm/dual:
    rtx-5090: { max_num_seqs: 2, compose_default: 2, validated: { concurrency_soak: "x" } }
EOF
grep -q "MAX_NUM_SEQS" <(pin vllm/dual "$S5090") && fail "value == compose_default must NOT inject (no gain)"
echo "  ✓ injection contract (inject · no-row · heterogeneous · user-env · no-gain)"

cp "$BACKUP" "$REAL"
[[ "$(sha256sum "$REAL" | cut -d' ' -f1)" == "$REAL_SUM" ]] || fail "real envelopes.yml not restored"

echo "test-envelopes: ok"
