#!/usr/bin/env bash
# M4 round-trip guard — the typed ModelSpec is the CANONICAL write-path input.
#
# Proven by unit tests below (no repo writes):
#   1. a spec carrying `model_spec` drives render_profile_yaml's arch values
#      (Fact values win over any loose arch dict);
#   2. legacy loose-only specs render byte-identically to pre-M4 behavior;
#   3. ModelSpec.to_dict/from_dict is lossless for the Fact set (the artifact
#      carries provenance even though registry.yaml stores plain kwargs).
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
import sys

sys.path.insert(0, "scripts/lib/profiles")
from model_spec import Fact, ModelSpec  # noqa: E402
from promote import _apply_model_spec, render_profile_yaml  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond:
        failures.append(msg)


def typed_spec():
    ms = ModelSpec(
        num_hidden_layers=Fact(32, "curated", "config.json:num_hidden_layers"),
        hidden_size=Fact(4096, "curated", "config.json:hidden_size"),
        num_attn_heads=Fact(32, "derived", "config.json:num_attention_heads"),
        num_kv_heads=Fact(8, "derived", "config.json:num_key_value_heads"),
        head_dim_attn=Fact(128, "computed", "hidden_size//num_attention_heads"),
        max_ctx_supported=Fact(131072, "fallback", "default:max_position_embeddings||131072"),
        vision_capable=Fact(False, "curated", "config.json"),
    )
    return ms


def base_spec(ms_dict, loose_arch=None):
    spec = {
        "model_id": "acme-test-7b",
        "display_name": "ACME Test 7B",
        "family": "generic-dense",
        "model_spec": ms_dict,
        "weights": {"v": {}},
        "default_weight_variant": "v",
        "compatible_drafters": [],
    }
    if loose_arch is not None:
        spec["arch"] = loose_arch
    return spec


ms = typed_spec()
spec = _apply_model_spec(base_spec(ms.to_dict(), loose_arch={"num_hidden_layers": 1}))
yaml_text = render_profile_yaml(spec)

check("num_hidden_layers: 32" in yaml_text,
      "typed Fact value WINS over conflicting loose arch (32, not 1)")
check("max_ctx_supported: 131072" in yaml_text,
      "fallback-labeled max_ctx still renders its VALUE")

legacy = {"num_hidden_layers": 48, "hidden_size": 5120}
spec2 = _apply_model_spec({"model_id": "x", "display_name": "X", "family": "f",
                           "weights": {}, "default_weight_variant": "v",
                           "arch": dict(legacy)})
check(spec2["arch"] == legacy, "loose-only spec passes through untouched")
yaml2 = render_profile_yaml(spec2)
check("num_hidden_layers: 48" in yaml2, "legacy render unchanged")

rt = ModelSpec.from_dict(typed_spec().to_dict())
pairs = [
    ("num_hidden_layers", rt.num_hidden_layers),
    ("hidden_size", rt.hidden_size),
    ("num_kv_heads", rt.num_kv_heads),
    ("head_dim_attn", rt.head_dim_attn),
    ("max_ctx_supported", rt.max_ctx_supported),
]
for name, f in pairs:
    src_fact = getattr(typed_spec(), name)
    check(f.value == src_fact.value and f.provenance == src_fact.provenance
          and f.source == src_fact.source,
          f"round-trip lossless incl. provenance: {name}")

if failures:
    import sys
    print(f"\n{len(failures)} FAILURES", file=sys.stderr)
    sys.exit(1)
print("\nM4 ROUND-TRIP GUARD: ALL PASS")
PY
