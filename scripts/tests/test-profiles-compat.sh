#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

run_test() {
    local name="$1"
    local tmp
    tmp="$(mktemp)"
    cat > "$tmp"
    if python3 "$tmp"; then
        echo "PASS: $name"
        rm -f "$tmp"
    else
        echo "FAIL: $name"
        rm -f "$tmp"
        exit 1
    fi
}

run_test "load_profiles parses all profile groups" <<'PY'
from scripts.lib.profiles.compat import load_profiles
p = load_profiles()
assert len(p.hardware) == 11  # +dgx-spark (#576 follow-up), +rtx-a6000 (#948 thread)
assert len(p.models) == 20   # +inkling-small, +qwen3.8-27b, +glm-5.3-flash, +qwen3.8-flash-next
assert len(p.workloads) == 5
assert len(p.engines) == 16   # +llamacpp-club3090-v1.1, +llamacpp-club3090-v1.5
assert len(p.drafters) == 18  # +syvai-qwen38-dflash2, +anbeeld-glm53-dflash2
assert len(p.calibration) == 6
PY

run_test "fits() happy path: Qwen dual on 2x3090" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits(
    hardware=[p.hardware["rtx-3090"], p.hardware["rtx-3090"]],
    model=p.models["qwen3.6-27b"],
    workload=p.workloads["long-ctx-single"],
    engine=p.engines["vllm-nightly-mtp"],
    drafter=p.drafters["qwen-mtp-builtin"],
    tp=2,
    pp=1,
    project_vram=False,
)
assert r.valid, r.reasons
assert r.recommended_kv_format == "turboquant_3bit_nc"
assert r.diagnostics["constraints_skipped"] == ["C12"]
PY

run_test "topology: single card classified" <<'PY'
from scripts.lib.profiles.compat import load_profiles, classify_hardware_topology, TopologyClass
p = load_profiles()
r = classify_hardware_topology([p.hardware["rtx-3090"]])
assert r == TopologyClass.SINGLE_CARD
PY

run_test "topology: 2x3090 classified homogeneous" <<'PY'
from scripts.lib.profiles.compat import load_profiles, classify_hardware_topology, TopologyClass
p = load_profiles()
r = classify_hardware_topology([p.hardware["rtx-3090"], p.hardware["rtx-3090"]])
assert r == TopologyClass.HOMOGENEOUS
PY

run_test "topology: 3090+4090 classified compute-mismatched" <<'PY'
from scripts.lib.profiles.compat import load_profiles, classify_hardware_topology, TopologyClass
p = load_profiles()
r = classify_hardware_topology([p.hardware["rtx-3090"], p.hardware["rtx-4090"]])
assert r == TopologyClass.VRAM_MATCHED_COMPUTE_MISMATCHED
PY

run_test "topology: 3090+3060 classified VRAM-mismatched" <<'PY'
from scripts.lib.profiles.compat import load_profiles, classify_hardware_topology, TopologyClass
p = load_profiles()
r = classify_hardware_topology([p.hardware["rtx-3090"], p.hardware["rtx-3060-12gb"]])
assert r == TopologyClass.VRAM_MISMATCHED
PY

run_test "topology: VRAM cluster wins over mixed compute" <<'PY'
from scripts.lib.profiles.compat import load_profiles, classify_hardware_topology, TopologyClass
p = load_profiles()
r = classify_hardware_topology([p.hardware["rtx-3090"], p.hardware["rtx-3060-12gb"], p.hardware["rtx-4090"]])
assert r == TopologyClass.VRAM_MISMATCHED
PY

run_test "C16 topology advisory emits note for compute mismatch" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits, TopologyClass
p = load_profiles()
r = fits(
    hardware=[p.hardware["rtx-3090"], p.hardware["rtx-4090"]],
    model=p.models["qwen3.6-27b"],
    workload=p.workloads["long-ctx-single"],
    engine=p.engines["vllm-nightly-mtp"],
    drafter=p.drafters["qwen-mtp-builtin"],
    tp=2,
    pp=1,
    project_vram=False,
)
assert r.topology_class == TopologyClass.VRAM_MATCHED_COMPUTE_MISMATCHED
assert "C16" in r.diagnostics["constraints_passed"]
assert any("C16" in n and "vram_matched_compute_mismatched" in n for n in r.notes), r.notes
PY

run_test "C16 topology advisory is silent for homogeneous GPUs" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits, TopologyClass
p = load_profiles()
r = fits(
    hardware=[p.hardware["rtx-3090"], p.hardware["rtx-3090"]],
    model=p.models["qwen3.6-27b"],
    workload=p.workloads["long-ctx-single"],
    engine=p.engines["vllm-nightly-mtp"],
    drafter=p.drafters["qwen-mtp-builtin"],
    tp=2,
    pp=1,
    project_vram=False,
)
assert r.topology_class == TopologyClass.HOMOGENEOUS
assert "C16" in r.diagnostics["constraints_passed"]
assert not any("C16" in n for n in r.notes), r.notes
PY

run_test "C1 card count: world size mismatch rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], tp=2, project_vram=False)
assert not r.valid
assert any(reason.startswith("C1:") for reason in r.reasons), r.reasons
PY

run_test "C2 head divisibility: invalid TP rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
hw = [p.hardware["h100-80gb"]] * 8
r = fits(hw, p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], tp=8, project_vram=False)
assert not r.valid
assert any(reason.startswith("C2:") for reason in r.reasons), r.reasons
PY

run_test "C3 SM floor: low-SM card rejected" <<'PY'
from dataclasses import replace
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
bad_hw = replace(p.hardware["rtx-3090"], id="gtx-1080", sm=6.1)
r = fits([bad_hw], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], kv_format="fp8_e5m2", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C3:") for reason in r.reasons), r.reasons
PY

run_test "C4 engine KV support: llama.cpp rejects bf16 KV" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["llama-cpp-local"], kv_format="bf16", weights_variant="gguf", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C4:") for reason in r.reasons), r.reasons
PY

run_test "C5 hardware KV support: Ampere rejects fp8_e4m3 on the Triton path (gemma)" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
# gemma-4 (W4A16 → Triton, whose fp8e4nv path needs sm_89+) is the Triton-path model:
# fp8_e4m3 KV is correctly REJECTED on Ampere. (Qwen3-Next routes to FlashInfer and is
# now allowed on Ampere — see the positive test below; changed 2026-07-14.)
r = fits([p.hardware["rtx-3090"]], p.models["gemma-4-31b"], p.workloads["long-ctx-single"], p.engines["vllm-gemma-stable"], kv_format="fp8_e4m3", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C5:") for reason in r.reasons), r.reasons
PY

run_test "C5 hardware KV support: Ampere ALLOWS fp8_e4m3 for Qwen3-Next (FlashInfer path)" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
# Qwen3-Next (DeltaNet hybrid attn) routes fp8_e4m3 KV to FlashInfer on sm_86 regardless of
# weight variant — live-verified 2026-07-14 (dual-fast autoround-int4 + 35b-a3b-dual both boot,
# FlashInfer selected, coherent). So C5 must PASS even with autoround-int4 weights on the 3090.
r = fits([p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-stable"], p.drafters["qwen-mtp-builtin"], kv_format="fp8_e4m3", tp=2, weights_variant="autoround-int4", project_vram=False)
assert "C5" not in r.diagnostics["constraints_failed"], r.reasons
PY

run_test "C6 Genesis one-way: TQ3 KV on non-Genesis engine rejected (via C15)" <<'PY'
# Under the TQ3-only Genesis policy, no model declares requires_genesis=true
# (so C6 has no current model-level trigger). Genesis is enforced at the
# *feature* level via C15: requesting turboquant_3bit_nc on an engine that
# doesn't expose it (e.g. vllm-stable-next) fails C15. The previous
# C6 assertion (Qwen on non-Genesis vLLM rejected) no longer holds —
# Qwen 27B with fp8 is valid on non-Genesis engines.
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
# Positive: Qwen 27B + fp8 on non-Genesis engine is now valid.
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-stable-next"], kv_format="fp8_e5m2", tp=1, project_vram=False)
assert r.valid, r.reasons
# Negative: Qwen 27B + TQ3 on non-Genesis engine fails C15.
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-stable-next"], kv_format="turboquant_3bit_nc", tp=1, project_vram=False, required_engine_features=["turboquant_3bit_nc"])
assert not r.valid
assert any(reason.startswith("C15:") for reason in r.reasons), r.reasons
PY

run_test "C7 drafter method: DFlash on MTP-only engine rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], p.drafters["zlab-qwen-dflash"], kv_format="fp8_e5m2", tp=2, project_vram=False)
assert not r.valid
assert any(reason.startswith("C7:") for reason in r.reasons), r.reasons
PY

run_test "C8 drafter model compatibility rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-full"], p.drafters["gemma-it-assistant"], kv_format="fp8_e5m2", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C8:") for reason in r.reasons), r.reasons
PY

run_test "C9 drafter engine type compatibility rejected" <<'PY'
from dataclasses import replace
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
engine = replace(p.engines["vllm-nightly-full"], supported_drafters=p.engines["vllm-nightly-full"].supported_drafters + ("mtp_gguf",))
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], engine, p.drafters["unsloth-mtp-gguf"], kv_format="fp8_e5m2", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C9:") for reason in r.reasons), r.reasons
PY

run_test "C10 model family support rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["gemma-4-31b"], p.workloads["long-ctx-single"], p.engines["llama-cpp-local"], kv_format="q4_0", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C10:") for reason in r.reasons), r.reasons
PY

run_test "C11 model max context rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], p.drafters["qwen-mtp-builtin"], kv_format="fp8_e5m2", tp=2, max_ctx=300000, project_vram=False)
assert not r.valid
assert any(reason.startswith("C11:") for reason in r.reasons), r.reasons
PY

run_test "C12 KV projection fail rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["gemma-4-31b"], p.workloads["fast-chat"], p.engines["vllm-nightly-mtp"], p.drafters["gemma-it-assistant"], kv_format="bf16", tp=1, max_ctx=8192, max_num_seqs=256, mem_util=0.95)
assert not r.valid
assert any(reason.startswith("C12:") for reason in r.reasons), r.reasons
assert r.diagnostics["kv_calc_invoked"] is True
PY

run_test "C12 PP estimate demotes hard failure to advisory" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p.models["gemma-4-31b"], p.workloads["fast-chat"], p.engines["vllm-nightly-mtp"], p.drafters["gemma-it-assistant"], kv_format="bf16", tp=1, pp=2, max_ctx=8192, max_num_seqs=256, mem_util=0.95)
assert r.valid, r.reasons
assert r.kv_projection["confidence"] == "PP_ESTIMATE"
assert r.kv_projection["verdict"] == "TIGHT"
PY

run_test "C12 skipped for non-vLLM engines" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["llama-cpp-local"], kv_format="q4_0", weights_variant="gguf", tp=1)
assert r.valid, r.reasons
assert "C12" in r.diagnostics["constraints_skipped"]
assert r.diagnostics["kv_calc_invoked"] is False
PY

# NOTE: the C13 "NVLink-required compose rejected" scenario was removed with the
# nvlink-* composes (vllm dual-nvlink prune, 2026-05-29). They were the only
# composes with requires_nvlink=True, so the C13/E3 estate-NVLink-gating code is
# now dormant (no compose users) — retained, cleanup tracked as a follow-up.

run_test "C14 explicit unsupported weight variant rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["llama-cpp-local"], kv_format="q4_0", weights_variant="autoround-int4", tp=1, project_vram=False)
assert not r.valid
assert any(reason.startswith("C14:") for reason in r.reasons), r.reasons
PY

run_test "C15 compose-required engine feature rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], p.drafters["qwen-mtp-builtin"], kv_format="fp8_e5m2", tp=2, required_engine_features=["int8_per_token_head"], project_vram=False)
assert not r.valid
assert any(reason.startswith("C15:") for reason in r.reasons), r.reasons
PY

run_test "weight fallthrough: Qwen llama.cpp resolves GGUF" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["llama-cpp-local"], kv_format="q4_0", tp=1, project_vram=False)
assert r.valid, r.reasons
assert r.weights_variant == "gguf"
PY

run_test "helpers expose compatible engines, drafters, KV formats, TP values" <<'PY'
from scripts.lib.profiles.compat import load_profiles, compatible_engines, compatible_drafters, compatible_kv_formats, valid_tp_values
p = load_profiles()
hw = [p.hardware["rtx-3090"], p.hardware["rtx-3090"]]
engines = [e.id for e in compatible_engines(hw, p.models["qwen3.6-27b"], profiles=p)]
assert "vllm-nightly-mtp" in engines
drafters = [d.id for d in compatible_drafters(p.models["qwen3.6-27b"], p.engines["vllm-nightly-mtp"], profiles=p)]
assert drafters == ["qwen-mtp-builtin"]
assert "turboquant_3bit_nc" in compatible_kv_formats(hw, p.engines["vllm-nightly-mtp"])
assert valid_tp_values(p.models["qwen3.6-27b"], 4) == [1, 2, 4]
PY

run_test "to_compose_name strict match resolves dual compose" <<'PY'
from scripts.lib.profiles.compat import load_profiles, to_compose_name
p = load_profiles()
name = to_compose_name(
    p.models["qwen3.6-27b"],
    p.engines["vllm-stable"],
    p.drafters["qwen-mtp-builtin"],
    "fp8_e4m3",
    2,
    1,
    workload=p.workloads["long-ctx-single"],
    weights_variant="autoround-int4",
    max_ctx=262144,
    max_num_seqs=2,
)
assert name == "vllm/dual", name
PY

run_test "FitsResult diagnostics populated on every call" <<'PY'
from scripts.lib.profiles.compat import load_profiles, fits
p = load_profiles()
r = fits([p.hardware["rtx-3090"]], p.models["qwen3.6-27b"], p.workloads["long-ctx-single"], p.engines["vllm-nightly-mtp"], tp=1, project_vram=False)
d = r.diagnostics
assert d["constraints_evaluated"] == [f"C{i}" for i in range(1, 17)]
assert "constraints_passed" in d and "constraints_failed" in d and "constraints_skipped" in d
assert isinstance(d["elapsed_ms"], float)
PY

run_test "canonical scenarios cover every registry tp" <<'PY'
# THE UNIFICATION GUARD (added 2026-08-15).
# CANONICAL_SCENARIOS is a hand-written enumeration; COMPOSE_REGISTRY grows
# independently. When vllm/qwen38-27b-multi8-max (tp=8) landed, the largest
# scenario was 4x -- so the self-test below reported 'no fitting canonical
# scenario' for a slug that diagnose-profile.sh passed at 15/16. The slug was
# fine; the scenario list simply could not describe an 8-card host.
#
# That failure mode is silent and MISLEADING (it reads as a broken compose), so
# assert coverage directly: every distinct tp in the registry must have at least
# one canonical scenario with that many GPUs.
#
# NOTE: diagnose_profile_cli.py deliberately does NOT read CANONICAL_SCENARIOS --
# it synthesises a scenario from the entry's own world size. The two answer
# different questions ('fits some real host' vs 'fits the host it implies'), so
# they are kept consistent by THIS test rather than by merging them.
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
from scripts.lib.profiles.canonical_scenarios import CANONICAL_SCENARIOS

sizes = {len(sc['hardware']) for sc in CANONICAL_SCENARIOS}
needed = {}
for name, e in COMPOSE_REGISTRY.items():
    tp = e.get('tp') if isinstance(e, dict) else getattr(e, 'tp', None)
    if tp:
        needed.setdefault(int(tp), []).append(name)
missing = {tp: slugs for tp, slugs in needed.items() if tp not in sizes}
for tp in sorted(needed):
    print(f"  tp={tp}: {len(needed[tp])} slug(s) -> canonical scenario " + ("MISSING" if tp in missing else "ok"))
assert not missing, (
    'registry uses tp values with no canonical scenario of that size: '
    + '; '.join(f"tp={tp} ({', '.join(sorted(s)[:3])})" for tp, s in sorted(missing.items()))
    + ' -- add a scenario to canonical_scenarios.py'
)
PY
run_test "self-test: all registry entries fit canonical scenarios" <<'PY'
from scripts.lib.profiles.compat import load_profiles, from_compose_name, calibration_status
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
from scripts.lib.profiles.canonical_scenarios import CANONICAL_SCENARIOS
p = load_profiles()
unfit = []
for compose_name in COMPOSE_REGISTRY:
    hit = None
    for sc in CANONICAL_SCENARIOS:
        hardware = [p.hardware[h] for h in sc["hardware"]]
        r = from_compose_name(compose_name, hardware=hardware, nvlink_active=sc["nvlink_active"], profiles=p)
        if r.valid:
            status, row = calibration_status(p, compose_name, hardware, r.effective_max_ctx)
            suffix = f"{status}"
            if row:
                suffix += f", {row.get('source')}"
            print(f"SELFTEST {compose_name} on {sc['name']}: PASS ({suffix})")
            hit = True
            break
    if not hit:
        unfit.append(compose_name)
assert not unfit, f"composes with no fitting canonical scenario: {unfit}"
PY

run_test "estate happy path: two disjoint instances on 4x3090" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [
    InstanceSpec("qwen", "vllm/dual", (0, 1), 8010),
    InstanceSpec("gemma", "vllm/gemma-bf16-mtp", (2, 3), 8030),
]
r = validate_estate(instances, [p.hardware["rtx-3090"]] * 4, p, nvlink_active=False)
assert r.valid, (r.cross_instance_failures, {k: v.reasons for k, v in r.per_instance.items()})
PY

run_test "E1 estate GPU collision rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [
    InstanceSpec("a", "llamacpp/default", (0,), 8020),
    InstanceSpec("b", "llamacpp/default", (0,), 8021),
]
r = validate_estate(instances, [p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p, nvlink_active=False)
assert not r.valid
assert any(msg.startswith("E1:") for msg in r.cross_instance_failures), r.cross_instance_failures
PY

run_test "E4 estate port collision rejected" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [
    InstanceSpec("a", "llamacpp/default", (0,), 8020),
    InstanceSpec("b", "llamacpp/default", (1,), 8020),
]
r = validate_estate(instances, [p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p, nvlink_active=False)
assert not r.valid
assert any(msg.startswith("E4:") for msg in r.cross_instance_failures), r.cross_instance_failures
PY

# (E3 estate NVLink-pairing scenarios removed with the nvlink-* composes —
#  see the C13 note above; the estate-NVLink machinery is dormant, follow-up tracked.)

run_test "estate per-instance failure bubbles up" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [InstanceSpec("bad", "vllm/dual", (0,), 8010)]
r = validate_estate(instances, [p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p, nvlink_active=False)
assert not r.valid
assert not r.per_instance["bad"].valid
PY

run_test "EstateResult diagnostics populated" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
r = validate_estate([InstanceSpec("a", "llamacpp/default", (0,), 8020)], [p.hardware["rtx-3090"]], p, nvlink_active=False)
d = r.diagnostics
assert d["constraints_evaluated"] == ["E1", "E2", "E3", "E4"]
assert "constraints_passed" in d and "constraints_failed" in d
assert isinstance(d["elapsed_ms"], float)
PY

run_test "estate self-test: two llama.cpp instances on 2x3090" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [
    InstanceSpec("llama-a", "llamacpp/default", (0,), 8020),
    InstanceSpec("llama-b", "llamacpp/mtp", (1,), 8021),
]
r = validate_estate(instances, [p.hardware["rtx-3090"], p.hardware["rtx-3090"]], p, nvlink_active=False)
assert r.valid, (r.cross_instance_failures, {k: v.reasons for k, v in r.per_instance.items()})
PY

run_test "estate self-test: Qwen plus Gemma on 4x3090" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [
    InstanceSpec("qwen", "vllm/dual", (0, 1), 8010),
    InstanceSpec("gemma", "vllm/gemma-int8-mtp", (2, 3), 8032),
]
r = validate_estate(instances, [p.hardware["rtx-3090"]] * 4, p, nvlink_active=False)
assert r.valid, (r.cross_instance_failures, {k: v.reasons for k, v in r.per_instance.items()})
PY

run_test "estate self-test: official Google Gemma QAT TP=4 on 4x3090" <<'PY'
from pathlib import Path
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
p = load_profiles()
entry = COMPOSE_REGISTRY["vllm/gemma-31b-multi-google-qat-w4a16"]
assert entry["drafter"] == "gemma-it-assistant", entry  # n=2 wins the measured n=1..4 sweep
compose = Path(entry["compose_path"]).read_text()
assert "${SPEC_N_MAX:-2}" in compose, "production MTP depth drifted from measured n=2"
assert "MTP_ACCEPT_MIN=${MTP_ACCEPT_MIN:-1.8}" in compose, "Gemma-specific AL floor missing"
instances = [
    InstanceSpec("gemma", "vllm/gemma-31b-multi-google-qat-w4a16", (0, 1, 2, 3), 8034),
]
r = validate_estate(instances, [p.hardware["rtx-3090"]] * 4, p, nvlink_active=False)
assert r.valid, (r.cross_instance_failures, {k: v.reasons for k, v in r.per_instance.items()})
PY

run_test "estate self-test: three-instance mix on 6x3090" <<'PY'
from scripts.lib.profiles.compat import load_profiles, InstanceSpec, validate_estate
p = load_profiles()
instances = [
    InstanceSpec("qwen", "vllm/dual", (0, 1), 8010),
    InstanceSpec("gemma", "vllm/gemma-bf16-mtp", (2, 3), 8030),
    InstanceSpec("llama", "llamacpp/default", (4,), 8020),
]
r = validate_estate(instances, [p.hardware["rtx-3090"]] * 6, p, nvlink_active=False)
assert r.valid, (r.cross_instance_failures, {k: v.reasons for k, v in r.per_instance.items()})
PY

run_test "strict keys: every shipped profile group loads under validation" <<'PY'
from scripts.lib.profiles.compat import load_profiles
p = load_profiles()  # raises UnknownProfileKeyError on any unknown key
assert len(p.models) == 20
PY

run_test "strict keys: typo'd top-level model key fails naming file + closest key" <<'PY'
import shutil, tempfile
from pathlib import Path
from scripts.lib.profiles.compat import load_profiles, UnknownProfileKeyError
tmp = Path(tempfile.mkdtemp())
shutil.copytree("scripts/lib/profiles", tmp / "profiles")
(tmp / "profiles/models/typo.yml").write_text("""schema_version: 1
id: typo-model
display_name: Typo
family: qwen
hidden_size: 1
num_hidden_layers: 1
num_attn_heads: 1
num_kv_heads: 1
max_ctx_supported: 1
attention_k_eq_v: false
weights: {default: {path: /x, size_gb: 1, format: gguf, status: verified}}
default_weight_variant: default
valid_tp: [1]
descrition: oops
""")
try:
    load_profiles(tmp / "profiles")
except UnknownProfileKeyError as e:
    msg = str(e)
    assert "models/typo.yml" in msg, msg          # names the file
    assert "descrition" in msg, msg               # names the unexpected key
    assert "description" in msg, msg              # suggests the closest valid key
else:
    raise AssertionError("typo'd top-level key loaded silently")
PY

run_test "strict keys: typo'd weights-variant verify_glob fails with suggestion" <<'PY'
import shutil, tempfile
from pathlib import Path
from scripts.lib.profiles.compat import load_profiles, UnknownProfileKeyError
tmp = Path(tempfile.mkdtemp())
shutil.copytree("scripts/lib/profiles", tmp / "profiles")
(tmp / "profiles/models/typo.yml").write_text("""schema_version: 1
id: typo-model
display_name: Typo
family: qwen
hidden_size: 1
num_hidden_layers: 1
num_attn_heads: 1
num_kv_heads: 1
max_ctx_supported: 1
attention_k_eq_v: false
weights:
  default:
    path: /x
    size_gb: 1
    format: gguf
    status: verified
    verify_glb: '*.gguf'
default_weight_variant: default
valid_tp: [1]
""")
try:
    load_profiles(tmp / "profiles")
except UnknownProfileKeyError as e:
    msg = str(e)
    assert "models/typo.yml" in msg, msg
    assert "weights.default" in msg, msg
    assert "`verify_glb`" in msg and "verify_glob" in msg, msg
else:
    raise AssertionError("typo'd verify_glob loaded silently")
PY

run_test "strict keys: typo'd setup subkey fails with suggestion" <<'PY'
import shutil, tempfile
from pathlib import Path
from scripts.lib.profiles.compat import load_profiles, UnknownProfileKeyError
tmp = Path(tempfile.mkdtemp())
shutil.copytree("scripts/lib/profiles", tmp / "profiles")
(tmp / "profiles/models/typo.yml").write_text("""schema_version: 1
id: typo-model
display_name: Typo
family: qwen
hidden_size: 1
num_hidden_layers: 1
num_attn_heads: 1
num_kv_heads: 1
max_ctx_supported: 1
attention_k_eq_v: false
weights: {default: {path: /x, size_gb: 1, format: gguf, status: verified}}
default_weight_variant: default
valid_tp: [1]
setup:
  primarry: default
""")
try:
    load_profiles(tmp / "profiles")
except UnknownProfileKeyError as e:
    msg = str(e)
    assert "setup: unknown key `primarry`" in msg, msg
    assert "primary" in msg, msg
else:
    raise AssertionError("typo'd setup subkey loaded silently")
PY
echo ""
echo "test-profiles-compat: ok"
