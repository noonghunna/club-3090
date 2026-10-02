#!/usr/bin/env python3
"""Profile-aware helpers for scripts/launch.sh.

This is intentionally a narrow CLI bridge: bash keeps the user-facing wizard,
while Python owns profile lookups and fits() diagnostics.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CLUB3090_LOG_LEVEL", "ERROR")

from scripts.lib.profiles.compat import (  # noqa: E402
    TOPOLOGY_ADVISORY,
    FitsResult,
    ProfileError,
    TopologyClass,
    classify_hardware_topology,
    fits,
    load_profiles,
    to_compose_name,
)
from scripts.lib.profiles.compose_registry import get_registry  # noqa: E402


class LaunchCompatError(Exception):
    """User-facing launch compatibility failure."""


def _quiet_compat_logger() -> None:
    logger = logging.getLogger("compat")
    logger.setLevel(logging.ERROR)
    logger.propagate = False


def _normalize_name(name: str) -> str:
    normalized = name.lower()
    for token in ("nvidia", "geforce", "gpu"):
        normalized = normalized.replace(token, " ")
    return " ".join(normalized.replace("_", " ").replace("-", " ").split())


def _hardware_id_from_gpu(name: str, mem_mib: int, sm: float) -> str:
    normalized = _normalize_name(name)
    vram_gb = round(mem_mib / 1024)

    # CMP 170HX: a GA100 (sm_8.0) mining board. Only the ~64 GB "unlocked" boards
    # have a profile. A stock 8 GB board falls through and gets none, rather than an
    # inflated one; and a 64 GB board must be caught HERE, before the sm_8.0
    # fallback below would read it as an NVLink A100.
    if "170hx" in normalized and vram_gb >= 60:
        return "cmp-170hx-64gb"

    aliases = (
        # RTX PRO 6000 Blackwell reports as "RTX PRO 6000 Blackwell" -> normalizes
        # to "rtx pro 6000 blackwell" (PRO before 6000), so match "pro 6000".
        # ("6000" alone is avoided — it would swallow the sm_89 "RTX 6000 Ada".)
        ("rtx pro 6000", "rtx-6000-pro-blackwell"),
        ("pro 6000", "rtx-6000-pro-blackwell"),
        # DGX Spark's GB10 superchip reports as "GB10" / "NVIDIA GB10".
        ("dgx spark", "dgx-spark"),
        ("gb10", "dgx-spark"),
        ("rtx 3090 ti", "rtx-3090-ti"),
        ("3090 ti", "rtx-3090-ti"),
        ("rtx 3090", "rtx-3090"),
        ("3090", "rtx-3090"),
        ("rtx 4090", "rtx-4090"),
        ("4090", "rtx-4090"),
        ("rtx 5090", "rtx-5090"),
        ("5090", "rtx-5090"),
        ("rtx a5000", "rtx-a5000"),
        ("a5000", "rtx-a5000"),
        # #1364: the rtx-a6000 profile (48 GB, sm_8.6) existed with three
        # envelopes.yml rows but NO alias, and the sm_8.6 fallback below returned
        # rtx-3090 for anything >=24 GB -- so the detector could never produce the
        # id and those rows could never fire.
        ("rtx a6000", "rtx-a6000"),
        ("a6000", "rtx-a6000"),
        ("rtx 3060", "rtx-3060-12gb"),
        ("3060", "rtx-3060-12gb"),
        ("a100", "a100-40gb"),
        ("h100", "h100-80gb"),
    )
    for needle, hardware_id in aliases:
        if needle in normalized:
            return hardware_id

    # sm >= 12 Blackwell family, split by SM then VRAM (name aliases above are
    # the primary signal; these are the fallback when the name string is odd):
    #   GB10 / DGX Spark = sm_121 (unified 128 GB)
    #   RTX PRO 6000 Blackwell = sm_120, 96 GB
    #   RTX 5090 = sm_120, 32 GB
    if 12.05 <= sm <= 12.2:
        return "dgx-spark"
    if sm >= 12 and vram_gb >= 64:
        return "rtx-6000-pro-blackwell"
    if sm >= 12 and vram_gb >= 24:
        return "rtx-5090"
    if sm >= 9 and vram_gb >= 80:
        return "h100-80gb"
    if sm >= 8.9 and vram_gb >= 24:
        return "rtx-4090"
    # #1364: sm_8.6 splits by VRAM, largest-first. This branch used to return
    # rtx-3090 for ANY >=24 GB Ampere, i.e. a 48 GB card resolved to the 24 GB
    # profile -- a VRAM-keyed envelope table sitting on a detector that discarded
    # VRAM above the floor, which is exactly the "same arch, more VRAM" case
    # #1360 hit.
    if 8.55 <= sm <= 8.65 and vram_gb >= 44:
        return "rtx-a6000"
    if 8.55 <= sm <= 8.65 and vram_gb >= 24:
        return "rtx-3090"
    if 7.9 <= sm <= 8.1 and vram_gb >= 40:
        return "a100-40gb"
    if 8.55 <= sm <= 8.65 and 11 <= vram_gb <= 13:
        return "rtx-3060-12gb"
    # ⚠️ DELIBERATE UNDER-PROMISE, not an oversight. Cards with no exact profile
    # fall to the largest SMALLER profile in their SM family:
    #     RTX 6000 Ada / L40S (48 GB, sm_8.9) -> rtx-4090 (24 GB)
    #     A100-80GB           (80 GB, sm_8.0) -> a100-40gb
    # That is conservative in the safe direction -- an envelope row sized for the
    # smaller card fits the bigger one; the reverse would not. It costs headroom,
    # never stability. Add a profile (and rows) to claim it; do NOT widen a
    # fallback to a LARGER profile than the card actually has.
    raise LaunchCompatError(
        f"could not map GPU `{name}` ({vram_gb} GB, sm_{sm:g}) to a hardware profile"
    )


def _parse_gpu_specs(value: str, profiles) -> list:
    hardware = []
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            _idx, name, mem_mib, sm = raw.split("|", 3)
        except ValueError as exc:
            raise LaunchCompatError(f"invalid --gpu-spec entry `{raw}`") from exc
        hardware_id = _hardware_id_from_gpu(name, int(mem_mib), float(sm))
        try:
            hardware.append(profiles.hardware[hardware_id])
        except KeyError as exc:
            raise LaunchCompatError(f"hardware profile `{hardware_id}` is not installed") from exc
    if not hardware:
        raise LaunchCompatError("no GPU specs were provided for profile validation")
    return hardware


def _parse_gpu_specs_with_indices(value: str, profiles) -> list[tuple[str, object]]:
    hardware = []
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            idx, name, mem_mib, sm = raw.split("|", 3)
        except ValueError as exc:
            raise LaunchCompatError(f"invalid --gpu-spec entry `{raw}`") from exc
        hardware_id = _hardware_id_from_gpu(name, int(mem_mib), float(sm))
        try:
            hardware.append((idx, profiles.hardware[hardware_id]))
        except KeyError as exc:
            raise LaunchCompatError(f"hardware profile `{hardware_id}` is not installed") from exc
    if not hardware:
        raise LaunchCompatError("no GPU specs were provided for topology classification")
    return hardware


def _engine_family(engine_type: str) -> str:
    return "llamacpp" if engine_type == "llama.cpp" else engine_type


def _entry_objects(entry: dict, profiles):
    drafter = profiles.drafters[entry["drafter"]] if entry.get("drafter") else None
    return (
        profiles.models[entry["model"]],
        profiles.workloads[entry["workload"]],
        profiles.engines[entry["engine"]],
        drafter,
    )


# #1365: `_ENGINE_IMAGE_ENV = {"beellama-local": "BEELLAMA_IMAGE"}` used to live
# here -- a hand-maintained code-side map that listed ONE of the 21 engines, so
# every other non-vLLM engine fell through to a raise. It is superseded by
# `EngineProfile.image_env`, declared by each profile next to its `install.spec`,
# so adding an engine can no longer forget to add it here. vLLM nightlies stay
# special-cased in `resolve_engine_pin` (they pin by SHA, not image tag).


# --- #246 Phase 1 KV-dtype injection: RETIRED 2026-09-21 (#1371) --------------
# `_arch_aware_env` + `ARCH_KV_PILOT_VARIANTS` + `_ARCH_KV_ALLOWED` lived here.
# They upgraded a pilot slug's KV dtype fp8_e5m2 -> fp8_e4m3 on a card whose
# balanced default was e4m3. Removed because the job is DONE, not because it
# broke: ZERO of 138 registry slugs still declare `kv_format: fp8_e5m2`, so the
# allow-map's only source key was orphaned and the function returned {} on every
# card. The migration it automated was completed statically, in the composes.
#
# ⚠️ DO NOT REVIVE IT AS A CARD-KEYED MAP. What decides fp8_e4m3 KV on Ampere is
# not the card class at all -- it is which ATTENTION BACKEND the checkpoint
# routes to. fp8-weights / nvfp4 / bf16 / qwen3-next go to FlashInfer (native fp8
# storage on sm_86, live-validated); gemma-style W4A16 goes to Triton, whose
# fp8e4nv path needs SM89+ and fails at KV-init on the same stack. That rule is
# already encoded, correctly, as `_fp8w_ampere_kv` in compat.py's C5 gate -- and
# it is why the Ampere hardware profiles DELIBERATELY omit fp8_e4m3 from
# `supported_kv_formats` (it keeps gemma rejected). A card-keyed injector is the
# wrong shape for a backend-routing question and would have to fight that gate.
#
# The general lesson, which cost nine months of silence here: a map keyed on a
# value ANOTHER FILE owns (the registry's `kv_format`) goes inert the moment that
# file changes, and inert is indistinguishable from working. See
# docs/DTYPE_MATRIX.md and learnings/ for the measured story.


# --- #246 Phase 2: memory-envelope injection (concurrency-only first pass) ---
# Weights-invariant. Injects MAX_NUM_SEQS from a per-(slug, card-class) ceiling
# in envelopes.yml (validated soak OR computed kv-calc), to spend a bigger
# card's KV pool on concurrency. no-row / user-env-set -> no injection.
# Heterogeneous rigs clamp to the smallest-VRAM card (which is exactly the pool
# vLLM allocates); a 24 GB card in the mix therefore lands on the compose default.
_ENVELOPES_PATH = Path(__file__).with_name("envelopes.yml")


def _load_envelopes() -> dict:
    try:
        import yaml
        doc = yaml.safe_load(_ENVELOPES_PATH.read_text()) or {}
        return doc.get("envelopes") or {}
    except (OSError, ImportError):
        return {}
# #1361: the concurrency knob has a different SPELLING per engine. The quantity
# is the same -- "how many sequences may run at once" -- but injecting vLLM's
# name into an SGLang compose is a silent no-op: the compose reads
# ${MAX_RUNNING_REQUESTS:-N} and never sees MAX_NUM_SEQS, so the row would look
# applied and do nothing.
#
# Keyed by EngineProfile.TYPE, not by engine id. The dialect is a property of the
# engine family, so `type` already carries it: 11 vllm ids collapse to one entry
# and a new vllm-* profile needs no Python edit. An engine family absent from
# this map gets NO injection rather than a guessed name -- a wrong key is worse
# than the compose default, because the default is at least a measured value.
# llama.cpp and exllamav3 are absent deliberately: their composes expose no
# concurrency cap at all (UBATCH_SIZE / KV_TYPE / THREADS / MOE_SPLIT instead).
_ENGINE_TYPE_CONCURRENCY_ENV = {
    "vllm": "MAX_NUM_SEQS",
    "sglang": "MAX_RUNNING_REQUESTS",
}

# #1365: same idea for the memory-fraction floor. vLLM reads
# GPU_MEMORY_UTILIZATION, SGLang reads MEM_FRACTION. Before the image pin became
# total this never surfaced, because sglang slugs raised before any injector ran;
# the #1363 matrix then caught GPU_MEMORY_UTILIZATION reaching 10 sgl (slug,card)
# pairs whose compose does not read it. llama.cpp/exllamav3 have no equivalent
# fraction knob -- absent means no injection, never a guessed name.
_ENGINE_TYPE_MEM_UTIL_ENV = {
    "vllm": "GPU_MEMORY_UTILIZATION",
    "sglang": "MEM_FRACTION",
}


def _engine_type(profiles, entry) -> str | None:
    if not isinstance(entry, dict):
        return None
    try:
        return getattr(profiles.engines[entry.get("engine")], "type", None)
    except (KeyError, AttributeError, TypeError):
        return None


def _concurrency_env_key(profiles, entry: dict | None) -> str | None:
    """Env var this entry's engine family reads for its concurrency cap, or None
    when the family has no such knob (-> no injection)."""
    if not isinstance(entry, dict):
        return None
    try:
        engine = profiles.engines[entry.get("engine")]
    except (KeyError, AttributeError, TypeError):
        return None
    return _ENGINE_TYPE_CONCURRENCY_ENV.get(getattr(engine, "type", None))


def _envelope_env(profiles, variant: str, gpu_spec: str,
                  entry: dict | None = None) -> dict[str, str]:
    """Phase 2 concurrency injection. Empty dict = no injection (the compose's
    own ${<KNOB>:-default} stands)."""
    if not gpu_spec:
        return {}
    env_key = _concurrency_env_key(profiles, entry)
    if not env_key:
        return {}  # unmapped engine -> compose default, never a guessed key
    if os.environ.get(env_key):
        return {}  # explicit user pin always wins
    row = _load_envelopes().get(variant)
    if not row:
        return {}
    try:
        hardware = _parse_gpu_specs(gpu_spec, profiles)
    except LaunchCompatError:
        return {}  # unmapped card -> compose default
    # Heterogeneous rigs: clamp to the SMALLEST-VRAM card. This mirrors vLLM
    # rather than guessing — with TP the KV cache is symmetric-sharded, so the
    # engine sizes the pool to min(free blocks) across ranks: the smallest card
    # already dictates the pool. A smallest-card ceiling is therefore the real
    # ceiling, not a conservative fudge. (Homogeneous rigs collapse to their one
    # class. TP=1 stays safe too: the ceiling fits whichever single card vLLM
    # runs on — all are >= the smallest.) No-op falls out for free when the
    # smallest card has no row: a 24 GB card in the mix -> compose default holds.
    smallest = min(hardware, key=lambda hw: hw.vram_gb)
    card_row = row.get(smallest.id)
    if not isinstance(card_row, dict):
        return {}
    seqs = card_row.get("max_num_seqs")
    default = card_row.get("compose_default")
    # only inject a validated value that actually raises the ceiling
    if not isinstance(seqs, int) or (isinstance(default, int) and seqs <= default):
        return {}
    return {env_key: str(seqs)}


def _mem_util_env(profiles, variant: str, gpu_spec: str,
                  entry: dict | None = None) -> dict[str, str]:
    """Phase 2 memory-fraction safety floor. Injects GPU_MEMORY_UTILIZATION
    DOWNWARD only — when a detected UNIFIED-MEMORY card cannot safely give the
    compose's default fraction (DGX Spark: its LPDDR5X is shared with the Grace
    CPU/OS, so mem_util_safe=0.85 < the 0.92 default — boot at 0.92 would starve
    the OS). It NEVER raises above the tested compose default: a bigger discrete
    card *could* give more, but that touches Cliff-2b margin + boot-OOM, so the
    upward move stays a validated opt-in, not an automatic bump. Empty dict = no
    injection (compose default stands).

    Discrete cards never set the floor (#1516). The RTX 3060's 0.88 predates
    this injector and was never measured as a boot limit, yet it fired here and
    pulled every 0.92 slug on a 3090+3060 rig down to 0.88 — including slugs
    that never touch the 3060, because the spec is every card in the rig, not
    the slug's ranks. Our composes carry their own tuned fraction; a discrete
    card's mem_util_safe stays a fit-estimate input (compat.fits)."""
    if not gpu_spec:
        return {}
    env_key = _ENGINE_TYPE_MEM_UTIL_ENV.get(_engine_type(profiles, entry))
    if not env_key:
        return {}  # engine family has no fraction knob -> compose default
    if os.environ.get(env_key):
        return {}  # explicit user pin always wins
    entry = get_registry().get(variant)
    if not entry:
        return {}
    compose_gmu = entry.get("mem_util")  # the compose's default --gpu-memory-utilization
    if not isinstance(compose_gmu, (int, float)):
        return {}
    try:
        hardware = _parse_gpu_specs(gpu_spec, profiles)
    except LaunchCompatError:
        return {}  # unmapped card -> compose default
    # One GMU applies across every rank, so the safe fraction is the LOWEST
    # unified-memory card's ceiling — such a card in a mixed rig forces the whole
    # rig down.
    ceilings = [hw.mem_util_safe for hw in hardware
                if hw.unified_memory and hw.mem_util_safe is not None]
    if not ceilings:
        return {}
    safe = min(ceilings)
    if safe < compose_gmu:  # downward only — never raise above the tested default
        return {env_key: f"{safe:g}"}
    return {}


_DEEPGEMM_DISABLE_SM = {8.9, 12.0, 12.1}


def _deepgemm_env(profiles, variant: str, entry: dict, gpu_spec: str) -> dict[str, str]:
    """Disable vLLM's DeepGEMM fp8-GEMM path on CONSUMER cards that serve fp8
    weights or ModelOpt NVFP4 weights with FP8 linears. DeepGEMM is built for
    Hopper (sm_90) + datacenter Blackwell
    (sm_100/103). Consumer Blackwell (sm_120/121) hard-fails with 'recipe not
    found' (confirmed on a 5090 — disc #571 guybrush01); Ada (sm_89) never routes
    fp8 through DeepGEMM at all (Marlin/CUTLASS), so injecting 0 there is a
    harmless no-op that pre-empts the same wall for 4090 owners. Hopper/datacenter
    (where DeepGEMM is the fast path) are left untouched. Fires for the whole
    fp8-family weight set — "fp8" AND "fp8-dynamic" (compressed-tensors FP8, e.g.
    agents-a1) — and for "nvfp4" ModelOpt checkpoints, which still route FP8
    attention/linear layers through the same DeepGEMM path. Empty dict = no injection."""
    # #1365: VLLM_USE_DEEP_GEMM is a vLLM env var. Before the image pin became
    # total this never surfaced; the #1363 matrix then caught it reaching 108
    # sgl (slug,card) pairs whose compose does not read it -- a silent no-op
    # accompanied by a boot line claiming the opposite.
    if _engine_type(profiles, entry) != "vllm":
        return {}
    if not gpu_spec:
        return {}
    if os.environ.get("VLLM_USE_DEEP_GEMM"):
        return {}  # explicit user pin always wins
    weights_variant = ((entry or {}).get("weights_variant") or "")
    if not (weights_variant.startswith("fp8") or weights_variant == "nvfp4"):
        return {}  # FP8-family + ModelOpt NVFP4 only; INT4/AWQ/W8A8/bf16 skip.
    try:
        hardware = _parse_gpu_specs(gpu_spec, profiles)
    except LaunchCompatError:
        return {}
    if any(float(hw.sm) in _DEEPGEMM_DISABLE_SM for hw in hardware):
        return {"VLLM_USE_DEEP_GEMM": "0"}
    return {}


def resolve_engine_pin(profiles, engine_id: str) -> dict[str, str]:
    """Resolve EngineProfile.install into compose environment exports."""
    try:
        engine = profiles.engines[engine_id]
    except KeyError as exc:
        raise ProfileError(f"unknown engine profile `{engine_id}`") from exc

    spec = str(engine.install.get("spec", ""))

    # #1365: NOT AN IMAGE PIN -> {} , never a raise. An engine with no image env
    # is a normal, expected state (pip installs; llama-cpp-local, whose two
    # binaries read different vars), and raising here made the whole slug
    # unresolvable -- which is why both launchers gated the entire call behind a
    # `vllm/* || beellama/*` prefix test and 73 of 138 slugs got no hardware
    # injection at all. The image pin and hardware-keyed env injection are
    # separate concerns; only the first can be absent.
    if engine.install.get("method") != "docker_image" or not engine.image_env:
        return {}

    if not spec or any(char.isspace() for char in spec):
        raise ProfileError(
            f"engine {engine_id!r} has an invalid docker image in install.spec: {spec!r}"
        )

    # vLLM nightlies pin by SHA rather than by image tag.
    if engine.type == "vllm" and ":nightly-" in spec:
        sha = spec.rsplit(":nightly-", 1)[1].strip()
        if not sha or any(char.isspace() for char in sha):
            raise ProfileError(
                f"engine {engine_id!r} has an invalid nightly SHA in install.spec: {spec!r}"
            )
        return {"VLLM_NIGHTLY_SHA": sha}

    return {engine.image_env: spec}


def resolve_variant_pin(profiles, variant: str, gpu_spec: str = "") -> dict[str, str]:
    entry = get_registry().get(variant)
    if not entry:
        raise ProfileError(f"unknown compose variant `{variant}`")
    exports = resolve_engine_pin(profiles, entry["engine"])
    # #246: arch-aware env rides the same export channel as the image pin.
    # Only emitted when a gpu_spec is passed (launchers do; the registry-emit
    # baselines join calls without one and sees pins only).
    exports.update(_envelope_env(profiles, variant, gpu_spec, entry))  # Phase 2 concurrency
    exports.update(_mem_util_env(profiles, variant, gpu_spec, entry))  # Phase 2 mem-fraction floor
    exports.update(_deepgemm_env(profiles, variant, entry, gpu_spec))  # fp8w consumer-Blackwell fix
    exports.update(_decode_granularity_env(profiles, entry))     # #809 dLLM decode class
    exports.update(_moe_cache_env(profiles, variant, entry, gpu_spec))  # expert-cache reserve floor
    return exports


# Expert-cache reserve measured on the reference rig (2×RTX 3090, 24 GB, sm 8.6).
# 1536 MiB was the sweep optimum there; 1024 measured ~11% SLOWER despite a larger
# pool and a higher hit rate, because the reserve stopped covering the allocator's
# working room. That regression is the failure mode this floor exists to avoid.
_MOE_RESERVE_REFERENCE_VRAM_GB = 24
_MOE_RESERVE_REFERENCE_MB = 1536
# Fraction of card VRAM used as the floor on cards larger than the reference. 1536
# of 24 GB is 6.25%; keeping the ratio means a bigger card gets proportionally more
# allocator headroom rather than inheriting a number derived on a smaller one.
_MOE_RESERVE_VRAM_FRACTION = 0.0625


def _moe_cache_env(profiles, variant: str, entry: dict, gpu_spec: str) -> dict[str, str]:
    """Expert-cache reserve floor. Injects MOE_RESERVE_MB UPWARD only, and only on
    cards larger than the rig the compose default was tuned on.

    `--moe-cache auto` grants free-minus-reserve, so the reserve is what the
    allocator keeps to absorb the compute buffer's swing between graph shapes
    (measured 1,128 MiB at -ub 4096 / 200K on the reference model) plus CUDA
    context, cudagraphs and fragmentation (~3.2 GB untracked there). Too small and
    the pool wins memory it then fights the allocator for: on 24 GB, 1024 MiB was
    ~11% slower than 1536 while showing a BIGGER pool and a HIGHER hit rate — cache
    counters improve as wall-clock regresses, so this cannot be tuned on hit rate.

    Direction is deliberately one-way. The swing is a property of graph shape, not
    of card capacity, so a larger card does not *need* more reserve; it can simply
    afford more, and erring high costs a few hundred pool slots while erring low
    costs double-digit throughput. Smaller-than-reference cards are left alone: the
    compose default already encodes a measured value, and lowering it on an untested
    card would be guessing in the direction that measurably hurts.

    ⚠️ Evidence scope: 24 GB Ampere only. The scaling on 32/96 GB cards is a safety
    heuristic, NOT a tuned optimum — it keeps the reserve/VRAM ratio the reference
    rig validated. Anyone with a bigger card should sweep on wall-clock and pin
    MOE_RESERVE_MB explicitly; an explicit pin always wins. Empty dict = no
    injection (compose default stands)."""
    if not gpu_spec:
        return {}
    if os.environ.get("MOE_RESERVE_MB"):
        return {}  # explicit user pin always wins
    if not entry.get("moe_cache"):
        return {}  # only composes that actually run the expert cache
    try:
        hardware = _parse_gpu_specs(gpu_spec, profiles)
    except LaunchCompatError:
        return {}  # unmapped card -> compose default
    vrams = [hw.vram_gb for hw in hardware if getattr(hw, "vram_gb", None)]
    if not vrams:
        return {}
    # One reserve applies per device, so size it for the SMALLEST card in the rig:
    # a mixed rig must not hand a 24 GB card a floor derived from a 96 GB sibling.
    smallest = min(vrams)
    if smallest <= _MOE_RESERVE_REFERENCE_VRAM_GB:
        return {}  # at or below the tuned reference -> the measured default stands
    scaled = int(smallest * 1024 * _MOE_RESERVE_VRAM_FRACTION)
    if scaled <= _MOE_RESERVE_REFERENCE_MB:
        return {}
    return {"MOE_RESERVE_MB": str(scaled)}


def _decode_granularity_env(profiles, entry: dict) -> dict[str, str]:
    """#809 — export DECODE_GRANULARITY for a model that declares a non-default
    decode class.

    A block-diffusion (dLLM) model denoises a whole canvas in parallel and the
    endpoint emits ~one chunk per canvas, so `decode_TPS = tokens/(wall - TTFT)`
    divides by a zero-width window. The harness can auto-detect that from the
    measured runs, but auto-detection is a majority vote over a shape — the
    model's own profile is the authority, and declaring it means the operator
    never has to remember the knob.

    NOT gpu_spec-gated (unlike the arch-aware exports): this is a property of the
    MODEL, identical on every card. Emitted only for the non-default value, so no
    other slug's export set changes by a byte.
    """
    model = profiles.models.get(entry["model"])
    gran = getattr(model, "decode_granularity", "token") if model else "token"
    if gran == "token":
        return {}
    return {"DECODE_GRANULARITY": gran}


def _print_env(exports: dict[str, str], fmt: str) -> None:
    if fmt == "value":
        # #1365: an EMPTY pin is now a normal answer (pip engine; or an engine whose
        # composes use more than one image var, so there is nothing to inject), not
        # a raise. Print nothing rather than StopIteration.
        if exports:
            print(next(iter(exports.values())))
    elif fmt == "json":
        import json

        print(json.dumps(exports, sort_keys=True))
    else:
        for key, value in exports.items():
            print(f"{key}={value}")


def _run_fits_for_entry(
    entry: dict,
    profiles,
    hardware: list,
    *,
    tp: int,
    pp: int,
    nvlink_active: bool,
    project_vram: bool,
    include_compose_requirements: bool,
) -> FitsResult:
    model, workload, engine, drafter = _entry_objects(entry, profiles)
    return fits(
        hardware=hardware,
        model=model,
        workload=workload,
        engine=engine,
        drafter=drafter,
        tp=tp,
        pp=pp,
        kv_format=entry["kv_format"],
        max_ctx=entry["max_ctx"],
        max_num_seqs=entry["max_num_seqs"],
        mem_util=entry.get("mem_util"),
        weights_variant=entry["weights_variant"],
        nvlink_active=nvlink_active,
        requires_nvlink=bool(entry.get("requires_nvlink", False)) if include_compose_requirements else False,
        required_engine_features=list(entry.get("required_engine_features", [])) if include_compose_requirements else [],
        required_sm=entry.get("required_sm") if include_compose_requirements else None,
        supported_sm=entry.get("supported_sm") if include_compose_requirements else None,
        project_vram=project_vram,
    )


def _format_reasons(result: FitsResult) -> list[str]:
    return [f"  - {reason}" for reason in result.reasons]


def _print_verbose_pass(label: str, result: FitsResult) -> None:
    diag = result.diagnostics
    passed = ", ".join(diag.get("constraints_passed", [])) or "(none)"
    skipped = ", ".join(diag.get("constraints_skipped", [])) or "(none)"
    if label:
        print(f"[wizard] {label}", file=sys.stderr)
    print(f"         constraints_passed: {passed}", file=sys.stderr)
    print(f"         constraints_skipped: {skipped}", file=sys.stderr)
    print(f"         kv_calc_invoked: {diag.get('kv_calc_invoked')}", file=sys.stderr)
    print(f"         elapsed_ms: {diag.get('elapsed_ms')}", file=sys.stderr)
    if result.kv_projection:
        kv = result.kv_projection
        print(
            "         verdict: "
            f"{kv.get('verdict')} — total {kv.get('total_gb')} GB/card, "
            f"budget {kv.get('budget_gb')} GB",
            file=sys.stderr,
        )
    for note in result.notes:
        print(f"         note: {note}", file=sys.stderr)


def _selected_runtime(tp: int, pp: int, entry: dict, use_runtime_parallelism: bool) -> tuple[int, int]:
    if use_runtime_parallelism:
        return tp, pp
    return int(entry["tp"]), int(entry.get("pp", 1))


def command_filter_candidates(args: argparse.Namespace) -> int:
    _quiet_compat_logger()
    profiles = load_profiles()
    hardware = _parse_gpu_specs(args.gpu_spec, profiles)
    selected = []
    variant_names = [name for name in args.variants.split(",") if name]

    for name in variant_names:
        entry = get_registry().get(name)
        if not entry or entry["model"] != args.model:
            continue
        if args.workload and entry["workload"] != args.workload:
            continue
        if args.drafter != "__unset__":
            desired = None if args.drafter in ("none", "off") else args.drafter
            if entry.get("drafter") != desired:
                continue
        if args.weights_variant and entry["weights_variant"] != args.weights_variant:
            continue

        engine = profiles.engines[entry["engine"]]
        if args.engine:
            if args.engine in ("vllm", "llamacpp"):
                if _engine_family(engine.type) != args.engine:
                    continue
            elif entry["engine"] != args.engine:
                continue
        if args.stable and engine.stability != "stable":
            continue
        if engine.type == "llama.cpp" and len(hardware) != 1:
            continue

        tp, pp = _selected_runtime(args.tp, args.pp, entry, args.use_runtime_parallelism)
        result = _run_fits_for_entry(
            entry,
            profiles,
            hardware,
            tp=tp,
            pp=pp,
            nvlink_active=args.nvlink_active,
            project_vram=False,
            include_compose_requirements=True,
        )
        if result.valid:
            selected.append(name)
        elif args.verbose:
            print(f"[wizard] reject {name}: {'; '.join(result.reasons)}", file=sys.stderr)

    print("\n".join(selected))
    return 0


def command_validate_variant(args: argparse.Namespace) -> int:
    _quiet_compat_logger()
    profiles = load_profiles()
    entry = get_registry().get(args.variant)
    if not entry:
        raise LaunchCompatError(f"unknown compose variant `{args.variant}`")

    hardware = _parse_gpu_specs(args.gpu_spec, profiles)
    tp = args.tp if args.tp > 0 else int(entry["tp"])
    pp = args.pp if args.pp > 0 else int(entry.get("pp", 1))
    model, workload, engine, drafter = _entry_objects(entry, profiles)

    pass1 = _run_fits_for_entry(
        entry,
        profiles,
        hardware,
        tp=tp,
        pp=pp,
        nvlink_active=args.nvlink_active,
        project_vram=args.project_vram,
        include_compose_requirements=False,
    )
    if args.verbose:
        print(
            "[wizard] Pass 1 fits() — "
            f"model={model.id} workload={workload.id} engine={engine.id} "
            f"drafter={drafter.id if drafter else 'none'} tp={tp} pp={pp}",
            file=sys.stderr,
        )
        _print_verbose_pass("", pass1)
    if not pass1.valid:
        print("[launch] ERROR: selected profile combination is invalid:", file=sys.stderr)
        print("\n".join(_format_reasons(pass1)), file=sys.stderr)
        return 2

    resolved = to_compose_name(
        model,
        engine,
        drafter,
        entry["kv_format"],
        tp,
        pp,
        workload=workload,
        weights_variant=entry["weights_variant"],
        nvlink_active=args.nvlink_active,
        max_ctx=entry["max_ctx"],
        max_num_seqs=entry["max_num_seqs"],
    )
    if args.verbose:
        print(f"[wizard] Resolved compose: {resolved or args.variant}", file=sys.stderr)

    pass2 = _run_fits_for_entry(
        entry,
        profiles,
        hardware,
        tp=tp,
        pp=pp,
        nvlink_active=args.nvlink_active,
        project_vram=args.project_vram,
        include_compose_requirements=True,
    )
    if args.verbose:
        features = entry.get("required_engine_features", [])
        print(
            "[wizard] Pass 2 fits() — "
            f"adding requires_nvlink={bool(entry.get('requires_nvlink', False))}, "
            f"required_engine_features={features}",
            file=sys.stderr,
        )
        _print_verbose_pass("", pass2)
    if not pass2.valid:
        print("[launch] ERROR: selected compose requirements are not satisfied:", file=sys.stderr)
        print("\n".join(_format_reasons(pass2)), file=sys.stderr)
        return 2

    return 0


def command_resolve_engine_pin(args: argparse.Namespace) -> int:
    _quiet_compat_logger()
    profiles = load_profiles()
    _print_env(resolve_engine_pin(profiles, args.engine_id), args.format)
    return 0


def command_resolve_variant_pin(args: argparse.Namespace) -> int:
    _quiet_compat_logger()
    profiles = load_profiles()
    _print_env(resolve_variant_pin(profiles, args.variant, args.gpu_spec), args.format)
    return 0


def _hardware_line(index: str, hardware) -> str:
    return f"  GPU {index}: {hardware.display_name} ({hardware.vram_gb:g} GB, sm {hardware.sm:g})"


def _standalone_recommendation(topology: TopologyClass, count: int) -> list[str]:
    if topology == TopologyClass.SINGLE_CARD:
        return [
            "Recommended:",
            "  1. Use the largest single-card compose your model fits.",
            "  2. Add another matched card for TP=2 when long-context concurrency matters.",
        ]
    if topology == TopologyClass.HOMOGENEOUS:
        return [
            "Recommended:",
            f"  1. TP={count} is the default path for matched cards; use the shipped vllm/dual* or multi-card composes.",
            "  2. Estate planner remains useful when you want separate models/endpoints instead of one larger TP instance.",
        ]
    if topology == TopologyClass.VRAM_MATCHED_COMPUTE_MISMATCHED:
        return [
            "Recommended:",
            f"  1. TP={count} works as-is. Compute mismatch means the faster card waits at every NCCL allreduce; effective throughput caps at the slower card's speed (~30% of faster card idle). Full per-card VRAM capacity preserved.",
            "  2. Estate planner — `bash scripts/launch.sh --estate` runs different models per card, each at full speed.",
            "",
            "Not recommended:",
            "  - PP=N: possible as a manual flag flip (`--pipeline-parallel-size N`) on a vllm/dual compose, but no PP compose ships today.",
        ]
    if topology == TopologyClass.VRAM_MISMATCHED:
        return [
            "Recommended:",
            "  1. llama.cpp `--tensor-split` for weighted layer split on mismatched VRAM.",
            "  2. PP=N as a manual vLLM flag flip (`--pipeline-parallel-size N`) if you are deliberately experimenting.",
            "  3. Estate planner — run different models per card or use the largest matched subset.",
            "",
            "Not recommended:",
            "  - TP=N on the full mismatched set: the smaller card caps usable model size and KV headroom.",
        ]
    return [
        "Recommended:",
        "  1. Manual selection. Use the largest matched subset for one model.",
        "  2. Estate planner — put different models on different card subsets.",
    ]


def command_topology(args: argparse.Namespace) -> int:
    _quiet_compat_logger()
    profiles = load_profiles()
    indexed_hardware = _parse_gpu_specs_with_indices(args.gpu_spec, profiles)
    hardware = [item[1] for item in indexed_hardware]
    topology = classify_hardware_topology(hardware)
    advisory = TOPOLOGY_ADVISORY.get(topology)

    if args.format == "wizard":
        if topology in (TopologyClass.SINGLE_CARD, TopologyClass.HOMOGENEOUS):
            return 0
        detected = " + ".join(
            f"1x {hw.display_name} ({hw.vram_gb:g} GB, sm {hw.sm:g})"
            for _idx, hw in indexed_hardware
        )
        print(f"Detected: {detected}")
        print("")
        print(f"Topology: {topology.value}")
        if advisory:
            print(f"  {advisory}")
        print("")
        print("Continue with the selected parallelism if that trade-off is acceptable.")
        return 0

    print("Detected hardware:")
    for idx, hw in indexed_hardware:
        print(_hardware_line(idx, hw))
    print("")
    print(f"Topology class: {topology.value}")
    print("")
    for line in _standalone_recommendation(topology, len(hardware)):
        print(line)
    print("")
    if advisory:
        print("Advisory:")
        print(f"  {advisory}")
        print("")
    print("For details, see docs/MULTI_CARD.md.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile bridge for scripts/launch.sh")
    sub = parser.add_subparsers(dest="command", required=True)

    filter_cmd = sub.add_parser("filter-candidates")
    filter_cmd.add_argument("--variants", required=True)
    filter_cmd.add_argument("--model", required=True)
    filter_cmd.add_argument("--gpu-spec", required=True)
    filter_cmd.add_argument("--tp", type=int, required=True)
    filter_cmd.add_argument("--pp", type=int, required=True)
    filter_cmd.add_argument("--engine", default="")
    filter_cmd.add_argument("--workload", default="")
    filter_cmd.add_argument("--drafter", default="__unset__")
    filter_cmd.add_argument("--weights-variant", default="")
    filter_cmd.add_argument("--stable", action="store_true")
    filter_cmd.add_argument("--use-runtime-parallelism", action="store_true")
    filter_cmd.add_argument("--nvlink-active", action="store_true")
    filter_cmd.add_argument("--verbose", action="store_true")
    filter_cmd.set_defaults(func=command_filter_candidates)

    validate = sub.add_parser("validate-variant")
    validate.add_argument("--variant", required=True)
    validate.add_argument("--gpu-spec", required=True)
    validate.add_argument("--tp", type=int, default=0)
    validate.add_argument("--pp", type=int, default=0)
    validate.add_argument("--project-vram", action=argparse.BooleanOptionalAction, default=True)
    validate.add_argument("--nvlink-active", action="store_true")
    validate.add_argument("--verbose", action="store_true")
    validate.set_defaults(func=command_validate_variant)

    engine_pin = sub.add_parser("resolve-engine-pin")
    engine_pin.add_argument("--engine-id", required=True)
    engine_pin.add_argument("--format", choices=("shell", "json", "value"), default="shell")
    engine_pin.set_defaults(func=command_resolve_engine_pin)

    variant_pin = sub.add_parser("resolve-variant-pin")
    variant_pin.add_argument("--variant", required=True)
    variant_pin.add_argument("--format", choices=("shell", "json", "value"), default="shell")
    # optional: detected GPUs (idx|name|mem_mib|sm;...) — enables the #246
    # arch-aware env for pilot variants; omitted -> pin exports only.
    variant_pin.add_argument("--gpu-spec", dest="gpu_spec", default="")
    variant_pin.set_defaults(func=command_resolve_variant_pin)

    topology = sub.add_parser("topology")
    topology.add_argument("--gpu-spec", required=True)
    topology.add_argument("--format", choices=("standalone", "wizard"), default="standalone")
    topology.set_defaults(func=command_topology)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (LaunchCompatError, ProfileError) as exc:
        print(f"[launch] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
