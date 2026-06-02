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
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402


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

    aliases = (
        ("rtx 6000 pro blackwell", "rtx-6000-pro-blackwell"),
        ("6000 pro blackwell", "rtx-6000-pro-blackwell"),
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
        ("rtx 3060", "rtx-3060-12gb"),
        ("3060", "rtx-3060-12gb"),
        ("a100", "a100-40gb"),
        ("h100", "h100-80gb"),
    )
    for needle, hardware_id in aliases:
        if needle in normalized:
            return hardware_id

    if sm >= 12 and vram_gb >= 32:
        return "rtx-5090"
    if sm >= 9 and vram_gb >= 80:
        return "h100-80gb"
    if sm >= 8.9 and vram_gb >= 24:
        return "rtx-4090"
    if 8.55 <= sm <= 8.65 and vram_gb >= 24:
        return "rtx-3090"
    if 7.9 <= sm <= 8.1 and vram_gb >= 40:
        return "a100-40gb"
    if 8.55 <= sm <= 8.65 and 11 <= vram_gb <= 13:
        return "rtx-3060-12gb"
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


# Non-vLLM docker-image engines → the compose env var their image is injected as.
# (vLLM is special-cased above: VLLM_IMAGE / VLLM_NIGHTLY_SHA.)
_ENGINE_IMAGE_ENV = {"beellama-local": "BEELLAMA_IMAGE"}


def resolve_engine_pin(profiles, engine_id: str) -> dict[str, str]:
    """Resolve EngineProfile.install into compose environment exports."""
    try:
        engine = profiles.engines[engine_id]
    except KeyError as exc:
        raise ProfileError(f"unknown engine profile `{engine_id}`") from exc

    spec = str(engine.install.get("spec", ""))
    if engine.install.get("method") != "docker_image":
        raise ProfileError(f"engine {engine_id!r} install.spec is not a docker image: {spec!r}")
    if engine.type == "vllm":
        if ":nightly-" in spec:
            sha = spec.rsplit(":nightly-", 1)[1].strip()
            if not sha or any(char.isspace() for char in sha):
                raise ProfileError(f"engine {engine_id!r} has an invalid nightly SHA in install.spec: {spec!r}")
            return {"VLLM_NIGHTLY_SHA": sha}
        if not spec or any(char.isspace() for char in spec):
            raise ProfileError(f"engine {engine_id!r} has an invalid docker image in install.spec: {spec!r}")
        return {"VLLM_IMAGE": spec}
    # Non-vLLM docker-image engines (e.g. beellama-local): inject a plain
    # <ENGINE>_IMAGE override, mirroring VLLM_IMAGE. The per-compose
    # ${<ENGINE>_IMAGE:-…} literal is then just a fallback for direct `docker compose`.
    env_key = _ENGINE_IMAGE_ENV.get(engine_id)
    if not env_key:
        raise ProfileError(f"engine {engine_id!r} install.spec is not a docker image: {spec!r}")
    if not spec or any(char.isspace() for char in spec):
        raise ProfileError(f"engine {engine_id!r} has an invalid docker image in install.spec: {spec!r}")
    return {env_key: spec}


def resolve_variant_pin(profiles, variant: str) -> dict[str, str]:
    entry = COMPOSE_REGISTRY.get(variant)
    if not entry:
        raise ProfileError(f"unknown compose variant `{variant}`")
    return resolve_engine_pin(profiles, entry["engine"])


def _print_env(exports: dict[str, str], fmt: str) -> None:
    if fmt == "value":
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
        entry = COMPOSE_REGISTRY.get(name)
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
    entry = COMPOSE_REGISTRY.get(args.variant)
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
    _print_env(resolve_variant_pin(profiles, args.variant), args.format)
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
