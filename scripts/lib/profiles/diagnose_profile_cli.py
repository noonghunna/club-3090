#!/usr/bin/env python3
"""Per-compose and free-form profile diagnostics for club-3090."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CLUB3090_LOG_LEVEL", "ERROR")

from scripts.lib.profiles.compat import (  # noqa: E402
    FitsResult,
    ProfileError,
    calibration_status,
    fits,
    from_compose_name,
    load_profiles,
    to_compose_name,
)
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402
from scripts.lib.profiles.launch_compat import resolve_engine_pin  # noqa: E402


class DiagnoseProfileError(Exception):
    """User-facing diagnose-profile failure."""


COMPOSE_ALIASES = {
    "gemma-dual-int8": "vllm/gemma-int8-mtp",
    "gemma-dual-int8-262k": "vllm/gemma-int8-mtp",
    "gemma-dual": "vllm/gemma-bf16-mtp",
}

OVERLAY_PATH_HINTS = {
    "vllm-pr40361-marlin-pad": ["models/qwen3.6-27b/vllm/patches/vllm-marlin-pad"],
    "vllm-pr35936-qwen3coder-tool-parser": [
        "models/qwen3.6-27b/vllm/patches/vllm-pr35936-required-fallback"
    ],
    "vllm-pr40391-rebased": [
        "models/gemma-4-31b/vllm/patches/vllm-pr40391-v0.22.0"
    ],
    "vllm-gemma4-tool-parser-fixes": [
        "models/gemma-4-31b/vllm/patches/vllm-pr42006-v0.22.0"
    ],
    "vllm-pr41800-truncate-prompt-tokens": [
        "models/gemma-4-31b/vllm/patches/vllm-pr41800-truncate-prompt-tokens",
        "models/qwen3.6-27b/vllm/patches/vllm-pr41800-truncate-prompt-tokens",
    ],
}


def marker(ok: bool) -> str:
    return "✓" if ok else "✗"


def print_available_composes() -> None:
    print("  available composes:")
    for name in sorted(COMPOSE_REGISTRY):
        print(f"    - {name}")
    if COMPOSE_ALIASES:
        print("  aliases:")
        for alias, target in sorted(COMPOSE_ALIASES.items()):
            print(f"    - {alias} -> {target}")


def normalize_compose_name(name: str) -> str:
    return COMPOSE_ALIASES.get(name, name)


def setup_genesis_pin() -> str | None:
    path = REPO_ROOT / "scripts/setup.sh"
    if not path.exists():
        return None
    match = re.search(r'^GENESIS_PIN="\$\{GENESIS_PIN:-(.+?)\}"', path.read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else None


def hardware_for_entry(entry: dict[str, Any], profiles) -> tuple[list, bool, str]:
    world_size = int(entry["tp"]) * int(entry.get("pp", 1))
    required_sm = float(entry.get("required_sm") or profiles.engines[entry["engine"]].min_sm)
    requires_nvlink = bool(entry.get("requires_nvlink", False))

    if required_sm >= 9.0:
        hardware_id = "rtx-5090"
    else:
        hardware_id = "rtx-3090"
    if world_size <= 0:
        world_size = 1
    hardware = [profiles.hardware[hardware_id] for _ in range(world_size)]
    if requires_nvlink and world_size == 2:
        scenario = f"{world_size}x-{hardware_id}-nvlink"
        nvlink_active = True
    else:
        scenario = f"{world_size}x-{hardware_id}-pcie" if world_size > 1 else f"1x-{hardware_id}"
        nvlink_active = False
    return hardware, nvlink_active, scenario


def hardware_for_combo(args: argparse.Namespace, profiles) -> tuple[list, bool, str]:
    count = max(int(args.tp) * int(args.pp), 1)
    hardware_id = args.hardware
    hardware = [profiles.hardware[hardware_id] for _ in range(count)]
    scenario = f"{count}x-{hardware_id}-pcie" if count > 1 else f"1x-{hardware_id}"
    return hardware, bool(args.nvlink_active), scenario


def entry_objects(entry: dict[str, Any], profiles):
    drafter = profiles.drafters[entry["drafter"]] if entry.get("drafter") else None
    return (
        profiles.models[entry["model"]],
        profiles.workloads[entry["workload"]],
        profiles.engines[entry["engine"]],
        drafter,
    )


def fits_for_entry(entry: dict[str, Any], profiles, hardware: list, nvlink_active: bool, *, project_vram: bool) -> FitsResult:
    model, workload, engine, drafter = entry_objects(entry, profiles)
    return fits(
        hardware=hardware,
        model=model,
        workload=workload,
        engine=engine,
        drafter=drafter,
        tp=int(entry["tp"]),
        pp=int(entry.get("pp", 1)),
        kv_format=entry["kv_format"],
        max_ctx=int(entry["max_ctx"]),
        max_num_seqs=int(entry["max_num_seqs"]),
        mem_util=entry.get("mem_util"),
        weights_variant=entry["weights_variant"],
        nvlink_active=nvlink_active,
        requires_nvlink=bool(entry.get("requires_nvlink", False)),
        required_engine_features=list(entry.get("required_engine_features", [])),
        required_sm=entry.get("required_sm"),
        project_vram=project_vram,
    )


def fits_for_combo(args: argparse.Namespace, profiles, hardware: list, nvlink_active: bool, *, project_vram: bool) -> FitsResult:
    model = profiles.models[args.model]
    workload = profiles.workloads[args.workload]
    engine = profiles.engines[args.engine]
    drafter = profiles.drafters[args.drafter] if args.drafter else None
    return fits(
        hardware=hardware,
        model=model,
        workload=workload,
        engine=engine,
        drafter=drafter,
        tp=args.tp,
        pp=args.pp,
        kv_format=args.kv_format,
        max_ctx=args.max_ctx,
        max_num_seqs=args.max_num_seqs,
        mem_util=args.mem_util,
        weights_variant=args.weights_variant,
        nvlink_active=nvlink_active,
        required_engine_features=list(args.required_feature or []),
        project_vram=project_vram,
    )


def constraints_line(result: FitsResult) -> str:
    diag = result.diagnostics
    passed = len(diag.get("constraints_passed", []))
    failed = len(diag.get("constraints_failed", []))
    skipped = len(diag.get("constraints_skipped", []))
    total = passed + failed + skipped
    return f"valid={str(result.valid).lower()}; constraints passed: {passed}/{total}; elapsed {diag.get('elapsed_ms')} ms"


def print_reasons(result: FitsResult) -> None:
    for reason in result.reasons:
        print(f"      - {reason}")


def print_notes(result: FitsResult) -> None:
    for note in result.notes:
        print(f"  note: {note}")


def print_kv_projection(result: FitsResult) -> bool:
    kv = result.kv_projection
    if not kv:
        skipped = result.diagnostics.get("constraints_skipped", [])
        if "C12" in skipped:
            c12_notes = [note for note in result.notes if "KV projection" in note]
            note = c12_notes[-1] if c12_notes else "KV projection skipped"
            print(f"  ⊘ {note}")
            return True
        print("  ⊘ no kv-calc projection available")
        return True
    verdict = kv.get("verdict", "UNKNOWN")
    total = kv.get("total_gb")
    budget = kv.get("budget_gb")
    pct = kv.get("pct_of_vram")
    if verdict == "FAIL":
        print(f"  ⊘ predicted total {total} GB/card ({pct}% budget), verdict FAIL")
        return False
    print(f"  ✓ predicted total {total} GB/card ({pct}% budget), verdict {verdict}; budget {budget} GB")
    return True


def newest_calibration_summary(profiles, model_id: str) -> str | None:
    cal = profiles.calibration.get(model_id)
    if not cal:
        return None
    rows = [row for row in cal.rows if row.get("status") == "active"]
    if not rows:
        return None
    rows.sort(key=lambda row: str(row.get("date", "")), reverse=True)
    row = rows[0]
    date = row.get("date", "unknown-date")
    compose = row.get("compose", "unknown-compose")
    return f"most recent calibration row {date} ({compose})"


def print_calibration(profiles, compose_name: str | None, entry: dict[str, Any] | None, hardware: list) -> None:
    if not compose_name or not entry:
        print("  ⊘ free-form combo; no exact compose calibration row")
        return
    status, row = calibration_status(profiles, compose_name, hardware, max_ctx=entry.get("max_ctx"))
    if row:
        print(f"  ✓ {status}; {row.get('source', 'calibration row present')}")
        return
    engine_pin = entry.get("engine", "unknown-engine")
    summary = newest_calibration_summary(profiles, entry["model"])
    if summary:
        print(f"  ⊘ engine_pin={engine_pin}; no exact row for this scenario; {summary}")
        print("      action: re-bench when convenient, or accept extrapolation")
    else:
        print(f"  ⊘ engine_pin={engine_pin}; no calibration rows for model {entry['model']}")


def overlay_exists(overlay_id: str) -> tuple[bool, str | None]:
    for rel in OVERLAY_PATH_HINTS.get(overlay_id, []):
        path = REPO_ROOT / rel
        if path.exists():
            return True, rel
    patches_root = REPO_ROOT / "models"
    needle = overlay_id.lower().replace("vllm-", "").replace("pr", "")
    for path in patches_root.glob("*/vllm/patches/*"):
        if needle and needle in path.name.lower():
            return True, str(path.relative_to(REPO_ROOT))
    return False, None


def print_overlays(engine) -> bool:
    ok = True
    required = list(engine.required_overlays or [])
    if required:
        print(f"  ⊘ required_overlays: {required}")
    else:
        print("  ✓ required_overlays: []")

    overlays = list(engine.vendored_overlays or [])
    if not overlays:
        print("  ✓ vendored_overlays: []")
    for overlay in overlays:
        overlay_id = overlay.get("id") if isinstance(overlay, dict) else str(overlay)
        if isinstance(overlay, dict) and overlay.get("image_baked"):
            # Capability provided by the pinned engine image (not a mountable
            # file overlay), so there is no on-disk source to verify.
            print(f"  ✓ {overlay_id}: image-baked (provided by pinned engine image)")
            continue
        exists, rel = overlay_exists(overlay_id)
        if exists:
            print(f"  ✓ {overlay_id}: {rel}")
        else:
            print(f"  ✗ {overlay_id}: source path not found")
            ok = False

    if engine.required_genesis:
        setup_pin = setup_genesis_pin()
        suffix = f" (setup.sh GENESIS_PIN={setup_pin})" if setup_pin else ""
        print(f"  ✓ Genesis pin: {engine.genesis_pin or 'unspecified'}{suffix}")
    else:
        print("  ✓ Genesis: not required")
    return ok


def resolve_free_form_compose(args: argparse.Namespace, profiles, result: FitsResult, nvlink_active: bool) -> str | None:
    if not result.valid:
        return None
    model = profiles.models[args.model]
    workload = profiles.workloads[args.workload]
    engine = profiles.engines[args.engine]
    drafter = profiles.drafters[args.drafter] if args.drafter else None
    weights = result.weights_variant or args.weights_variant or model.default_weight_variant
    return to_compose_name(
        model,
        engine,
        drafter,
        result.recommended_kv_format or args.kv_format,
        args.tp,
        args.pp,
        workload=workload,
        weights_variant=weights,
        nvlink_active=nvlink_active,
        max_ctx=args.max_ctx,
        max_num_seqs=args.max_num_seqs,
    )


def command_diagnose(args: argparse.Namespace) -> int:
    compose_name = normalize_compose_name(args.compose) if args.compose else None
    free_form = compose_name is None

    try:
        profiles = load_profiles()
    except ProfileError as exc:
        target = args.compose or "free-form combo"
        print(f"Profile triage: {target}")
        print("=" * (16 + len(target)))
        print("[1/6] Compose registry entry exists")
        print("  ⊘ free-form combo" if free_form else "  ✓ compose argument provided")
        print("[2/6] Cross-references resolve")
        print(f"  ✗ {exc}")
        print("")
        print("Triage summary: RED")
        return 2

    target = compose_name or "free-form combo"
    print(f"Profile triage: {target}")
    print("=" * (16 + len(target)))

    entry = None
    hardware = []
    nvlink_active = False
    scenario = "unknown"
    print("[1/6] Compose registry entry exists")
    if compose_name:
        if compose_name not in COMPOSE_REGISTRY:
            print(f"  ✗ {compose_name} not found")
            print_available_composes()
            print("")
            print("Triage summary: RED")
            return 3
        entry = COMPOSE_REGISTRY[compose_name]
        print(
            f"  ✓ {compose_name} found "
            f"(model={entry['model']}, workload={entry['workload']}, engine={entry['engine']})"
        )
    else:
        missing = [name for name in ("model", "engine") if not getattr(args, name)]
        if missing:
            print(f"  ✗ free-form combo missing required flag(s): {', '.join('--' + name for name in missing)}")
            print("")
            print("Triage summary: RED")
            return 3
        print("  ⊘ free-form combo; compose registry lookup skipped")

    print("")
    print("[2/6] Cross-references resolve")
    try:
        if entry:
            model, workload, engine, drafter = entry_objects(entry, profiles)
            hardware, nvlink_active, scenario = hardware_for_entry(entry, profiles)
            refs = ["model", "workload", "engine"] + (["drafter"] if drafter else [])
        else:
            model = profiles.models[args.model]
            workload = profiles.workloads[args.workload]
            engine = profiles.engines[args.engine]
            drafter = profiles.drafters[args.drafter] if args.drafter else None
            hardware, nvlink_active, scenario = hardware_for_combo(args, profiles)
            refs = ["model", "workload", "engine"] + (["drafter"] if drafter else [])
        print(f"  ✓ all referenced profiles exist ({', '.join(refs)}, hardware-default={scenario})")
    except KeyError as exc:
        print(f"  ✗ missing referenced profile: {exc}")
        print("")
        print("Triage summary: RED")
        return 2

    print("")
    print(f"[3/6] fits() on canonical {scenario} scenario")
    if entry:
        fast_result = from_compose_name(compose_name, hardware, nvlink_active, profiles, project_vram=False)
    else:
        fast_result = fits_for_combo(args, profiles, hardware, nvlink_active, project_vram=False)
    print(f"  {marker(fast_result.valid)} {constraints_line(fast_result)}")
    print_notes(fast_result)
    if not fast_result.valid:
        print_reasons(fast_result)

    print("")
    print("[4/6] kv-calc projection")
    if entry:
        full_result = fits_for_entry(entry, profiles, hardware, nvlink_active, project_vram=True)
    else:
        full_result = fits_for_combo(args, profiles, hardware, nvlink_active, project_vram=True)
    kv_ok = print_kv_projection(full_result)
    for note in full_result.notes:
        if note.startswith("kv-calc:"):
            print(f"  note: {note}")

    resolved_compose = compose_name
    if free_form:
        resolved_compose = resolve_free_form_compose(args, profiles, fast_result, nvlink_active)
        if resolved_compose:
            entry = COMPOSE_REGISTRY[resolved_compose]

    print("")
    print("[5/6] Calibration freshness")
    print_calibration(profiles, resolved_compose, entry, hardware)

    print("")
    print("[6/6] Vendored overlays applied")
    overlays_ok = print_overlays(engine)
    try:
        if engine.type == "vllm":
            pins = resolve_engine_pin(profiles, engine.id)
            for key, value in pins.items():
                print(f"  ✓ {key} resolves: {value}")
    except ProfileError as exc:
        print(f"  ✗ {exc}")
        overlays_ok = False

    print("")
    if not fast_result.valid or not overlays_ok:
        print("Triage summary: RED")
        return 2
    if not kv_ok:
        print("Triage summary: YELLOW")
        return 1
    print("Triage summary: GREEN")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="club-3090 per-compose profile diagnostics")
    parser.add_argument("compose", nargs="?", help="Compose registry name, e.g. vllm/dual")
    parser.add_argument("--model")
    parser.add_argument("--workload", default="long-ctx-single")
    parser.add_argument("--engine")
    parser.add_argument("--drafter", default="")
    parser.add_argument("--kv-format", default="fp8_e5m2")
    parser.add_argument("--weights-variant")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--mem-util", type=float)
    parser.add_argument("--hardware", default="rtx-3090")
    parser.add_argument("--nvlink-active", action="store_true")
    parser.add_argument("--required-feature", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        return 3 if int(exc.code) != 0 else 0
    if args.compose and any(token.startswith("--") for token in raw_argv):
        print("[diagnose-profile] ERROR: pass either a compose name or free-form flags, not both", file=sys.stderr)
        return 3
    if not args.compose and not (args.model or args.engine):
        parser.print_usage(sys.stderr)
        return 3
    try:
        return command_diagnose(args)
    except DiagnoseProfileError as exc:
        print(f"[diagnose-profile] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
