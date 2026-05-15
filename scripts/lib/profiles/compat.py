"""Profile compatibility helpers for club-3090 v0.7.0.

Requires PyYAML. On Debian/Ubuntu this is usually available from
``python3-yaml``; otherwise install with ``pip install pyyaml``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only on missing dep
    raise RuntimeError("scripts.lib.profiles.compat requires PyYAML; install python3-yaml or pip install pyyaml") from exc

from .compose_registry import COMPOSE_REGISTRY


SUPPORTED_SCHEMA_VERSIONS = {1}
PROFILE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTRAINT_IDS = [f"C{i}" for i in range(1, 17)]
ESTATE_CONSTRAINT_IDS = [f"E{i}" for i in range(1, 5)]


class ProfileError(Exception):
    """Base class for profile loading/validation errors."""


class UnsupportedSchemaVersionError(ProfileError):
    """Raised when a profile schema_version is not supported."""


class CrossReferenceError(ProfileError):
    """Raised when a profile references a missing profile id."""


class TopologyClass(str, Enum):
    SINGLE_CARD = "single_card"
    HOMOGENEOUS = "homogeneous"
    VRAM_MATCHED_COMPUTE_MISMATCHED = "vram_matched_compute_mismatched"
    VRAM_MISMATCHED = "vram_mismatched"
    HETEROGENEOUS_MIXED = "heterogeneous_mixed"


TOPOLOGY_ADVISORY = {
    TopologyClass.SINGLE_CARD: None,
    TopologyClass.HOMOGENEOUS: None,
    TopologyClass.VRAM_MATCHED_COMPUTE_MISMATCHED: (
        "Compute mismatch detected (VRAM matched). TP=N works fine but the faster card "
        "waits at every NCCL allreduce — effective throughput caps at slower card's speed "
        "(~30% of faster card idle at allreduce). Full per-card VRAM capacity preserved. "
        "Alternative: estate planner (--estate) to run different models per card at full speed."
    ),
    TopologyClass.VRAM_MISMATCHED: (
        "VRAM mismatch detected. TP=N would cap to smaller card's usable model size. "
        "Recommended paths: (a) llama.cpp `--tensor-split` for weighted layer split, "
        "(b) PP=N (manual flag flip — `--pipeline-parallel-size N` on a vllm/dual compose; "
        "no shipping PP compose), (c) estate planner (--estate) to run different models per card."
    ),
    TopologyClass.HETEROGENEOUS_MIXED: (
        "Heterogeneous hardware detected (multiple VRAM and compute tiers). Manual selection "
        "recommended. Consider the estate planner (--estate) to put different models on "
        "different card subsets, or run a single model on the largest matched subset."
    ),
}


def _logger() -> logging.Logger:
    logger = logging.getLogger("compat")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[compat] %(message)s"))
        logger.addHandler(handler)
    level = os.environ.get("CLUB3090_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    return logger


def _tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _dict(value: Any) -> dict:
    return dict(value or {})


def _number_or_none(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) else None


@dataclass(frozen=True)
class HardwareProfile:
    schema_version: int
    id: str
    display_name: str
    sm: float
    vram_gb: float
    arch: str
    mem_util_safe: float
    supported_kv_formats: tuple[str, ...]
    kv_format_default: dict[str, str]
    cudagraph: str
    driver_pin_recommended: dict[str, Any]
    nvlink_capable: bool
    power_cap_w_optimal: Optional[int] = None
    power_cap_w_prefill: Optional[int] = None
    power_cap_w_max: Optional[int] = None
    notes: Optional[str] = None


def classify_hardware_topology(hardware: list[HardwareProfile]) -> TopologyClass:
    """Classify selected GPUs for TP-vs-PP/estate advisory output."""
    if not hardware:
        raise ProfileError("classify_hardware_topology requires at least one HardwareProfile")
    if len(hardware) == 1:
        return TopologyClass.SINGLE_CARD

    vrams = sorted(hw.vram_gb for hw in hardware)
    sms = {hw.sm for hw in hardware}

    vram_clusters = 1
    for i in range(1, len(vrams)):
        if vrams[i] - vrams[i - 1] > 1.0:
            vram_clusters += 1

    if vram_clusters == 1 and len(sms) == 1:
        return TopologyClass.HOMOGENEOUS
    if vram_clusters == 1 and len(sms) > 1:
        return TopologyClass.VRAM_MATCHED_COMPUTE_MISMATCHED
    if vram_clusters > 1:
        return TopologyClass.VRAM_MISMATCHED
    return TopologyClass.HETEROGENEOUS_MIXED


@dataclass(frozen=True)
class ModelProfile:
    schema_version: int
    id: str
    display_name: str
    family: str
    hidden_size: int
    num_hidden_layers: int
    num_attn_heads: int
    num_kv_heads: int
    max_ctx_supported: int
    attention_k_eq_v: bool
    weights: dict[str, dict[str, Any]]
    default_weight_variant: str
    compatible_drafters: tuple[str, ...]
    valid_tp: tuple[int, ...]
    requires_genesis: bool
    intermediate_size: Optional[int] = None
    num_gdn_layers: Optional[int] = None
    num_attn_layers: Optional[int] = None
    num_full_attn_layers: Optional[int] = None
    num_sliding_attn_layers: Optional[int] = None
    head_dim_attn: Optional[int] = None
    linear_num_v_heads: Optional[int] = None
    linear_num_k_heads: Optional[int] = None
    linear_v_head_dim: Optional[int] = None
    linear_k_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None
    head_dim_sliding: Optional[int] = None
    global_head_dim: Optional[int] = None
    sliding_window: Optional[int] = None
    # Asymmetric KV head counts for SWA-hybrid models where global layers
    # have a different KV head count than sliding layers (e.g. Gemma 4
    # 26B-A4B: 8 sliding, 2 global). Leave None for symmetric models.
    num_global_kv_heads: Optional[int] = None
    # MoE fields (None for dense models; set for MoE variants)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    active_params_b: Optional[float] = None
    # Optional architectural metadata
    mtp_num_hidden_layers: Optional[int] = None
    attn_output_gate: Optional[bool] = None
    vision_capable: Optional[bool] = None
    # C12 (KV projection via tools/kv-calc.py) only supports models whose
    # architecture has been added to MODEL_SPECS in kv-calc.py. New MoE /
    # hybrid models can set kv_calc_supported=false to skip C12 until
    # kv-calc gains MoE-aware activation/KV formulas.
    kv_calc_supported: bool = True


@dataclass(frozen=True)
class WorkloadProfile:
    schema_version: int
    id: str
    display_name: str
    description: str
    priorities: dict[str, str]
    defaults: dict[str, Any]
    requires_features: tuple[str, ...]


@dataclass(frozen=True)
class EngineProfile:
    schema_version: int
    id: str
    display_name: str
    type: str
    stability: str
    install: dict[str, Any]
    min_sm: float
    supported_model_families: tuple[str, ...]
    features: dict[str, Any]
    supported_kv_formats: tuple[str, ...]
    supported_drafters: tuple[str, ...]
    supported_weight_formats: tuple[str, ...]
    required_overlays: tuple[Any, ...]
    vendored_overlays: tuple[Any, ...]
    required_genesis: bool
    feature_provenance: dict[str, Any] = field(default_factory=dict)
    genesis_pin: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class DrafterProfile:
    schema_version: int
    id: str
    display_name: str
    spec_method: str
    model_compat: tuple[str, ...]
    n_default: int
    n_max: int
    download: Any
    vram_footprint_gb: Any
    status: str
    engine_compat: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationProfile:
    schema_version: int
    model: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Profiles:
    hardware: dict[str, HardwareProfile]
    models: dict[str, ModelProfile]
    workloads: dict[str, WorkloadProfile]
    engines: dict[str, EngineProfile]
    drafters: dict[str, DrafterProfile]
    calibration: dict[str, CalibrationProfile] = field(default_factory=dict)


@dataclass
class FitsResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    recommended_kv_format: Optional[str] = None
    effective_mem_util: Optional[float] = None
    effective_max_ctx: Optional[int] = None
    effective_max_num_seqs: Optional[int] = None
    effective_cudagraph_mode: Optional[str] = None
    world_size: Optional[int] = None
    bottleneck_vram_gb: Optional[float] = None
    homogeneous: Optional[bool] = None
    topology_class: Optional[TopologyClass] = None
    kv_projection: Optional[dict[str, Any]] = None
    compose_name: Optional[str] = None
    weights_variant: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceSpec:
    """One endpoint in a multi-model estate."""

    name: str
    compose_name: str
    gpu_indices: tuple[int, ...]
    port: int


@dataclass
class EstateResult:
    valid: bool
    per_instance: dict[str, FitsResult]
    cross_instance_failures: list[str]
    notes: list[str]
    diagnostics: dict[str, Any]


def _check_schema(path: Path, data: dict[str, Any]) -> None:
    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        msg = (
            f"{path.relative_to(PROFILE_ROOT)} has unsupported schema_version={version}. "
            "Upgrade club-3090 profile tooling or pin older profiles."
        )
        _logger().error(msg)
        raise UnsupportedSchemaVersionError(msg)


def _load_yaml(path: Path) -> dict[str, Any]:
    log = _logger()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        log.error("failed to load %s: %s", path, exc)
        raise ProfileError(f"failed to load {path}: {exc}") from exc
    _check_schema(path, data)
    log.debug("loaded %s", path.relative_to(PROFILE_ROOT))
    return data


def _load_dir(root: Path, subdir: str, factory) -> dict[str, Any]:
    out = {}
    for path in sorted((root / subdir).glob("*.yml")):
        data = _load_yaml(path)
        profile = factory(data)
        out[profile.id if hasattr(profile, "id") else profile.model] = profile
    return out


def _hardware(data: dict[str, Any]) -> HardwareProfile:
    return HardwareProfile(
        schema_version=data["schema_version"],
        id=data["id"],
        display_name=data["display_name"],
        sm=float(data["sm"]),
        vram_gb=float(data["vram_gb"]),
        arch=data["arch"],
        mem_util_safe=float(data["mem_util_safe"]),
        supported_kv_formats=_tuple(data.get("supported_kv_formats")),
        kv_format_default=_dict(data.get("kv_format_default")),
        cudagraph=data.get("cudagraph", "full"),
        driver_pin_recommended=_dict(data.get("driver_pin_recommended")),
        nvlink_capable=bool(data.get("nvlink_capable", False)),
        power_cap_w_optimal=data.get("power_cap_w_optimal"),
        power_cap_w_prefill=data.get("power_cap_w_prefill"),
        power_cap_w_max=data.get("power_cap_w_max"),
        notes=data.get("notes"),
    )


def _model(data: dict[str, Any]) -> ModelProfile:
    return ModelProfile(
        schema_version=data["schema_version"],
        id=data["id"],
        display_name=data["display_name"],
        family=data["family"],
        hidden_size=int(data["hidden_size"]),
        intermediate_size=data.get("intermediate_size"),
        num_hidden_layers=int(data["num_hidden_layers"]),
        num_gdn_layers=data.get("num_gdn_layers"),
        num_attn_layers=data.get("num_attn_layers"),
        num_full_attn_layers=data.get("num_full_attn_layers"),
        num_sliding_attn_layers=data.get("num_sliding_attn_layers"),
        num_attn_heads=int(data["num_attn_heads"]),
        num_kv_heads=int(data["num_kv_heads"]),
        head_dim_attn=data.get("head_dim_attn"),
        linear_num_v_heads=data.get("linear_num_v_heads"),
        linear_num_k_heads=data.get("linear_num_k_heads"),
        linear_v_head_dim=data.get("linear_v_head_dim"),
        linear_k_head_dim=data.get("linear_k_head_dim"),
        linear_conv_kernel_dim=data.get("linear_conv_kernel_dim"),
        head_dim_sliding=data.get("head_dim_sliding"),
        global_head_dim=data.get("global_head_dim"),
        sliding_window=data.get("sliding_window"),
        num_global_kv_heads=data.get("num_global_kv_heads"),
        num_experts=data.get("num_experts"),
        num_experts_per_tok=data.get("num_experts_per_tok"),
        moe_intermediate_size=data.get("moe_intermediate_size"),
        shared_expert_intermediate_size=data.get("shared_expert_intermediate_size"),
        active_params_b=data.get("active_params_b"),
        mtp_num_hidden_layers=data.get("mtp_num_hidden_layers"),
        attn_output_gate=data.get("attn_output_gate"),
        vision_capable=data.get("vision_capable"),
        max_ctx_supported=int(data["max_ctx_supported"]),
        attention_k_eq_v=bool(data["attention_k_eq_v"]),
        weights=_dict(data.get("weights")),
        default_weight_variant=data["default_weight_variant"],
        compatible_drafters=_tuple(data.get("compatible_drafters")),
        valid_tp=tuple(int(x) for x in _tuple(data.get("valid_tp"))),
        requires_genesis=bool(data.get("requires_genesis", False)),
        kv_calc_supported=bool(data.get("kv_calc_supported", True)),
    )


def _workload(data: dict[str, Any]) -> WorkloadProfile:
    return WorkloadProfile(
        schema_version=data["schema_version"],
        id=data["id"],
        display_name=data["display_name"],
        description=data.get("description", ""),
        priorities=_dict(data.get("priorities")),
        defaults=_dict(data.get("defaults")),
        requires_features=_tuple(data.get("requires_features")),
    )


def _engine(data: dict[str, Any]) -> EngineProfile:
    return EngineProfile(
        schema_version=data["schema_version"],
        id=data["id"],
        display_name=data["display_name"],
        type=data["type"],
        stability=data["stability"],
        install=_dict(data.get("install")),
        min_sm=float(data["min_sm"]),
        supported_model_families=_tuple(data.get("supported_model_families")),
        features=_dict(data.get("features")),
        feature_provenance=_dict(data.get("feature_provenance")),
        supported_kv_formats=_tuple(data.get("supported_kv_formats")),
        supported_drafters=_tuple(data.get("supported_drafters")),
        supported_weight_formats=_tuple(data.get("supported_weight_formats")),
        required_overlays=_tuple(data.get("required_overlays")),
        vendored_overlays=_tuple(data.get("vendored_overlays")),
        required_genesis=bool(data.get("required_genesis", False)),
        genesis_pin=data.get("genesis_pin"),
        notes=data.get("notes"),
    )


def _drafter(data: dict[str, Any]) -> DrafterProfile:
    return DrafterProfile(
        schema_version=data["schema_version"],
        id=data["id"],
        display_name=data["display_name"],
        spec_method=data["spec_method"],
        model_compat=_tuple(data.get("model_compat")),
        engine_compat=_dict(data.get("engine_compat")),
        n_default=int(data["n_default"]),
        n_max=int(data["n_max"]),
        download=data.get("download"),
        vram_footprint_gb=data.get("vram_footprint_gb"),
        status=data["status"],
    )


def _calibration(data: dict[str, Any]) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=data["schema_version"],
        model=data["model"],
        rows=tuple(dict(row) for row in data.get("rows", [])),
    )


def _known(items: dict[str, Any]) -> str:
    return ", ".join(sorted(items))


def _validate_cross_refs(profiles: Profiles) -> None:
    failures: list[str] = []

    for model in profiles.models.values():
        if model.default_weight_variant not in model.weights:
            failures.append(
                f"models/{model.id}.yml default_weight_variant `{model.default_weight_variant}` "
                f"not in weights. Available variants: {', '.join(sorted(model.weights))}"
            )
        for drafter_id in model.compatible_drafters:
            if drafter_id not in profiles.drafters:
                failures.append(
                    f"models/{model.id}.yml references unknown drafter `{drafter_id}`. "
                    f"Available drafters: {_known(profiles.drafters)}"
                )

    for drafter in profiles.drafters.values():
        for model_id in drafter.model_compat:
            if model_id not in profiles.models:
                failures.append(
                    f"drafters/{drafter.id}.yml references unknown model `{model_id}`. "
                    f"Available models: {_known(profiles.models)}"
                )

    for cal in profiles.calibration.values():
        if cal.model not in profiles.models:
            failures.append(
                f"calibration/{cal.model}.yml references unknown model `{cal.model}`. "
                f"Available models: {_known(profiles.models)}"
            )
        for row in cal.rows:
            compose = row.get("compose")
            if compose not in COMPOSE_REGISTRY:
                failures.append(f"calibration/{cal.model}.yml references unknown compose `{compose}`")

    for name, entry in COMPOSE_REGISTRY.items():
        for field_name, table in (
            ("model", profiles.models),
            ("workload", profiles.workloads),
            ("engine", profiles.engines),
        ):
            ref = entry.get(field_name)
            if ref not in table:
                failures.append(f"COMPOSE_REGISTRY[{name!r}].{field_name} references unknown `{ref}`")
        drafter_id = entry.get("drafter")
        if drafter_id is not None and drafter_id not in profiles.drafters:
            failures.append(f"COMPOSE_REGISTRY[{name!r}].drafter references unknown `{drafter_id}`")
        model = profiles.models.get(entry.get("model"))
        if model and entry.get("weights_variant") not in model.weights:
            failures.append(
                f"COMPOSE_REGISTRY[{name!r}].weights_variant references unknown "
                f"`{entry.get('weights_variant')}` for {model.id}"
            )
        if "default_port" not in entry:
            failures.append(f"COMPOSE_REGISTRY[{name!r}] missing default_port")
        if "gpu_assignment_mode" not in entry:
            failures.append(f"COMPOSE_REGISTRY[{name!r}] missing gpu_assignment_mode")

    if failures:
        msg = "cross-reference validation failed:\n  " + "\n  ".join(failures)
        _logger().error(msg)
        raise CrossReferenceError(msg)


def load_profiles(root: Path = PROFILE_ROOT) -> Profiles:
    """Load profile YAML files and validate all cross-references."""

    log = _logger()
    root = Path(root)
    profiles = Profiles(
        hardware=_load_dir(root, "hardware", _hardware),
        models=_load_dir(root, "models", _model),
        workloads=_load_dir(root, "workloads", _workload),
        engines=_load_dir(root, "engines", _engine),
        drafters=_load_dir(root, "drafters", _drafter),
        calibration=_load_dir(root, "calibration", _calibration),
    )
    _validate_cross_refs(profiles)

    cal_counts = ", ".join(f"{model}={len(cal.rows)}" for model, cal in sorted(profiles.calibration.items()))
    log.info(
        "Loaded %d profiles from %s",
        len(profiles.hardware) + len(profiles.models) + len(profiles.workloads)
        + len(profiles.engines) + len(profiles.drafters) + len(profiles.calibration),
        root,
    )
    log.info(
        "  hardware: %d, models: %d, workloads: %d, engines: %d, drafters: %d",
        len(profiles.hardware),
        len(profiles.models),
        len(profiles.workloads),
        len(profiles.engines),
        len(profiles.drafters),
    )
    log.info("  calibration rows: %s", cal_counts)
    log.info("  compose_registry entries: %d", len(COMPOSE_REGISTRY))
    return profiles


def _cudagraph_mode(hardware: list[HardwareProfile]) -> Optional[str]:
    if not hardware:
        return None
    rank = {"full": 0, "partial": 1, "enforce-eager": 2, "enforce-eager-required": 2}
    return max((hw.cudagraph for hw in hardware), key=lambda mode: rank.get(mode, 0))


def resolve_kv_format(workload: WorkloadProfile, hardware: list[HardwareProfile]) -> str:
    if not hardware:
        return "bf16"
    if workload.priorities.get("max_ctx") == "high":
        context = "long_context"
    elif workload.priorities.get("concurrency") == "high":
        context = "multi_stream"
    else:
        context = "balanced"

    candidates = [hw.kv_format_default.get(context) for hw in hardware]
    if len(set(candidates)) == 1 and candidates[0]:
        return candidates[0]
    if all("fp8_e5m2" in hw.supported_kv_formats for hw in hardware):
        return "fp8_e5m2"
    return "bf16"


def resolve_weights_variant(model: ModelProfile, engine: EngineProfile, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    default = model.default_weight_variant
    default_format = model.weights.get(default, {}).get("format")
    if default_format in engine.supported_weight_formats:
        return default

    status_rank = {"production": 0, "experimental": 1, "historical": 2}
    variants = sorted(
        model.weights.items(),
        key=lambda item: (status_rank.get(item[1].get("status"), 99), list(model.weights).index(item[0])),
    )
    for name, meta in variants:
        if meta.get("format") in engine.supported_weight_formats:
            return name
    return default


def compatible_kv_formats(hardware: list[HardwareProfile], engine: EngineProfile) -> list[str]:
    supported = set(engine.supported_kv_formats)
    for hw in hardware:
        supported &= set(hw.supported_kv_formats)
    return sorted(supported)


def valid_tp_values(model: ModelProfile, world_size: int) -> list[int]:
    return [tp for tp in model.valid_tp if tp > 0 and world_size % tp == 0]


def compatible_drafters(
    model: ModelProfile,
    engine: EngineProfile,
    profiles: Optional[Profiles] = None,
) -> list[DrafterProfile]:
    profiles = profiles or load_profiles()
    out = []
    for drafter_id in model.compatible_drafters:
        drafter = profiles.drafters[drafter_id]
        if drafter.spec_method not in engine.supported_drafters:
            continue
        engine_type = drafter.engine_compat.get("engine_type")
        if engine_type and engine_type != engine.type:
            continue
        out.append(drafter)
    return out


def compatible_engines(
    hardware: list[HardwareProfile],
    model: ModelProfile,
    profiles: Optional[Profiles] = None,
) -> list[EngineProfile]:
    profiles = profiles or load_profiles()
    out = []
    for engine in profiles.engines.values():
        if any(hw.sm < engine.min_sm for hw in hardware):
            continue
        if engine.type == "vllm" and model.requires_genesis and not engine.required_genesis:
            continue
        if model.family not in engine.supported_model_families:
            continue
        out.append(engine)
    return out


def _kv_calc_weights_variant(model: ModelProfile, variant: str) -> str:
    if model.family == "gemma4-swa-dense":
        if variant == "awq":
            return "awq"
        if variant == "bf16":
            return "bf16"
        return "int4"
    return "default"


def _drafter_gb(drafter: Optional[DrafterProfile]) -> float:
    if drafter is None:
        return 0.0
    return float(_number_or_none(drafter.vram_footprint_gb) or 0.0)


def _run_kv_calc(
    *,
    model: ModelProfile,
    drafter: Optional[DrafterProfile],
    kv_format: str,
    max_ctx: int,
    max_num_seqs: int,
    tp: int,
    mem_util: float,
    vram_gb: float,
    weights_variant: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    args = [
        "bash",
        str(REPO_ROOT / "tools/kv-calc.py"),
        "--model",
        model.id,
        "--kv-format",
        kv_format,
        "--max-ctx",
        str(max_ctx),
        "--max-num-seqs",
        str(max_num_seqs),
        "--tp",
        str(tp),
        "--mem-util",
        str(mem_util),
        "--vram",
        str(vram_gb),
        "--weights-variant",
        _kv_calc_weights_variant(model, weights_variant),
        "--json",
    ]
    if drafter and drafter.spec_method in {"mtp", "mtp_assistant"}:
        args.append("--mtp")
    else:
        args.append("--no-mtp")
    drafter_gb = _drafter_gb(drafter)
    if drafter_gb:
        args.extend(["--drafter-gb", str(drafter_gb)])

    proc = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if not proc.stdout.strip():
        return None, f"kv-calc.py produced no JSON (exit {proc.returncode}): {proc.stderr.strip()}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f"kv-calc.py produced invalid JSON: {exc}: {proc.stdout[:200]}"
    if proc.returncode not in (0, 1):
        return data, f"kv-calc.py failed with exit {proc.returncode}: {proc.stderr.strip()}"
    return data, None


def fits(
    hardware: list[HardwareProfile],
    model: ModelProfile,
    workload: WorkloadProfile,
    engine: EngineProfile,
    drafter: Optional[DrafterProfile] = None,
    *,
    tp: int = 1,
    pp: int = 1,
    kv_format: Optional[str] = None,
    max_ctx: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    mem_util: Optional[float] = None,
    weights_variant: Optional[str] = None,
    nvlink_active: bool = False,
    requires_nvlink: bool = False,
    required_engine_features: Optional[list[str]] = None,
    required_sm: Optional[float] = None,
    project_vram: bool = True,
) -> FitsResult:
    start = time.monotonic()
    reasons: list[str] = []
    notes: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    kv_projection: Optional[dict[str, Any]] = None
    kv_calc_invoked = False

    def ok(cid: str) -> None:
        passed.append(cid)

    def fail(cid: str, reason: str) -> None:
        failed.append(cid)
        reasons.append(f"{cid}: {reason}")

    def skip(cid: str, note: str) -> None:
        skipped.append(cid)
        notes.append(note)

    required_engine_features = list(required_engine_features or [])
    world_size = tp * pp
    effective_kv = kv_format or resolve_kv_format(workload, hardware)
    effective_mem_util = mem_util if mem_util is not None else min((hw.mem_util_safe for hw in hardware), default=None)
    effective_max_ctx = max_ctx if max_ctx is not None else model.max_ctx_supported
    effective_max_num_seqs = max_num_seqs if max_num_seqs is not None else int(workload.defaults.get("max_num_seqs", 1))
    effective_weights = resolve_weights_variant(model, engine, weights_variant)
    homogeneous = len({hw.id for hw in hardware}) <= 1
    topology_class = classify_hardware_topology(hardware) if hardware else None
    bottleneck = min((hw.vram_gb for hw in hardware), default=None)
    effective_cudagraph = _cudagraph_mode(hardware)

    if not homogeneous and hardware:
        smallest = min(hardware, key=lambda hw: hw.vram_gb)
        notes.append(f"Heterogeneous GPU set; bottleneck is {smallest.id} ({smallest.vram_gb:g} GB)")
        notes.append("Power-cap settings are not unified; tune per card")
        if tp > 1:
            notes.append("Tensor parallel on heterogeneous cards wastes VRAM on larger cards")
    if pp > 1:
        notes.append("Pipeline parallel (PP>1) is allowed but layer split is not benchmarked on this stack")

    if len(hardware) != world_size:
        fail("C1", f"tp={tp} * pp={pp} = {world_size} != {len(hardware)} cards selected")
    else:
        ok("C1")

    if tp not in model.valid_tp:
        fail("C2", f"tp={tp} not in model.valid_tp {list(model.valid_tp)} for {model.id}")
    else:
        ok("C2")

    min_sm = max(float(engine.min_sm), float(required_sm or engine.min_sm))
    low_sm = [hw for hw in hardware if hw.sm < min_sm]
    if low_sm:
        fail("C3", f"engine/compose requires sm >= {min_sm:g}; below floor: " + ", ".join(f"{hw.id}=sm_{hw.sm:g}" for hw in low_sm))
    else:
        ok("C3")

    if effective_kv not in engine.supported_kv_formats:
        fail("C4", f"kv_format={effective_kv} not in {engine.id}.supported_kv_formats {list(engine.supported_kv_formats)}")
    else:
        ok("C4")

    unsupported_hw = [hw.id for hw in hardware if effective_kv not in hw.supported_kv_formats]
    if unsupported_hw:
        fail("C5", f"kv_format={effective_kv} not supported by hardware: {', '.join(unsupported_hw)}")
    else:
        ok("C5")

    if engine.type == "vllm" and model.requires_genesis and not engine.required_genesis:
        fail("C6", f"{model.id}.requires_genesis=true; engine {engine.id}.required_genesis=false")
    else:
        ok("C6")

    if drafter is None:
        ok("C7")
        ok("C8")
        ok("C9")
    else:
        if drafter.spec_method not in engine.supported_drafters:
            fail("C7", f"drafter {drafter.id} has spec_method={drafter.spec_method}; {engine.id}.supported_drafters={list(engine.supported_drafters)} excludes it")
        else:
            ok("C7")
        if model.id not in drafter.model_compat:
            fail("C8", f"drafter {drafter.id}.model_compat={list(drafter.model_compat)} excludes {model.id}")
        else:
            ok("C8")
        engine_type = drafter.engine_compat.get("engine_type")
        if engine_type and engine_type != engine.type:
            fail("C9", f"drafter {drafter.id}.engine_compat.engine_type={engine_type}; selected engine type is {engine.type}")
        else:
            ok("C9")

    if model.family not in engine.supported_model_families:
        fail("C10", f"engine {engine.id} supported_model_families={list(engine.supported_model_families)} excludes {model.family}")
    else:
        ok("C10")

    if effective_max_ctx > model.max_ctx_supported:
        fail("C11", f"max_ctx={effective_max_ctx} > model.max_ctx_supported={model.max_ctx_supported}")
    else:
        ok("C11")

    if requires_nvlink and not nvlink_active:
        fail("C13", "compose requires active NVLink, but nvlink_active=false")
    elif requires_nvlink and any(not hw.nvlink_capable for hw in hardware):
        fail("C13", "compose requires NVLink, but not all selected cards are NVLink-capable")
    else:
        ok("C13")

    weight_meta = model.weights.get(effective_weights)
    if weight_meta is None:
        fail("C14", f"weights_variant={effective_weights} not in model.weights {list(model.weights)}")
    elif weight_meta.get("format") not in engine.supported_weight_formats:
        fail(
            "C14",
            f"model.weights.{effective_weights}.format={weight_meta.get('format')} not in "
            f"{engine.id}.supported_weight_formats {list(engine.supported_weight_formats)}",
        )
    else:
        ok("C14")

    missing_features = [feature for feature in required_engine_features if not engine.features.get(feature)]
    if missing_features:
        fail("C15", f"engine {engine.id} does not satisfy required features {missing_features}")
    else:
        ok("C15")

    if failed:
        skip("C12", "KV projection not run; resolve fast-constraint failures first.")
    elif not project_vram:
        skipped.append("C12")
        notes.append("KV projection skipped because project_vram=false")
    elif engine.type != "vllm":
        skipped.append("C12")
        notes.append("KV projection not available for non-vLLM engines")
    elif not model.kv_calc_supported:
        skipped.append("C12")
        notes.append(f"KV projection skipped: {model.id} not yet wired into tools/kv-calc.py")
    else:
        kv_calc_invoked = True
        if effective_mem_util is None or bottleneck is None:
            fail("C12", "cannot run KV projection without mem_util and bottleneck VRAM")
        else:
            kv_projection, err = _run_kv_calc(
                model=model,
                drafter=drafter,
                kv_format=effective_kv,
                max_ctx=effective_max_ctx,
                max_num_seqs=effective_max_num_seqs,
                tp=tp,
                mem_util=effective_mem_util,
                vram_gb=bottleneck,
                weights_variant=effective_weights,
            )
            if err:
                fail("C12", err)
            else:
                kv_projection = dict(kv_projection or {})
                kv_projection.setdefault("confidence", "HIGH")
                for note in kv_projection.get("notes", []):
                    notes.append(f"kv-calc: {note}")
                verdict = kv_projection.get("verdict")
                if verdict == "FAIL" and pp > 1:
                    kv_projection["verdict"] = "TIGHT"
                    kv_projection["confidence"] = "PP_ESTIMATE"
                    notes.append(
                        "PP>1 is not modelled by kv-calc; demoted C12 FAIL to advisory TIGHT "
                        "because per-card weights are conservatively over-counted"
                    )
                    ok("C12")
                elif verdict == "FAIL":
                    fail("C12", f"KV projection verdict FAIL: predicted total {kv_projection.get('total_gb')} GB/card > budget {kv_projection.get('budget_gb')} GB")
                else:
                    ok("C12")

    if topology_class is None:
        skip("C16", "Topology advisory not run; no hardware profiles provided.")
    else:
        ok("C16")
        advisory = TOPOLOGY_ADVISORY.get(topology_class)
        if advisory:
            notes.append(f"C16 topology={topology_class.value}: {advisory}")

    diagnostics = {
        "constraints_evaluated": list(CONSTRAINT_IDS),
        "constraints_passed": passed,
        "constraints_failed": failed,
        "constraints_skipped": skipped,
        "kv_calc_invoked": kv_calc_invoked,
        "elapsed_ms": round((time.monotonic() - start) * 1000, 3),
    }
    return FitsResult(
        valid=not reasons,
        reasons=reasons,
        notes=notes,
        recommended_kv_format=effective_kv,
        effective_mem_util=effective_mem_util,
        effective_max_ctx=effective_max_ctx,
        effective_max_num_seqs=effective_max_num_seqs,
        effective_cudagraph_mode=effective_cudagraph,
        world_size=world_size,
        bottleneck_vram_gb=bottleneck,
        homogeneous=homogeneous,
        topology_class=topology_class,
        kv_projection=kv_projection,
        weights_variant=effective_weights,
        diagnostics=diagnostics,
    )


def from_compose_name(
    name: str,
    hardware: list[HardwareProfile],
    nvlink_active: bool,
    profiles: Optional[Profiles] = None,
    *,
    project_vram: bool = True,
) -> FitsResult:
    profiles = profiles or load_profiles()
    if name not in COMPOSE_REGISTRY:
        return FitsResult(
            valid=False,
            reasons=[f"unknown compose `{name}`. Available composes: {', '.join(COMPOSE_REGISTRY)}"],
            diagnostics={
                "constraints_evaluated": [],
                "constraints_passed": [],
                "constraints_failed": [],
                "constraints_skipped": list(CONSTRAINT_IDS),
                "kv_calc_invoked": False,
                "elapsed_ms": 0.0,
            },
        )
    entry = COMPOSE_REGISTRY[name]
    drafter = profiles.drafters[entry["drafter"]] if entry.get("drafter") else None
    result = fits(
        hardware=hardware,
        model=profiles.models[entry["model"]],
        workload=profiles.workloads[entry["workload"]],
        engine=profiles.engines[entry["engine"]],
        drafter=drafter,
        tp=entry["tp"],
        pp=entry.get("pp", 1),
        kv_format=entry["kv_format"],
        max_ctx=entry["max_ctx"],
        max_num_seqs=entry["max_num_seqs"],
        mem_util=entry.get("mem_util"),
        weights_variant=entry["weights_variant"],
        nvlink_active=nvlink_active,
        requires_nvlink=bool(entry.get("requires_nvlink", False)),
        required_engine_features=list(entry.get("required_engine_features", [])),
        required_sm=entry.get("required_sm"),
        project_vram=project_vram,
    )
    result.compose_name = name
    return result


def to_compose_name(
    model: ModelProfile,
    engine: EngineProfile,
    drafter: Optional[DrafterProfile],
    kv_format: str,
    tp: int,
    pp: int,
    *,
    workload: WorkloadProfile,
    weights_variant: str,
    nvlink_active: bool = False,
    max_ctx: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
) -> Optional[str]:
    drafter_id = drafter.id if drafter else None
    for name, entry in COMPOSE_REGISTRY.items():
        if entry["model"] != model.id:
            continue
        if entry["engine"] != engine.id:
            continue
        if entry.get("drafter") != drafter_id:
            continue
        if entry["kv_format"] != kv_format:
            continue
        if entry["tp"] != tp or entry.get("pp", 1) != pp:
            continue
        if entry["workload"] != workload.id:
            continue
        if entry["weights_variant"] != weights_variant:
            continue
        if max_ctx is not None and entry["max_ctx"] != max_ctx:
            continue
        if max_num_seqs is not None and entry["max_num_seqs"] != max_num_seqs:
            continue
        if entry.get("requires_nvlink", False) and not nvlink_active:
            continue
        return name
    return None


def calibration_status(profiles: Profiles, compose_name: str, hardware: list[HardwareProfile], max_ctx: Optional[int] = None) -> tuple[str, Optional[dict[str, Any]]]:
    entry = COMPOSE_REGISTRY.get(compose_name)
    if not entry:
        return "predicted", None
    cal = profiles.calibration.get(entry["model"])
    if not cal:
        return "predicted", None
    vram = min((hw.vram_gb for hw in hardware), default=None)
    for row in cal.rows:
        if row.get("status") != "active":
            continue
        if row.get("compose") != compose_name:
            continue
        if vram is not None and float(row.get("vram_gb", -1)) != float(vram):
            continue
        ctx_override = row.get("ctx_override")
        if ctx_override is not None and max_ctx is not None and int(ctx_override) != int(max_ctx):
            continue
        return "verified", row
    return "predicted", None


def _dummy_instance_result(reason: str) -> FitsResult:
    return FitsResult(
        valid=False,
        reasons=[reason],
        diagnostics={
            "constraints_evaluated": [],
            "constraints_passed": [],
            "constraints_failed": [],
            "constraints_skipped": list(CONSTRAINT_IDS),
            "kv_calc_invoked": False,
            "elapsed_ms": 0.0,
        },
    )


def validate_estate(
    instances: list[InstanceSpec],
    hardware: list[HardwareProfile],
    profiles: Profiles,
    nvlink_active: bool,
    nvlink_pairs: Optional[list[tuple[int, int]]] = None,
) -> EstateResult:
    start = time.monotonic()
    per_instance: dict[str, FitsResult] = {}
    failures: list[str] = []
    notes: list[str] = []
    passed: list[str] = []
    failed: list[str] = []

    claimed: dict[int, list[str]] = {}
    ports: dict[int, list[str]] = {}
    normalized_pairs = {tuple(sorted(pair)) for pair in (nvlink_pairs or [])}

    for inst in instances:
        for idx in inst.gpu_indices:
            claimed.setdefault(idx, []).append(inst.name)
        ports.setdefault(inst.port, []).append(inst.name)

        bad_indices = [idx for idx in inst.gpu_indices if idx < 0 or idx >= len(hardware)]
        if bad_indices:
            per_instance[inst.name] = _dummy_instance_result(
                f"instance {inst.name} references GPU indices outside rig: {bad_indices}"
            )
            continue
        selected = [hardware[idx] for idx in inst.gpu_indices]
        per_instance[inst.name] = from_compose_name(
            inst.compose_name,
            hardware=selected,
            nvlink_active=nvlink_active,
            profiles=profiles,
        )

    e1_failures = []
    for idx, names in sorted(claimed.items()):
        if idx < 0 or idx >= len(hardware):
            e1_failures.append(f"GPU index {idx} is outside rig size {len(hardware)}")
        if len(names) > 1:
            e1_failures.append(f"GPU {idx} claimed by {', '.join(names)}")
    if e1_failures:
        failures.extend(f"E1: {msg}" for msg in e1_failures)
        failed.append("E1")
    else:
        passed.append("E1")

    passed.append("E2")
    notes.append("E2 no-op: all v0.7.0 compose entries use contiguous exclusive GPU assignment")

    e3_failures = []
    for inst in instances:
        entry = COMPOSE_REGISTRY.get(inst.compose_name, {})
        if not entry.get("requires_nvlink"):
            continue
        pair = tuple(sorted(inst.gpu_indices))
        if nvlink_active and len(pair) == 2 and normalized_pairs and pair not in normalized_pairs:
            e3_failures.append(
                f"instance {inst.name} uses {inst.compose_name} on GPUs {pair}, not in nvlink_pairs={sorted(normalized_pairs)}"
            )
        if nvlink_active and len(pair) != 2:
            e3_failures.append(f"instance {inst.name} requires one NVLink pair but has GPUs {pair}")
        if nvlink_active and normalized_pairs:
            for nv_pair in normalized_pairs:
                if set(pair) & set(nv_pair) and not set(nv_pair).issubset(set(pair)):
                    e3_failures.append(f"instance {inst.name} splits NVLink pair {nv_pair}")
    if e3_failures:
        failures.extend(f"E3: {msg}" for msg in e3_failures)
        failed.append("E3")
    else:
        passed.append("E3")

    e4_failures = [f"port {port} claimed by {', '.join(names)}" for port, names in sorted(ports.items()) if len(names) > 1]
    if e4_failures:
        failures.extend(f"E4: {msg}" for msg in e4_failures)
        failed.append("E4")
    else:
        passed.append("E4")

    if any(not result.valid for result in per_instance.values()):
        notes.append("One or more estate instances failed per-instance fits() validation")

    diagnostics = {
        "constraints_evaluated": list(ESTATE_CONSTRAINT_IDS),
        "constraints_passed": passed,
        "constraints_failed": failed,
        "constraints_skipped": [],
        "elapsed_ms": round((time.monotonic() - start) * 1000, 3),
    }
    return EstateResult(
        valid=not failures and all(result.valid for result in per_instance.values()),
        per_instance=per_instance,
        cross_instance_failures=failures,
        notes=notes,
        diagnostics=diagnostics,
    )
