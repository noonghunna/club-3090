"""Auto-detection of serving model + endpoint — re-exports from shared core."""

from club3090_tui_core.detect import (
    GpuInfo,
    ServingTarget,
    detect_endpoint,
    get_gpu_info,
    detect_from_registry,
    match_target_to_registry,
    ENGINE_PREFIXES,
    ENGINE_INTERNAL_PORTS,
    PORT_MAP_BROAD_RE,
    PORT_MAP_RE,
    _classify_engine,
    _classify_engine_from_container,
)

__all__ = [
    "GpuInfo",
    "ServingTarget",
    "detect_endpoint",
    "get_gpu_info",
    "detect_from_registry",
    "match_target_to_registry",
    "ENGINE_PREFIXES",
    "ENGINE_INTERNAL_PORTS",
    "PORT_MAP_BROAD_RE",
    "PORT_MAP_RE",
    "_classify_engine",
    "_classify_engine_from_container",
]
