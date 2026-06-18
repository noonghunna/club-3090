"""club3090-tui-core: Shared TUI primitives for club-3090 apps."""

from .detect import (
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
from .parsers import (
    TestType,
    Status,
    ParseEvent,
    BenchParser,
    VerifyParser,
    StressParser,
    QualityParser,
    SoakParser,
    RebenchParser,
    get_parser,
    strip_ansi,
)
from .registry import (
    VariantRow,
    parse_variant_rows,
    detect_from_registry_async,
    load_catalog_sync,
)
from .runner import SubprocessRunner, CoreRunState

__version__ = "0.1.0"
