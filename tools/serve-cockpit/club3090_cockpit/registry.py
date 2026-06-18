"""Registry catalog loader — re-exports from shared core.

The canonical implementation lives in club3090_tui_core.registry.
This module is kept as a thin wrapper so existing cockpit imports continue
to work: ``from club3090_cockpit.registry import VariantRow, parse_variant_rows``.
"""

from club3090_tui_core.registry import (
    VariantRow,
    parse_variant_rows,
    detect_from_registry_async,
    load_catalog_sync,
)

__all__ = [
    "VariantRow",
    "parse_variant_rows",
    "detect_from_registry_async",
    "load_catalog_sync",
]
