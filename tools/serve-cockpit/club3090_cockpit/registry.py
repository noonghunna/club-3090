"""Registry catalog loader — the ONE allowed script call in Phase 1.

Invokes:
    bash -c 'source "<repo_root>/scripts/lib/registry-emit.sh"
             && registry_variant_rows "<repo_root>"'

Parses the tab-delimited VARIANT rows.  Every other data source is stubbed.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VariantRow:
    """One registry variant row parsed from registry_variant_rows output."""

    slug: str
    switch_engine: str
    launch_engine: str
    compose_dir: str
    file: str
    port: int
    model: str
    engine: str
    kvcalc_key: str
    container: str
    compose_path: str
    status: str
    ctx_label: str
    status_note: str

    # Stub columns — populated by later phases
    fit: str = "·"
    tps: str = "—"
    quality_8pk: str = "—"
    source: str = "·"


def parse_variant_rows(output: str) -> list[VariantRow]:
    """Parse tab-delimited registry_variant_rows stdout into VariantRow objects.

    Only lines whose first field is VARIANT are consumed; DEFAULT and other
    marker lines are ignored.  This is the same filtering used by
    ``detect.py::detect_from_registry`` in c3t.

    Fields (0-indexed after the leading VARIANT marker):
      0  VARIANT
      1  slug
      2  switch_engine
      3  launch_engine
      4  compose_dir
      5  file
      6  port
      7  model
      8  engine
      9  kvcalc_key
     10  container
     11  compose_path
     12  status
     13  ctx_label      (optional)
     14  status_note    (optional)
    """
    rows: list[VariantRow] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 13 or parts[0] != "VARIANT":
            continue
        rows.append(
            VariantRow(
                slug=parts[1],
                switch_engine=parts[2],
                launch_engine=parts[3],
                compose_dir=parts[4],
                file=parts[5],
                port=int(parts[6]) if parts[6].isdigit() else 0,
                model=parts[7],
                engine=parts[8],
                kvcalc_key=parts[9],
                container=parts[10],
                compose_path=parts[11],
                status=parts[12],
                ctx_label=parts[13] if len(parts) > 13 else "",
                status_note=parts[14] if len(parts) > 14 else "",
            )
        )
    return rows


def load_catalog_sync(repo_root: Path) -> tuple[list[VariantRow], Optional[str]]:
    """Synchronously load the catalog.  Returns (rows, error_message).

    Used from on_mount workers and the ``r`` refresh action.
    This is the only subprocess invocation in the entire Phase 1 app.
    """
    cmd = (
        f'source "{repo_root}/scripts/lib/registry-emit.sh" '
        f'&& registry_variant_rows "{repo_root}"'
    )
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(repo_root),
        )
        if result.returncode != 0 and not result.stdout.strip():
            return [], f"registry-emit failed (rc={result.returncode}): {result.stderr[:200]}"
        rows = parse_variant_rows(result.stdout)
        if not rows:
            return [], "No VARIANT rows returned — registry may be empty or parse failed"
        return rows, None
    except subprocess.TimeoutExpired:
        return [], "registry-emit timed out (>15s)"
    except FileNotFoundError:
        return [], "bash not found — cannot call registry-emit.sh"
    except Exception as exc:
        return [], f"Unexpected error loading catalog: {exc}"
