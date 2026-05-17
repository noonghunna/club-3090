"""v0.8.0 Pull-Emit-Derived `[E]` — CONTRACT-1 `EInput` dataclass.

Module path (stable for E2-E5/E4): `scripts.lib.profiles.einput.EInput`.

Why a NEW module (documented choice, per the STEP-E1 brief): the shipped
`PullResult` (`pull.py`) carries only outcome/compose/notice/diagnostic
fields, NOT `DeriveResult.profile`, the selected weight files, the resolved
`HF_HOME`, the `--profile-like` runtime entry, the `[C2a]` sizing, or the
fetched HF API data E2/E3 need (CONTRACT-1 §). `EInput` is the explicit,
isolation-testable typed input the post-`[C1]` `[E]` stage receives. It is
**pure data** — E1 only DEFINES it and consumes it as the typed argument to
`derived_emittable()` / `generate_from_profile()`; E4 (NOT E1) is what
populates it from inside `run_pull()` where `der`/`s2`/`c2a`/`runtime`/
`selected_files`/`hf_home` are still in scope. Placing it in its own module
(rather than in `pull.py`) keeps E1 additive-only against `pull.py` (no
`run_pull` wiring this STEP) and keeps the dataclass importable by every
later STEP without importing the orchestrator.

The field list is EXACTLY CONTRACT-1's (no more, no less); see the inline
comments for each field's source (which E4 will wire).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EInput:
    """CONTRACT-1 — the explicit, isolation-testable input to the post-`[C1]`
    `[E]` stage. Pure data. Constructed by E4 inside `run_pull()`; in E1 it is
    constructed directly by `test-generate-from-profile.sh` fixtures."""

    slug: str
    # one of: proceed | confirm→proceed(+yes) | override-accepted
    terminal: str
    # §5.3 force-capture trigger (override-accepted terminal)
    is_override_accepted: bool
    # DeriveResult: .profile / .spec / .confidence / .generic_dense_eligible
    # / .tier1 / .slug — typed as Any so this module never imports the frozen
    # deriver (kept dependency-light; tests pass a constructed stand-in or the
    # real DeriveResult).
    der: Any
    # the --profile-like COMPOSE_REGISTRY runtime entry
    # (engine / kv_format / tp / max_ctx / max_num_seqs / mem_util / drafter /
    #  required_engine_features / default_port).
    runtime: dict
    # the CONTRACT-3 allowlist (weights + required metadata) — populated by E2.
    selected_files: list[str]
    # deriver.resolve_hf_home(...) — host HF_HOME root.
    hf_home: Path
    # C2aResult — disk verdict already computed on selected_files (P3 gates).
    c2a: Any
    # min compute_cap across selected GPUs (P3 gates.py detection path / the
    # --hardware override) — reuse, never re-invent.
    hardware_sm: float
    # nvidia-smi -L count (or --hardware-gpus override).
    visible_gpu_count: int
    # nvidia-smi --query-gpu=memory.total over ALL visible GPUs — topology.
    per_gpu_vram_mib: list[int]
    # the EXACT tp GPU indices the derived compose will bind.
    selected_gpu_indices: list[int]
    # VRAM of selected_gpu_indices — the ONLY list [B] prices against.
    selected_gpu_vram_mib: list[int]
    # canonical sorted (gpu_name, vram_mib) tuples — §6.2 topology_summary.
    topology_summary: str
    # `git -C <repo> rev-parse HEAD` captured on the HOST in run_pull() and
    # passed in (NEVER git-in-container) — feeds the §6.2 fingerprint.
    club3090_commit: str
    diagnostics: dict = field(default_factory=dict)
