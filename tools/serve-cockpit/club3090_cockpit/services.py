"""Cockpit service layer — the data/service API the panes call.

Wraps the Phase-2 ``--json`` contracts + the shared-core detect into a clean,
dependency-injectable Python API.  Panes (and the Wire step) call ``CockpitData``
methods; tests construct it with a ``FakeRunner`` so no subprocess / GPU / docker
is ever touched.

Contracts wrapped (all READ-only, safe to call live):
  - ``scripts/lib/registry-emit.sh --json``           → load_catalog / containers
  - ``scripts/switch.sh --explain <slug> --json``      → explain / fit + measurement join
  - ``tools/kv-calc.py --fit <slug> --card <c> --json``→ fit
  - ``scripts/pull.sh <repo> --profile-like <k> --dry-run --json`` → byo_check
  - ``scripts/lib/profiles/estate_cli.py report-state --json`` → estate_state
  - ``scripts/gpu-mode.sh --list-modes --json``        → estate_state (scene catalog)
  - ``scripts/health.sh`` (text)                       → estate_state (Doctor read)
  - core ``detect_endpoint`` / ``get_gpu_info``        → estate_state / containers / reconcile

WRITES (serve / scene_switch / set_default / clear_default / estate_down /
container_action) are WIRED as ``ActionPlan`` builders + an ``execute_action``
that streams via the core SubprocessRunner.  In this phase they are NEVER run
live; tests mock execution.  ``execute_action`` always re-runs the reconcile
gate first and refuses if unsafe (unless an explicit, reasoned force override
is supplied).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol

from club3090_tui_core.detect import (
    GpuInfo,
    ServingTarget,
    detect_endpoint as core_detect_endpoint,
    get_gpu_info as core_get_gpu_info,
    match_target_to_registry,
)
from club3090_tui_core.registry import VariantRow, parse_variant_rows
from club3090_tui_core.runner import SubprocessRunner

from .data import (
    ActionPlan,
    BenchRow,
    ByoResult,
    CatalogEntry,
    ContainerInfo,
    ContainerTop,
    DiskUsage,
    DoctorRead,
    DoctorReport,
    EstateDiagnose,
    EstateState,
    EstateTelemetry,
    EvaluateHandoff,
    EvidenceReport,
    EvidenceTag,
    FitVerdict,
    GpuCompApp,
    GpuConflict,
    Measurement,
    MeasuredNumbers,
    MeasureVsBar,
    OptimizerReport,
    PowerCapState,
    ProfileTriage,
    PromoteScaffold,
    RamUsage,
    ReconcileResult,
    Scene,
    SceneServiceState,
    ServedProbe,
    VerifyFull,
    VerifySmoke,
    WEIGHTS_ABSENT,
    WEIGHTS_PARTIAL,
    WEIGHTS_PRESENT,
    WEIGHTS_UNKNOWN,
    WeightsMeta,
    _bench_row_matches,
    _canon_engine_family,
    _canon_model_key,
    _measure_verdict,
    attribute_gpu_apps,
    bench_row_from_corpus_record,
    bench_rows_from_benchmarks_md,
    compute_promote_scaffold,
    measured_from_internal_json,
    measured_from_report_md,
    measurement_from_explain_benchmarks,
    parse_benchmarks_md_for_slug,
    parse_compute_apps,
    parse_df_output,
    parse_docker_ps_id_names,
    parse_docker_top,
    parse_health_text,
    parse_meminfo,
    parse_power_cap_status,
    parse_profile_triage,
    parse_verify_full,
    parse_verify_smoke,
)

# ── Local card name (this rig) ──────────────────────────────────────────────────

# The locally-detected per-card name used for fit joins.  RTX 3090 is the rig
# default; ``CockpitData(card=...)`` overrides it.  Detection from nvidia-smi is
# done lazily in ``detect_local_card`` so headless tests never shell out.
DEFAULT_CARD = "rtx-3090"

# The default WEIGHTS ROOT — the dir that DIRECTLY holds the model subdirs
# (``<root>/<subdir>``), the SAME convention setup.sh uses for ``${MODEL_DIR}``
# (NO extra ``huggingface`` segment appended by us).  On this rig that's
# ``/mnt/models/huggingface`` (``/mnt/models`` is the volume; ``/mnt/models/gguf``
# is a back-compat symlink → ``huggingface``).  Resolution prefers the repo .env
# (the shared config setup.sh reads) over this default — see weights_model_dir.
# SAFE to surface (it's the model dir, not a secret).
MODEL_DIR = "/mnt/models/huggingface"


# ── Subprocess runner protocol (dependency injection seam) ───────────────────────


class RunResult:
    """Result of a read-only subprocess call."""

    def __init__(self, returncode: int, stdout: str, stderr: str, timed_out: bool = False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class Runner(Protocol):
    """Async subprocess seam.  Real impl shells out; fake impl returns canned
    output keyed on the command.  All READ contracts go through ``run``."""

    async def run(
        self, cmd: list[str], *, cwd: str, timeout: float = 30.0
    ) -> RunResult: ...


class RealRunner:
    """Production runner — actually shells out (READ contracts only)."""

    async def run(
        self, cmd: list[str], *, cwd: str, timeout: float = 30.0
    ) -> RunResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return RunResult(
                returncode=proc.returncode if proc.returncode is not None else -1,
                stdout=out.decode("utf-8", errors="replace"),
                stderr=err.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            return RunResult(returncode=-1, stdout="", stderr="timeout", timed_out=True)
        except FileNotFoundError as exc:
            return RunResult(returncode=127, stdout="", stderr=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return RunResult(returncode=-1, stdout="", stderr=str(exc))


# Detect seam: async callables matching the core signatures.
DetectEndpointFn = Callable[[], Awaitable[ServingTarget]]
GetGpuInfoFn = Callable[[], Awaitable[list[GpuInfo]]]
# A7: probe the live engine for its ACTUAL running config (ctx + image).  Takes
# the detected ServingTarget (for url / container) and returns a ServedProbe.
ProbeServedFn = Callable[[Any], Awaitable["ServedProbe"]]


# ── The service class ───────────────────────────────────────────────────────────


class CockpitData:
    """Clean, dependency-injectable data/service API for the cockpit panes.

    Injectable seams (all default to the real implementations):
      - ``runner``            : Runner  — read-only subprocess calls
      - ``detect_endpoint_fn``: core detect_endpoint (running container probe)
      - ``get_gpu_info_fn``   : core get_gpu_info (nvidia-smi)
      - ``write_runner``      : SubprocessRunner — streams WRITE actions (never
                                executed in tests / this phase)
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        card: str = DEFAULT_CARD,
        runner: Optional[Runner] = None,
        detect_endpoint_fn: Optional[DetectEndpointFn] = None,
        get_gpu_info_fn: Optional[GetGpuInfoFn] = None,
        probe_served_fn: Optional[ProbeServedFn] = None,
        write_runner: Optional[SubprocessRunner] = None,
        download_runner: Optional[SubprocessRunner] = None,
    ):
        self.repo_root = Path(repo_root)
        self.card = card
        # FIX 2 — the registry's top-level ``defaults`` array (curated
        # per-(model,engine,topology) recommendations) from registry-emit --json.
        # Surfaced here (not on the per-row CatalogEntry) because it's a catalog
        # property; ``profile_templates`` reads it to pick a FUNCTIONAL,
        # registry-recommended representative per (family, topology).  Refreshed
        # on each ``load_catalog_rows``; empty on the raw-tab fallback path.
        self.catalog_defaults: list[dict] = []
        self._runner: Runner = runner or RealRunner()
        self._detect_endpoint: DetectEndpointFn = detect_endpoint_fn or core_detect_endpoint
        self._get_gpu_info: GetGpuInfoFn = get_gpu_info_fn or core_get_gpu_info
        # A7: the live-config probe seam.  Defaults to the real httpx + docker
        # inspect probe; tests inject a fake so no network / docker is touched.
        self._probe_served: ProbeServedFn = probe_served_fn or self._real_probe_served
        # Write runner is constructed lazily and NEVER invoked in this phase.
        self._write_runner = write_runner or SubprocessRunner(self.repo_root)
        # SEPARATE runner for weight downloads (Download UX): a download streams
        # for minutes/hours, so it must NOT occupy the write-runner (which serves
        # GPU writes) — otherwise a serve couldn't run while a download is going,
        # and a second start_raw would clobber the download's process handle.
        # conftest blocks SubprocessRunner.start_raw at the class level, so this
        # default is test-safe; download tests inject a fake.
        self._download_runner = download_runner or SubprocessRunner(self.repo_root)
        # Dual-writer serialization (design §3.2): the reconcile→execute window
        # must be ATOMIC.  Without this, two confirmed plans can both reconcile
        # "safe" before either claims VRAM (TOCTOU).  Held across the whole gate
        # + dispatch in execute_action so a second write cannot run its gate
        # until the first has finished claiming the cards.
        self._write_lock = asyncio.Lock()
        # In-process pending-claim registry (§3.2 TOCTOU fix).
        # Maps token -> (gpu_set, expiry_monotonic).  Registered UNDER the lock
        # before start_raw is called; cleared when the write subprocess exits.
        # A TTL (600 s) ensures a leaked claim cannot block the gate forever.
        self._pending_claims: dict[str, tuple[frozenset[int], float]] = {}
        # Leak backstop ONLY — the real lifecycle is "clear on the write
        # subprocess's completion" (see _release_claim_when_done). switch.sh's
        # own READY_TIMEOUT is already 600s, so a TTL near that would prune a
        # still-booting claim; keep it well above a worst-case boot.
        self._claim_ttl = 1800.0  # seconds

    # ── small JSON helper ──────────────────────────────────────────────────────

    async def _run_json(
        self, cmd: list[str], *, timeout: float = 30.0
    ) -> tuple[Any, Optional[str]]:
        """Run a read contract, parse stdout as JSON.  Returns (data, error)."""
        res = await self._runner.run(cmd, cwd=str(self.repo_root), timeout=timeout)
        if res.timed_out:
            return None, f"timed out after {timeout:.0f}s: {' '.join(cmd[:2])}"
        if not res.stdout.strip():
            # Some contracts print diagnostics to stderr only on failure.
            return None, (res.stderr.strip()[:200] or f"empty output (rc={res.returncode})")
        try:
            return json.loads(res.stdout), None
        except json.JSONDecodeError as exc:
            # Contracts may prepend non-JSON banner lines on stderr only, but if
            # stdout itself is dirty, try to recover the first JSON value.
            recovered = _extract_first_json(res.stdout)
            if recovered is not None:
                return recovered, None
            return None, f"JSON parse error: {exc}"

    # ── READ: catalog ────────────────────────────────────────────────────────────

    async def load_catalog(
        self, *, enrich_fit: bool = True, enrich_measurement: bool = True
    ) -> tuple[list[CatalogEntry], Optional[str]]:
        """Fully-enriched variant rows: load_catalog_rows() + optional fit
        (batched kv-calc --fit-all) + measurement (parallel) enrichment.

        Enrichment is best-effort: a failed join leaves the stub glyph rather
        than failing the whole load. The cockpit paints load_catalog_rows()
        first and enriches in the background; this combined entry point is kept
        for callers/tests that want a fully-enriched result in one await.
        """
        entries, err = await self.load_catalog_rows()
        if err and not entries:
            return [], err

        if enrich_fit:
            await self.enrich_fits(entries)
        if enrich_measurement:
            await self.enrich_measurements(entries)

        if not entries:
            return [], "No variants returned — registry may be empty"
        return entries, None

    async def load_catalog_rows(self) -> tuple[list[CatalogEntry], Optional[str]]:
        """Registry rows ONLY — the fast first paint (no fit/measurement
        enrichment). One read of compose_registry.py via registry-emit.sh
        --json, with the raw-tab fallback if the --json wrapper regresses."""
        data, err = await self._run_json(
            ["bash", "scripts/lib/registry-emit.sh", "--json"], timeout=30.0
        )
        if err and not data:
            # Fall back to the raw tab emitter (registry_variant_rows) so the
            # catalog still loads even if the --json wrapper regresses.  The raw
            # emitter has no `defaults` array, so the profile-template picker
            # degrades to the status floor (still functional-only).
            self.catalog_defaults = []
            rows, ferr = await self._load_catalog_rows_fallback()
            if ferr:
                return [], err
        else:
            data = data or {}
            rows = [_variant_row_from_dict(d) for d in data.get("variants", [])]
            # FIX 2 — surface the curated defaults so profile_templates can pick
            # the registry's own recommendation per (family, topology).
            d = data.get("defaults")
            self.catalog_defaults = list(d) if isinstance(d, list) else []

        return [CatalogEntry(row=r) for r in rows], None

    async def _load_catalog_rows_fallback(self) -> tuple[list[VariantRow], Optional[str]]:
        cmd = [
            "bash",
            "-c",
            'source "$1/scripts/lib/registry-emit.sh" && registry_variant_rows "$1"',
            "bash",
            str(self.repo_root),
        ]
        res = await self._runner.run(cmd, cwd=str(self.repo_root), timeout=30.0)
        if not res.stdout.strip():
            return [], res.stderr.strip()[:200] or "no rows"
        return parse_variant_rows(res.stdout), None

    # ── Download state (Download UX): which slugs' weights are on disk ────────────

    async def weights_index(self) -> dict[tuple[str, str], WeightsMeta]:
        """Static weights metadata for every ``(model, variant)`` — ONE
        ``weights.py list --json`` subprocess, cached for the process (the model
        profiles are static).  ``{}`` on error (download-state degrades to
        'unknown'; the catalog still renders)."""
        cache = getattr(self, "_weights_index_cache", None)
        if cache is not None:
            return cache
        index: dict[tuple[str, str], WeightsMeta] = {}
        try:
            res = await self._runner.run(
                ["python3", "scripts/lib/profiles/weights.py", "list", "--json"],
                cwd=str(self.repo_root),
                timeout=20.0,
            )
            rows = json.loads(res.stdout) if res.ok and res.stdout.strip() else []
        except Exception:
            rows = []
        for d in rows if isinstance(rows, list) else []:
            try:
                m = WeightsMeta.from_dict(d)
            except Exception:
                continue
            if m.model and m.variant:
                index[(m.model, m.variant)] = m
        self._weights_index_cache = index
        return index

    def weights_state_for(
        self,
        entry: CatalogEntry,
        index: dict[tuple[str, str], WeightsMeta],
        *,
        model_dir: Optional[str] = None,
    ) -> tuple[str, Optional[WeightsMeta]]:
        """Join ``entry`` to its weights meta + stat the model dir → ``(state,
        meta)``.  PRESENT = subdir has ≥1 verify_glob match; PARTIAL = subdir
        exists but no match (interrupted / wrong); ABSENT = subdir missing;
        UNKNOWN = no weights entry to join (e.g. a self-grabbed GGUF compose)."""
        meta = index.get((entry.model, entry.weights_variant))
        if meta is None or not meta.subdir:
            return WEIGHTS_UNKNOWN, meta
        root = Path(model_dir or self.weights_model_dir())
        base = root / meta.subdir
        try:
            if not base.is_dir():
                return WEIGHTS_ABSENT, meta
            if not any(base.glob(meta.verify_glob)):
                return WEIGHTS_PARTIAL, meta
            # Core is present — but a slug with companions (a DFlash draft / mmproj
            # projector its compose mounts) is only truly READY when those are on
            # disk too; otherwise Start would serve-fail.  Treat a present-core but
            # missing-companion as PARTIAL so the Download action still fires.  A
            # companion we can't resolve in the index doesn't block (degrade open).
            for ck in (getattr(entry, "weights_companions", None) or []):
                cvar = ck.split(":", 1)[1] if ":" in ck else ck
                cmeta = index.get((entry.model, cvar))
                if cmeta is None or not cmeta.subdir:
                    continue
                cbase = root / cmeta.subdir
                if not (cbase.is_dir() and any(cbase.glob(cmeta.verify_glob))):
                    return WEIGHTS_PARTIAL, meta
            return WEIGHTS_PRESENT, meta
        except OSError:
            return WEIGHTS_UNKNOWN, meta

    async def enrich_weights(
        self, entries: list[CatalogEntry], *, model_dir: Optional[str] = None
    ) -> None:
        """Set each entry's ``weights_state`` + ``weights`` from the index (ONE
        weights.py call + a stat per entry).  Best-effort — an error leaves the
        entry at its default 'unknown'.  Mirrors enrich_fits/enrich_measurements:
        a post-first-paint enrichment, NOT part of the fast registry-only paint."""
        index = await self.weights_index()
        for e in entries:
            try:
                e.weights_state, e.weights = self.weights_state_for(
                    e, index, model_dir=model_dir
                )
            except Exception:
                pass

    # ── Download (Download UX): fetch a slug's weights via setup.sh ───────────────

    def weights_download_plan(self, model: str, variant: str) -> ActionPlan:
        """The download ActionPlan: ``WEIGHT_KEY=<model>:<variant> bash
        scripts/setup.sh <model>`` (the user's chosen in-repo fetch: whole-repo
        HF pull + per-file SHA verify).  setup.sh accepts ``WEIGHT_KEY`` directly
        for an exact catalog entry.  A DISK write, NOT a GPU write — no reconcile
        gate; the pop-up's Download button is itself the confirm.  ``WEIGHT_KEY``
        is injected into the child env at run time (run_weights_download), kept
        off ``cmd`` so the plan stays inspectable."""
        return ActionPlan(
            kind="download",
            cmd=["bash", "scripts/setup.sh", model],
            description=f"download {model}:{variant} weights (setup.sh)",
            requires_reconcile=False,
            requires_confirm=False,
        )

    async def run_weights_download(
        self,
        model: str,
        variant: str,
        *,
        companions: Optional[list[str]] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Launch the weights download, streamed via the core runner (no GPU
        claim).  ``WEIGHT_KEY`` selects the core variant; ``companions`` (the
        slug's registry ``weights_companions`` — a DFlash draft / mmproj projector)
        are passed as ``WEIGHT_EXTRA_KEYS`` so setup.sh fetches them ALONGSIDE the
        core (otherwise the slug reads "present" then fails to boot).  ``HF_HOME``
        is derived from the model dir so HF's staging cache lands on the big models
        disk (off root).  WIRED-BUT-MOCK-ONLY in tests (conftest blocks the real
        spawn); returns the streaming run handle."""
        plan = self.weights_download_plan(model, variant)
        env = dict(os.environ)
        env["WEIGHT_KEY"] = f"{model}:{variant}"
        # Pin setup.sh to the SAME weights root the cockpit reads, so the download
        # can't land somewhere the cockpit then can't see (setup.sh writes to
        # ``${MODEL_DIR}/<subdir>``, the same root convention weights_model_dir
        # now uses).  This is the cockpit↔scripts single-source-of-truth seam.
        root = self.weights_model_dir()
        env["MODEL_DIR"] = root
        comp_keys = [c if ":" in c else f"{model}:{c}" for c in (companions or []) if c]
        if comp_keys:
            env["WEIGHT_EXTRA_KEYS"] = " ".join(comp_keys)
        env.setdefault("HF_HOME", str(Path(root) / ".cache" / "huggingface"))
        if on_line is not None:
            self._download_runner.set_callbacks(on_line=on_line)
        return await self._download_runner.start_raw(
            plan.cmd, env=env, run_type=plan.kind, parser=None
        )

    async def cancel_weights_download(self) -> None:
        """Cancel the in-flight download (kills the setup.sh/hf process group via
        the download runner's SIGINT→TERM→KILL).  Best-effort; safe to call when
        nothing is running."""
        try:
            await self._download_runner.cancel()
        except Exception:
            pass

    def _dotenv_model_dir(self) -> str:
        """``MODEL_DIR`` from the repo ``.env`` — the SHARED config setup.sh reads.

        Reading it here lets the cockpit resolve the SAME weights root as the
        scripts (the single-source-of-truth goal) without re-implementing the
        convention.  ``""`` if the file is absent / unreadable / has no MODEL_DIR."""
        try:
            envf = self.repo_root / ".env"
            if not envf.is_file():
                return ""
            for raw in envf.read_text(encoding="utf-8", errors="replace").splitlines():
                s = raw.strip()
                if s.startswith("MODEL_DIR=") and not s.startswith("#"):
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            return ""
        return ""

    def weights_model_dir(self) -> str:
        """The WEIGHTS ROOT — the dir that DIRECTLY holds the model subdirs
        (``<root>/<subdir>``), the SAME convention setup.sh uses (no extra
        ``huggingface`` segment — that double-append was the cockpit↔scripts
        divergence this resolves).

        Precedence (highest first): an in-app / persisted value (``_model_dir``,
        set at launch by ``apply_persisted_settings`` from the ``$MODEL_DIR`` env
        var or the saved settings, or live by the Settings screen) > the
        ``$MODEL_DIR`` env var (also covers a bare ``CockpitData`` built outside
        ``__main__`` — tests / embedding) > ``MODEL_DIR`` in the repo ``.env`` (so
        the cockpit and setup.sh land on the SAME root) > the bundled default."""
        configured = getattr(self, "_model_dir", None)
        if configured:
            return configured
        env_dir = (os.environ.get("MODEL_DIR") or "").strip()
        if env_dir:
            return env_dir
        dotenv = self._dotenv_model_dir()
        if dotenv:
            return dotenv
        return MODEL_DIR

    # Backup / cruft a user leaves in the model dir when swapping weights (rename
    # the old GGUF to *.bak, etc.).  EXCLUDED from the byte count — otherwise a
    # leftover ~= full-size .bak pins download progress at a false ~99%.  NOTE:
    # hf's in-progress ``*.incomplete`` staging files are deliberately NOT excluded
    # (they ARE the live download — counting them is what makes the % move).
    _BACKUP_SUFFIXES = (".bak", ".old", ".orig", ".disabled", ".save")

    def weights_bytes_on_disk(self, meta: WeightsMeta, *, model_dir: Optional[str] = None) -> int:
        """Total bytes currently under ``<model_dir>/huggingface/<subdir>`` — the
        download-progress signal.  0 when the dir is absent.  Backup/cruft files
        (``*.bak`` / ``*.old`` / ``*.orig`` / ``foo~`` …) are skipped so a leftover
        old-quant copy can't inflate the count; hf's ``*.incomplete`` staging IS
        counted so live progress tracks the transfer."""
        base = Path(model_dir or self.weights_model_dir()) / meta.subdir
        if not base.is_dir():
            return 0
        total = 0
        try:
            for f in base.rglob("*"):
                try:
                    if not f.is_file():
                        continue
                    name = f.name.lower()
                    if name.endswith(self._BACKUP_SUFFIXES) or name.endswith("~"):
                        continue
                    total += f.stat().st_size
                except OSError:
                    continue
        except OSError:
            return 0
        return total

    def weights_download_progress(
        self, meta: WeightsMeta, *, model_dir: Optional[str] = None
    ) -> Optional[int]:
        """Download progress % from bytes-on-disk vs ``size_gb`` — ``None`` when
        size is unknown.  Capped at 99 (100 is reserved for verify-confirmed
        present, set by re-stat on completion)."""
        if not meta.size_gb or meta.size_gb <= 0:
            return None
        got = self.weights_bytes_on_disk(meta, model_dir=model_dir)
        pct = int(got / (float(meta.size_gb) * 1e9) * 100)
        return max(0, min(99, pct))

    def download_set_metas(
        self, entry: CatalogEntry, index: dict[tuple[str, str], WeightsMeta]
    ) -> list[WeightsMeta]:
        """The FULL weight-meta set a slug's download fetches: the core
        ``weights_variant`` PLUS each companion (resolved via the index).  The
        download-progress signal must aggregate over this whole set — otherwise a
        slug whose core is already on disk (only a companion is missing) reads a
        static ~99% off the core subdir while the companion downloads into its own
        subdir uncounted.  Companions that don't resolve in the index are skipped."""
        metas: list[WeightsMeta] = []
        core = index.get((entry.model, entry.weights_variant))
        if core is not None:
            metas.append(core)
        for ck in (entry.weights_companions or []):
            cvar = ck.split(":", 1)[1] if ":" in ck else ck
            cm = index.get((entry.model, cvar))
            if cm is not None:
                metas.append(cm)
        return metas

    def weights_download_progress_set(
        self, metas: list[WeightsMeta], *, model_dir: Optional[str] = None
    ) -> Optional[int]:
        """Aggregate download progress across a set (core + companions): total
        bytes-on-disk / total ``size_gb``, capped at 99.  ``None`` when no size is
        known.  This is the value the cockpit shows for an in-flight download so
        the % MOVES as each artifact lands (vs the core-only static-99 trap)."""
        total_size = sum(float(m.size_gb) for m in metas if m.size_gb)
        if total_size <= 0:
            return None
        got = sum(self.weights_bytes_on_disk(m, model_dir=model_dir) for m in metas)
        return max(0, min(99, int(got / (total_size * 1e9) * 100)))

    def weights_fits_disk(
        self, meta: WeightsMeta, *, model_dir: Optional[str] = None
    ) -> tuple[bool, float, float]:
        """Disk pre-check before a download → ``(fits, free_gb, need_gb)``.
        ``need`` carries a 10% headroom over ``size_gb``.  Fits=True (unknown
        free / unknown size) when it can't be determined — never block on a
        read error, just skip the guard."""
        need = float(meta.size_gb or 0) * 1.10
        try:
            free_gb = shutil.disk_usage(model_dir or self.weights_model_dir()).free / 1e9
        except OSError:
            return True, 0.0, need
        if need <= 0:
            return True, free_gb, need
        return (free_gb >= need), free_gb, need

    # enrich_measurements still fans out one switch --explain per slug; a
    # cap-bounded asyncio.gather keeps that to a few seconds without flooding
    # the box. enrich_fits no longer fans out — it uses the single kv-calc
    # --fit-all batch (one process for the whole catalog).
    _ENRICH_CONCURRENCY = 12

    async def fit_all(self, card: Optional[str] = None) -> dict:
        """kv-calc.py --fit-all --card <card> --json — fit verdict for EVERY
        registry slug in ONE process. Returns {slug: verdict_dict}, or {} on
        error (catalog still renders, fit column shows the stub glyph)."""
        cmd = ["python3", "tools/kv-calc.py", "--fit-all", "--json"]
        c = card or self.card
        if c:
            cmd += ["--card", c]
        data, _err = await self._run_json(cmd, timeout=30.0)
        return (data or {}).get("variants", {}) or {}

    async def enrich_fits(self, entries: list[CatalogEntry]) -> None:
        """Fit column for the whole catalog via ONE kv-calc --fit-all call
        (the batch replaces the former per-slug fan-out)."""
        variants = await self.fit_all(self.card)
        for e in entries:
            vd = variants.get(e.slug)
            if vd is not None:
                e.fit = FitVerdict.from_dict(vd, card=self.card)
            elif (e.row.kvcalc_key or "").upper() == "SKIP":
                # ik/llama composes (kvcalc_key SKIP) — no vLLM kv-calc fit.
                e.fit = FitVerdict(verdict="skip", card=self.card)

    async def enrich_measurements(self, entries: list[CatalogEntry]) -> None:
        # Read BENCHMARKS.md once up front — the per-slug md fallback below is
        # then a pure in-memory parse (no I/O per slug).
        bench_md = self._read_benchmarks_md()
        sem = asyncio.Semaphore(self._ENRICH_CONCURRENCY)

        async def _one(e: CatalogEntry) -> None:
            # Preferred: structured benchmarks from the explain contract.  The
            # REAL shape is [{"row","columns"}]; measurement_from_explain_*
            # parses TPS out of columns[].  Only COMMIT the explain result when
            # it actually yields a TPS — otherwise an empty benchmarks[] (or a
            # row that is stress/soak-only) must NOT suppress the markdown
            # fallback (the `continue`-suppresses-fallback bug this fixes).
            async with sem:
                explain, _err = await self.explain(e.slug)
            if explain and explain.get("benchmarks"):
                m = measurement_from_explain_benchmarks(explain["benchmarks"])
                if m.narr_tps is not None or m.code_tps is not None:
                    e.measurement = m
                    return
            # Fallback: coarse BENCHMARKS.md scrape (flagged in source).
            m = parse_benchmarks_md_for_slug(bench_md or "", e.slug)
            if m:
                e.measurement = m

        await asyncio.gather(*(_one(e) for e in entries))

    def _read_benchmarks_md(self) -> str:
        path = self.repo_root / "BENCHMARKS.md"
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # ── READ: explain (Tier-3 detail) ────────────────────────────────────────────

    async def explain(self, slug: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """switch.sh --explain <slug> --json — full per-slug story."""
        data, err = await self._run_json(
            ["bash", "scripts/switch.sh", "--explain", slug, "--json"], timeout=45.0
        )
        if data is None:
            return None, err
        return data, None

    # ── READ: fit ──────────────────────────────────────────────────────────────────

    async def fit(self, slug: str, card: Optional[str] = None) -> FitVerdict:
        """kv-calc.py --fit <slug> --card <card> --json."""
        card = card or self.card
        data, err = await self._run_json(
            ["python3", "tools/kv-calc.py", "--fit", slug, "--card", card, "--json"],
            timeout=40.0,
        )
        if data is None:
            return FitVerdict(verdict="unknown", card=card, error=err or "")
        return FitVerdict.from_dict(data, card=card)

    # ── READ: BYO check ──────────────────────────────────────────────────────────────

    async def byo_check(self, repo: str, profile_like: str) -> ByoResult:
        """pull.sh <repo> --profile-like <key> --dry-run --json.

        ``--dry-run`` forces Path B (evaluate only, never download/emit), so this
        is safe to call live.  The structured ``swap_path`` block drives the
        Route-C reuse suggestion in the BYO pane.
        """
        data, err = await self._run_json(
            [
                "bash",
                "scripts/pull.sh",
                repo,
                "--profile-like",
                profile_like,
                "--dry-run",
                "--json",
            ],
            timeout=90.0,
        )
        if data is None:
            return ByoResult(repo=repo, profile_like=profile_like, error=err or "no output")
        return ByoResult.from_dict(repo, profile_like, data)

    # ── READ: containers ────────────────────────────────────────────────────────────

    async def containers(
        self, variants: Optional[list[VariantRow]] = None
    ) -> list[ContainerInfo]:
        """The stack containers that hold GPUs, read-only via docker ps.

        Three classes are surfaced (each is a potential GPU user the reconcile
        gate must see):
          - **engine** containers (``vllm-`` / ``llama-cpp-`` / ``ik-llama-`` /
            ``sglang-`` / ``beellama-``) — slug-matched against the registry;
          - **estate** containers (``club3090-<name>`` — booted by the estate
            planner);
          - **service** containers — rig services that hold a GPU (ComfyUI /
            Step-Audio).

        Engine containers get their slug matched against the registry when
        ``variants`` is supplied."""
        infos: list[ContainerInfo] = []
        for name, host_port, internal_port, engine, kind in await self._docker_ps_stack_containers():
            slug = ""
            if kind == "engine" and variants:
                tmp = ServingTarget(container=name, host_port=host_port)
                tmp = match_target_to_registry(tmp, variants)
                slug = tmp.slug
            infos.append(
                ContainerInfo(
                    name=name,
                    kind=kind,
                    host_port=host_port,
                    internal_port=internal_port,
                    engine=engine,
                    slug=slug,
                )
            )
        return infos

    def _known_service_dirs(self) -> list[str]:
        """Enumerate the KNOWN supporting services from the repo's
        ``services/<name>/docker-compose.yml`` tree (READ — a filesystem scan).

        These are the rig's full supporting estate (ComfyUI / LiteLLM / OpenWebUI
        / Qdrant / SearXNG / Studio …); a service is "known" if its
        directory carries a ``docker-compose.yml``.  Returns the sorted service
        names; empty when the tree is absent (e.g. the test fake root)."""
        base = self.repo_root / "services"
        names: list[str] = []
        try:
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "docker-compose.yml").is_file() or (
                    child / "docker-compose.yaml"
                ).is_file():
                    names.append(child.name)
        except (OSError, FileNotFoundError):
            return []
        return names

    async def _running_container_ports(self) -> list[tuple[str, int]]:
        """READ — every running container as ``(name, first-published-host-port)``
        via ``docker ps``, INDEPENDENT of ``_classify_container_kind`` (so the
        non-GPU supporting services ``_docker_ps_stack_containers`` filters out are
        included).  The de-dup + port source for :meth:`_merge_known_services`.
        Goes through the injected runner so tests stay mockable; failures degrade
        to an empty list (callers bias toward running / no-port).

        The port is the FIRST published host port (``:(\\d+)->`` in the docker-ps
        ``Ports`` field — a GENERAL match, not the engine-only ``PORT_MAP_BROAD_RE``
        whose internal-port allow-list would miss a web service like litellm:4000)."""
        res = await self._runner.run(
            ["docker", "ps", "--format", "{{.Names}}|{{.Ports}}"],
            cwd=str(self.repo_root),
            timeout=10.0,
        )
        if not res.ok:
            return []
        out: list[tuple[str, int]] = []
        for line in res.stdout.splitlines():
            if "|" not in line:
                # Tolerate a name-only canned response (tests / `{{.Names}}`).
                name = line.strip()
                if name:
                    out.append((name, 0))
                continue
            name, ports_str = line.split("|", 1)
            name = name.strip()
            if not name:
                continue
            m = _HOST_PORT_RE.search(ports_str)
            out.append((name, int(m.group(1)) if m else 0))
        return out

    async def _merge_known_services(
        self, running: list[ContainerInfo]
    ) -> list[ContainerInfo]:
        """Union the running stack containers with the KNOWN ``services/`` estate.

        The set = {running stack containers} ∪ {known ``services/`` entries}.
        Each known service dir resolves to one of three outcomes:

        - **already a stack row** (a GPU service surfaced by
          ``_docker_ps_stack_containers`` — ComfyUI / Step-Audio): omitted here,
          it's already represented (running) in ``running``;
        - **running, but NOT a stack row** (a non-GPU supporting service —
          ``litellm`` / ``qdrant`` / ``searxng`` / ``open-webui``):
          appended as a **running** ``status="running"`` ContainerInfo so it
          shows live with working actions (logs / top / restart / stop / rm);
        - **not running at all**: appended as a greyed, read-only
          ``status="stopped"`` ContainerInfo.

        The running/stopped decision keys off the FULL ``docker ps`` set
        (``_running_container_ports``), NOT ``_docker_ps_stack_containers`` —
        the latter drops every non-GPU supporting service via
        ``_classify_container_kind``, so a RUNNING ``litellm`` would otherwise be
        falsely rendered "stopped" and its actions wrongly suppressed.  Matching
        NORMALIZES both sides (lowercase, strip ``-``/``_``) so a service dir
        ``open-webui`` matches a running container named ``openwebui``.  Bias is
        toward running: the running-set read failing degrades to an EMPTY set, so
        on a transient read miss only genuinely-known services fall through to
        "stopped" — a possibly-live service is never action-suppressed on a stale
        read of the un-classified set (the classified stack rows still count)."""
        out = list(running)
        # (actual container name, host port) for every running container, so a
        # matched supporting service carries BOTH the port it serves on AND its
        # real container name.
        running_pairs = await self._running_container_ports()
        # Names already present as stack rows (GPU services + engines/estate) —
        # those service dirs are de-duped (omitted) since they're already a row.
        stack_norm = {_normalize_service_name(c.name) for c in running}
        stack_norm.discard("")
        for svc in self._known_service_dirs():
            if any(_service_dir_matches_running(svc, sn) for sn in stack_norm):
                continue  # already a (running) stack row — don't duplicate
            # Find the matching RUNNING container — capture its ACTUAL name + port.
            # The container name MAY differ from the service dir (services/openwebui/
            # → container_name "open-webui"), so the row MUST carry the real name or
            # stop / restart / logs would target a non-existent container.
            match_name: Optional[str] = None
            match_port = 0
            for orig, port in running_pairs:
                norm = _normalize_service_name(orig)
                if norm and _service_dir_matches_running(svc, norm):
                    match_name, match_port = orig, port
                    break
            is_running = match_name is not None
            # Non-GPU supporting services (litellm / qdrant / searxng / open-webui)
            # carry an ``engine="web"`` tag + the port they serve on, so the
            # Containers table shows what they are instead of a bare "—".  A
            # GPU-holding rig service (ComfyUI / Step-Audio — classifies non-None)
            # keeps no web tag; it surfaces with its real engine as a stack row.
            engine = "" if _classify_container_kind(svc) is not None else "web"
            out.append(
                ContainerInfo(
                    # real container name when up (for stop/restart/logs); the dir
                    # name when down (service_start uses it for the compose path).
                    name=match_name if is_running else svc,
                    kind="service",
                    status="running" if is_running else "stopped",
                    engine=engine,
                    host_port=match_port,
                )
            )
        return out

    async def _docker_ps_stack_containers(
        self,
    ) -> list[tuple[str, int, int, str, str]]:
        """Read-only docker ps for every stack container that can hold a GPU.

        Returns ``(name, host_port, internal_port, engine, kind)`` where
        ``kind ∈ {engine, estate, service}``.  Engine prefixes come from the
        core ``ENGINE_PREFIXES``; estate containers carry the ``club3090-``
        prefix; a small set of GPU-holding rig services is matched by name so
        the gate sees ComfyUI / Step-Audio too (they don't share a prefix)."""
        from club3090_tui_core.detect import (
            ENGINE_PREFIXES,
            PORT_MAP_BROAD_RE,
            _classify_engine,
            _classify_engine_from_container,
        )

        res = await self._runner.run(
            ["docker", "ps", "--format", "{{.Names}}|{{.Ports}}"],
            cwd=str(self.repo_root),
            timeout=10.0,
        )
        # Fix 2 (docker ps fail-closed): a timed-out or failed docker ps must
        # NOT yield an empty container list (which looks like "no conflicts").
        # Raise so reconcile_before_write can catch it and set safe=False.
        if not res.ok:
            raise RuntimeError(
                f"docker ps read failed (rc={res.returncode}, "
                f"timed_out={res.timed_out}): {res.stderr.strip()[:120]}"
            )
        out: list[tuple[str, int, int, str, str]] = []
        seen: set[tuple[str, int]] = set()
        for line in res.stdout.splitlines():
            if "|" not in line:
                continue
            name, ports_str = line.split("|", 1)
            kind = _classify_container_kind(name)
            if kind is None:
                continue
            engine = ""
            if kind == "engine":
                engine = _classify_engine_from_container(name)
            matched_port = False
            for match in PORT_MAP_BROAD_RE.finditer(ports_str):
                host_port = int(match.group(1))
                internal_port = int(match.group(2))
                key = (name, host_port)
                if key in seen:
                    continue
                seen.add(key)
                eng = engine
                if kind == "engine" and eng in ("", "unknown"):
                    eng = _classify_engine(str(internal_port))
                out.append((name, host_port, internal_port, eng, kind))
                matched_port = True
            if not matched_port:
                # A GPU-holding container with no published port (common for
                # estate / service containers) is still a conflict.
                key = (name, 0)
                if key not in seen:
                    seen.add(key)
                    out.append((name, 0, 0, engine, kind))
        return out

    # ── READ: container logs ──────────────────────────────────────────────────────

    async def container_logs(
        self, name: str, *, tail: int = 200
    ) -> dict[str, Any]:
        """`docker logs --tail <N> <name>` — a READ (safe to run live).

        Returns ``{"lines": [...], "error": <str|None>}``.  Goes through the
        injected read runner so tests stay subprocess-free.  This is NOT a write
        — it does not touch container state."""
        res = await self._runner.run(
            ["docker", "logs", "--tail", str(tail), name],
            cwd=str(self.repo_root),
            timeout=15.0,
        )
        if res.timed_out:
            return {"lines": [], "error": f"timed out reading logs for {name}"}
        # docker logs writes app stdout to stdout and app stderr to stderr; show
        # both, stdout first.  A non-zero rc with no output is an error.
        text = res.stdout or ""
        if res.stderr and not text:
            # No stdout — surface stderr (it may BE the log, or an error).
            if res.returncode != 0 and "No such container" in res.stderr:
                return {"lines": [], "error": res.stderr.strip()[:200]}
            text = res.stderr
        lines = text.splitlines()
        if not lines and res.returncode != 0:
            return {"lines": [], "error": (res.stderr.strip()[:200] or f"rc={res.returncode}")}
        return {"lines": lines, "error": None}

    # ── READ: estate state ────────────────────────────────────────────────────────

    async def estate_state(
        self, variants: Optional[list[VariantRow]] = None
    ) -> EstateState:
        """Live estate snapshot: detect (GPUs + running engine + matched slug) +
        health.sh Doctor read + gpu-mode scene catalog + estate-planner report."""
        state = EstateState()

        # detect: running engine + GPUs
        try:
            target = await self._detect_endpoint()
        except Exception as exc:  # pragma: no cover - defensive
            state.error = f"detect failed: {exc}"
            target = ServingTarget()
        if variants:
            target = match_target_to_registry(target, variants)
            state.matched_slug = target.slug
        state.target = target

        # A7: PROBE the live engine for its ACTUAL running config (ctx + image),
        # so the serving panel shows what's REALLY running, not the catalog slug's
        # claim.  READ-only (httpx GET /v1/models + docker inspect); best-effort —
        # a failed probe leaves an empty ServedProbe and the panel falls back to
        # the catalog claim (clearly labelled).  Only probe when something is
        # actually serving (a url or container was detected).
        if getattr(target, "url", "") or getattr(target, "container", ""):
            try:
                state.served = await self._probe_served(target)
            except Exception as exc:  # pragma: no cover - defensive
                state.served = ServedProbe(error=str(exc)[:120])
        state.gpus = list(getattr(target, "gpus", []) or [])
        if not state.gpus:
            # detect may not populate GPUs if no engine running; query directly.
            try:
                state.gpus = await self._get_gpu_info()
            except Exception as exc:
                state.gpus = []
                # NH4: a pure nvidia-smi failure would otherwise leave the rail
                # silently GPU-less with no cue.  Record a cue (don't clobber a
                # docker/detect error that's already more specific).
                if not state.error:
                    state.error = (
                        "nvidia-smi unreachable — GPU read failed "
                        f"({str(exc).strip()[:120]})"
                    )

        # containers — running stack containers, plus the KNOWN supporting
        # services (services/<name>/docker-compose.yml) that are NOT currently
        # running, rendered as stopped so the user sees the full estate.
        #
        # A2/N2: this is the READ path.  ``containers()`` →
        # ``_docker_ps_stack_containers`` RAISES on a failed/timed-out docker ps
        # (fail-closed — load-bearing for the reconcile WRITE gate, which catches
        # it itself).  Here on the READ side a raise would crash the load_estate
        # worker and leave the panes blank (or worse, show a calm "no model
        # serving" false-idle).  CATCH it, record state.error, and return a
        # PARTIAL snapshot (GPUs/doctor/scenes still rendered) so the UI can show
        # an honest "docker unreachable" strip instead of crashing or lying.  The
        # WRITE gate keeps its own raise → writes still fail loudly.
        try:
            running = await self.containers(variants=variants)
            state.containers = await self._merge_known_services(running)
        except Exception as exc:
            state.error = (
                "docker unreachable — daemon running? in the docker group? "
                f"({str(exc).strip()[:120]})"
            )
            state.containers = []

        # scenes (gpu-mode --list-modes --json)
        state.scenes = await self.scenes()

        # doctor (health.sh — text)
        state.doctor = await self.doctor_read(url=target.url or None)

        # estate planner (report-state --json)
        report, _ = await self._run_json(
            ["python3", "scripts/lib/profiles/estate_cli.py", "report-state", "--json"],
            timeout=40.0,
        )
        state.estate_report = report or {}

        return state

    # ── UX Batch 5: estate telemetry (disk / RAM / GPU-VRAM attribution) ──────────

    async def estate_telemetry(self) -> EstateTelemetry:
        """READ the Batch-5 host telemetry in ONE batched pass (piggybacks the
        Operate 4 s tick — no new timer, no per-keystroke storm).

        Three read groups, each best-effort + caught (a single failure records a
        cue string and degrades that group, never crashes the pane, never shows a
        silent false-zero — the B2 "A2" honesty rule):

          1. **Disk (#12)** — ``df -B1 <repo> /mnt/models`` (ONE call, both
             mounts).  Same-device de-dup happens in ``parse_df_output``.
          2. **RAM (N5)**  — ``cat /proc/meminfo`` → MemTotal / MemAvailable.
          3. **GPU attribution** — ``nvidia-smi --query-compute-apps`` for the
             per-card VRAM holders, then a per-pid ``cat /proc/<pid>/cgroup`` +
             ``docker ps --no-trunc`` id→name map to attribute each holder to its
             container.  Graceful degradation: if the cgroup/ps map fails the
             holder still shows (pid + VRAM, no name).

        All I/O goes through the injected runner so tests feed canned df /
        meminfo / nvidia-smi / cgroup stdout — no stdlib disk/file read that a
        test can't intercept."""
        tel = EstateTelemetry()
        errors: list[str] = []

        # ── (1) disk (#12) ────────────────────────────────────────────────────
        try:
            repo_path = str(self.repo_root)
            models_path = self.weights_model_dir()   # the configured weights root
            label_for_path = {repo_path: "repo", models_path: "models"}
            # ``-P`` forces POSIX one-line-per-filesystem output (no device-name
            # wrap → the parser keys off field POSITION, never breaks).  We parse
            # stdout whenever it is non-empty REGARDLESS of returncode: df exits
            # rc=1 when one of the paths (e.g. /mnt/models, a separate drive) is
            # missing/unmounted, but STILL prints the valid rows for the paths it
            # COULD stat to stdout (the error goes to stderr).  Gating on rc would
            # discard a perfectly-good repo bar.
            res = await self._runner.run(
                ["df", "-P", "-B1", repo_path, models_path],
                cwd=str(self.repo_root),
                timeout=10.0,
            )
            if res.stdout.strip():
                tel.disks = parse_df_output(res.stdout, label_for_path)
            if not tel.disks:
                errors.append("disk read failed (df)")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"disk read failed ({str(exc).strip()[:60]})")

        # ── (2) RAM (N5) ──────────────────────────────────────────────────────
        try:
            res = await self._runner.run(
                ["cat", "/proc/meminfo"],
                cwd=str(self.repo_root),
                timeout=10.0,
            )
            if res.ok and res.stdout.strip():
                tel.ram = parse_meminfo(res.stdout)
            else:
                tel.ram = RamUsage(error="meminfo read failed")
            if tel.ram.error:
                errors.append(tel.ram.error)
        except Exception as exc:  # pragma: no cover - defensive
            tel.ram = RamUsage(error=f"meminfo read failed ({str(exc).strip()[:60]})")
            errors.append(tel.ram.error)

        # ── (3) GPU-VRAM → container attribution ──────────────────────────────
        try:
            tel.gpu_apps, tel.attribution_degraded = await self._gpu_attribution()
        except Exception as exc:  # pragma: no cover - defensive
            tel.gpu_apps = {}
            tel.attribution_degraded = True
            errors.append(f"gpu attribution failed ({str(exc).strip()[:60]})")

        tel.error = " · ".join(errors)
        return tel

    async def _gpu_attribution(self) -> tuple[dict[Optional[int], list[GpuCompApp]], bool]:
        """Map the live CUDA compute-apps → owning containers (best-effort).

        Reads ``nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory`` (the
        gpu_uuid leg lets us bucket a holder onto its physical card; absent that
        the holder lands under the ``None`` bucket — still attributed by VRAM,
        just not pinned to a specific card, rather than mis-pinned to GPU0),
        then resolves each pid → container via ``/proc/<pid>/cgroup`` (docker id)
        ⨯ a ``docker ps --no-trunc`` id→name map.

        GRACEFUL DEGRADATION: a failed nvidia-smi → empty map (no crash); a pid
        whose cgroup is unreadable / non-docker / not in the ps map → the holder
        keeps an empty container name (the card shows pid + VRAM, no fabricated
        owner)."""
        # The compute-apps query — gpu_uuid lets us bucket by physical card.
        res = await self._runner.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(self.repo_root),
            timeout=10.0,
        )
        if not res.ok or not res.stdout.strip():
            return {}, False
        # Build (gpu_uuid, app) pairs.  parse_compute_apps wants pid,used cols, so
        # we strip the uuid prefix per line but keep it for bucketing.
        uuid_for_app: dict[int, str] = {}
        app_lines: list[str] = []
        for line in res.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1].isdigit():
                uuid_for_app[int(parts[1])] = parts[0]
                app_lines.append(f"{parts[1]}, {parts[2]}")
            elif len(parts) == 2 and parts[0].isdigit():
                # No uuid column (older nvidia-smi) — pid,used only.
                app_lines.append(line)
        apps = parse_compute_apps("\n".join(app_lines))
        if not apps:
            return {}, False

        # Map gpu_uuid → index via a separate query (cheap; one extra read).
        uuid_to_index = await self._gpu_uuid_index_map()

        # Resolve each pid → cgroup body, and read the docker ps id→name map ONCE.
        cgroup_by_pid: dict[int, str] = {}
        for app in apps:
            try:
                cg = await self._runner.run(
                    ["cat", f"/proc/{app.pid}/cgroup"],
                    cwd=str(self.repo_root),
                    timeout=5.0,
                )
                cgroup_by_pid[app.pid] = cg.stdout if cg.ok else ""
            except Exception:  # pragma: no cover - defensive
                cgroup_by_pid[app.pid] = ""
        id_to_name = await self._docker_ps_id_names()

        apps, degraded = attribute_gpu_apps(apps, cgroup_by_pid, id_to_name)

        # Bucket the attributed apps onto their physical card.  When the uuid→index
        # map failed (secondary nvidia-smi read skewed/empty) we DON'T default to
        # index 0 — that would mis-pin a GPU1 holder under GPU0's "held by:" line.
        # Instead the holder buckets under ``None`` (a neutral, card-agnostic
        # heading); VRAM totals stay honest, the card attribution just isn't claimed.
        by_index: dict[Optional[int], list[GpuCompApp]] = {}
        for app in apps:
            uuid = uuid_for_app.get(app.pid, "")
            idx = uuid_to_index.get(uuid)  # None when unresolved → not pinned to a card
            app.gpu_index = idx
            by_index.setdefault(idx, []).append(app)
        return by_index, degraded

    async def _gpu_uuid_index_map(self) -> dict[str, int]:
        """``nvidia-smi --query-gpu=uuid,index`` → {uuid: index} (best-effort, {}
        on failure → holders bucket under ``None``, still attributed by VRAM but
        not pinned to a specific card)."""
        try:
            res = await self._runner.run(
                ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader,nounits"],
                cwd=str(self.repo_root),
                timeout=10.0,
            )
        except Exception:  # pragma: no cover - defensive
            return {}
        if not res.ok:
            return {}
        out: dict[str, int] = {}
        for line in res.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                out[parts[0]] = int(parts[1])
        return out

    async def _docker_ps_id_names(self) -> dict[str, str]:
        """``docker ps --no-trunc --format '{{.ID}} {{.Names}}'`` → {id: name}
        for pid→container resolution.  Degrades to {} on failure (holders show
        nameless — graceful, never a crash)."""
        try:
            res = await self._runner.run(
                ["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"],
                cwd=str(self.repo_root),
                timeout=10.0,
            )
        except Exception:  # pragma: no cover - defensive
            return {}
        if not res.ok:
            return {}
        return parse_docker_ps_id_names(res.stdout)

    async def _real_probe_served(self, target: Any) -> ServedProbe:
        """A7: probe the live engine for its ACTUAL running config (READ-only).

        Two legs, both best-effort (a failed leg leaves its field empty so the
        UI falls back to the catalog claim, clearly labelled):
          - ``GET <url>/v1/models`` → ``max_model_len`` + served model id (vLLM
            exposes the running context per model id; llama.cpp omits it, so the
            field stays None and the panel labels ctx "(per catalog slug)").
          - ``docker inspect <container> --format '{{.Image}}'`` → the engine
            image (CLAUDE.md: ``vllm.__version__`` lags the docker tag, so the
            image digest is ground truth).

        NEVER run in tests — the probe seam is injected with a fake (conftest also
        hard-blocks the real subprocess/httpx path)."""
        probe = ServedProbe()
        url = (getattr(target, "url", "") or "").strip()
        container = (getattr(target, "container", "") or "").strip()
        # Leg 1: /v1/models (httpx).
        if url:
            try:
                import httpx  # local import — only the real probe needs it

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{url}/v1/models")
                    if resp.status_code == 200:
                        data = (resp.json() or {}).get("data", []) or []
                        if data:
                            first = data[0] or {}
                            probe.served_model = str(first.get("id", "") or "")
                            mml = first.get("max_model_len")
                            if mml is not None:
                                try:
                                    probe.max_model_len = int(mml)
                                except (TypeError, ValueError):
                                    pass
            except Exception as exc:
                probe.error = f"models probe: {str(exc)[:80]}"
        # Leg 2: docker inspect image (READ).
        if container:
            res = await self._runner.run(
                ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
                cwd=str(self.repo_root),
                timeout=10.0,
            )
            img = (res.stdout or "").strip()
            if img and res.returncode == 0:
                probe.image = img
        return probe

    async def scenes(self) -> list[Scene]:
        """gpu-mode --list-modes --json → Scene list.

        Hides the COMMAND modes — ``power-cap`` (its own [c] menu), ``prune`` /
        ``prune-all`` (removed from the cockpit): they're not GPU *scenes* you'd
        ⏎-switch to, yet --list-modes lists them (it mirrors the CLI dispatch).
        Everything else is a real scene, incl. ``off`` and ``chat`` (the ops-group
        browser-chat stack) and the models / studio scenes."""
        data, _ = await self._run_json(
            ["bash", "scripts/gpu-mode.sh", "--list-modes", "--json"], timeout=30.0
        )
        if not isinstance(data, list):
            return []
        hidden = {"power-cap", "prune", "prune-all"}
        kept = [s for s in (Scene.from_dict(d) for d in data) if s.name not in hidden]
        # Cluster by group (models → studio → ops) for the Orchestration table —
        # stable within a group (preserves catalog order).  Without this the table
        # renders raw catalog order, which splits the ops scenes apart (chat at the
        # top, off near the bottom) and reads as ungrouped.
        rank = {"models": 0, "studio": 1, "ops": 2}
        kept.sort(key=lambda s: rank.get(s.group, 3))   # list.sort is stable
        return kept

    async def doctor_read(self, url: Optional[str] = None) -> DoctorRead:
        """health.sh (text-only) → parsed DoctorRead."""
        env_prefix: list[str] = []
        cmd = ["bash", "scripts/health.sh"]
        if url:
            # health.sh reads URL from env; pass via a wrapper to keep the
            # Runner protocol env-free.
            cmd = ["bash", "-c", 'URL="$1" bash scripts/health.sh', "bash", url]
        res = await self._runner.run(cmd, cwd=str(self.repo_root), timeout=30.0)
        return parse_health_text(res.stdout or res.stderr)

    # ── THE DUAL-WRITER GATE ─────────────────────────────────────────────────────

    async def reconcile_before_write(
        self,
        action: str,
        *,
        pending_gpus: Optional[list[int]] = None,
        variants: Optional[list[VariantRow]] = None,
    ) -> ReconcileResult:
        """Re-run detect to see what is ACTUALLY on the cards NOW, and report the
        set of running containers / GPU users / estate claims a pending write
        would collide with.  This is the safety core (design §3.2): a pending
        serve / scene-switch must call this immediately before executing, so a
        concurrent writer (e.g. estate_cli already booted GPU0) is caught.

        ``pending_gpus`` = the GPUs the action wants.  ``None`` means "any GPU"
        (treated as wanting both 0 and 1 — the conservative default), so ANY
        live GPU user is a conflict.

        ``safe`` is True only when nothing live overlaps the requested GPUs.
        """
        result = ReconcileResult(safe=True, action=action)

        # Fresh detect — never trust a cached snapshot for the gate.
        try:
            target = await self._detect_endpoint()
        except Exception as exc:  # pragma: no cover - defensive
            result.note = f"detect failed: {exc}"
            # A failed detect is NOT safe — we can't prove the cards are free.
            result.safe = False
            return result

        # GPU read — FAIL CLOSED.  For a SAFETY gate, an error reading the cards
        # must mean "I cannot prove they are free" → UNSAFE, never "nothing in
        # use".  (Previously this swallowed the exception → gpus=[] → safe.)
        gpus = list(getattr(target, "gpus", []) or [])
        if not gpus:
            try:
                gpus = await self._get_gpu_info()
            except Exception as exc:
                result.note = f"GPU read failed: {exc}"
                result.safe = False
                return result
            if not gpus:
                # No detect GPUs AND no nvidia-smi readout → we have no evidence
                # the cards are free.  Fail closed.
                result.note = "GPU read returned no cards — cannot prove free"
                result.safe = False
                return result

        # Determine which GPUs the action wants.
        if pending_gpus is None:
            wanted = {0, 1}            # conservative: assume both cards
        else:
            wanted = set(pending_gpus)
        result.pending_gpus = sorted(wanted)

        # 1. Running stack containers (docker ps) — engine containers, estate
        #    `club3090-<name>` containers, and rig services (ComfyUI / Step-Audio)
        #    are all live GPU users.  When a container's GPU set is KNOWN and is
        #    disjoint from the wanted cards, it does not conflict; when UNKNOWN
        #    (the common case — docker ps doesn't expose the device list) we stay
        #    conservative and treat it as a conflict.  Detector #2 (raw GPU,
        #    fail-closed) is the backstop for any GPU holder this misses.
        #    Fix 2 (fail-closed): _docker_ps_stack_containers raises RuntimeError
        #    on a failed/timed-out docker ps → catch it and fail closed.
        try:
            containers = await self.containers(variants=variants)
        except Exception as exc:
            result.note = f"docker ps read failed: {exc}"
            result.safe = False
            return result
        # Is a WANTED card actually occupied right now (>512 MiB rules out
        # driver/compositor noise)?  An unknown-GPU container is only a teardown
        # CONFLICT when there is real contention — a running-but-idle service
        # (e.g. the studio stack at 0 GiB on free cards) is nothing a serve onto
        # free cards needs to stop, so it must NOT be cried-wolf as a conflict.
        # When a wanted card IS busy we stay conservative (the holder may be
        # unnamed); the raw-VRAM gpu_conflicts pass below is the fail-closed
        # backstop in EITHER case, so the `safe` verdict never weakens.
        wanted_busy = any(
            getattr(g, "index", -1) in wanted and getattr(g, "mem_used_mib", 0) > 512
            for g in gpus
        )
        for c in containers:
            known = _container_gpu_set(c)
            if known is not None:
                if known & wanted:
                    result.conflicts.append(c)  # provably on a wanted card
                continue  # known set: authoritative (empty/other-card → not a conflict)
            # Unknown GPU set → conflict only when a wanted card is occupied.
            if wanted_busy:
                result.conflicts.append(c)

        # 2. Raw GPU occupancy — a card with meaningful VRAM in use is occupied
        #    even if we can't name the container (e.g. a bare llama-server).
        #    Threshold: >512 MiB rules out driver/compositor noise.
        for g in gpus:
            idx = getattr(g, "index", -1)
            mem = getattr(g, "mem_used_mib", 0)
            if idx in wanted and mem > 512:
                result.gpu_conflicts.append(
                    GpuConflict(
                        gpu_index=idx,
                        mem_used_mib=mem,
                        note="GPU in use",
                    )
                )

        # 3. Active estate claims — the estate planner may have booted instances
        #    that hold cards even if our engine-prefix detect missed them.
        #    FAIL CLOSED: if the estate read errors, we cannot rule out a hidden
        #    estate claim on the wanted cards → UNSAFE.  (Previously the error
        #    was discarded → report={} → "no claims" → falsely safe.)
        report, estate_err = await self._run_json(
            ["python3", "scripts/lib/profiles/estate_cli.py", "report-state", "--json"],
            timeout=40.0,
        )
        if estate_err and report is None:
            result.note = f"estate read failed: {estate_err}"
            result.safe = False
            return result
        active = (report or {}).get("active_estate") or {}
        if active.get("present") and active.get("instances"):
            for inst in active["instances"]:
                inst_gpus = set(inst.get("gpus", []) or [])
                if inst_gpus & wanted:
                    result.estate_claims.append(inst)

        # 4. In-process pending-claim check (Fix 1 — TOCTOU).
        #    A concurrent write that has passed the gate and started its subprocess
        #    registers a pending claim UNDER the write lock before releasing it.
        #    Because reconcile_before_write is called UNDER the same lock by
        #    execute_action, a second writer will see the first's claim here and
        #    report a conflict even though docker ps / nvidia-smi may still show
        #    the cards as free (the first process hasn't booted yet).
        #    Expired claims (> TTL) are silently pruned.
        now = time.monotonic()
        expired = [tok for tok, (_, exp) in self._pending_claims.items() if now > exp]
        for tok in expired:
            self._pending_claims.pop(tok, None)
        for tok, (claimed_gpus, _exp) in self._pending_claims.items():
            if claimed_gpus & wanted:
                result.pending_claim_tokens.append(tok)
                result.note = (
                    result.note or f"in-flight write already claimed GPUs {sorted(claimed_gpus)}"
                )

        # Safe only if NOTHING overlaps.
        result.safe = not (
            result.conflicts or result.gpu_conflicts or result.estate_claims
            or result.pending_claim_tokens
        )
        if not result.safe and not result.note:
            result.note = f"would collide with: {result.conflict_summary}"
        return result

    # ── WRITE: action builders (wired, execution-gated) ──────────────────────────

    def serve(self, slug: str, *, force: bool = False, force_reason: str = "") -> ActionPlan:
        """Build the GATED switch.sh <slug> action.  ``--force`` is only added
        when explicitly requested WITH a reason (surfaced to the user)."""
        cmd = ["bash", "scripts/switch.sh"]
        if force:
            if not force_reason:
                raise ValueError("force=True requires a force_reason (surfaced to user)")
            cmd.append("--force")
        cmd.append(slug)
        desc = f"switch.sh {'--force ' if force else ''}{slug}"
        return ActionPlan(
            kind="serve",
            cmd=cmd,
            description=desc,
            force=force,
            force_reason=force_reason,
            requires_reconcile=True,
        )

    # ── Producer lane ② Serve — generate a compose, then serve it untested ─────────

    async def generate_compose(
        self, slug: str, *, accept_degraded: bool = False
    ) -> dict[str, Any]:
        """Generate a minimal compose for the CATALOG profile ``slug`` via
        scripts/generate-compose.sh.

        Producer-lane ② Serve step.  Shells ``generate-compose.sh --profile <slug>
        --out <tmpfile> [--accept-degraded]`` (cwd repo_root, via the injected read
        runner — it is a generation step that writes only a TEMP compose: no GPU, no
        network), then reads the generated YAML back.

        ⚠️  HONESTY (R3b-1): ``slug`` is a CATALOG profile slug.  generate-compose.sh
        has no --repo / weight-swap, so the emitted compose is a reproduction of the
        resolved CATALOG profile's compose — NOT a brought (BYO) model's weights.
        The brought-model serve (pull-to-disk + a generate-compose.sh --repo
        extension) is a deferred follow-up.

        Mission (locked decision #2 in generate-compose.sh): reproduce + flag,
        NEVER repair — the emitted compose is returned VERBATIM; the caller shows
        it as an untested config reproduction and does NOT fit-adapt it.

        Returns ``{"compose_path", "compose_yaml", "error"}``.  On a failed /
        drift-guard-flagged generation, ``error`` is set and ``compose_yaml`` is
        empty (the generator's stderr is surfaced).  The temp compose PERSISTS on
        success (``serve_generated`` serves it via ``docker compose -f <path>``);
        it is unlinked on every error path AND when the preview is declined without
        serving (the app unlinks it on dismiss)."""
        import os
        import tempfile

        def _fail(msg: str) -> dict[str, Any]:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return {"compose_path": "", "compose_yaml": "", "error": msg}

        # A temp file under the repo root so the generator's relative ../ mount
        # paths (if any are emitted) resolve the same as a real compose would.
        fd, tmp_path = tempfile.mkstemp(
            prefix="c3-genc-", suffix=".yml", dir=str(self.repo_root)
        )
        os.close(fd)
        cmd = [
            "bash",
            "scripts/generate-compose.sh",
            "--profile",
            slug,
            "--out",
            tmp_path,
        ]
        if accept_degraded:
            cmd.append("--accept-degraded")
        res = await self._runner.run(cmd, cwd=str(self.repo_root), timeout=60.0)
        if res.timed_out:
            return _fail(f"generate-compose timed out for {slug}")
        if not res.ok:
            err = (res.stderr.strip() or res.stdout.strip())[:300]
            return _fail(err or f"generate-compose failed (rc={res.returncode})")
        try:
            yaml_text = Path(tmp_path).read_text(encoding="utf-8")
        except OSError as exc:
            return _fail(f"generated compose unreadable: {exc}")
        if not yaml_text.strip():
            # Generator returned 0 but emitted nothing to --out: surface its
            # stdout (a convenience-tuple / candidate list, or a soft refuse).
            return _fail(res.stdout.strip()[:300] or "generator emitted no compose")
        return {"compose_path": tmp_path, "compose_yaml": yaml_text, "error": ""}

    def serve_generated(self, compose_path: str) -> ActionPlan:
        """Serve a GENERATED (producer-lane ②) compose, badged untested.

        Serving a generated compose CLAIMS the GPU exactly like any serve, so the
        plan is ``requires_reconcile=True`` + ``requires_confirm=True`` and routes
        through the SAME ConfirmActionScreen → run_reconcile_for_modal →
        dispatch_action gate (the dual-writer lease MUST hold).  We launch it via
        ``docker compose -f <path> up`` — the generated compose is a verbatim
        minimal reproduction, NOT a registry slug switch.sh knows about."""
        return ActionPlan(
            kind="serve",
            cmd=["docker", "compose", "-f", compose_path, "up", "-d"],
            description=f"serve generated compose {Path(compose_path).name} (untested)",
            requires_reconcile=True,
            requires_confirm=True,
        )

    def set_default(self, slug: str) -> ActionPlan:
        return ActionPlan(
            kind="set_default",
            cmd=["bash", "scripts/switch.sh", "--set-default", slug],
            description=f"switch.sh --set-default {slug}",
            requires_reconcile=False,   # .env pin write — no GPU contention
        )

    def clear_default(self, model: str) -> ActionPlan:
        return ActionPlan(
            kind="clear_default",
            cmd=["bash", "scripts/switch.sh", "--clear-default", model],
            description=f"switch.sh --clear-default {model}",
            requires_reconcile=False,
        )

    def scene_switch(self, mode: str) -> ActionPlan:
        return ActionPlan(
            kind="scene",
            cmd=["bash", "scripts/gpu-mode.sh", mode],
            description=f"gpu-mode {mode}",
            requires_reconcile=True,
        )

    def stop_all(self) -> ActionPlan:
        """Comprehensive 'off' — ``gpu-mode off`` (stops the named scene presets)
        AND ``estate_cli down`` (estate-managed instances), so neither path's blind
        spot survives.  A FORCED 'off' ALSO tears down any other detected running
        container via the reconcile gate (execute_action → _teardown_conflicts),
        covering a cockpit-served catalog slug that is neither a preset nor estate-
        managed.  requires_reconcile=True so the gate detects + (on force) clears
        whatever is actually running."""
        return ActionPlan(
            kind="scene",
            cmd=["bash", "-c",
                 "bash scripts/gpu-mode.sh off; "
                 "python3 scripts/lib/profiles/estate_cli.py down"],
            description="stop all (gpu-mode off + estate down)",
            requires_reconcile=True,
        )

    def estate_down(self) -> ActionPlan:
        return ActionPlan(
            kind="estate_down",
            cmd=["python3", "scripts/lib/profiles/estate_cli.py", "down"],
            description="estate_cli down",
            requires_reconcile=True,
        )

    def container_action(self, name: str, op: str) -> ActionPlan:
        """op ∈ {restart, stop}.  Builds a docker write — execution-gated."""
        if op not in ("restart", "stop"):
            raise ValueError(f"container op must be restart|stop, got {op!r}")
        return ActionPlan(
            kind="container",
            cmd=["docker", op, name],
            description=f"docker {op} {name}",
            requires_reconcile=(op == "stop"),
        )

    def service_start(self, name: str) -> ActionPlan:
        """Bring a KNOWN supporting service UP — ``docker compose up -d`` for its
        ``services/<name>/`` dir (the verb for a STOPPED service row in the
        Containers tab; running rows use stop/restart).

        Mirrors gpu-mode's ``compose_at`` so the cockpit and gpu-mode operate on
        the SAME containers:
          - ``-p <name>`` pins the compose PROJECT to the service dir name.
            gpu-mode starts these from inside ``services/<name>/`` (so its project
            == the dir); the cockpit runs with ``cwd=repo_root``, where the project
            would otherwise default to the repo dir — and ``up -d`` would try to
            CREATE a second container with the same ``container_name`` → "name
            already in use", which is exactly why comfyui/qdrant/searxng wouldn't
            start.
          - ``--env-file .env`` (when present) resolves repo vars like
            ``${MODEL_DIR}`` / ``${HF_TOKEN}`` that some composes (comfyui) need.

        Reconcile-gated: a GPU-holding service (ComfyUI / Step-Audio) can't
        silently collide with whatever holds the cards, while a non-GPU web
        service (litellm / qdrant / …) clears the gate immediately.  Tolerates the
        ``.yml`` / ``.yaml`` spelling."""
        rel = f"services/{name}/docker-compose.yml"
        if (
            not (self.repo_root / rel).is_file()
            and (self.repo_root / f"services/{name}/docker-compose.yaml").is_file()
        ):
            rel = f"services/{name}/docker-compose.yaml"
        cmd = ["docker", "compose"]
        if (self.repo_root / ".env").is_file():
            cmd += ["--env-file", ".env"]
        cmd += ["-f", rel, "-p", name, "up", "-d"]
        return ActionPlan(
            kind="service-up",
            cmd=cmd,
            description=f"docker compose up {name}",
            requires_reconcile=True,
        )

    @staticmethod
    def scene_service_states_sync(
        scene: Scene, running_names: list[str]
    ) -> list[SceneServiceState]:
        """Per-service running state for ``scene`` against an ALREADY-FETCHED set
        of running container names — a pure, I/O-free match so the scene-preview
        highlight path can re-render every keystroke without a docker call.

        Matches each ``scene.services`` name with the SAME normalized matcher the
        Containers tab uses (``_service_dir_matches_running`` — tolerant of the
        ``-server`` / ``-dual`` suffix a container name carries off the service
        name).  A service absent from the running set reads ``running=False``."""
        running_norm = {_normalize_service_name(n) for n in running_names}
        running_norm.discard("")
        return [
            SceneServiceState(
                name=svc,
                running=any(
                    _service_dir_matches_running(svc, rn) for rn in running_norm
                ),
            )
            for svc in scene.services
        ]

    async def _teardown_conflicts(self, reconcile: "ReconcileResult") -> list[str]:
        """Stop the running containers a FORCED write collides with (the gate's
        ``conflicts``) so the write clears the way regardless of whether its own
        command knows them.  Best-effort + idempotent: each ``docker stop`` is
        awaited (synchronous → the GPU is freed before the command claims it); a
        failure to stop one doesn't abort the others or the write.  Returns the
        container names it attempted to stop (for logging/tests).  Routed through
        the read runner so tests observe/mock it — NEVER stops a real container in
        the test suite."""
        names: list[str] = []
        for c in (getattr(reconcile, "conflicts", None) or []):
            n = (getattr(c, "name", "") or "").strip()
            if n and n not in names:
                names.append(n)
        for name in names:
            try:
                await self._runner.run(
                    ["docker", "stop", name], cwd=str(self.repo_root), timeout=60.0
                )
            except Exception:
                continue
        return names

    # ── WRITE: execution (gated — NEVER run in tests / this phase) ────────────────

    async def execute_action(
        self,
        plan: ActionPlan,
        *,
        parser: Any = None,
        run_type: Optional[str] = None,
        variants: Optional[list[VariantRow]] = None,
        skip_reconcile: bool = False,
    ) -> tuple[bool, Optional[ReconcileResult], Any]:
        """Execute a WRITE ActionPlan via the core SubprocessRunner.

        ⚠️  WRITE PATH.  The maintainer validates the first real serve / scene
        switch later; in this phase this is wired but execution is mocked in
        tests and NEVER run live.

        Always re-runs the reconcile gate first (unless the plan opts out or
        ``skip_reconcile`` is set — only honored when ``plan.force`` is True with
        a reason).  Returns ``(executed, reconcile_result, run_state)``.

        If the gate is unsafe and force isn't set, returns
        ``(False, reconcile_result, None)`` — refusing to write.

        SERIALIZED (design §3.2): the gate→write window is held under a single
        write lock so two confirmed plans can't both reconcile "safe" before
        either claims VRAM (the dual-writer TOCTOU).  A second write blocks on
        the lock until the first has finished claiming the cards, then runs its
        OWN fresh gate against the now-updated state.
        """
        # skip_reconcile is ONLY honored as an explicit, reasoned force override
        # (the docstring contract).  Enforce the coupling in code so a caller
        # can't silently bypass the safety gate.
        effective_skip = skip_reconcile and plan.force and bool(plan.force_reason)

        async with self._write_lock:
            reconcile: Optional[ReconcileResult] = None
            if plan.requires_reconcile and not effective_skip:
                reconcile = await self.reconcile_before_write(
                    f"{plan.kind}:{plan.description}", variants=variants
                )
                if not reconcile.safe:
                    # An in-flight write's pending claim is a HARD block — force
                    # CANNOT override it: a not-yet-booted write has nothing
                    # deterministic to tear down, so the user must cancel the
                    # in-flight write first (an explicit cancel path), not force
                    # over it. force only overrides *materialized* conflicts.
                    if reconcile.pending_claim_tokens:
                        return False, reconcile, None
                    if not plan.force:
                        return False, reconcile, None
                    if not plan.force_reason:
                        raise ValueError("force override requires force_reason")
                    # FORCE override: actually CLEAR the way — stop the materialized
                    # conflicting containers the gate found, so the write doesn't
                    # depend on its OWN command knowing them.  This is what makes a
                    # forced scene switch ('off' → gpu-mode off, which does NOT stop
                    # a cockpit-served catalog slug) reliably free the cards.  docker
                    # stop is synchronous, and we're under the write lock, so the
                    # cards are freed before the command claims them.
                    await self._teardown_conflicts(reconcile)

            # Fix 1 (TOCTOU): register a pending claim for the GPUs this write
            # wants BEFORE calling start_raw and BEFORE releasing the lock.  The
            # reconcile above (while still under the lock) already checked
            # _pending_claims; by adding our claim here, the NEXT writer's
            # reconcile (also serialized by the lock) will see it — even if our
            # subprocess hasn't allocated any VRAM yet.
            claim_token = str(uuid.uuid4())
            if plan.requires_reconcile:
                # Infer the GPU set the plan wants.  pending_gpus from the last
                # reconcile is most accurate; fall back to conservative {0, 1}.
                if reconcile is not None and reconcile.pending_gpus:
                    claimed = frozenset(reconcile.pending_gpus)
                else:
                    claimed = frozenset({0, 1})
                expiry = time.monotonic() + self._claim_ttl
                self._pending_claims[claim_token] = (claimed, expiry)

            # Stream via the core runner.  No no-op parser → use a passthrough.
            run_parser = parser or _NullParser()
            import os as _os

            try:
                state = await self._write_runner.start_raw(
                    plan.cmd,
                    env=dict(_os.environ),
                    run_type=run_type or plan.kind,
                    parser=run_parser,
                )
            except Exception:
                # The spawn ITSELF failed — no card was claimed; release now.
                self._pending_claims.pop(claim_token, None)
                raise

        # NOTE: do NOT clear the claim here. start_raw only SPAWNS the write
        # (switch.sh/gpu-mode is still booting; no container/VRAM yet — start_raw
        # returns immediately, runner.py). Holding the claim until the subprocess
        # COMPLETES bridges the materialization gap: by the time the process
        # exits (switch.sh's wait_ready ⇒ container up + /v1/models answering),
        # docker ps / nvidia-smi see it, so the next reconcile is covered with no
        # gap. The write lock is released here (block exit) so a concurrent
        # writer gets an immediate "in-flight conflict" rather than hanging.
        if claim_token in self._pending_claims:
            if getattr(state, "is_finished", False):
                # start_raw returned an already-finished (spawn-failure) state.
                self._pending_claims.pop(claim_token, None)
            else:
                asyncio.create_task(self._release_claim_when_done(claim_token, state))

        return True, reconcile, state

    async def _release_claim_when_done(self, token: str, state: Any) -> None:
        """Hold the pending-GPU claim until the write subprocess COMPLETES, then
        release it. By completion the materialized container/VRAM is visible to
        the next reconcile (the clean handoff). Awaits the run's PER-RUN ``done``
        event (not the runner's shared ``current_run``), so overlapping runs are
        never misattributed. The TTL is only a leak backstop for a state that
        never signals done (a stub runner / a vanished process)."""
        done = getattr(state, "done", None)
        try:
            if done is not None:
                await asyncio.wait_for(done.wait(), timeout=self._claim_ttl)
        except asyncio.TimeoutError:
            pass
        finally:
            self._pending_claims.pop(token, None)

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Validate surface (Run · Doctor · Benchmarks · Evidence + ops)
    # ════════════════════════════════════════════════════════════════════════════

    # ── Validate / Run: launch a validation script (WIRED, execution MOCKED) ──────

    # The validation scripts the Run pane can launch.  Each maps to its core
    # parser (where one exists) so the streamed output becomes structured
    # progress.  ALL of these LAUNCH a heavy process that stresses / hits a live
    # serving model — they are WIRED but execution is MOCKED in tests and NEVER
    # run live this phase (conftest blocks the real spawn).
    #   kind → (script-relative-cmd, parser_test_type|None)
    #
    # Verified live (2026-06-18): the script filenames + arg conventions below
    # are the REAL on-disk ones.  Most scripts read the target endpoint/model
    # from the environment (``MODEL=``/``URL=``); two do NOT and take CLI args
    # instead — those are handled specially in ``validation_plan``:
    #   - ``stream-toolcall-probe`` is a ``.py`` (not ``.sh``) and takes
    #     ``--url``/``--model`` flags, not env (see its ``Usage:`` header).
    #   - ``quality-baseline.sh`` exists as its own wrapper (#252) and REQUIRES
    #     ``--slug``; endpoint/model are inherited from env via quality-test.sh.
    _VALIDATION_KINDS: dict[str, tuple[list[str], Optional[str]]] = {
        "verify-full": (["bash", "scripts/verify-full.sh"], "verify-full"),
        "verify-stress": (["bash", "scripts/verify-stress.sh"], "verify-stress"),
        "bench": (["bash", "scripts/bench.sh"], "bench"),
        "quality-test": (["bash", "scripts/quality-test.sh", "--quick"], "quality"),
        "soak-test": (["bash", "scripts/soak-test.sh"], "soak"),
        "rebench-full": (["bash", "scripts/rebench-full.sh"], "rebench-full"),
        # Extra tools (no dedicated core parser → stream raw via NullParser):
        "quality-baseline": (["bash", "scripts/quality-baseline.sh"], None),
        "bench-agentic": (["bash", "scripts/bench-agentic.sh"], None),
        "stream-toolcall-probe": (["python3", "scripts/stream-toolcall-probe.py"], None),
    }

    def validation_plan(
        self,
        kind: str,
        *,
        model: Optional[str] = None,
        url: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> ActionPlan:
        """Build the ActionPlan for a validation-script launch (WIRED, gated).

        Validation scripts hit / stress a live serving model but do NOT
        claim/free a GPU, so ``requires_reconcile=False`` — yet they are heavy
        and must still go through a confirm modal (``requires_confirm=True``).

        Most scripts read ``MODEL`` / ``URL`` of the current target from the
        environment; the actual env is injected by ``run_validation`` at
        execution time, NOT baked into the cmd here, so the plan stays
        inspectable/loggable without leaking the target.  Two are exceptions
        (verified live):
          - ``stream-toolcall-probe.py`` takes ``--url`` / ``--model`` CLI
            args (not env), so they are appended to the cmd when supplied;
          - ``quality-baseline.sh`` REQUIRES ``--slug`` (the registry slug),
            which is appended when supplied."""
        if kind not in self._VALIDATION_KINDS:
            raise ValueError(
                f"unknown validation kind {kind!r}; "
                f"expected one of {sorted(self._VALIDATION_KINDS)}"
            )
        cmd, _parser = self._VALIDATION_KINDS[kind]
        cmd = list(cmd)
        # stream-toolcall-probe.py reads --url/--model from the CLI, not env.
        if kind == "stream-toolcall-probe":
            if url:
                cmd += ["--url", url]
            if model:
                cmd += ["--model", model]
        # quality-baseline.sh requires --slug (endpoint/model still env-inherited).
        elif kind == "quality-baseline" and slug:
            cmd += ["--slug", slug]
        target_bits: list[str] = []
        if slug and kind == "quality-baseline":
            target_bits.append(f"slug={slug}")
        if model:
            target_bits.append(f"MODEL={model}")
        if url:
            target_bits.append(f"URL={url}")
        target = (" → " + " ".join(target_bits)) if target_bits else ""
        return ActionPlan(
            kind="validation",
            cmd=cmd,
            description=f"{kind}{target}".strip(),
            requires_reconcile=False,   # hits the model; does not claim a GPU
            requires_confirm=True,      # heavy — confirm before launching
        )

    def _validation_parser(self, kind: str) -> Any:
        """The core parser for a validation kind, or a NullParser when none
        exists (extra tools).  Imported lazily so the data layer stays
        Textual/parser-import-free until a launch is actually requested."""
        _cmd, test_type = self._VALIDATION_KINDS.get(kind, ([], None))
        if not test_type:
            return _NullParser()
        from club3090_tui_core.parsers import TestType, get_parser

        return get_parser(TestType(test_type))

    async def run_validation(
        self,
        kind: str,
        *,
        model: Optional[str] = None,
        url: Optional[str] = None,
        slug: Optional[str] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Launch a validation script via the core SubprocessRunner, streamed.

        ⚠️  WIRED-BUT-MOCK-ONLY.  These scripts stress / hit a serving model and
        are heavy; tests mock the write runner (conftest blocks the real spawn).
        NEVER run live this phase.

        Parses the streamed output into structured progress/result via the core
        parser for the kind (``verify-full`` / ``bench`` / ``verify-stress`` /
        ``quality`` / ``soak`` / ``rebench-full``); extra tools stream raw.
        ``MODEL`` / ``URL`` of the current target are injected into the child
        env so the scripts hit the right endpoint.

        Returns the core ``CoreRunState`` (the streaming handle).  Confirmation
        is the CALLER's job (the Run pane wires a confirm modal before calling
        this — these launches always ``requires_confirm``)."""
        import os as _os

        plan = self.validation_plan(kind, model=model, url=url, slug=slug)
        env = dict(_os.environ)
        if model:
            env["MODEL"] = model
        if url:
            env["URL"] = url
        parser = self._validation_parser(kind)
        if on_event is not None or on_line is not None:
            # Per-launch callbacks for the live pane.  set_callbacks is on the
            # shared runner; the caller owns wiring/teardown.
            self._write_runner.set_callbacks(on_event=on_event, on_line=on_line)
        # No reconcile gate (validation does not claim a GPU); straight to the
        # streamer.  In tests this is the FakeWriteRunner; live it is blocked.
        return await self._write_runner.start_raw(
            plan.cmd, env=env, run_type=plan.kind, parser=parser
        )

    # ── Producer / ③ Gate: the FULL validation battery (report.sh --full) ─────────

    def full_validation_report_plan(
        self, *, model: Optional[str] = None, url: Optional[str] = None
    ) -> ActionPlan:
        """Build the ActionPlan for ``report.sh --full`` — the ~43-min
        verify+stress+soak+bench+agentic PRODUCER battery (Q1 maintainer call).

        Distinct from the consumer share-back ``rig_report`` (bare ``report.sh``,
        a ~2 s redacted snapshot, R2b).  ``--full`` runs the heavy battery against
        the ALREADY-SERVING model, so it does NOT claim a GPU
        (``requires_reconcile=False``) but is HEAVY + long-running and MUST be
        confirm-gated (``requires_confirm=True``).  ``MODEL`` / ``URL`` are
        injected into the child env at execution time (NOT baked into the cmd) so
        the plan stays inspectable without leaking the target."""
        target_bits: list[str] = []
        if model:
            target_bits.append(f"MODEL={model}")
        if url:
            target_bits.append(f"URL={url}")
        target = (" → " + " ".join(target_bits)) if target_bits else ""
        return ActionPlan(
            kind="full_report",
            cmd=["bash", "scripts/report.sh", "--full"],
            description=f"report.sh --full (~43-min full validation battery){target}".strip(),
            requires_reconcile=False,   # hits the serving model; does not claim a GPU
            requires_confirm=True,      # heavy + long-running — confirm before launching
        )

    async def run_full_validation_report(
        self,
        *,
        model: Optional[str] = None,
        url: Optional[str] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Launch ``report.sh --full`` as a BACKGROUND streaming worker.

        ⚠️  WIRED-BUT-MOCK-ONLY.  The ~43-min battery stresses the serving model;
        the write runner is NEVER executed live this phase — conftest blocks the
        real spawn and tests inject a FakeWriteRunner.  Streams raw output (no
        dedicated core parser — it is a multi-stage shell battery) so the UI shows
        live progress over the long run; it does NOT block the event loop.  Uses
        the SERVING model (injects ``MODEL`` / ``URL`` into the child env) and does
        NOT claim a GPU.  Confirmation is the CALLER's job."""
        import os as _os

        plan = self.full_validation_report_plan(model=model, url=url)
        env = dict(_os.environ)
        if model:
            env["MODEL"] = model
        if url:
            env["URL"] = url
        if on_event is not None or on_line is not None:
            self._write_runner.set_callbacks(on_event=on_event, on_line=on_line)
        # No reconcile gate (uses the serving model; claims no GPU); straight to
        # the streamer.  In tests this is the FakeWriteRunner; live it is blocked.
        return await self._write_runner.start_raw(
            plan.cmd, env=env, run_type=plan.kind, parser=_NullParser()
        )

    # ── Validate / Doctor: health + estate-diagnose + profile-triage (READS) ──────

    async def doctor(
        self, *, url: Optional[str] = None, slug: Optional[str] = None
    ) -> DoctorReport:
        """Full Doctor read (ALL legs are READ-only, safe to call live):

          - ``health.sh`` (text) → ``DoctorRead`` (reuses the existing parser);
          - ``diagnose-estate.sh --json`` → ``EstateDiagnose``;
          - ``diagnose-profile.sh <slug>`` (text-only — no --json) →
            ``ProfileTriage`` (only when a target ``slug`` is supplied).

        Each leg is best-effort: a failed leg carries its own error and does not
        fail the others."""
        report = DoctorReport()
        report.health = await self.doctor_read(url=url)
        report.estate = await self.estate_diagnose()
        if slug:
            report.profile = await self.profile_triage(slug)
        return report

    async def estate_diagnose(self) -> EstateDiagnose:
        """diagnose-estate.sh --json → EstateDiagnose (READ)."""
        data, err = await self._run_json(
            ["bash", "scripts/diagnose-estate.sh", "--json"], timeout=40.0
        )
        if data is None:
            return EstateDiagnose(error=err or "no output")
        return EstateDiagnose.from_dict(data)

    async def profile_triage(self, slug: str) -> ProfileTriage:
        """diagnose-profile.sh <slug> (text-only — NO --json) → ProfileTriage.

        Verified live: this script has no JSON mode, so we parse the 6-step text
        triage.  A non-zero exit still parses (the triage prints steps before a
        RED verdict)."""
        res = await self._runner.run(
            ["bash", "scripts/diagnose-profile.sh", slug],
            cwd=str(self.repo_root),
            timeout=60.0,
        )
        if res.timed_out:
            return ProfileTriage(slug=slug, error=f"timed out triaging {slug}")
        text = res.stdout or res.stderr
        tri = parse_profile_triage(text, slug)
        if not tri.steps and not tri.summary:
            tri.error = (res.stderr.strip()[:200] or f"no triage output (rc={res.returncode})")
        return tri

    async def verify_smoke(self) -> VerifySmoke:
        """Run ``scripts/verify.sh`` (the ~15s "is the model serving correctly?"
        smoke: server reachable → Genesis patch → basic completion → tool-call)
        and parse its ✓/✗ checks.  READ-only — it sends a couple of test queries
        to the LIVE model but claims/frees no GPU.

        verify.sh auto-detects the running container/port/served-model via
        preflight (it is designed to run post-setup with no args), so no env
        injection is needed — when nothing is serving, check 1 fails and the
        result reports unreachable, which is the correct 'not serving' signal."""
        res = await self._runner.run(
            ["bash", "scripts/verify.sh"], cwd=str(self.repo_root), timeout=90.0
        )
        if res.timed_out:
            vs = VerifySmoke(raw=res.stdout or "")
            vs.error = "verify.sh timed out (model unreachable or very slow)"
            return vs
        return parse_verify_smoke(res.stdout or res.stderr, rc=res.returncode)

    async def verify_full(self) -> VerifyFull:
        """Run ``scripts/verify-full.sh`` (the ~1-2 min functional battery:
        9 steps incl. streaming / thinking / cascade / MTP-acceptance) and parse
        its ``[N/9]`` steps + final summary.  READ-only (queries the live model;
        no GPU claim).  Heavier than verify_smoke but still a read — step
        ``[4/9]`` tool-calling is expected to fail on the default compose."""
        res = await self._runner.run(
            ["bash", "scripts/verify-full.sh"], cwd=str(self.repo_root), timeout=240.0
        )
        if res.timed_out:
            vf = VerifyFull(raw=res.stdout or "")
            vf.error = "verify-full.sh timed out (>4 min — model unreachable or stalled)"
            return vf
        return parse_verify_full(res.stdout or res.stderr, rc=res.returncode)

    # ── Validate / Benchmarks: explorer (corpus → BENCHMARKS.md fallback) (READ) ──

    async def benchmarks_explorer(
        self, *, prefer_corpus: bool = True
    ) -> tuple[list[BenchRow], Optional[str]]:
        """Filterable benchmarks rows for the explorer (READ).

        Preference order (verified live shapes):
          1. the structured #249 measurement-record corpus
             (``results/measurement-records/*.jsonl``) — authoritative
             TPS / ctx per (model, engine, topology), but NO 8-pack;
          2. a coarse BENCHMARKS.md scrape — carries the 8-pack and covers
             configs that were never run through the #249 producer.

        The corpus is per-rig + gitignored and may be EMPTY (it is on a fresh
        rig — verified: no ``results/measurement-records/`` dir), so the markdown
        fallback is the common path.  When corpus rows exist they take
        precedence for their (model, engine, topology) key, and the markdown
        fallback fills the 8-pack the corpus lacks + adds any configs the corpus
        doesn't cover.  Returns ``(rows, error)``; ``error`` is set only when
        BOTH sources are unavailable."""
        corpus_rows: list[BenchRow] = []
        if prefer_corpus:
            corpus_rows = self._read_measurement_corpus()

        md_rows = bench_rows_from_benchmarks_md(self._read_benchmarks_md())

        if not corpus_rows and not md_rows:
            return [], "no benchmark data (empty #249 corpus and no BENCHMARKS.md rows)"

        # Index markdown rows by (model, engine, topology) so corpus rows can
        # borrow the 8-pack the bench-only corpus record lacks.
        md_by_key: dict[tuple[str, str, str], BenchRow] = {}
        for r in md_rows:
            md_by_key.setdefault((r.model, r.engine, r.topology), r)

        out: list[BenchRow] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for cr in corpus_rows:
            key = (cr.model, cr.engine, cr.topology)
            seen_keys.add(key)
            md = md_by_key.get(key)
            if md and not cr.quality_8pk and md.quality_8pk:
                cr.quality_8pk = md.quality_8pk   # borrow the 8-pack
            out.append(cr)
        # Append markdown rows the corpus didn't cover.
        for r in md_rows:
            if (r.model, r.engine, r.topology) in seen_keys:
                continue
            out.append(r)
        return out, None

    def _read_measurement_corpus(self) -> list[BenchRow]:
        """Read every JSONL record from the #249 corpus dir into BenchRows.

        Pure file read (no subprocess): the corpus lives at
        ``results/measurement-records/<tag>__<fp>.jsonl``.  Each line is one
        record; malformed lines are skipped (never crash the explorer).  Newer
        lines win for a (model, engine, topology) key (the file is appended)."""
        corpus_dir = self.repo_root / "results" / "measurement-records"
        if not corpus_dir.is_dir():
            return []
        by_key: dict[tuple[str, str, str], BenchRow] = {}
        for path in sorted(corpus_dir.glob("*.jsonl")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = bench_row_from_corpus_record(rec)
                if row is not None:
                    by_key[(row.model, row.engine, row.topology)] = row
        return list(by_key.values())

    # ── Validate / Evidence: rebench run tags + paste-ready report (READ) ─────────

    async def evidence_list(self) -> list[EvidenceTag]:
        """Enumerate ``results/rebench/<tag>/`` run directories (READ).

        Pure filesystem walk: each subdirectory of ``results/rebench/`` is a run
        tag.  We surface what artifacts it carries (REPORT.md / _internal.json /
        soak) + a coarse date + a one-line TL;DR scraped from REPORT.md if
        present.  Sorted newest-first by directory mtime."""
        base = self.repo_root / "results" / "rebench"
        if not base.is_dir():
            return []
        tags: list[EvidenceTag] = []
        for d in sorted(
            (p for p in base.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            report = d / "REPORT.md"
            internal = d / "_internal.json"
            has_report = report.is_file()
            et = EvidenceTag(
                tag=d.name,
                path=str(d),
                has_report=has_report,
                has_internal=internal.is_file(),
                has_soak=(d / "soak.log").is_file() or (d / "soak-artifacts").is_dir(),
            )
            if has_report:
                et.date, et.tldr = self._scrape_report_meta(report)
            if not et.date:
                # mtime fallback (YYYY-MM-DD).
                import datetime as _dt

                et.date = _dt.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d")
            tags.append(et)
        return tags

    def _scrape_report_meta(self, report_path: Path) -> tuple[str, str]:
        """Pull (date, tldr) from a REPORT.md without importing the generator.

        REAL shape (verified live): a ``## TL;DR`` section of ``- `` bullets and
        a ``## Meta`` section with ``- **Date:** YYYY-MM-DD``."""
        try:
            text = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "", ""
        date = ""
        m = re.search(r"\*\*Date:\*\*\s*`?(\d{4}-\d{2}-\d{2})`?", text)
        if m:
            date = m.group(1)
        # First TL;DR bullet → coarse one-liner (strip markdown emphasis).
        tldr = ""
        in_tldr = False
        for line in text.splitlines():
            if line.strip().lower().startswith("## tl;dr"):
                in_tldr = True
                continue
            if in_tldr:
                s = line.strip()
                if s.startswith("- "):
                    tldr = re.sub(r"[*`]", "", s[2:]).strip()
                    break
                if s.startswith("#"):
                    break
        return date, tldr

    async def evidence_report(
        self, tag: str, *, compare_to: Optional[str] = None
    ) -> EvidenceReport:
        """Generate a paste-ready report for a run tag (READ — reads results).

        Uses ``scripts/rebench-report.py <tag_dir>`` (the canonical generator —
        report generation reads results, allowed live this phase).  It writes
        ``REPORT.md`` into the tag dir and we read it back; if generation fails
        but a REPORT.md already exists, we fall back to the existing file."""
        base = self.repo_root / "results" / "rebench" / tag
        if not base.is_dir():
            return EvidenceReport(tag=tag, error=f"no run dir results/rebench/{tag}")
        cmd = ["python3", "scripts/rebench-report.py", str(base), "--no-discuss"]
        if compare_to:
            cmp_dir = self.repo_root / "results" / "rebench" / compare_to
            cmd += ["--compare-to", str(cmp_dir)]
        res = await self._runner.run(cmd, cwd=str(self.repo_root), timeout=120.0)
        report_md = base / "REPORT.md"
        if report_md.is_file():
            try:
                body = report_md.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return EvidenceReport(tag=tag, report_path=str(report_md), error=str(exc))
            return EvidenceReport(tag=tag, report_path=str(report_md), body=body)
        # Generation produced no REPORT.md and none pre-existed.
        return EvidenceReport(
            tag=tag,
            error=(res.stderr.strip()[:200] or f"report generation failed (rc={res.returncode})"),
        )

    # ── Validate / ④ Measure: producer-measured vs the curated catalog bar (READ) ─

    async def measure_vs_bar(
        self, tag: str, *, variants: Optional[list[VariantRow]] = None
    ) -> MeasureVsBar:
        """Compare a rebench tag's MEASURED numbers against the curated catalog's
        published bar for the SAME class — "did this config earn catalog-grade?"
        (design §3.3 ④).  READ-only: pure filesystem reads (the tag dir) + the
        existing benchmarks explorer; NO GPU / network / write.

        Steps:
          1. parse the producer's MEASURED numbers from ``results/rebench/<tag>/``
             — ``_internal.json`` (authoritative) with a ``REPORT.md`` TL;DR
             scrape fallback (reuses the same artifacts evidence_report reads);
          2. resolve the tag's MODEL **and engine-FAMILY** (from REPORT.md Meta —
             Container/vLLM-image — then the served slug → registry-variant
             engine, best-effort) and fetch the CURATED bar via
             ``benchmarks_explorer()`` (prefer the #249 corpus rows, BENCHMARKS.md
             fallback), matched by ``_bench_row_matches`` on (model, engine-family)
             so a vLLM-dual run is NOT graded against a single-card llama.cpp row.
             When the engine can't be resolved (or no bar exists for it) the bar is
             picked DETERMINISTICALLY and the matched row's engine/topology is
             surfaced in the struct + caveats so the user sees which bar was used;
          3. return a structured comparison with a SIMPLE, honest ``verdict`` and
             ``protocol_caveats`` LISTING what the cockpit cannot verify (matched
             power? same harness? same prompts?).  It FLAGS the protocol — it
             NEVER fabricates a "catalog-grade" it can't substantiate, and
             discloses the bar's source (corpus vs BENCHMARKS.md)."""
        base = self.repo_root / "results" / "rebench" / tag
        if not base.is_dir():
            return MeasureVsBar(tag=tag, error=f"no run dir results/rebench/{tag}")

        # (1) MEASURED numbers — _internal.json first, REPORT.md fallback.
        measured = MeasuredNumbers()
        report_md_text = ""
        report_md = base / "REPORT.md"
        if report_md.is_file():
            try:
                report_md_text = report_md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report_md_text = ""
        internal = base / "_internal.json"
        if internal.is_file():
            try:
                blob = json.loads(internal.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                blob = None
            measured = measured_from_internal_json(blob)
        # If the sidecar gave nothing usable, fall back to REPORT.md numbers.
        if not measured.has_any and report_md_text:
            measured = measured_from_report_md(report_md_text)
        # The model AND engine-family are only carried in REPORT.md Meta (the
        # _internal.json sidecar has neither) — resolve them from there even when
        # the numbers came from the sidecar.
        if report_md_text:
            from_report = measured_from_report_md(report_md_text)
            if not measured.model:
                measured.model = from_report.model
            if not measured.engine:
                measured.engine = from_report.engine

        # (2) Resolve the curated bar for this model — ENGINE-AWARE so a vLLM-dual
        # run is not graded against a single-card llama.cpp row just because they
        # share a model.  Resolve the RUN's engine-family from the rebench
        # artifacts (REPORT.md Meta Container/vLLM-image first; then the served
        # slug → registry-variant engine), and pass THAT as the engine arg to
        # _bench_row_matches.  When the engine genuinely can't be resolved we pick
        # deterministically (corpus-then-doc-order) BUT flag it + surface the
        # matched bar's engine/topology so the user sees which bar was used.
        run_engine = self._resolve_run_engine(measured, variants)
        rows, _err = await self.benchmarks_explorer()
        bar: Optional[BenchRow] = None
        engine_resolved = False
        if measured.model and rows:
            mkey = _canon_model_key(measured.model)
            # Model-matched candidates first (served-name → slug via canon).
            model_rows = [r for r in rows if _bench_row_matches(r, measured.model, "")]
            if not model_rows:
                model_rows = [r for r in rows if _canon_model_key(r.model) == mkey and mkey]
            # Engine discrimination: among the model-matched rows, keep only those
            # of the run's engine-family.  (We compare families directly here, NOT
            # via _bench_row_matches, because that re-checks the served-name model
            # which differs from the slug — model-matching was already done above.)
            # If that empties the set (no published bar for this engine) we fall
            # back to the model-only set and flag that we could not engine-match.
            if run_engine:
                eng_rows = [
                    r for r in model_rows
                    if _canon_engine_family(r.engine) == run_engine
                ]
                if eng_rows:
                    model_rows = eng_rows
                    engine_resolved = True
            # Prefer a corpus row (authoritative TPS) over a BENCHMARKS.md row.
            model_rows.sort(key=lambda r: 0 if r.source == "corpus" else 1)
            bar = model_rows[0] if model_rows else None

        vsbar = MeasureVsBar(
            tag=tag,
            measured=measured,
            bar=bar,
            bar_source=(bar.source if bar else ""),
            run_engine=run_engine,
            engine_resolved=engine_resolved,
        )
        if bar is not None:
            # Fix 2 (corpus metric mismatch): a corpus bar's narr_tps is WALL TPS
            # (bench_row_from_corpus_record maps wall_tps→narr_tps) while the
            # measured narr_tps is narrative DECODE — subtracting them shows a
            # fabricated green delta.  The corpus carries a single canonical-short
            # bench (no narrative/code split), so SUPPRESS the narrative-TPS delta
            # for a corpus bar and flag it in protocol_caveats; the code/decode
            # delta is left (both decode).  A BENCHMARKS.md bar carries the proper
            # narrative/code decode pair → keep its narrative delta.
            corpus_bar = vsbar.bar_source == "corpus"
            if (not corpus_bar) and measured.narr_tps is not None and bar.narr_tps is not None:
                vsbar.narr_tps_delta = measured.narr_tps - bar.narr_tps
            if measured.code_tps is not None and bar.code_tps is not None:
                vsbar.code_tps_delta = measured.code_tps - bar.code_tps

        # (3) honest verdict + protocol caveats (FLAG, never fabricate).
        vsbar.verdict = _measure_verdict(vsbar)
        vsbar.protocol_caveats = self._measure_protocol_caveats(vsbar)
        return vsbar

    @staticmethod
    def _resolve_run_engine(
        measured: MeasuredNumbers, variants: Optional[list[VariantRow]]
    ) -> str:
        """Resolve the RUN's engine-FAMILY so the curated bar is engine-matched.

        Two signals, best-first:
          1. REPORT.md Meta carried it (``measured.engine`` — the Container prefix
             / vLLM-image tell, parsed in ``measured_from_report_md``);
          2. else map the run's served slug/name to a registry variant and take
             that variant's engine-family.  The served-name (``qwen3.6-27b-…``)
             rarely equals a slug, so match by canon-model AND require a UNIQUE
             engine across that model's variants — if a model has variants on
             multiple engines (e.g. vllm + ik-llama), the served-name alone can't
             disambiguate, so return "" (the caller picks deterministically and
             flags that it could not engine-match).
        Returns a coarse family token via ``_canon_engine_family``, or ""."""
        if measured.engine:
            return _canon_engine_family(measured.engine)
        if not variants or not measured.model:
            return ""
        mkey = _canon_model_key(measured.model)
        fams: set[str] = set()
        for v in variants:
            vmodel = getattr(v, "model", "") or ""
            if mkey and _canon_model_key(vmodel) == mkey:
                fam = _canon_engine_family(getattr(v, "engine", "") or "")
                if fam:
                    fams.add(fam)
        # Only confident when the model's variants are all one engine-family.
        return next(iter(fams)) if len(fams) == 1 else ""

    @staticmethod
    def _measure_protocol_caveats(vsbar: MeasureVsBar) -> list[str]:
        """List what the cockpit CANNOT verify about the comparison — so the
        verdict is read as a flag, not a certification.  Honesty over a fabricated
        "catalog-grade"."""
        caveats: list[str] = []
        if vsbar.bar is None:
            if not vsbar.measured.model:
                caveats.append(
                    "Could not resolve this run's MODEL from REPORT.md — no curated bar matched."
                )
            else:
                caveats.append(
                    f"No curated catalog bar found for model '{vsbar.measured.model}'."
                )
            return caveats
        # We have a bar — surface WHICH bar was used (engine + topology) so the
        # comparison is legible, then flag every protocol dimension we can't
        # confirm.
        src = "the #249 measurement corpus" if vsbar.bar_source == "corpus" else "BENCHMARKS.md"
        bar_eng = _canon_engine_family(vsbar.bar.engine) or vsbar.bar.engine or "?"
        bar_topo = vsbar.bar.topology or "?"
        caveats.append(
            f"Bar source: {src} — matched row: engine '{bar_eng}', topology "
            f"'{bar_topo}', published by another run, not this one."
        )
        # Whether the run's engine actually drove the bar selection.
        if not vsbar.engine_resolved:
            if vsbar.run_engine:
                caveats.append(
                    f"No curated bar matched this run's engine '{vsbar.run_engine}' — "
                    f"compared against an engine '{bar_eng}' bar instead (deterministic "
                    "fallback; NOT an engine-matched grade)."
                )
            else:
                caveats.append(
                    "Could NOT resolve this run's ENGINE from REPORT.md — the bar was "
                    f"picked deterministically (engine '{bar_eng}', topology '{bar_topo}'); "
                    "it may be a different engine/topology than this run."
                )
        # Corpus bar carries a single canonical-short bench (wall-vs-decode, no
        # narrative/code split) — the narrative-TPS delta is suppressed (Fix 2).
        if vsbar.bar_source == "corpus":
            caveats.append(
                "Bar is a #249 corpus row: a single canonical-short bench (its "
                "narr_tps is WALL, not decode, and it carries no narrative/code "
                "split) — the narrative-TPS delta is SUPPRESSED to avoid a "
                "wall-vs-decode comparison; only the decode delta is shown."
            )
        caveats.append(
            "Cannot verify MATCHED POWER CAP — the bar may have been measured at a "
            "different W/card (this rig systemd-caps to 230W; bench scripts record "
            "but do not enforce power)."
        )
        caveats.append(
            "Cannot verify SAME HARNESS / engine pin — a TPS gap can be an image / "
            "flag delta, not a config regression."
        )
        caveats.append(
            "Cannot verify SAME PROMPTS / sampling — the 8-pack + bench prompts must "
            "match for an apples-to-apples grade."
        )
        if vsbar.measured.source == "REPORT.md":
            caveats.append(
                "Measured numbers scraped from REPORT.md (no _internal.json) — less precise."
            )
        caveats.append(
            "This is a FLAG, not a catalog-grade certification — run the full gate "
            "(③) + measure (rebench) at matched power before promoting."
        )
        return caveats

    # ── Validate / Evidence: submit-bench (OUTWARD-FACING WRITE — gated) ───────────

    async def submit_bench_preview(self, tag: str) -> dict[str, Any]:
        """Generate the BENCHMARKS.md row for a tag WITHOUT submitting (READ-ish).

        ``submit-bench.sh --tag <tag>`` (no ``--auto-submit``) only writes a
        local ``BENCHMARKS-row.md`` into the tag dir and prints the row — it does
        NOT touch the network or open a PR (verified live: the network/PR path is
        gated behind ``--auto-submit``).  This lets the UI show the row before
        the user confirms the outward submit.  Returns ``{"row","error"}``."""
        res = await self._runner.run(
            ["bash", "scripts/submit-bench.sh", "--tag", tag],
            cwd=str(self.repo_root),
            timeout=60.0,
        )
        if res.timed_out:
            return {"row": "", "error": f"timed out generating row for {tag}"}
        # The row is also written to results/rebench/<tag>/BENCHMARKS-row.md.
        row_file = self.repo_root / "results" / "rebench" / tag / "BENCHMARKS-row.md"
        row = ""
        if row_file.is_file():
            try:
                row = row_file.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                row = ""
        if not row:
            row = (res.stdout or "").strip()
        if not row:
            return {"row": "", "error": (res.stderr.strip()[:200] or "no row generated")}
        return {"row": row, "error": None}

    def submit_bench(self, tag: str, *, as_pr: bool = False) -> ActionPlan:
        """Build the OUTWARD-FACING submit-bench ActionPlan (NEVER auto-fired).

        ``submit-bench.sh --tag <tag> --auto-submit [--as-pr]`` opens the network
        path (``gh pr create`` / the localmaxxing POST).  This is an outward
        write that LEAVES THE RIG, so the plan is built but NEVER executed
        automatically: ``requires_confirm=True`` + ``network=True`` so the UI
        shows a network-warning confirm, and tests mock the network.  It does not
        claim a GPU → ``requires_reconcile=False``."""
        cmd = ["bash", "scripts/submit-bench.sh", "--tag", tag, "--auto-submit"]
        if as_pr:
            cmd.append("--as-pr")
        return ActionPlan(
            kind="submit_bench",
            cmd=cmd,
            description=f"submit-bench --tag {tag} --auto-submit{' --as-pr' if as_pr else ''}",
            requires_reconcile=False,
            requires_confirm=True,
            network=True,
        )

    # ── Share-back (R2b): rig report + problem report (READ — paste-ready) ────────

    async def rig_report(self) -> dict[str, Any]:
        """Generate a paste-ready rig/bench report (READ — no network/GPU write).

        Bare ``scripts/report.sh`` assembles a redacted, paste-ready snapshot
        (hardware + stack + boot-log highlights, ~2 s) that a consumer can drop
        into a GitHub ``numbers-from-your-rig`` issue — the lightweight "one
        keystroke while running" affordance.  We DO NOT pass ``--full``: that is
        the ~43-min verify+stress+soak+bench+agentic battery, a producer-lane
        Gate action (R3), NOT a lightweight share-back — it would contend with
        the user's running model for the GPU.  Nor ``--submit`` / ``--auto`` —
        generation is a local read; the user copies the text and posts it.
        Returns ``{"report", "error"}``."""
        res = await self._runner.run(
            ["bash", "scripts/report.sh"],
            cwd=str(self.repo_root),
            timeout=60.0,
        )
        if res.timed_out:
            return {"report": "", "error": "timed out generating rig report"}
        body = (res.stdout or "").strip()
        if not body:
            return {
                "report": "",
                "error": (res.stderr.strip()[:300] or f"report.sh failed (rc={res.returncode})"),
            }
        return {"report": body, "error": None}

    async def problem_report(
        self,
        slug: str,
        *,
        boot_log: str = "",
        url: Optional[str] = None,
        variants: Optional[list[VariantRow]] = None,
    ) -> dict[str, Any]:
        """Assemble a paste-ready problem/issue report from readily-available
        failure context (READ — gathers LOCAL context only, no auto-network).

        Pulls together: (a) the recent boot-log lines passed in (from the
        serve-live LivePane) and, when a container is resolvable, a ``docker
        logs`` tail of the failed container; (b) the slug's compose path
        (best-effort from the registry/variants); (c) a rig snapshot (GPU cards
        + driver from ``estate_state``/detect).  The user copies the text and
        opens the issue themselves — nothing leaves the rig."""
        sections: list[str] = []
        sections.append("## Problem report")
        sections.append(f"- **Slug:** `{slug or '—'}`")

        # (b) compose path (best-effort, registry/variants).
        variant = None
        if variants:
            variant = next((v for v in variants if getattr(v, "slug", "") == slug), None)
        compose_path = getattr(variant, "compose_path", "") if variant is not None else ""
        sections.append(f"- **Compose:** `{compose_path or '—'}`")
        if variant is not None:
            eng = getattr(variant, "engine", "") or getattr(variant, "switch_engine", "")
            sections.append(f"- **Engine:** `{eng or '—'}`")

        # (c) rig snapshot — GPU cards + driver.
        try:
            state = await self.estate_state(variants=variants)
        except Exception:  # pragma: no cover - defensive
            state = EstateState()
        gpus = list(getattr(state, "gpus", []) or [])
        sections.append("")
        sections.append("### Rig snapshot")
        driver = self._driver_version_from_gpus(gpus)
        sections.append(f"- **GPUs:** {len(gpus)} card(s)")
        if driver:
            sections.append(f"- **Driver:** {driver}")
        for g in gpus:
            idx = getattr(g, "index", "?")
            tot = getattr(g, "mem_total_mib", None)
            tot_s = f"{tot} MiB" if tot else "unknown"
            sections.append(f"  - GPU {idx}: {tot_s} total")

        # (a) boot log — passed-in lines first, then docker logs of the failed
        # container if one is resolvable (READ, never a write).
        log_lines: list[str] = []
        if boot_log.strip():
            log_lines.extend(boot_log.strip().splitlines())
        container = getattr(variant, "container", "") if variant is not None else ""
        if container:
            try:
                res = await self.container_logs(container, tail=50)
            except Exception:  # pragma: no cover - defensive
                res = {"lines": [], "error": "log read failed"}
            if res.get("error"):
                log_lines.append(f"[docker logs unavailable: {res['error']}]")
            else:
                log_lines.extend(res.get("lines", [])[-50:])
        sections.append("")
        sections.append("### Boot / failure log")
        if log_lines:
            sections.append("```")
            sections.extend(log_lines[-80:])
            sections.append("```")
        else:
            sections.append("(no boot-log context captured)")

        return {"report": "\n".join(sections), "error": None}

    def _driver_version_from_gpus(self, gpus: list[Any]) -> str:
        """Best-effort NVIDIA driver version from a GpuInfo list.

        GpuInfo carries no driver field today, so we read whatever attribute the
        core detect may have attached (``driver`` / ``driver_version``) and
        degrade to '' silently — the report stays useful without it."""
        for g in gpus:
            for attr in ("driver_version", "driver"):
                v = getattr(g, attr, None)
                if v:
                    return str(v)
        return ""

    # ── Power cap: read (safe) + write/sweep (WIRED, mock-only, confirm) ──────────

    async def power_cap_get(self) -> PowerCapState:
        """gpu-mode power-cap status → PowerCapState (READ — safe to call live).

        Verified live: prints a banner + a per-GPU ``index, limit W, default W,
        min W, max W`` table."""
        res = await self._runner.run(
            ["bash", "scripts/gpu-mode.sh", "power-cap", "status"],
            cwd=str(self.repo_root),
            timeout=20.0,
        )
        if res.timed_out:
            st = PowerCapState(error="timed out reading power-cap status")
            return st
        return parse_power_cap_status(res.stdout or res.stderr)

    def power_cap_set(self, state) -> ActionPlan:
        """Build the power-cap WRITE ActionPlan (WIRED, mock-only — rig mutation).

        ``gpu-mode power-cap`` takes ``on`` (re-apply the default 230W cap) /
        ``off`` (uncap to hardware default) / a positive integer **wattage**
        (a custom cap applied to both cards, validated by gpu-mode against each
        card's [min,max] — the cockpit power-cap menu's "custom" option).
        Mutating a GPU power limit is a rig change, so this is built but NEVER
        run live this phase; it goes through a confirm modal.  It does not claim
        a GPU → ``requires_reconcile=False``."""
        if isinstance(state, bool):  # guard: bool is an int subclass
            raise ValueError(f"power-cap state must be 'on'/'off'/<watts>, got {state!r}")
        if isinstance(state, int) or (isinstance(state, str) and state.isdigit()):
            watts = int(state)
            if watts <= 0:
                raise ValueError(f"power-cap wattage must be a positive integer, got {watts!r}")
            arg, desc = str(watts), f"gpu-mode power-cap {watts}W (custom)"
        elif state in ("on", "off"):
            arg = state
            desc = f"gpu-mode power-cap {state} ({'default 230W' if state == 'on' else 'uncap'})"
        else:
            raise ValueError(
                f"power-cap state must be 'on' (default 230W) / 'off' (uncap) / <watts>, got {state!r}"
            )
        return ActionPlan(
            kind="power_cap",
            cmd=["bash", "scripts/gpu-mode.sh", "power-cap", arg],
            description=desc,
            requires_reconcile=False,
            requires_confirm=True,
        )

    def power_cap_sweep(self, *, step_size: Optional[int] = None,
                        caps: Optional[list[int]] = None) -> ActionPlan:
        """Build the power-cap-sweep ActionPlan (WIRED, mock-only — rig mutation).

        ``power-cap-sweep.sh`` runs a power-limit A/B sweep (needs sudo on the
        real rig) — it mutates the GPU power cap repeatedly AND runs benches at
        each cap.  Heavy + mutating, so built-but-NEVER-run-live; confirm-gated.
        ``--caps`` / ``--step-size`` are passed through when supplied."""
        cmd = ["sudo", "bash", "scripts/power-cap-sweep.sh"]
        if caps:
            cmd += ["--caps", ",".join(str(c) for c in caps)]
        if step_size:
            cmd += ["--step-size", str(step_size)]
        return ActionPlan(
            kind="power_cap_sweep",
            cmd=cmd,
            description=f"power-cap-sweep{(' caps=' + ','.join(map(str, caps))) if caps else ''}".strip(),
            requires_reconcile=False,
            requires_confirm=True,
        )

    # (image prune / prune-all removed from the cockpit — it was Orchestration-only
    # and the maintainer dropped it from the tab; ``gpu-mode prune[-all]`` stays
    # available on the CLI for manual cleanup.)

    # ── Container: top (READ) + rm (WIRED, mock-only, reconcile-gated) ────────────

    async def container_top(self, name: str) -> ContainerTop:
        """docker top <name> → ContainerTop (READ — never mutates the container)."""
        res = await self._runner.run(
            ["docker", "top", name],
            cwd=str(self.repo_root),
            timeout=15.0,
        )
        if res.timed_out:
            return ContainerTop(name=name, error=f"timed out reading top for {name}")
        text = res.stdout or ""
        if res.returncode != 0 and not text.strip():
            return ContainerTop(name=name, error=(res.stderr.strip()[:200] or f"rc={res.returncode}"))
        return parse_docker_top(name, text)

    def container_rm(self, name: str, *, force: bool = False, force_reason: str = "") -> ActionPlan:
        """Build the ``docker rm`` ActionPlan (WIRED, mock-only — RECONCILE-GATED).

        Removing a container frees a GPU it held, so this MUST route through the
        reconcile gate (``requires_reconcile=True``) exactly like a stop — the
        gate sees that the rm collides with the running container and surfaces
        it.  ``docker rm`` cannot remove a running container without ``-f``;
        the ``force`` flag adds ``-f`` AND becomes the reconcile force override
        (so the rm of a live container is an explicit, reasoned action)."""
        cmd = ["docker", "rm"]
        if force:
            if not force_reason:
                raise ValueError("force=True requires a force_reason (surfaced to user)")
            cmd.append("-f")
        cmd.append(name)
        return ActionPlan(
            kind="container_rm",
            cmd=cmd,
            description=f"docker rm {'-f ' if force else ''}{name}",
            requires_reconcile=True,     # frees a GPU → gate it
            requires_confirm=True,
            force=force,
            force_reason=force_reason,
        )

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 5 — the three v2 hooks (Evaluate · Promote-to-catalog · Optimize)
    # ════════════════════════════════════════════════════════════════════════════

    # ── Hook 1: Evaluate — hand the SHARED ServingTarget to c3t (design §4) ────────

    def evaluate_handoff(self, target: Optional[ServingTarget]) -> EvaluateHandoff:
        """Build the c3t Evaluate hand-off for a running target (Estate → ▸ Evaluate).

        Hands the SHARED ``club3090_tui_core.detect.ServingTarget`` (the SAME
        dataclass c3t speaks — design §4/§6.6) to the post-boot evaluator at
        ``tools/test-console``.  The launch is HEAVY (c3t runs tests against the
        live serving model), so the plan is ``requires_confirm=True`` and is
        execution-MOCKED this phase — ``launch_evaluate`` streams it via the write
        runner which conftest blocks / tests fake.  It does NOT claim or free a
        GPU (c3t only HITS the endpoint) → ``requires_reconcile=False``.

        ``target`` is carried by IDENTITY on the returned handoff so the receiver
        evaluates exactly what's running.  The launch invokes ``scripts/c3t``
        (the isolated-env launcher) with the target's endpoint/model/container
        passed through env so c3t scopes to the current target rather than
        re-detecting; the env is injected at LAUNCH time (``launch_evaluate``),
        not baked into the inspectable cmd here."""
        if target is None or not getattr(target, "url", ""):
            # Nothing serving — there is no running model for c3t to evaluate.
            return EvaluateHandoff(
                target=target,
                plan=ActionPlan(
                    kind="evaluate",
                    cmd=["bash", "scripts/c3t"],
                    description="c3t (no running target)",
                    requires_reconcile=False,
                    requires_confirm=True,
                ),
                available=False,
                reason="no running serving target detected — start a model first",
            )
        model = getattr(target, "model", "") or ""
        url = getattr(target, "url", "") or ""
        plan = ActionPlan(
            kind="evaluate",
            cmd=["bash", "scripts/c3t"],
            description=f"c3t evaluate → {model or url}",
            requires_reconcile=False,    # c3t hits the endpoint; claims no GPU
            requires_confirm=True,       # heavy — runs tests against the model
        )
        return EvaluateHandoff(target=target, plan=plan, available=True)

    async def launch_evaluate(
        self,
        target: Optional[ServingTarget],
        *,
        on_event: Optional[Callable[[Any], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Launch c3t scoped to the SHARED ServingTarget, streamed (MOCK-ONLY).

        ⚠️  WIRED-BUT-MOCK-ONLY.  c3t runs the post-boot evaluator against the
        live serving model — heavy.  The write runner is NEVER executed live this
        phase; conftest blocks the real spawn and tests inject a FakeWriteRunner.

        Scopes c3t to ``target`` by passing the endpoint/model/container through
        the child env (``C3T_REPO_ROOT`` + ``C3T_TARGET_*``) so the test-console
        preselects the SAME running model rather than re-detecting.  Confirmation
        is the CALLER's job (the Estate pane wires a confirm modal before calling
        this — the handoff plan always ``requires_confirm``)."""
        import os as _os

        handoff = self.evaluate_handoff(target)
        env = dict(_os.environ)
        env["C3T_REPO_ROOT"] = str(self.repo_root)
        if target is not None:
            # Scope c3t to the SAME target (shared ServingTarget fields).
            if getattr(target, "url", ""):
                env["C3T_TARGET_URL"] = target.url
            if getattr(target, "model", ""):
                env["C3T_TARGET_MODEL"] = target.model
            if getattr(target, "container", ""):
                env["C3T_TARGET_CONTAINER"] = target.container
            if getattr(target, "slug", ""):
                env["C3T_TARGET_SLUG"] = target.slug
        if on_event is not None or on_line is not None:
            self._write_runner.set_callbacks(on_event=on_event, on_line=on_line)
        return await self._write_runner.start_raw(
            handoff.plan.cmd, env=env, run_type=handoff.plan.kind, parser=_NullParser()
        )

    # ── Hook 2: Promote to catalog — SCAFFOLD + GATE (design §3.5b) ────────────────

    def promote_scaffold(
        self,
        *,
        byo: Optional[ByoResult],
        measurement: Optional[Measurement] = None,
        model_id: str = "",
        sibling_compose_path: str = "",
    ) -> PromoteScaffold:
        """COMPUTE + PREVIEW the catalog-promotion scaffold (design §3.5b).

        For a served/validated BYO model, compute a ModelProfile YAML skeleton +
        a ``compose_registry.py`` ``_entry(...)`` row from facts the app already
        holds (the BYO pull-gate arch facts in ``byo`` + the Evidence
        ``measurement`` numbers), match the REAL shapes
        (``scripts/lib/profiles/models/*.yml`` + ``_entry(...)`` +
        ``docs/ADDING_MODELS.md``), and attach a GATED hand-off plan.

        In THIS phase: compute + preview ONLY.  The write-into-``scripts/`` + the
        guard-suite run is the attached ``write_plan`` (built by
        ``promote_write_plan``), which is MOCKED / never-executed and NEVER
        auto-fires.  This method does NOT touch the filesystem."""
        scaffold = compute_promote_scaffold(
            byo=byo,
            measurement=measurement,
            model_id=model_id,
            sibling_compose_path=sibling_compose_path,
        )
        if scaffold.computed:
            scaffold.write_plan = self.promote_write_plan(scaffold)
        return scaffold

    def promote_write_plan(self, scaffold: PromoteScaffold) -> ActionPlan:
        """Build the GATED, MOCK-ONLY write+guard ActionPlan for a scaffold.

        ⚠️  REPO MUTATION — NEVER auto-fired / executed this phase.  This would
        (a) write the profile YAML + registry row into ``scripts/lib/profiles/``
        and (b) run the guard suite (``for t in scripts/tests/*.sh``).  Because it
        mutates ``scripts/`` (a repo write) it is built but NEVER executed live —
        ``requires_confirm=True``; tests assert it is mock-only and never reaches
        the write runner.  It does NOT claim a GPU → ``requires_reconcile=False``.

        The cmd is a guard-suite invocation as a PLACEHOLDER for the gated
        action; the actual file-write is performed by the (future) promote tool,
        not auto-written by the cockpit (do NOT auto-write into scripts/)."""
        return ActionPlan(
            kind="promote_catalog",
            cmd=list(scaffold.guard_suite_cmd)
            or ["bash", "-c", 'for t in scripts/tests/*.sh; do bash "$t"; done'],
            description=(
                f"promote {scaffold.model_id} → catalog "
                f"(write {scaffold.profile_path} + registry {scaffold.registry_slug}, "
                "then guard suite)"
            ),
            requires_reconcile=False,    # no GPU contention — a repo write
            requires_confirm=True,       # repo mutation — confirm, never auto
        )

    # ── Hook 3: Optimize for my card — DORMANT v0.10.0 seam (design §5.2 seam 1) ────

    async def optimize_for_card(
        self, *, slug: str = "", card: Optional[str] = None
    ) -> OptimizerReport:
        """The ▸ Optimize-for-my-card seam — DORMANT until v0.10.0 (design §5.2).

        The optimizer (``recommend --optimize`` / ``generate_compose.py
        --optimize``) does NOT exist yet.  This detects its absence and returns an
        ``OptimizerReport(available=False, message='optimizer not available
        (v0.10.0)')`` — it NEVER fabricates optimizer output.  The honesty-gate
        fields on the report (boot-fit predicted|measured · runtime
        soak-validated · confidence tier · cliff-class --accept-runtime-risk) are
        the reserved INTERFACE, rendered only once the engine lands.

        Absence is detected via the read runner probing for the optimizer's
        ``--optimize`` flag; any non-zero / missing result keeps the seam
        dormant.  Until the engine exists this is, in practice, always
        unavailable."""
        # Probe for the optimizer flag.  When it lands it will print a JSON
        # OptimizerReport on `recommend --optimize --json`; until then the probe
        # returns non-zero / empty and we stay honestly dormant.
        try:
            res = await self._runner.run(
                ["bash", "scripts/recommend.sh", "--optimize", "--probe"],
                cwd=str(self.repo_root),
                timeout=15.0,
            )
        except Exception:
            return OptimizerReport(available=False)
        if not res.ok or "optimize" not in (res.stdout or "").lower():
            # No optimizer engine → dormant.  Do NOT fabricate a recommendation.
            return OptimizerReport(available=False)
        # (Reserved) — when the engine lands, parse its JSON honesty-gate output
        # here.  Until then this branch is unreachable; keep the seam honest.
        return OptimizerReport(available=False)


# ── helpers ──────────────────────────────────────────────────────────────────────


class _NullParser:
    """Passthrough parser for the write runner — emits no structured events."""

    def parse_line(self, line: str):  # noqa: D401 - protocol shim
        return None


# First published HOST port in a docker-ps ``Ports`` field (``0.0.0.0:4000->4000/
# tcp`` → 4000).  GENERAL match (any internal port), unlike the engine-only
# ``PORT_MAP_BROAD_RE`` — used to label a supporting service with the port it serves.
_HOST_PORT_RE = re.compile(r":(\d+)->")

# Rig services that hold a GPU but share no naming prefix with the engines /
# estate containers — matched by name so the reconcile gate sees them.
_GPU_SERVICE_NAMES = ("comfyui", "step-audio", "step-audio-editx")

# UX Batch 5 (studio-* / #2-ext): the rig's AI-studio stack containers occupy
# GPU0 but match no engine/estate prefix and no _GPU_SERVICE_NAMES entry, so they
# were INVISIBLE in the Operate service list (GPU0's holder unattributed — the
# "what about all the OTHER services" gap from Batch 1).  These name-fragments
# classify a RUNNING studio/comfy/ai-studio container as kind ``"stack"`` so each
# shows as its own row (NOT collapsed into one greyed ``services/studio`` entry)
# and the reconcile gate sees it as the GPU holder it is.  Scope stays HONEST:
# this only matches containers docker ps reports RUNNING — it is NOT a
# ``docker ps -a`` of every dead container.
_STACK_CONTAINER_FRAGMENTS = ("studio", "comfy")


def _normalize_service_name(name: str) -> str:
    """Normalize a service-dir / container name for cross-matching: lowercase
    and strip ``-`` / ``_`` separators.  So ``open-webui`` (the services/ dir)
    and ``openwebui`` (the running container_name) collapse to the same key."""
    return name.lower().replace("-", "").replace("_", "")


def _service_dir_matches_running(svc_dir: str, running_norm: str) -> bool:
    """True when a ``services/<svc_dir>`` entry is represented by a running
    container whose ALREADY-NORMALIZED name is ``running_norm``.

    Containers commonly carry a suffix/prefix off the service name
    (``comfyui-server`` covers ``comfyui``), so this is a normalized substring
    match in either direction — but only on non-empty keys (an empty service
    slug must not match everything)."""
    svc_norm = _normalize_service_name(svc_dir)
    if not svc_norm or not running_norm:
        return False
    return svc_norm in running_norm or running_norm in svc_norm


def _classify_container_kind(name: str) -> Optional[str]:
    """Classify a docker-ps container name into a GPU-holder kind.

    Returns ``"engine"`` for the core engine prefixes, ``"estate"`` for the
    estate planner's ``club3090-`` containers, ``"service"`` for the named
    GPU-holding rig services (ComfyUI / Step-Audio), ``"stack"`` for a RUNNING
    studio/AI-studio stack container (Batch 5: the studio-* GPU0 occupants that
    matched no prefix and were previously invisible), else ``None`` (not a GPU
    holder we surface — open-webui / redis / qdrant …).

    Order matters: the named ``service`` check precedes the ``stack`` fragment
    check so ``comfyui`` keeps its ``service`` kind (it's a first-class GPU
    service), and only the genuinely-unclassified studio-* fall through to
    ``stack``."""
    from club3090_tui_core.detect import ENGINE_PREFIXES

    if ENGINE_PREFIXES.match(name):
        return "engine"
    if name.startswith("club3090-"):
        return "estate"
    lname = name.lower()
    if any(svc in lname for svc in _GPU_SERVICE_NAMES):
        return "service"
    if any(frag in lname for frag in _STACK_CONTAINER_FRAGMENTS):
        return "stack"
    return None


def _container_gpu_set(c: ContainerInfo) -> Optional[set[int]]:
    """The set of GPU indices a container provably holds, or None if unknown.

    docker ps does not expose the device list, so this is None unless something
    upstream populated ``ContainerInfo.gpus`` (e.g. ``"0,1"`` / ``"1"``).  None
    means "unknown" → the gate stays conservative and treats it as a conflict."""
    raw = (getattr(c, "gpus", "") or "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out or None


def _variant_row_from_dict(d: dict[str, Any]) -> VariantRow:
    """Build a VariantRow from the registry-emit --json 'variants' dict.

    The --json variants block carries an extra ``source`` field not on the
    tab-row dataclass; it's attached as an attribute for the catalog 'source'
    column without altering the shared-core schema.
    """
    row = VariantRow(
        slug=str(d.get("slug", "")),
        switch_engine=str(d.get("switch_engine", "")),
        launch_engine=str(d.get("launch_engine", "")),
        compose_dir=str(d.get("compose_dir", "")),
        file=str(d.get("file", "")),
        port=int(d["port"]) if str(d.get("port", "")).isdigit() else 0,
        model=str(d.get("model", "")),
        engine=str(d.get("engine", "")),
        kvcalc_key=str(d.get("kvcalc_key", "")),
        container=str(d.get("container", "")),
        compose_path=str(d.get("compose_path", "")),
        status=str(d.get("status", "")),
        ctx_label=str(d.get("ctx_label", "")),
        # A7/MUST-FIX 2: the EXACT numeric configured ctx (registry max_ctx int).
        # None when the contract didn't carry it (older --json / tab fallback) so
        # the badge degrades to the colloquial-label fallback rather than fabricating.
        configured_ctx=(
            int(d["configured_ctx"])
            if isinstance(d.get("configured_ctx"), (int, float))
            or (isinstance(d.get("configured_ctx"), str) and str(d.get("configured_ctx")).isdigit())
            else None
        ),
        status_note=str(d.get("status_note", "")),
    )
    # 'source' provenance (curated/community/local) — attach without schema change.
    src = d.get("source")
    if src:
        try:
            object.__setattr__(row, "source", str(src))
        except Exception:
            pass
    # Per-slug download artifacts + facets, attached without touching the shared
    # tui-core schema (same pattern as 'source'): weights_companions = the extra
    # weight-variant keys (DFlash draft / mmproj projector) the slug needs beyond
    # its core weights_variant; drafter / vision = display + companion derivation.
    try:
        comp = d.get("weights_companions") or []
        object.__setattr__(row, "weights_companions", [str(c) for c in comp])
        object.__setattr__(row, "drafter", str(d.get("drafter") or ""))
        object.__setattr__(row, "vision", bool(d.get("vision")))
    except Exception:
        pass
    return row


def _extract_first_json(text: str) -> Any:
    """Recover the first balanced JSON value (object or array) from dirty stdout.

    Some contracts may interleave a banner line; this finds the first '{' or '['
    and decodes from there using raw_decode."""
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None
