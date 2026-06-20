"""Headless tests for the CockpitApp (Phase 3 — wired).

Verifies:
  1. The app mounts without error (no TTY, no GPU, no Docker, no live script).
     The data layer is a CockpitData backed by a FakeRunner + fake detect +
     a FakeWriteRunner, so NO subprocess is ever spawned.
  2. All three modes (Run · Operate · Validate) are reachable via digit-key
     bindings 1/2/3; nav nodes exist.  (R1 folded Discover + Serve + Benchmarks
     into a single Run mode; R2a renamed Estate → Operate and moved Doctor into it.)
  3. Run · Catalog populates from real enriched entries (fit glyph, TPS,
     8pk, source) and filters live.
  4. BYO renders the swap_path route from byo_check.
  5. ⏎ on a Run · Catalog row opens the reconcile-gated confirm modal directly.
  6. Operate · Orchestration + Containers + Doctor populate from estate_state /
     doctor().
  7. EVERY write path goes through the reconcile gate, and NO test ever
     executes a live write — the FakeWriteRunner records start_raw calls and
     never spawns a process; an unsafe gate refuses to even reach it.

The whole service layer is dependency-injected; tests never touch the real
RealRunner / SubprocessRunner.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import pytest

from textual.widgets import Button, DataTable, Input, Select, Static, TabbedContent, TabPane, Label
from textual.widgets._footer import FooterKey

from club3090_tui_core.detect import GpuInfo, ServingTarget

from club3090_cockpit.app import (
    CockpitApp,
    CatalogPane,
    CockpitCommands,
    ConfirmActionScreen,
    ExplainScreen,
    FocusableFooter,
    HelpScreen,
    ModeSwitcher,
    ByoPane,
    OperateOrchPane,
    OperateContainersPane,
    ValidateRunPane,
    DoctorPane,
    ValidateEvidencePane,
    EvidenceReportScreen,
    MeasureVsBarScreen,
    ShareBackReportScreen,
    RailStatus,
    _PALETTE_COMMANDS,
    _PALETTE_PRODUCER_ONLY,
)
from club3090_cockpit.data import ContainerInfo, EstateState, ServedProbe
from club3090_cockpit.services import CockpitData, RunResult
from club3090_cockpit.__main__ import (
    resolve_surface,
    config_path,
    load_surface_setting,
    save_surface_setting,
)


FAKE_REPO_ROOT = Path("/tmp/fake-club-3090-test-root")


# ---------------------------------------------------------------------------
# Fake service-layer seams (no subprocess, no GPU, no docker, no TTY)
# ---------------------------------------------------------------------------


class FakeRunner:
    """Canned-output read runner keyed on a substring of the command.

    A WRITE command must NEVER reach here (writes go through the write_runner);
    if one did it would still be a no-op canned response, but the write path is
    separately asserted to use FakeWriteRunner only.
    """

    def __init__(self, responses: Optional[dict[str, RunResult]] = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    async def run(self, cmd, *, cwd, timeout=30.0) -> RunResult:
        self.calls.append(list(cmd))
        joined = " ".join(cmd)
        for token, res in self.responses.items():
            if token in joined:
                return res
        return RunResult(returncode=0, stdout="", stderr="no canned response")


class FakeWriteRunner:
    """Stand-in for the core SubprocessRunner — records start_raw calls but
    NEVER spawns a process.  This is the assertion that no live write happens.

    ``set_callbacks`` mirrors the real runner's signature (the Run pane wires
    on_line/on_event for the live stream) — it just records them, no spawn."""

    def __init__(self):
        self.started: list[dict[str, Any]] = []
        self.callbacks: dict[str, Any] = {}

    def set_callbacks(self, on_event=None, on_line=None, on_complete=None):
        self.callbacks = {"on_event": on_event, "on_line": on_line, "on_complete": on_complete}

    async def start_raw(self, cmd, env, run_type, parser):
        self.started.append({"cmd": cmd, "run_type": run_type})
        return {"mock_state": True, "cmd": cmd}


class FakeGenComposeRunner(FakeRunner):
    """A FakeRunner that, on a generate-compose.sh call, writes a canned compose
    to the ``--out`` path (mocking the real generator's file emit) so the data
    layer's read-back of the temp file returns YAML — no real subprocess.  Other
    commands fall through to the canned-response behaviour."""

    def __init__(self, compose_yaml: str, responses=None):
        # Default to the standard canned read responses so non-generate reads
        # (pull.sh for byo_check, registry-emit for the catalog, etc.) still work.
        super().__init__(responses if responses is not None else fake_responses())
        self._compose_yaml = compose_yaml

    async def run(self, cmd, *, cwd, timeout=30.0) -> RunResult:
        self.calls.append(list(cmd))
        if "generate-compose.sh" in " ".join(cmd):
            # Write the canned compose to the --out path the data layer chose.
            try:
                out_idx = cmd.index("--out")
                Path(cmd[out_idx + 1]).write_text(self._compose_yaml, encoding="utf-8")
            except (ValueError, IndexError, OSError):
                pass
            return RunResult(returncode=0, stdout="", stderr="")
        return await super().run(cmd, cwd=cwd, timeout=timeout)


def ok(stdout: str) -> RunResult:
    return RunResult(returncode=0, stdout=stdout, stderr="")


def make_detect(target: ServingTarget):
    async def _detect() -> ServingTarget:
        return target
    return _detect


def make_probe(probe):
    """A7: a fake live-config probe (returns a canned ServedProbe) so no real
    httpx / docker inspect is touched.  Defaults to an empty probe (the
    'nothing probed' case) when ``probe`` is None."""
    from club3090_cockpit.data import ServedProbe

    async def _probe(_target) -> ServedProbe:
        return probe if probe is not None else ServedProbe()
    return _probe


def make_gpu_info(gpus: list[GpuInfo]):
    async def _gpus() -> list[GpuInfo]:
        return gpus
    return _gpus


# ---------------------------------------------------------------------------
# Canned contract outputs
# ---------------------------------------------------------------------------

REGISTRY_JSON = json.dumps(
    {
        "defaults": [],
        "profiles": {},
        "variants": [
            {
                "slug": "vllm/dual",
                "switch_engine": "vllm",
                "launch_engine": "vllm",
                "compose_dir": "models/qwen3.6-27b/vllm/compose/dual/autoround-int4",
                "file": "fp8-mtp.yml",
                "port": 8010,
                "model": "qwen3.6-27b",
                "engine": "vllm-stable",
                "kvcalc_key": "qwen3.6-27b:dual",
                "container": "vllm_qwen36_27b",
                "compose_path": "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
                "status": "production",
                "ctx_label": "262K",
                "configured_ctx": 262144,
                "status_note": "",
                "source": "curated",
            },
            {
                "slug": "ik-llama/iq4ks-mtp",
                "switch_engine": "ik-llama",
                "launch_engine": "ik-llama",
                "compose_dir": "models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks",
                "file": "mtp.yml",
                "port": 8063,
                "model": "qwen3.6-27b",
                "engine": "ik-llama",
                "kvcalc_key": "SKIP",
                "container": "ik_llama_qwen_single",
                "compose_path": "models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml",
                "status": "production",
                "ctx_label": "200K",
                "configured_ctx": 200000,
                "status_note": "",
                "source": "curated",
            },
        ],
    }
)

FIT_JSON = json.dumps(
    {"verdict": "fits-clean", "vram_est_gb": 19.881, "band_gb": 1.5, "max_ctx": 262144}
)

# kv-calc --fit-all batch (one call enriches the whole catalog's fit column).
FIT_ALL_JSON = json.dumps(
    {
        "card": "rtx-3090",
        "card_vram_gb": 24.0,
        "variants": {
            "vllm/dual": {"verdict": "fits-clean", "vram_est_gb": 19.881, "band_gb": 1.5, "max_ctx": 262144},
            "ik-llama/iq4ks-mtp": {"verdict": "skip"},
        },
    }
)

# REAL switch.sh --explain --json benchmarks shape: [{"row","columns"}].
# TPS lives in columns[4] ("Narr / Code TPS"); 8-pack is scraped from the row.
EXPLAIN_BENCH_ROW = {
    "row": (
        "| `dual.yml` ⭐ | @noonghunna (2× 3090 PCIe) | fp8 | 262K | "
        "**174.0 / 42.0** | — | ~23.6 GB | 2026-05-30 | 8-pack 109/150 |"
    ),
    "columns": [
        "`dual.yml` ⭐",
        "@noonghunna (2× 3090 PCIe)",
        "fp8",
        "262K",
        "**174.0 / 42.0**",
        "—",
        "~23.6 GB",
        "2026-05-30",
        "8-pack 109/150",
    ],
}

EXPLAIN_JSON = json.dumps(
    {
        "slug": "vllm/dual",
        "registry": {"slug": "vllm/dual", "model": "qwen3.6-27b", "engine": "vllm-stable", "status": "production"},
        "card": "rtx-3090",
        "fit": {"verdict": "fits-constrained", "vram_est_gb": 19.881, "band_gb": 1.5, "max_ctx": 262144},
        "benchmarks": [EXPLAIN_BENCH_ROW],
    }
)

EXPLAIN_NO_BENCH_JSON = json.dumps(
    {"slug": "ik-llama/iq4ks-mtp", "registry": {}, "card": "rtx-3090", "fit": {}, "benchmarks": []}
)

SCENES_JSON = json.dumps(
    [
        {"name": "27b", "group": "serving", "description": "Qwen", "services": ["vllm-qwen36-27b-dual"], "ports": ["8010"], "gpus": "both"},
        {"name": "off", "group": "ops", "description": "Stop all", "services": [], "ports": [], "gpus": "none"},
    ]
)

PULL_JSON = json.dumps(
    {
        "arch": "Qwen3_5ForConditionalGeneration",
        "eligible": True,
        "fit_verdict": "fits-clean",
        "note": "reuse compose + swap weights",
        "swap_path": {
            "drop_spec_config": True,
            "quant_match": "int4",
            "route": "C",
            "sibling_slug": "vllm/dual",
        },
    }
)

ESTATE_REPORT_FREE = json.dumps({"active_estate": {"present": False, "instances": []}})
ESTATE_REPORT_BUSY = json.dumps(
    {
        "active_estate": {
            "present": True,
            "instances": [
                {"name": "llama-gpu0", "compose": "llamacpp/default", "gpus": [0], "port": 8010},
            ],
        }
    }
)

HEALTH_SERVING = (
    "club-3090 health check\n"
    "Endpoint: http://localhost:8010\n"
    "  \x1b[0;32m✓\x1b[0m serving\n"
    "  KV pool 61%\n"
    "  spec-dec firing (MTP n=2, 73% accept)\n"
    "  0 recent errors\n"
)
HEALTH_DOWN = (
    "club-3090 health check\n"
    "  ✗ API not reachable at http://localhost:8020 — is the container running?\n"
)

DOCKER_PS_ENGINE = (
    "vllm-qwen36-27b-dual|0.0.0.0:8010->8000/tcp, [::]:8010->8000/tcp\n"
    "open-webui|0.0.0.0:3000->8080/tcp\n"
)
# Two recognized engine containers → a 2-row containers table (open-webui above
# is filtered out as an unrecognized service, so it can't test a row CHANGE).
DOCKER_PS_TWO = (
    "vllm-qwen36-27b-dual|0.0.0.0:8010->8000/tcp\n"
    "vllm-gemma-4-31b-dual|0.0.0.0:8011->8000/tcp\n"
)
DOCKER_PS_EMPTY = ""

# REAL diagnose-estate.sh --json shape (verified live 2026-06-18).
DIAGNOSE_ESTATE_JSON = json.dumps(
    {
        "estate_file": "/home/u/.club3090/estate.yml",
        "live": False,
        "valid": True,
        "summary": "GREEN",
        "checks": {
            "schema": {"ok": True, "schema_version": 1, "instance_count": 2},
            "per_instance_fits": [
                {"name": "llama-gpu0", "valid": True},
                {"name": "llama-gpu1", "valid": True},
            ],
            "cross_checks": {"ok": True, "failures": []},
        },
    }
)

# REAL diagnose-profile.sh text shape (verified live): [N/6] steps + verdict.
DIAGNOSE_PROFILE_TEXT = (
    "Profile triage: vllm/dual\n"
    "=========================\n"
    "[1/6] Compose registry entry exists\n"
    "  ✓ vllm/dual found (model=qwen3.6-27b)\n"
    "\n"
    "[2/6] Cross-references resolve\n"
    "  ✓ all referenced profiles exist\n"
    "\n"
    "[3/6] fits() on canonical scenario\n"
    "  ✓ valid=true; constraints passed: 15/16\n"
    "\n"
    "[4/6] kv-calc projection\n"
    "  ✓ verdict PASS; budget 22.08 GB\n"
    "\n"
    "[5/6] Calibration freshness\n"
    "  ✓ verified; BENCHMARKS.md\n"
    "\n"
    "[6/6] Vendored overlays applied\n"
    "  ✓ VLLM_IMAGE resolves: vllm/vllm-openai:v0.22.0\n"
    "\n"
    "Triage summary: GREEN\n"
)

# REAL gpu-mode power-cap status shape (verified live): banner + per-GPU rows.
# GPU0 capped (limit < default), GPU1 uncapped (limit == default).
POWER_CAP_STATUS = (
    "\x1b[0;36m═══ GPU Power Limits ═══\x1b[0m\n"
    "index, power.limit [W], power.default_limit [W], power.min_limit [W], power.max_limit [W]\n"
    "0, 230.00 W, 370.00 W, 100.00 W, 390.00 W\n"
    "1, 420.00 W, 420.00 W, 100.00 W, 450.00 W\n"
)

# docker top — ps-style table (READ).
DOCKER_TOP = (
    "UID    PID    PPID   C   STIME   TTY   TIME       CMD\n"
    "root   1234   1200   9   10:01   ?     00:12:30   python3 -m vllm.entrypoints.openai.api_server\n"
)

# docker logs --tail N <name> (READ).
DOCKER_LOGS = (
    "INFO 06-18 boot: starting vLLM engine\n"
    "INFO 06-18 ready: serving on :8000\n"
)

# Minimal BENCHMARKS.md the explorer can scrape (model + topo headers + a row).
BENCHMARKS_MD = (
    "# BENCHMARKS\n"
    "\n"
    "## Qwen3.6-27B\n"
    "\n"
    "### Dual-card (2× RTX 3090, TP=2)\n"
    "\n"
    "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP | VRAM | Date | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
    "| `vllm/dual` | @noonghunna | fp8 | 262K | **174.0 / 42.0** | — | 23.6 GB | 2026-05-30 | 8-pack 109/150 |\n"
    "\n"
    "### Single-card (1× RTX 3090)\n"
    "\n"
    "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP | VRAM | Date | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
    "| `llamacpp/mtp` | @somerig | q8 | 262K | **53.0 / 30.0** | — | 15.0 GB | 2026-06-03 | 8-pack 99/150 |\n"
    "\n"
    "## Gemma-4-31B\n"
    "\n"
    "### Dual-card (2× RTX 3090, TP=2)\n"
    "\n"
    "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP | VRAM | Date | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
    "| `vllm/dual` | @anotherrig | int8 | 192K | **36.8 / 20.0** | — | 22.0 GB | 2026-05-31 | 8-pack 118/150 |\n"
)

# Minimal rebench REPORT.md for the evidence-report read.
REBENCH_REPORT_MD = (
    "# Rebench report — vllm-dual-test\n"
    "\n"
    "## TL;DR\n"
    "\n"
    "- TPS narrative **174.0** / code **42.0**.\n"
    "\n"
    "## Meta\n"
    "\n"
    "- **Date:** 2026-06-18\n"
)

# R3b-2 — a rebench tag with a resolvable MODEL (Meta `Served as`) so
# measure_vs_bar can match the curated bar.  The decode numbers (150/40) are
# BELOW the BENCHMARKS_MD bar (174/42) → "under the bar".
MEASURE_REPORT_MD = (
    "# Rebench report — measure-tag\n"
    "\n"
    "## TL;DR\n"
    "\n"
    "- TPS narrative **150.0** / code **40.0**.\n"
    "\n"
    "## Meta\n"
    "\n"
    "- **Served as:** `qwen3.6-27b-autoround` from `/mnt/models/x`\n"
    "- **Model arch:** qwen3_next (Qwen3NextForCausalLM)\n"
    "- **vLLM image:** `vllm/vllm-openai:v0.22.0`\n"
    "- **Container:** `vllm-qwen36-dual`\n"
    "- **Date:** 2026-06-18\n"
    "\n"
    "## Quality — `quality-test.sh --full`\n"
    "\n"
    "| Pack | passed | pct |\n"
    "|---|---|---|\n"
    "| **TOTAL** | **100 / 150** | **67%** |\n"
)

MEASURE_INTERNAL_JSON = json.dumps(
    {
        "bench": {
            "narrative": {"decode_tps_mean": 150.0, "wall_tps_mean": 148.0},
            "code": {"decode_tps_mean": 40.0, "wall_tps_mean": 39.0},
        },
        "quality": {"total_passed": 100, "total_total": 150, "total_pct": 67},
    }
)


def seed_measure_tag(root: Path, tag: str = "measure-tag", *, internal: bool = True) -> None:
    """Seed a rebench tag with a resolvable model + measured numbers for the
    ④ Measure-vs-bar read.  ``internal=False`` omits _internal.json (forces the
    REPORT.md fallback)."""
    (root / "BENCHMARKS.md").write_text(BENCHMARKS_MD, encoding="utf-8")
    d = root / "results" / "rebench" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "REPORT.md").write_text(MEASURE_REPORT_MD, encoding="utf-8")
    if internal:
        (d / "_internal.json").write_text(MEASURE_INTERNAL_JSON, encoding="utf-8")


def fake_responses(**overrides) -> dict[str, RunResult]:
    responses = {
        "registry-emit.sh --json": ok(REGISTRY_JSON),
        # --fit-all MUST precede --fit (the batch cmd contains "kv-calc.py --fit"
        # as a substring; FakeRunner returns the first match).
        "kv-calc.py --fit-all": ok(FIT_ALL_JSON),
        "kv-calc.py --fit": ok(FIT_JSON),
        "--explain vllm/dual --json": ok(EXPLAIN_JSON),
        "--explain ik-llama/iq4ks-mtp --json": ok(EXPLAIN_NO_BENCH_JSON),
        "gpu-mode.sh --list-modes --json": ok(SCENES_JSON),
        "pull.sh": ok(PULL_JSON),
        "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
        "health.sh": ok(HEALTH_SERVING),
        "docker ps": ok(DOCKER_PS_EMPTY),
        # Phase-4 reads:
        "diagnose-estate.sh --json": ok(DIAGNOSE_ESTATE_JSON),
        "diagnose-profile.sh": ok(DIAGNOSE_PROFILE_TEXT),
        "power-cap status": ok(POWER_CAP_STATUS),
        "docker top": ok(DOCKER_TOP),
        "docker logs": ok(DOCKER_LOGS),
    }
    responses.update(overrides)
    return responses


def make_app(
    *,
    responses: Optional[dict[str, RunResult]] = None,
    gpus: Optional[list[GpuInfo]] = None,
    target: Optional[ServingTarget] = None,
    write_runner: Optional[FakeWriteRunner] = None,
    repo_root: Optional[Path] = None,
    surface: str = "consumer",
    runner: Optional[FakeRunner] = None,
    probe_served: Optional[Any] = None,
) -> tuple[CockpitApp, FakeRunner, FakeWriteRunner]:
    """Build a CockpitApp wired to a fully-faked CockpitData.

    Returns (app, read_runner, write_runner) so tests can assert on calls.
    ``repo_root`` overrides the fake root for the filesystem-backed reads
    (benchmarks explorer / evidence list) — seed it with BENCHMARKS.md and a
    results/rebench/ tree for those panes.  ``runner`` injects a custom read
    runner (e.g. FakeGenComposeRunner for the ② Serve generate path).
    """
    root = repo_root or FAKE_REPO_ROOT
    runner = runner or FakeRunner(responses or fake_responses())
    gpus = gpus if gpus is not None else [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
    target = target if target is not None else ServingTarget(gpus=gpus)
    write_runner = write_runner or FakeWriteRunner()
    # A7: always inject a fake probe so the real httpx + docker-inspect probe is
    # never reached in tests.  ``probe_served`` may be a ServedProbe (wrapped) or
    # an async callable (used as-is); default = an empty probe.
    if callable(probe_served):
        probe_fn = probe_served
    else:
        probe_fn = make_probe(probe_served)
    data = CockpitData(
        root,
        runner=runner,
        detect_endpoint_fn=make_detect(target),
        get_gpu_info_fn=make_gpu_info(gpus),
        probe_served_fn=probe_fn,
        write_runner=write_runner,
    )
    app = CockpitApp(repo_root=root, data=data, surface=surface)
    return app, runner, write_runner


def seed_repo(root: Path) -> None:
    """Seed a tmp root with the filesystem state the explorer/evidence read."""
    (root / "BENCHMARKS.md").write_text(BENCHMARKS_MD, encoding="utf-8")
    tag_dir = root / "results" / "rebench" / "vllm-dual-test"
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "REPORT.md").write_text(REBENCH_REPORT_MD, encoding="utf-8")
    (tag_dir / "_internal.json").write_text("{}", encoding="utf-8")


async def _settle(pilot) -> None:
    """Let background workers (catalog / estate) finish."""
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


PANEL_IDS = ["panel-run", "panel-operate", "panel-validate"]


# ===========================================================================
# Mount / navigation
# ===========================================================================


class TestAppMounts:
    @pytest.mark.asyncio
    async def test_app_mounts(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app is not None

    @pytest.mark.asyncio
    async def test_persistent_rail_status_present(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one("#mode-switcher") is not None
            assert app.query_one("#rail-status", RailStatus) is not None

    @pytest.mark.asyncio
    async def test_run_panel_visible_on_start(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert "active" in app.query_one("#panel-run").classes

    @pytest.mark.asyncio
    async def test_other_panels_hidden_on_start(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for pid in ["panel-operate", "panel-validate"]:
                assert "active" not in app.query_one(f"#{pid}").classes

    @pytest.mark.asyncio
    async def test_no_live_write_runner_constructed(self):
        """The injected fake write runner is the one in use — never a real one."""
        app, _, wr = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._data._write_runner is wr
            assert wr.started == []  # nothing executed on mount


class TestModeNavigation:
    @pytest.mark.asyncio
    async def test_key_1_switches_to_run_mode(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # leave Run first
            await pilot.press("1")
            assert "active" in app.query_one("#panel-run").classes
            assert "active" not in app.query_one("#panel-operate").classes

    @pytest.mark.asyncio
    async def test_switch_to_estate_mode(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            assert "active" in app.query_one("#panel-operate").classes

    @pytest.mark.asyncio
    async def test_switch_to_validate_mode(self):
        # Validate is the producer Bring & Validate lane (R3a) — reachable only
        # on the producer surface.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            assert "active" in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_switch_back_to_run(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await pilot.press("1")
            assert "active" in app.query_one("#panel-run").classes
            assert "active" not in app.query_one("#panel-operate").classes

    @pytest.mark.asyncio
    async def test_old_serve_mode_no_longer_exists(self):
        """R1 retired the standalone Serve mode + its panel/action/binding."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            # No serve panel.
            assert not app.query("#panel-serve")
            # No serve mode action.
            assert not hasattr(app, "action_mode_serve")
            # Pressing 1/2/3 only ever yields Run/Operate/Validate (never Serve);
            # there is no 4th mode key.
            await pilot.press("4")  # unbound now — should not switch anything
            await pilot.pause()
            active = [pid for pid in PANEL_IDS if "active" in app.query_one(f"#{pid}").classes]
            assert active == ["panel-run"]

    @pytest.mark.asyncio
    async def test_all_three_modes_cycle(self):
        # All three modes (incl. the producer Validate lane) on the producer surface.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            for key, expected_active in [("1", 0), ("2", 1), ("3", 2), ("1", 0)]:
                await pilot.press(key)
                await pilot.pause()
                active = [pid for pid in PANEL_IDS if "active" in app.query_one(f"#{pid}").classes]
                assert len(active) == 1, f"after {key!r}: {active}"
                assert active[0] == PANEL_IDS[expected_active]


class TestNavNodesExist:
    @pytest.mark.asyncio
    async def test_run_tabs_exist(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#run-tabs", TabbedContent)
            app.query_one("#tab-catalog", TabPane)
            app.query_one("#tab-byo", TabPane)

    @pytest.mark.asyncio
    async def test_operate_tabs_exist(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            tc = app.query_one("#operate-tabs", TabbedContent)
            app.query_one("#tab-orchestration", TabPane)
            app.query_one("#tab-containers", TabPane)
            # R2a moved Doctor into Operate (Orchestration · Containers · Doctor).
            app.query_one("#tab-doctor", TabPane)
            # Only the operate-tabs' OWN tab panes (the Containers pane nests a
            # drill TabbedContent of its own, so filter to the mode-level ids).
            mode_tab_ids = [
                p.id for p in tc.query(TabPane)
                if p.id in {"tab-orchestration", "tab-containers", "tab-doctor"}
            ]
            assert mode_tab_ids == ["tab-orchestration", "tab-containers", "tab-doctor"], mode_tab_ids

    @pytest.mark.asyncio
    async def test_validate_tabs_exist(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            tc = app.query_one("#validate-tabs", TabbedContent)
            # R3b-1: the producer lane is the ordered ①→⑤ pipeline.  ③ Gate keeps
            # the tab id tab-run, ④ Measure keeps tab-evidence (grandfathered);
            # ① Bring / ② Serve / ⑤ Promote are the new stage tabs.
            app.query_one("#tab-bring", TabPane)
            app.query_one("#tab-serve", TabPane)
            app.query_one("#tab-run", TabPane)
            app.query_one("#tab-evidence", TabPane)
            app.query_one("#tab-promote", TabPane)
            pane_ids = [p.id for p in tc.query(TabPane)]
            assert pane_ids == [
                "tab-bring", "tab-serve", "tab-run", "tab-evidence", "tab-promote"
            ], pane_ids

    @pytest.mark.asyncio
    async def test_doctor_renders_under_operate_not_validate(self):
        """R2a structural: the Doctor surface lives under Operate (mode 1,
        tab-doctor) and is GONE from Validate (mode 2)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            # tab-doctor is a child of operate-tabs, NOT validate-tabs.
            operate = app.query_one("#operate-tabs", TabbedContent)
            validate = app.query_one("#validate-tabs", TabbedContent)
            operate_panes = [p.id for p in operate.query(TabPane)]
            validate_panes = [p.id for p in validate.query(TabPane)]
            assert "tab-doctor" in operate_panes, operate_panes
            assert "tab-doctor" not in validate_panes, validate_panes
            # The Doctor pane itself renders under the Operate panel subtree.
            doctor = app.query_one("#doctor-pane", DoctorPane)
            panel_operate = app.query_one("#panel-operate")
            assert doctor in panel_operate.query("#doctor-pane")
            # Activate the Doctor tab in Operate and confirm it switches cleanly.
            await pilot.press("2")
            await _settle(pilot)
            operate.active = "tab-doctor"
            await pilot.pause()
            assert operate.active == "tab-doctor"

    @pytest.mark.asyncio
    async def test_benchmarks_tab_is_gone(self):
        """Fold 3 removed the standalone Validate · Benchmarks tab + its pane."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#tab-benchmarks")
            assert not app.query("#validate-benchmarks-pane")
            assert not app.query("#bmk-table")

    @pytest.mark.asyncio
    async def test_catalog_datatable_has_columns(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            table = app.query_one("#catalog-table", DataTable)
            col_labels = [str(c.label) for c in table.columns.values()]
            # Fold 3: TPS / 8pk columns are explicitly labelled as our-rig.
            for expected in (
                "slug", "engine", "fit", "ctx",
                "TPS (our rig)", "8pk (our rig)", "status", "source",
            ):
                assert expected in col_labels, f"missing {expected!r}: {col_labels}"

    @pytest.mark.asyncio
    async def test_mode_switcher_exists(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#mode-switcher", ModeSwitcher)

    @pytest.mark.asyncio
    async def test_catalog_status_label_exists(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#catalog-status", Label)

    @pytest.mark.asyncio
    async def test_catalog_action_hint_exists(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#catalog-hint", Label)


# ===========================================================================
# Run · Catalog (now wired to real enriched entries)
# ===========================================================================


class TestCatalogWired:
    @pytest.mark.asyncio
    async def test_catalog_populates_from_service(self):
        """On mount the catalog worker pulls enriched entries from CockpitData."""
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            table = app.query_one("#catalog-table", DataTable)
            assert table.row_count == 2  # vllm/dual + ik-llama/iq4ks-mtp
            # registry-emit was actually consulted (real read, faked).
            assert any("registry-emit.sh" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_catalog_shows_fit_glyph_and_tps(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            entry = next(e for e in pane._entries if e.slug == "vllm/dual")
            assert entry.fit.glyph == "●"            # fits-clean
            assert entry.measurement.tps_label == "174/42"
            assert entry.measurement.quality_label == "109/150"

    @pytest.mark.asyncio
    async def test_catalog_ik_llama_fit_is_skip(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            ik = next(e for e in pane._entries if e.slug == "ik-llama/iq4ks-mtp")
            assert ik.fit.verdict == "skip"

    @pytest.mark.asyncio
    async def test_catalog_filter_narrows_rows(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.set_filter("ik-llama")
            table = app.query_one("#catalog-table", DataTable)
            assert table.row_count == 1
            assert pane.selected_entry() is None or pane.selected_entry().slug == "ik-llama/iq4ks-mtp"
            pane.set_filter("")
            assert app.query_one("#catalog-table", DataTable).row_count == 2

    @pytest.mark.asyncio
    async def test_catalog_error_surfaces(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(json.dumps({"variants": []}))})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            status = app.query_one("#catalog-status", Label)
            text = str(status.render()).lower()
            assert "error" in text

    @pytest.mark.asyncio
    async def test_explain_modal_opens_and_populates(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("e")
            await _settle(pilot)
            assert isinstance(app.screen, ExplainScreen)


# ===========================================================================
# Run · BYO (wired to byo_check)
# ===========================================================================


class TestByoWired:
    @pytest.mark.asyncio
    async def test_byo_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#byo-panel", ByoPane)
            app.query_one("#byo-url-input", Input)
            # #6 — the profile-like input is now a registry-derived template Select.
            app.query_one("#byo-profile-input", Select)
            app.query_one("#byo-fit-btn", Button)
            app.query_one("#byo-result-card", Static)

    @pytest.mark.asyncio
    async def test_byo_fit_check_renders_route(self):
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#byo-url-input", Input).value = "org/Model"
            app.query_one("#byo-fit-btn", Button).press()
            await _settle(pilot)
            card = app.query_one("#byo-result-card", Static)
            text = str(card.render())
            assert "Route C" in text or "vllm/dual" in text
            # pull.sh was invoked with --dry-run (never downloads).
            pull = next(c for c in runner.calls if "pull.sh" in " ".join(c))
            assert "--dry-run" in pull


# ===========================================================================
# Serve folded into Run (⏎ on a catalog row → reconcile-gated confirm modal)
# ===========================================================================


class TestServeFoldedIntoRun:
    @pytest.mark.asyncio
    async def test_run_has_transient_boot_pane_hidden_on_start(self):
        """The Serve mode's boot LivePane is re-homed into the Run panel and
        starts hidden (no .serving class) until a serve commits."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            live = app.query_one("#serve-live")  # re-homed core LivePane
            # parented under the Run panel, not a defunct serve panel.
            assert app.query_one("#panel-run") in live.ancestors
            assert "serving" not in live.classes

    @pytest.mark.asyncio
    async def test_enter_on_catalog_row_opens_confirm_directly(self):
        """Fold 2: ⏎ on a Run · Catalog row stages the slug AND opens the
        reconcile-gated ConfirmActionScreen directly — no Serve-mode hop."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app._staged_entry is not None
            assert app._staged_entry.slug == "vllm/dual"
            # Still on the Run panel (no separate Serve panel exists).
            assert "active" in app.query_one("#panel-run").classes

    @pytest.mark.asyncio
    async def test_enter_serve_plan_is_gated_not_force(self):
        """The serve plan ⏎ builds is the GATED switch.sh <slug> (NOT --force),
        and it is the reconcile-gated plan inside the confirm modal."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            plan = app.screen._plan
            assert plan.kind == "serve"
            assert "--force" not in plan.cmd
            assert plan.requires_reconcile is True

    @pytest.mark.asyncio
    async def test_enter_on_byo_tab_does_not_serve(self):
        """⏎ only serves from the Catalog tab; on BYO it no-ops (no confirm)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#run-tabs", TabbedContent).active = "tab-byo"
            await pilot.pause()
            app.action_primary_action()
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)

    @pytest.mark.asyncio
    async def test_serve_dispatch_reveals_boot_pane(self):
        """On a safe gate, confirming the serve dispatches through the gated
        executor and reveals the transient Run boot pane (Fold 2)."""
        app, _, wr = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")            # → confirm modal
            await pilot.pause()
            await app.workers.wait_for_complete()  # reconcile resolves (safe)
            await pilot.pause()
            await pilot.press("enter")            # confirm → dispatch
            await app.workers.wait_for_complete()
            await pilot.pause()
            live = app.query_one("#serve-live")
            assert "serving" in live.classes
            # Gated executor ran (mocked write runner), never a live spawn.
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/switch.sh", "vllm/dual"]


# ===========================================================================
# Operate · Orchestration + Containers (wired to estate_state)
# ===========================================================================


class TestEstateWired:
    @pytest.mark.asyncio
    async def test_estate_orch_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-orch-pane", OperateOrchPane)
            app.query_one("#gpu0-card")
            app.query_one("#gpu1-card")
            app.query_one("#doctor-line")
            app.query_one("#scene-table", DataTable)
            app.query_one("#services-strip")

    @pytest.mark.asyncio
    async def test_estate_scene_table_populates_from_gpu_mode(self):
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            t = app.query_one("#scene-table", DataTable)
            assert t.row_count == 2  # 27b + off (from SCENES_JSON)
            assert any("gpu-mode.sh --list-modes" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_estate_gpu_card_reflects_detect(self):
        gpus = [
            GpuInfo(index=0, mem_used_mib=18 * 1024, mem_total_mib=24 * 1024, utilization=71, power_draw_w=312, power_limit_w=370, temp_c=64),
            GpuInfo(index=1, mem_used_mib=12 * 1024, mem_total_mib=24 * 1024, utilization=45),
        ]
        app, _, _ = make_app(gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "18.0 / 24.0 GiB" in bar
            assert "71%" in bar

    @pytest.mark.asyncio
    async def test_estate_doctor_line_serving(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            line = str(app.query_one("#doctor-line", Static).render())
            assert "serving" in line.lower()

    @pytest.mark.asyncio
    async def test_estate_containers_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-containers-pane", OperateContainersPane)
            app.query_one("#containers-table", DataTable)
            app.query_one("#drill-tabs", TabbedContent)
            app.query_one("#drill-tab-logs", TabPane)
            app.query_one("#drill-tab-stats", TabPane)
            app.query_one("#drill-tab-config", TabPane)

    @pytest.mark.asyncio
    async def test_estate_containers_populate_from_docker_ps(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            names = [c.name for c in pane._containers]
            assert "vllm-qwen36-27b-dual" in names
            assert "open-webui" not in names  # not an engine prefix

    @pytest.mark.asyncio
    async def test_estate_scene_switch_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#scene-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)


# ===========================================================================
# Batch 1 — Operate / BYO UX (bugs + polish from real-rig feedback)
# ===========================================================================


def _seed_services(root: Path, names: list[str]) -> None:
    """Seed services/<name>/docker-compose.yml so _known_service_dirs finds them."""
    for n in names:
        d = root / "services" / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")


class TestBatch1OperateServingPanel:
    """#1 — Operate · Orchestration surfaces WHAT'S SERVING."""

    @pytest.mark.asyncio
    async def test_serving_panel_shows_matched_target(self):
        # A target on port 8010 matches the vllm/dual registry row → matched_slug.
        tgt = ServingTarget(
            url="http://localhost:8010", model="qwen3.6-27b", host_port=8010,
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )
        app, _, _ = make_app(target=tgt)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            assert "Serving" in line
            assert "qwen3.6-27b" in line
            assert "vllm/dual" in line
            assert ":8010" in line

    @pytest.mark.asyncio
    async def test_serving_panel_no_model(self):
        # Default target has no model / no matching port → "no model serving".
        app, _, _ = make_app(target=ServingTarget())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            assert "no model serving" in line.lower()


class TestBatch1KnownServices:
    """#2 — the Containers view shows known-but-stopped supporting services."""

    @pytest.mark.asyncio
    async def test_stopped_service_appears_greyed(self, tmp_path):
        _seed_services(tmp_path, ["comfyui", "litellm"])
        seed_repo(tmp_path)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            names = [c.name for c in pane._containers]
            # The running engine container is still present...
            assert "vllm-qwen36-27b-dual" in names
            # ...alongside the known-but-stopped supporting services.
            assert "comfyui" in names
            assert "litellm" in names
            stopped = [c for c in pane._containers if c.name == "litellm"]
            assert stopped and stopped[0].status == "stopped"
            # Rendered greyed/"stopped" in the table.
            tbl = app.query_one("#containers-table", DataTable)
            blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "stopped" in blob

    @pytest.mark.asyncio
    async def test_running_service_not_duplicated_as_stopped(self, tmp_path):
        # A running comfyui-* container should NOT also show a stopped "comfyui".
        _seed_services(tmp_path, ["comfyui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok("comfyui-server|0.0.0.0:8188->8188/tcp\n")}
        )
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            stopped = [c for c in pane._containers if c.status == "stopped"]
            assert not any(c.name == "comfyui" for c in stopped)

    @pytest.mark.asyncio
    async def test_running_non_gpu_service_not_stopped_actions_live(self, tmp_path):
        """MUST-FIX #2 (a): a RUNNING non-GPU supporting service (litellm — NOT
        in _GPU_SERVICE_NAMES, so dropped from the GPU stack-container list) must
        NOT be rendered "stopped" and its container actions must NOT be
        suppressed.  Before the fix, the de-dup keyed off the GPU-filtered list →
        litellm never appeared as running → it was appended as a greyed,
        read-only "stopped" row even while live."""
        _seed_services(tmp_path, ["litellm"])
        seed_repo(tmp_path)
        # docker ps reports litellm RUNNING (its container_name == "litellm").
        responses = fake_responses(
            **{"docker ps": ok("litellm|0.0.0.0:4000->4000/tcp\n")}
        )
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            litellm = next((c for c in pane._containers if c.name == "litellm"), None)
            assert litellm is not None, "litellm service not surfaced at all"
            # NOT stopped (it's live) and NOT in the stopped set.
            assert litellm.status != "stopped"
            assert litellm not in [c for c in pane._containers if c.status == "stopped"]
            # Actions are NOT suppressed — the stopped-service guard is False.
            assert app._is_stopped_service(litellm) is False
            # A write op (restart) routes to the confirm gate, not the
            # "<name> is not running" warning short-circuit.
            tbl = pane.query_one("#containers-table", DataTable)
            idx = pane._containers.index(litellm)
            tbl.move_cursor(row=idx)
            await pilot.press("s")  # restart
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["docker", "restart", "litellm"]

    @pytest.mark.asyncio
    async def test_separator_mismatch_service_matched_as_running(self, tmp_path):
        """MUST-FIX #2 (b): a service dir ``open-webui`` whose running container
        is named ``openwebui`` (separator mismatch) must NORMALIZE-match → shown
        running, not stopped.  A bare substring match would miss this."""
        _seed_services(tmp_path, ["open-webui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok("openwebui|0.0.0.0:3000->8080/tcp\n")}
        )
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            stopped = [c for c in pane._containers if c.status == "stopped"]
            assert not any(c.name == "open-webui" for c in stopped), (
                "open-webui dir wrongly marked stopped despite running 'openwebui'"
            )
            owui = next((c for c in pane._containers if c.name == "open-webui"), None)
            assert owui is not None and owui.status != "stopped"
            assert app._is_stopped_service(owui) is False


class TestBatch1PowerCapCard:
    """#10 — GPU cards show power+cap; a cap write re-polls the estate."""

    @pytest.mark.asyncio
    async def test_gpu_card_shows_power_and_cap(self):
        gpus = [
            GpuInfo(index=0, mem_used_mib=18 * 1024, mem_total_mib=24 * 1024,
                    utilization=71, power_draw_w=312, power_limit_w=370, temp_c=64),
            GpuInfo(index=1, mem_used_mib=12 * 1024, mem_total_mib=24 * 1024, utilization=45),
        ]
        # POWER_CAP_STATUS has GPU0 capped at 230 (default 370).
        app, _, _ = make_app(gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "power:" in bar
            assert "312 / 370 W" in bar
            assert "cap 230W" in bar  # capped card annotates the active cap

    @pytest.mark.asyncio
    async def test_cap_write_repolls_estate(self, monkeypatch):
        wr = FakeWriteRunner()
        app, runner, _ = make_app(write_runner=wr)
        # Count load_estate invocations directly (it's a @work worker — wrap the
        # underlying coroutine so the re-poll is observable).
        calls = {"n": 0}
        import club3090_cockpit.app as appmod
        orig = appmod.CockpitApp.load_estate

        def counting(self):
            calls["n"] += 1
            return orig(self)

        monkeypatch.setattr(appmod.CockpitApp, "load_estate", counting)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # entering Operate triggers a load_estate
            await _settle(pilot)
            before = calls["n"]
            plan = app._data.power_cap_set("off")
            assert plan.kind == "power_cap"
            app.dispatch_action(plan)
            await _settle(pilot)
            # The cap WRITE re-polled the estate (load_estate fired again).
            assert calls["n"] > before
            # The write still went through the (mocked) write runner — gate intact.
            assert len(wr.started) == 1

    @pytest.mark.asyncio
    async def test_cap_toggle_is_confirm_gated(self):
        # [c] toggle routes through the confirm modal (gate NOT bypassed).
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await _settle(pilot)
            app.action_power_cap_toggle()
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.kind == "power_cap"


class TestBatch1ByoPlaceholder:
    """#7 — BYO placeholder names a HuggingFace model slug."""

    @pytest.mark.asyncio
    async def test_byo_placeholder_says_huggingface_slug(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # BYO is a Run sub-tab (consumer surface).
            app.query_one("#run-tabs", TabbedContent).active = "tab-byo"
            await _settle(pilot)
            ph = app.query_one("#byo-url-input", Input).placeholder
            assert "HuggingFace model slug" in ph
            assert "unsloth/Qwen3-27B-abliterated-GGUF" in ph


# ===========================================================================
# THE RECONCILE GATE — every write path goes through it
# ===========================================================================


class TestEveryWriteGoesThroughReconcile:
    @pytest.mark.asyncio
    async def test_confirm_modal_runs_reconcile_on_mount(self):
        """The confirm modal re-runs the fresh reconcile gate before enabling
        any commit button.  On a free rig the gate is clear → Confirm enabled."""
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            assert screen._reconcile is not None
            assert screen._reconcile.safe is True
            ok_btn = screen.query_one("#confirm-ok-btn", Button)
            assert ok_btn.disabled is False

    @pytest.mark.asyncio
    async def test_unsafe_gate_disables_confirm_enables_force(self):
        """When a container is running, the gate is unsafe → Confirm disabled,
        Force enabled.  The teardown list is surfaced."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            screen = app.screen
            assert screen._reconcile.safe is False
            assert screen.query_one("#confirm-ok-btn", Button).disabled is True
            assert screen.query_one("#confirm-force-btn", Button).disabled is False
            body = str(screen.query_one("#confirm-body", Static).render())
            assert "tear down" in body.lower() or "collide" in body.lower()

    @pytest.mark.asyncio
    async def test_confirm_dispatches_through_gated_executor_safe(self):
        """Confirm on a safe gate → execute_action reaches the (mocked) write
        runner exactly once.  NO live process is ever spawned."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            screen = app.screen
            screen.query_one("#confirm-ok-btn", Button).press()
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/switch.sh", "vllm/dual"]

    @pytest.mark.asyncio
    async def test_dispatch_refuses_when_unsafe_no_force(self):
        """If the gate is unsafe and the plan is not forced, the executor refuses
        and the write runner is NEVER reached."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")  # not forced
            app.dispatch_action(plan)
            await _settle(pilot)
            assert wr.started == []  # refused at the gate

    @pytest.mark.asyncio
    async def test_force_override_proceeds_despite_unsafe(self):
        """A forced plan (reason surfaced) proceeds even when the gate is unsafe,
        still via the MOCKED write runner — never a live process."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual", force=True, force_reason="user accepted teardown")
            app.dispatch_action(plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "--force" in wr.started[0]["cmd"]

    @pytest.mark.asyncio
    async def test_force_button_reissues_forced_plan(self):
        """Pressing Force in the confirm modal re-issues the plan as forced
        (--force inserted) and dispatches it through the mocked runner."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")  # un-forced
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            screen = app.screen
            screen.query_one("#confirm-force-btn", Button).press()
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "--force" in wr.started[0]["cmd"]

    @pytest.mark.asyncio
    async def test_scene_switch_dispatch_through_gate(self):
        """Scene-switch is gated too — a free rig dispatches gpu-mode <mode>."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.scene_switch("27b")
            assert plan.requires_reconcile is True
            app.dispatch_action(plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/gpu-mode.sh", "27b"]


class TestAllPanesWired:
    """Phase-3 acceptance (§9.2 all_panes_wired): every advertised UI hint is
    backed by a real handler routed through the SAME gate."""

    @pytest.mark.asyncio
    async def test_container_restart_opens_confirm_modal(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            # Restart is a Containers-tab action — activate it. (R2a gates the
            # container write to tab-containers so [s] can't restart from the
            # read-only Doctor tab or from Orchestration.)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await pilot.pause()
            app.query_one("#operate-containers-pane", OperateContainersPane).query_one(
                "#containers-table", DataTable
            ).move_cursor(row=0)
            await pilot.press("s")  # restart
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["docker", "restart", "vllm-qwen36-27b-dual"]

    @pytest.mark.asyncio
    async def test_container_stop_dispatches_through_gate(self):
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_EMPTY)})
        app, _, _ = make_app(responses=responses, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            plan = app._data.container_action("vllm-x", "stop")
            assert plan.requires_reconcile is True
            app.dispatch_action(plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["docker", "stop", "vllm-x"]

    @pytest.mark.asyncio
    async def test_container_logs_stream_into_live_pane(self):
        responses = fake_responses(
            **{"docker ps": ok(DOCKER_PS_ENGINE), "docker logs": ok("boot line A\nboot line B\n")}
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-containers-pane", OperateContainersPane).query_one(
                "#containers-table", DataTable
            ).move_cursor(row=0)
            await pilot.press("l")  # logs (READ)
            await _settle(pilot)
            # No modal — logs is a read, not a gated write.
            assert not isinstance(app.screen, ConfirmActionScreen)

    @pytest.mark.asyncio
    async def test_estate_off_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("o")  # stop all
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert "down" in app.screen._plan.cmd

    @pytest.mark.asyncio
    async def test_estate_off_dispatches_through_gate(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.estate_down()
            app.dispatch_action(plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "down" in wr.started[0]["cmd"]

    @pytest.mark.asyncio
    async def test_set_default_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("d")  # set-default
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/switch.sh", "--set-default", "vllm/dual"]

    @pytest.mark.asyncio
    async def test_clear_default_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("D")  # clear-default
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/switch.sh", "--clear-default", "qwen3.6-27b"]

    @pytest.mark.asyncio
    async def test_set_default_dispatches_through_gate(self):
        """set_default routes through the same gate; requires_reconcile=False so
        it dispatches straight to the mocked write runner."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.set_default("vllm/dual")
            app.dispatch_action(plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/switch.sh", "--set-default", "vllm/dual"]


class TestNoLiveWriteEverExecuted:
    """Belt-and-suspenders: across the whole app surface, no FakeWriteRunner
    call is a real process and the read FakeRunner never receives a write."""

    @pytest.mark.asyncio
    async def test_full_serve_flow_only_touches_fakes(self):
        wr = FakeWriteRunner()
        app, runner, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # ⏎ on a Run · Catalog row → reconcile-gated confirm modal directly.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            # commit through the (mocked) gated executor.
            screen.query_one("#confirm-ok-btn", Button).press()
            await _settle(pilot)
            # The only write went to the fake; no switch.sh appears in READ calls.
            assert all("scripts/switch.sh vllm/dual" not in " ".join(c) for c in runner.calls)
            assert len(wr.started) == 1


# ===========================================================================
# Validate panes (Phase 4 — illustrative; nodes still present)
# ===========================================================================


class TestValidatePanes:
    @pytest.mark.asyncio
    async def test_validate_run_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#validate-run-pane", ValidateRunPane)
            t = app.query_one("#run-ladder-table", DataTable)
            # 6 ladder steps + 3 extras = 9 launchable kinds.
            assert t.row_count == 9
            app.query_one("#run-gotchas")   # §3.5 tune gotchas surfaced inline
            app.query_one("#run-output")     # core LivePane for streamed output

    @pytest.mark.asyncio
    async def test_operate_doctor_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#doctor-pane", DoctorPane)
            app.query_one("#doctor-card-health")
            app.query_one("#doctor-card-estate")
            app.query_one("#doctor-card-profile")

    @pytest.mark.asyncio
    async def test_operate_doctor_health_line_goes_live_on_estate_poll(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # Operate estate poll feeds the doctor pane too
            await _settle(pilot)
            body = str(app.query_one("#doctor-health-body", Static).render())
            assert "serving" in body.lower()

    @pytest.mark.asyncio
    async def test_benchmarks_pane_and_table_are_gone(self):
        """Fold 3: the standalone Validate · Benchmarks pane/table no longer
        exist anywhere in the widget tree."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#validate-benchmarks-pane")
            assert not app.query("#bmk-table")
            assert not app.query("#tab-benchmarks")

    @pytest.mark.asyncio
    async def test_validate_evidence_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#validate-evidence-pane", ValidateEvidencePane)
            app.query_one("#evidence-table", DataTable)


# ===========================================================================
# Primary action does not crash in any mode
# ===========================================================================


class TestPrimaryActionSafe:
    @pytest.mark.asyncio
    async def test_enter_in_discover_with_no_selection_is_safe(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # filter to nothing selectable then press enter — must not crash
            await pilot.press("enter")

    @pytest.mark.asyncio
    async def test_enter_in_validate_is_safe(self):
        # Validate (mode 2) is the producer lane (R3a).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await pilot.press("enter")


# ===========================================================================
# Operate · Doctor (wired to doctor() — health + estate + profile cards)
# R2a moved Doctor from Validate into Operate (mode 1); load_doctor fires on
# Operate entry, alongside the estate poll.
# ===========================================================================


class TestOperateDoctorWired:
    @pytest.mark.asyncio
    async def test_doctor_cards_populate_from_doctor_read(self):
        """Entering Operate runs the full Doctor read → estate + profile cards
        fill from diagnose-estate.sh --json + diagnose-profile.sh (text)."""
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            estate = str(app.query_one("#doctor-estate-body", Static).render())
            assert "GREEN" in estate and "2/2" in estate  # 2/2 instances fit
            assert any("diagnose-estate.sh --json" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_doctor_profile_triage_after_estate_target(self):
        """When a running engine is detected (matched slug), Doctor triages it
        via diagnose-profile.sh and renders the 6 steps + verdict."""
        # A detect target on port 8010 matches vllm/dual in the registry → the
        # Operate estate poll captures the slug, and load_doctor (also fired on
        # Operate entry) triages it — both happen in the SAME mode now (R2a).
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses, gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")            # Operate: estate poll captures slug + doctor read
            await _settle(pilot)
            assert app._target_slug == "vllm/dual"
            profile = str(app.query_one("#doctor-profile-body", Static).render())
            assert "GREEN" in profile
            assert any("diagnose-profile.sh" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_s_key_on_doctor_tab_does_not_restart(self):
        """R2a put the READ-ONLY Doctor surface in Operate (mode 1).  [s] — whose
        binding spans modes 1+2 with no sub-tab constraint — must NOT fire a
        `docker restart` on the Doctor tab, even with a container selected in the
        (hidden) Containers tab.  The container write is gated to tab-containers."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")            # Operate
            await _settle(pilot)
            # Select a container, so the ONLY thing preventing a restart is the
            # sub-tab guard (not an empty selection)…
            app.query_one("#operate-containers-pane", OperateContainersPane).query_one(
                "#containers-table", DataTable
            ).move_cursor(row=0)
            # …then move to the read-only Doctor tab and press [s].
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)  # Doctor is read-only


# ===========================================================================
# Cross-rig benchmarks folded into the explain drill-down (Fold 3)
# ===========================================================================


class TestCrossRigBenchmarksViaExplain:
    """Fold 3: the cross-rig benchmark rows the retired Validate · Benchmarks
    tab used to show are now reachable per-slug via the explain (e) drill-down."""

    @pytest.mark.asyncio
    async def test_explain_surfaces_cross_rig_rows(self, tmp_path):
        seed_repo(tmp_path)  # writes a BENCHMARKS.md (cross-rig scrape) row
        app, _, _ = make_app(repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # vllm/dual is the qwen3.6-27b row the BENCHMARKS.md scrape covers.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("e")
            await _settle(pilot)
            assert isinstance(app.screen, ExplainScreen)
            body = str(app.screen.query_one("#explain-body", Static).render())
            # The cross-rig section is present and labelled (not our-rig).
            assert "Cross-rig benchmarks" in body
            assert "109/150" in body  # the 8-pack from the BENCHMARKS.md scrape

    @pytest.mark.asyncio
    async def test_explain_cross_rig_only_matches_this_slug_model(self, tmp_path):
        """The folded-in rows are scoped to the slug's model — with BOTH qwen and
        gemma rows seeded, explaining the gemma slug surfaces ONLY gemma rows; the
        qwen rows must not leak in (data not silently mixed across models)."""
        seed_repo(tmp_path)  # BENCHMARKS_MD has both qwen3.6-27b AND gemma-4-31b rows
        app, _, _ = make_app(repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.push_screen(
                ExplainScreen("vllm/dual", model="gemma-4-31b", engine="vllm-stable")
            )
            await _settle(pilot)
            screen = app.screen
            # Assert on the FILTERED cross-rig list, not the rendered body — the
            # body also carries the our-rig detail for the slug. There IS a gemma
            # row in the scrape, so this is not vacuous; the qwen rows (also in
            # the scrape) must be filtered out → exactly the gemma row remains.
            assert [r.model for r in screen._cross_rig] == ["gemma-4-31b"]  # no qwen leak

    @pytest.mark.asyncio
    async def test_explain_llama_cpp_family_surfaces_cross_rig(self, tmp_path):
        """Regression (verify-cockpit-r1 HIGH): the registry emits engine
        'llama-cpp-local' for EVERY llamacpp/* and ik-llama/* slug, while the
        BENCHMARKS.md scrape labels those rows 'llamacpp'/'ik-llama'.  The old
        loose-substring matcher matched NEITHER direction, silently dropping the
        whole llama.cpp family's cross-rig data (green-but-broken — the fixture
        had used 'ik-llama').  The family-canon matcher must surface it."""
        seed_repo(tmp_path)  # BENCHMARKS_MD includes a `llamacpp/mtp` qwen row
        app, _, _ = make_app(repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # The slug is only used for the our-rig detail; the cross-rig fold
            # filters on the passed (model, engine) — here the real registry
            # engine label for a llama.cpp slug.
            app.push_screen(
                ExplainScreen("vllm/dual", model="qwen3.6-27b", engine="llama-cpp-local")
            )
            await _settle(pilot)
            screen = app.screen
            assert screen._cross_rig, "llama.cpp-family cross-rig rows were dropped"
            assert all(r.model == "qwen3.6-27b" for r in screen._cross_rig)
            # The `llamacpp/mtp` row IS surfaced; the vllm/dual qwen row (a
            # DIFFERENT engine family) is NOT — proving family scoping, not just
            # model. (Asserted on the filtered list, not the body, which also
            # carries the slug's own our-rig detail.)
            assert any(r.engine == "llamacpp" for r in screen._cross_rig)
            assert all(r.engine != "vllm" for r in screen._cross_rig)

    def test_bench_row_matcher_engine_family(self):
        """The (model, engine) matcher: model exact, engine by FAMILY.

        Registry labels ('vllm-stable', 'llama-cpp-local', 'beellama-local') and
        BENCHMARKS.md scrape labels ('vllm', 'llamacpp', 'ik-llama', 'beellama')
        live in different spaces; both reduce to a coarse family before
        comparing.  The old loose-substring test silently dropped the entire
        llama.cpp family ('llama-cpp-local' substring-matches neither 'llamacpp'
        nor 'ik-llama')."""
        from club3090_cockpit.app import _bench_row_matches
        from club3090_cockpit.data import BenchRow

        m = "qwen3.6-27b"
        def row(engine):
            return BenchRow(model=m, engine=engine, topology="single")

        # vLLM family (registry 'vllm-stable' ↔ scrape 'vllm').
        assert _bench_row_matches(row("vllm"), m, "vllm-stable") is True
        # llama.cpp family — BOTH scrape labels vs the single registry label
        # (the exact regression the family-canon fix addresses).
        assert _bench_row_matches(row("llamacpp"), m, "llama-cpp-local") is True
        assert _bench_row_matches(row("ik-llama"), m, "llama-cpp-local") is True
        # beellama family.
        assert _bench_row_matches(row("beellama"), m, "beellama-local") is True
        # cross-family must NOT match.
        assert _bench_row_matches(row("vllm"), m, "llama-cpp-local") is False
        assert _bench_row_matches(row("llamacpp"), m, "vllm-stable") is False
        # wrong model → no match regardless of engine.
        assert _bench_row_matches(row("vllm"), "gemma-4-31b", "vllm-stable") is False
        # blank engine on either side matches on model alone (bare *.yml cell).
        assert _bench_row_matches(row(""), m, "vllm-stable") is True
        assert _bench_row_matches(row("vllm"), m, "") is True


# ===========================================================================
# Validate · Evidence (wired to evidence_list / evidence_report / submit)
# ===========================================================================


class TestValidateEvidenceWired:
    @pytest.mark.asyncio
    async def test_evidence_list_populates_from_rebench_dir(self, tmp_path):
        # Validate (mode 2) is the producer lane (R3a) — enter on producer.
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            t = app.query_one("#evidence-table", DataTable)
            assert t.row_count == 1
            pane = app.query_one("#validate-evidence-pane", ValidateEvidencePane)
            assert pane._tags[0].tag == "vllm-dual-test"
            assert pane._tags[0].date == "2026-06-18"

    @pytest.mark.asyncio
    async def test_evidence_enter_opens_report_modal(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            app.query_one("#evidence-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, EvidenceReportScreen)
            body = str(app.screen.query_one("#evidence-report-body", Static).render())
            assert "Rebench report" in body

    @pytest.mark.asyncio
    async def test_evidence_submit_opens_gated_confirm_never_auto(self, tmp_path):
        """[s] in Evidence stages the OUTWARD submit behind a confirm modal —
        the network is NEVER auto-fired."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(repo_root=tmp_path, write_runner=wr, surface="producer")
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            app.query_one("#evidence-table", DataTable).move_cursor(row=0)
            await pilot.press("s")  # submit
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.network is True
            assert "--auto-submit" in app.screen._plan.cmd
            assert wr.started == []  # nothing fired — only the modal opened

    @pytest.mark.asyncio
    async def test_evidence_submit_dispatches_through_gate(self, tmp_path):
        """Confirming the submit reaches ONLY the mocked write runner (network
        is never touched live — conftest blocks the real spawn)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(repo_root=tmp_path, write_runner=wr)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.submit_bench("vllm-dual-test")
            app.dispatch_action(plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "--auto-submit" in wr.started[0]["cmd"]


# ===========================================================================
# Phase R / R3b-2 — ④ Measure-vs-curated-bar (READ · producer-only)
#
# The producer's MEASURED numbers (rebench tag) compared apples-to-apples to the
# curated catalog's published bar for the same class.  A READ: parses the tag's
# _internal.json / REPORT.md + the benchmarks explorer; NO GPU / network / write.
# The cockpit FLAGS the protocol (matched power? same harness? same prompts?) —
# it does NOT fabricate a "catalog-grade" verdict.
# ===========================================================================


class TestMeasureVsBarData:
    """The CockpitData.measure_vs_bar READ — parses + matches + verdicts honestly."""

    @pytest.mark.asyncio
    async def test_measure_vs_bar_parses_and_matches_curated_bar(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_measure_tag(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vsbar = await app._data.measure_vs_bar("measure-tag")
            # Measured numbers parsed from _internal.json (authoritative source).
            assert vsbar.measured.source == "_internal.json"
            assert vsbar.measured.narr_tps == 150.0
            assert vsbar.measured.code_tps == 40.0
            assert vsbar.measured.quality_8pk == "100/150"
            assert vsbar.measured.model == "qwen3.6-27b-autoround"
            # Matched the curated bar (qwen3.6-27b vllm/dual @ 174/42) by
            # canon-model AND the run's engine-family (vllm, resolved from the
            # REPORT.md Meta Container/vLLM-image) — NOT the single-card
            # llamacpp 53/30 row that shares the model.
            assert vsbar.bar is not None
            assert vsbar.bar.model == "qwen3.6-27b"
            assert vsbar.run_engine == "vllm"
            assert vsbar.engine_resolved is True
            assert vsbar.bar.engine == "vllm"        # the engine-matched bar
            assert vsbar.bar.topology == "dual"
            assert vsbar.bar_source == "benchmarks.md"
            # Deltas: 150−174 = −24, 40−42 = −2.
            assert vsbar.narr_tps_delta == -24.0
            assert vsbar.code_tps_delta == -2.0
            # 150/174 = 0.86 < 0.90 → under the bar.
            assert vsbar.verdict == "under the bar"
            # Protocol caveats FLAG what the cockpit can't verify (power/harness/
            # prompts) + disclose the bar source — never a fabricated grade.
            joined = " ".join(vsbar.protocol_caveats).lower()
            assert "matched power" in joined
            assert "harness" in joined
            assert "prompts" in joined
            assert "not a catalog-grade" in joined

    @pytest.mark.asyncio
    async def test_measure_vs_bar_report_md_fallback(self, tmp_path):
        """With no _internal.json the numbers come from REPORT.md (less precise)."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_measure_tag(tmp_path, internal=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vsbar = await app._data.measure_vs_bar("measure-tag")
            assert vsbar.measured.source == "REPORT.md"
            assert vsbar.measured.narr_tps == 150.0
            assert vsbar.measured.code_tps == 40.0
            assert vsbar.measured.quality_8pk == "100/150"
            assert vsbar.bar is not None
            assert any("REPORT.md" in c for c in vsbar.protocol_caveats)

    @pytest.mark.asyncio
    async def test_measure_vs_bar_insufficient_data_is_honest(self, tmp_path):
        """Unparseable numbers / no resolvable model → 'insufficient data', no
        fabricated bar match, and an honest caveat (no false catalog-grade)."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        # REPORT.md with NO TL;DR TPS, NO model — and no _internal.json.
        d = tmp_path / "results" / "rebench" / "blank-tag"
        d.mkdir(parents=True)
        (d / "REPORT.md").write_text("# Rebench report\n\n## Meta\n\n- **Date:** 2026-01-01\n")
        (tmp_path / "BENCHMARKS.md").write_text(BENCHMARKS_MD)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vsbar = await app._data.measure_vs_bar("blank-tag")
            assert vsbar.verdict == "insufficient data"
            assert vsbar.bar is None
            assert not vsbar.measured.has_any
            # Honest: it says it couldn't resolve / match — NOT a fabricated grade.
            joined = " ".join(vsbar.protocol_caveats).lower()
            assert "could not resolve" in joined or "no curated catalog bar" in joined

    @pytest.mark.asyncio
    async def test_measure_vs_bar_missing_tag_dir_errors(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        (tmp_path / "BENCHMARKS.md").write_text(BENCHMARKS_MD)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vsbar = await app._data.measure_vs_bar("does-not-exist")
            assert vsbar.error
            assert "no run dir" in vsbar.error

    @pytest.mark.asyncio
    async def test_measure_vs_bar_engine_match_is_order_independent(self, tmp_path):
        """Reorder the BENCHMARKS.md sections so the single-card llamacpp 53/30
        row comes FIRST: the vLLM-dual run must STILL match the vLLM 174/42 bar
        (engine-discriminated, not document-order) — no verdict flip, no
        fabricated '+97 narr' delta."""
        # llamacpp single-card section first, vLLM-dual second (reversed order).
        reordered = (
            "# BENCHMARKS\n\n"
            "## Qwen3.6-27B\n\n"
            "### Single-card (1× RTX 3090)\n\n"
            "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP | VRAM | Date | Notes |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
            "| `llamacpp/mtp` | @somerig | q8 | 262K | **53.0 / 30.0** | — | 15.0 GB | 2026-06-03 | 8-pack 99/150 |\n\n"
            "### Dual-card (2× RTX 3090, TP=2)\n\n"
            "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP | VRAM | Date | Notes |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
            "| `vllm/dual` | @noonghunna | fp8 | 262K | **174.0 / 42.0** | — | 23.6 GB | 2026-05-30 | 8-pack 109/150 |\n"
        )
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_measure_tag(tmp_path)
        (tmp_path / "BENCHMARKS.md").write_text(reordered, encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vsbar = await app._data.measure_vs_bar("measure-tag")
            # Engine-matched the vLLM-dual bar despite the llamacpp row being first.
            assert vsbar.bar is not None
            assert vsbar.bar.engine == "vllm"
            assert vsbar.bar.topology == "dual"
            assert vsbar.bar.narr_tps == 174.0
            # Honest delta + verdict — NOT the +97 narr / 'within tolerance' flip
            # the llamacpp 53/30 bar would have fabricated.
            assert vsbar.narr_tps_delta == -24.0
            assert vsbar.verdict == "under the bar"

    @pytest.mark.asyncio
    async def test_measure_vs_bar_corpus_suppresses_narrative_delta(self, tmp_path):
        """A corpus bar's narr_tps is WALL, not decode — the narrative-TPS delta
        is SUPPRESSED (no fabricated green +N) + a wall-vs-decode caveat is added;
        the decode (code) delta is kept (both decode)."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_measure_tag(tmp_path)
        # Seed a #249 corpus record: wall_tps 200 (would fake a +50 narr delta vs
        # measured narrative-decode 150) + canonical-short decode 45.
        recdir = tmp_path / "results" / "measurement-records"
        recdir.mkdir(parents=True, exist_ok=True)
        rec = {
            "model_slug": "qwen3.6-27b",
            "engine_id": "vllm-stable",
            "topology": "dual",
            "max_model_len": 262144,
            "measured_extensions": {
                "decode_tps_by_ctx": {"canonical-short": 45.0},
                "wall_tps": 200.0,
            },
            "provenance": {"last_confirmed": "2026-06-10"},
            "_tag": "corpus-run",
        }
        (recdir / "qwen.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vsbar = await app._data.measure_vs_bar("measure-tag")
            assert vsbar.bar is not None
            assert vsbar.bar_source == "corpus"
            # Narrative delta SUPPRESSED (not 150−200=−50, not a +N) ...
            assert vsbar.narr_tps_delta is None
            # ... decode delta kept: 40 − 45 = −5.
            assert vsbar.code_tps_delta == -5.0
            joined = " ".join(vsbar.protocol_caveats).lower()
            assert "wall" in joined and "decode" in joined

    @pytest.mark.asyncio
    async def test_measure_vs_bar_quality_only_below_bar_is_honest(self, tmp_path):
        """When only quality (no comparable TPS) is present, the 8-pack pass
        counts ARE compared — 50/150 measured vs 140/150 bar is materially UNDER
        the bar, NOT 'within tolerance'."""
        from club3090_cockpit.data import MeasureVsBar, MeasuredNumbers, BenchRow, _measure_verdict

        vsbar = MeasureVsBar(
            tag="q",
            measured=MeasuredNumbers(quality_8pk="50/150", model="qwen3.6-27b"),
            bar=BenchRow(model="qwen3.6-27b", quality_8pk="140/150", source="benchmarks.md"),
            bar_source="benchmarks.md",
        )
        assert _measure_verdict(vsbar) == "under the bar"
        # And a near-match (130/150 vs 140/150 = 0.93 ≥ 0.90) is within tolerance.
        vsbar.measured.quality_8pk = "130/150"
        assert _measure_verdict(vsbar) == "within tolerance of the bar"


class TestMeasureVsBarUI:
    """The ④ Measure 'vs catalog bar' view (producer-only key [m])."""

    @pytest.mark.asyncio
    async def test_measure_vs_bar_action_is_producer_only(self):
        """[m] measure_vs_bar is gated off on the consumer surface."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert "measure_vs_bar" in app._PRODUCER_ONLY
            assert app.check_action("measure_vs_bar", ()) is False

    @pytest.mark.asyncio
    async def test_measure_vs_bar_enabled_on_producer_measure_tab(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            assert app.check_action("measure_vs_bar", ()) is True

    @pytest.mark.asyncio
    async def test_m_key_opens_vs_bar_view_with_verdict_and_caveats(self, tmp_path):
        """[m] on a selected ④ Measure tag opens the vs-bar modal rendering the
        measured-vs-bar comparison + honest verdict + protocol caveats."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_measure_tag(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            app.query_one("#evidence-table", DataTable).move_cursor(row=0)
            await pilot.press("m")
            await _settle(pilot)
            assert isinstance(app.screen, MeasureVsBarScreen)
            body = str(app.screen.query_one("#vsbar-body", Static).render())
            # The verdict + the side-by-side measured/bar + a protocol caveat.
            assert "under the bar" in body
            assert "Catalog bar" in body
            assert "150" in body and "174" in body
            assert "cannot verify" in body.lower()

    @pytest.mark.asyncio
    async def test_m_key_no_selection_notifies(self, tmp_path):
        """No selectable tag → [m] notifies, no modal."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        (tmp_path / "BENCHMARKS.md").write_text(BENCHMARKS_MD)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            await pilot.press("m")
            await pilot.pause()
            # No tags seeded → the "—" placeholder row → no real selection → no modal.
            assert not isinstance(app.screen, MeasureVsBarScreen)


# ===========================================================================
# Phase R / R3b-2 — producer FULL validation battery (report.sh --full)
#
# A PRODUCER-lane action ([F] on ③ Gate): the ~43-min verify+stress+soak+bench+
# agentic battery against the SERVING model.  Confirm-gated (heavy + needs a
# model serving), bg-streamed into the Gate LivePane, uses the serving model
# (claims NO GPU), NEVER auto-fired, MOCK-ONLY in tests (conftest blocks the
# real spawn — the FakeWriteRunner records start_raw without spawning).
# ===========================================================================


class TestFullValidationReport:
    @pytest.mark.asyncio
    async def test_full_report_plan_is_confirm_gated_no_reconcile(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.full_validation_report_plan(model="qwen3.6-27b")
            assert plan.cmd == ["bash", "scripts/report.sh", "--full"]
            assert plan.requires_confirm is True       # heavy — confirm
            assert plan.requires_reconcile is False     # uses serving model; no GPU claim
            assert "--full" in plan.description

    @pytest.mark.asyncio
    async def test_full_report_action_is_producer_only(self):
        """[F] full_report is gated off on the consumer surface."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert "full_report" in app._PRODUCER_ONLY
            assert app.check_action("full_report", ()) is False

    @pytest.mark.asyncio
    async def test_full_report_enabled_on_producer_gate_tab(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            assert app.check_action("full_report", ()) is True

    @pytest.mark.asyncio
    async def test_full_report_opens_confirm_never_auto_fires(self):
        """[F] on ③ Gate opens a confirm modal and does NOT fire the battery
        until confirmed (the heavy ~43-min run is never auto-launched)."""
        wr = FakeWriteRunner()
        # A serving target is required (the battery hits the serving model); the
        # no-target guard is covered by test_full_report_no_serving_model_notifies.
        app, _, _ = make_app(write_runner=wr, surface="producer", target=SERVING_TARGET)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            await pilot.press("F")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/report.sh", "--full"]
            assert app.screen._plan.requires_confirm is True
            assert app.screen._plan.requires_reconcile is False
            assert wr.started == []  # nothing fired — only the modal opened

    @pytest.mark.asyncio
    async def test_full_report_no_serving_model_notifies_no_confirm(self):
        """[F] with NO serving target → notify + return, NO ConfirmActionScreen
        (the ~43-min battery must not run against an empty MODEL=/URL=)."""
        wr = FakeWriteRunner()
        # An empty ServingTarget → _target_url / _target_model resolve blank.
        app, _, _ = make_app(
            write_runner=wr, surface="producer", target=ServingTarget(gpus=[])
        )
        notes: list[str] = []
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            assert not app._target_url and not app._target_model
            orig_notify = app.notify
            app.notify = lambda *a, **k: (notes.append((a[0] if a else k.get("message", ""))), orig_notify(*a, **k))[1]
            await pilot.press("F")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []
            assert any("no serving model" in n.lower() for n in notes)

    @pytest.mark.asyncio
    async def test_full_report_confirm_launches_via_mocked_write_runner(self):
        """Confirming streams the battery via the MOCKED write runner — NO live
        process is spawned (conftest blocks it)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            # Drive the launch directly (mirrors the run_validation_launch test —
            # the confirm modal's reconcile-on-mount is exercised separately).
            app.run_full_report_launch()
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/report.sh", "--full"]
            assert wr.started[0]["run_type"] == "full_report"

    @pytest.mark.asyncio
    async def test_full_report_distinct_from_consumer_bare_rig_report(self):
        """R2b consumer [R] rig_report stays BARE report.sh (no --full); the
        producer --full battery is a SEPARATE action."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            bare = app._data.full_validation_report_plan()
            assert "--full" in bare.cmd
            # The consumer rig_report path shells bare report.sh (asserted by the
            # R2b suite); here we assert the two cmds differ.
            assert bare.cmd != ["bash", "scripts/report.sh"]


# ===========================================================================
# Phase R / R2b — consumer share-back affordances (R [rig] · B [submit] · ! [problem])
#
# These are LIGHTWEIGHT, CONSUMER-RESIDENT contributions: a consumer shares a
# rig report / bench / problem report WITHOUT switching to the producer surface.
# So they are CONSUMER actions — NOT in _PRODUCER_ONLY — and work on the default
# (consumer) surface.  rig_report + problem_report are READS (no confirm/network);
# ONLY submit_bench [B] is an outward write and keeps its confirm + network gate.
# All backends are mocked through the injected runner — the network is never hit.
# ===========================================================================


RIG_REPORT_TEXT = (
    "# Rig report (paste-ready)\n"
    "- GPUs: 2x RTX 3090\n"
    "- decode 178 TPS @ 262K ctx\n"
)


class TestShareBackR2b:
    @pytest.mark.asyncio
    async def test_rig_report_modal_renders_mocked_report_sh(self):
        """[R] in Run opens the paste-ready rig-report modal, which loads
        bare report.sh snapshot output (mocked) — a READ, no ConfirmActionScreen."""
        responses = fake_responses(**{"scripts/report.sh": ok(RIG_REPORT_TEXT)})
        app, _, wr = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("R")
            await _settle(pilot)
            assert isinstance(app.screen, ShareBackReportScreen)
            # READ — never a confirm/network write.
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []
            body = str(app.screen.query_one("#share-report-body", Static).render())
            assert "Rig report (paste-ready)" in body
            assert "178 TPS" in body

    @pytest.mark.asyncio
    async def test_rig_report_invokes_bare_report_sh_no_full_no_submit(self):
        """[R] shells the LIGHTWEIGHT bare report.sh (~2 s snapshot) — NOT
        report.sh --full (the ~43-min verify+stress+soak+bench+agentic battery,
        a producer-Gate action that would contend with the running model) — and
        never a submit/auto flag."""
        responses = fake_responses(**{"scripts/report.sh": ok(RIG_REPORT_TEXT)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("R")
            await _settle(pilot)
            report_calls = [c for c in runner.calls if "report.sh" in " ".join(c)]
            assert report_calls, "report.sh was not invoked"
            for c in report_calls:
                joined = " ".join(c)
                assert "--full" not in joined    # lightweight, not the 43-min battery
                assert "--submit" not in joined
                assert "--auto" not in joined

    @pytest.mark.asyncio
    async def test_submit_bench_operate_opens_gated_confirm_never_auto(self, tmp_path):
        """[B] in Operate resolves the latest evidence tag and stages the OUTWARD
        submit behind a confirm modal — the network is NEVER auto-fired."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(repo_root=tmp_path, write_runner=wr)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # Operate
            await _settle(pilot)
            await pilot.press("B")
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.network is True
            assert app.screen._plan.requires_confirm is True
            assert "--auto-submit" in app.screen._plan.cmd
            # Resolved to the most-recent (only) tag in results/rebench/.
            assert "vllm-dual-test" in app.screen._plan.cmd
            assert wr.started == []  # nothing fired — only the modal opened

    @pytest.mark.asyncio
    async def test_submit_bench_no_tags_notifies(self, tmp_path):
        """[B] with no benched results notifies and opens no confirm modal."""
        # tmp_path has NO results/rebench/ tree → evidence_list is empty.
        wr = FakeWriteRunner()
        app, _, _ = make_app(repo_root=tmp_path, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # Operate
            await _settle(pilot)
            await pilot.press("B")
            await _settle(pilot)
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []

    @pytest.mark.asyncio
    async def test_problem_report_modal_renders(self):
        """[!] opens the paste-ready problem-report modal — a READ, no confirm."""
        app, _, wr = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("exclamation_mark")
            await _settle(pilot)
            assert isinstance(app.screen, ShareBackReportScreen)
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []
            body = str(app.screen.query_one("#share-report-body", Static).render())
            assert "Problem report" in body
            assert "Rig snapshot" in body

    @pytest.mark.asyncio
    async def test_failed_serve_surfaces_report_affordance(self):
        """A FAILED (gate-refused) serve surfaces the 'press !' affordance in the
        serve-live pane AND captures the failure context for [!]."""
        wr = FakeWriteRunner()
        # Make the gate unsafe: a live engine container + a busy GPU.
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(
            responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Spy on the serve-live pane's append_line to capture emitted lines.
            live = app.query_one("#serve-live")
            emitted: list[str] = []
            orig = live.append_line
            live.append_line = lambda line, _o=orig, _e=emitted: (_e.append(line), _o(line))[1]
            plan = app._data.serve("vllm/dual")  # not forced → gate refuses
            app.dispatch_action(plan)
            await _settle(pilot)
            assert wr.started == []  # refused at the gate (failure)
            assert any("press ! to report this" in ln for ln in emitted)
            # Failure context captured for the [!] problem report.
            assert app._problem_slug == "vllm/dual"
            assert app._problem_boot_log

    @pytest.mark.asyncio
    async def test_problem_report_uses_captured_failure_context(self):
        """After a failed serve, [!] assembles the report from the captured slug
        + boot-log context (READ — local context, no network)."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(
            responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.dispatch_action(plan)
            await _settle(pilot)
            await pilot.press("exclamation_mark")
            await _settle(pilot)
            assert isinstance(app.screen, ShareBackReportScreen)
            body = str(app.screen.query_one("#share-report-body", Static).render())
            assert "vllm/dual" in body  # the failed slug
            assert "Boot / failure log" in body

    @pytest.mark.asyncio
    async def test_share_back_actions_enabled_on_consumer_surface(self):
        """The 3 share-back actions are CONSUMER actions — check_action returns
        True on the default consumer surface in their modes (NOT producer-gated)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "consumer"
            # NOT producer-gated.
            assert "rig_report" not in app._PRODUCER_ONLY
            assert "submit_bench" not in app._PRODUCER_ONLY
            assert "report_problem" not in app._PRODUCER_ONLY
            # Run (mode 0): rig_report + report_problem enabled.
            assert app._active_mode == 0
            assert app.check_action("rig_report", ()) is True
            assert app.check_action("report_problem", ()) is True
            # Operate (mode 1): all three enabled.
            await pilot.press("2")
            await _settle(pilot)
            assert app.check_action("rig_report", ()) is True
            assert app.check_action("submit_bench", ()) is True
            assert app.check_action("report_problem", ()) is True
            # submit_bench is Operate-only — hidden in Run.
            await pilot.press("1")
            await _settle(pilot)
            assert app.check_action("submit_bench", ()) is False


# ===========================================================================
# Validate · Run (launch a validation step — confirm-gated, MOCKED stream)
# ===========================================================================


class TestValidateRunWired:
    @pytest.mark.asyncio
    async def test_run_enter_opens_confirm_modal(self):
        # Validate · Run is the producer lane (R3a) — enter on producer.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            # R3b-1: ③ Gate (the ladder) is no longer the lane's default tab (① Bring
            # is) — activate it explicitly.
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            app.query_one("#run-ladder-table", DataTable).move_cursor(row=0)  # verify-full
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert "verify-full" in app.screen._plan.description
            assert app.screen._plan.requires_confirm is True
            assert app.screen._plan.requires_reconcile is False

    @pytest.mark.asyncio
    async def test_run_confirm_launches_via_mocked_write_runner(self):
        """Confirming a Run step streams via run_validation → the MOCKED write
        runner.  NO live process is spawned (conftest blocks it)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            # R3b-1: activate ③ Gate (no longer the lane default tab).
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            app.query_one("#run-ladder-table", DataTable).move_cursor(row=2)  # bench
            await pilot.press("enter")
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            screen.query_one("#confirm-ok-btn", Button).press()
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/bench.sh"]
            assert wr.started[0]["run_type"] == "validation"

    @pytest.mark.asyncio
    async def test_run_step_does_not_go_through_dispatch_action(self):
        """A validation launch uses the on_confirm seam (run_validation), NOT the
        gated execute_action — it never claims a GPU."""
        wr = FakeWriteRunner()

        async def detect_should_not_be_called():
            raise AssertionError("a validation run must not reconcile")

        app, _, _ = make_app(write_runner=wr, surface="producer")
        # Swap the detect to one that screams if the reconcile gate runs on confirm.
        app._data._detect_endpoint = detect_should_not_be_called
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            # The confirm modal DOES run reconcile-on-mount for display; but the
            # commit must not re-enter execute_action.  Drive the kind directly.
            app.run_validation_launch("verify-full")
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/verify-full.sh"]


# ===========================================================================
# Estate write-extras (power-cap / prune / container top + rm) — all gated
# ===========================================================================


class TestEstateExtrasWired:
    @pytest.mark.asyncio
    async def test_power_cap_strip_populates(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            strip = str(app.query_one("#powercap-strip", Static).render())
            assert "GPU0" in strip and "230W" in strip
            assert "capped" in strip  # GPU0 limit 230 < default 370

    @pytest.mark.asyncio
    async def test_power_cap_toggle_opens_confirm_off(self):
        """GPU0 is capped → [c] stages a 'power-cap off' (uncap) confirm."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/gpu-mode.sh", "power-cap", "off"]
            assert app.screen._plan.requires_confirm is True
            assert app.screen._plan.requires_reconcile is False

    @pytest.mark.asyncio
    async def test_power_cap_sweep_opens_confirm(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("w")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert "power-cap-sweep" in " ".join(app.screen._plan.cmd)

    @pytest.mark.asyncio
    async def test_prune_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("p")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/gpu-mode.sh", "prune"]

    @pytest.mark.asyncio
    async def test_prune_dispatches_through_gate(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.dispatch_action(app._data.prune())
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/gpu-mode.sh", "prune"]

    @pytest.mark.asyncio
    async def test_container_top_reads_into_drill(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await pilot.pause()
            app.query_one("#containers-table", DataTable).move_cursor(row=0)
            await pilot.press("t")  # top (READ)
            await _settle(pilot)
            # No modal — top is a read.
            assert not isinstance(app.screen, ConfirmActionScreen)
            body = str(app.query_one("#drill-stats", Static).render())
            assert "PID" in body or "vllm" in body
            # [t] also fills the Config tab from the matched registry row.
            cfg = str(app.query_one("#drill-config", Static).render())
            assert "vllm-qwen36-27b-dual" in cfg

    @pytest.mark.asyncio
    async def test_container_rm_opens_reconcile_gated_confirm(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await pilot.pause()
            app.query_one("#containers-table", DataTable).move_cursor(row=0)
            await pilot.press("X")  # rm
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["docker", "rm", "vllm-qwen36-27b-dual"]
            assert app.screen._plan.requires_reconcile is True  # frees a GPU → gated

    @pytest.mark.asyncio
    async def test_container_rm_refused_when_unsafe(self):
        """rm of a live container → reconcile unsafe → refused (no write)."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(
            responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.dispatch_action(app._data.container_rm("vllm-qwen36-27b-dual"))
            await _settle(pilot)
            assert wr.started == []  # refused at the gate


# ===========================================================================
# Belt-and-suspenders: no live write / network across the Validate surface
# ===========================================================================


class TestValidateNoLiveWriteOrNetwork:
    @pytest.mark.asyncio
    async def test_full_validate_browse_touches_only_fakes(self, tmp_path):
        wr = FakeWriteRunner()
        # Validate (mode 2) is the producer lane (R3a).
        app, runner, _ = make_app(repo_root=tmp_path, write_runner=wr, surface="producer")
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            # Browse every Validate tab + Operate (incl. Doctor) — pure reads,
            # no writes.  R2a: Doctor lives under Operate now (tab-doctor).
            await pilot.press("3")
            await _settle(pilot)
            # R3b-1: browse every lane stage ①→⑤ — all pure reads, no writes.
            for tab in ("tab-bring", "tab-serve", "tab-run", "tab-evidence", "tab-promote"):
                app.query_one("#validate-tabs", TabbedContent).active = tab
                await pilot.pause()
            await pilot.press("2")
            await _settle(pilot)
            for tab in ("tab-orchestration", "tab-containers", "tab-doctor"):
                app.query_one("#operate-tabs", TabbedContent).active = tab
                await pilot.pause()
            # Nothing was written; submit-bench / prune / power-cap never auto-fired.
            assert wr.started == []
            assert all("--auto-submit" not in " ".join(c) for c in runner.calls)
            assert all("power-cap on" not in " ".join(c) and "power-cap off" not in " ".join(c)
                       for c in runner.calls)


# ===========================================================================
# PHASE 5 — the three v2 hooks (app wiring)
# ===========================================================================


from club3090_cockpit.app import (  # noqa: E402
    PromoteScaffoldScreen,
    OptimizeScreen,
    UntestedComposePreviewScreen,
    LaneBringPane,
    LaneServePane,
    LanePromotePane,
)


SERVING_TARGET = ServingTarget(
    url="http://localhost:8010",
    model="qwen3.6-27b",
    container="vllm-qwen36-27b-dual",
    slug="vllm/dual",
    gpus=[GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=22000)],
)


class TestEvaluateHookWired:
    """Hook 1 — Operate → ▸ Evaluate (c3t hand-off, confirm-gated, MOCK-ONLY)."""

    @pytest.mark.asyncio
    async def test_evaluate_opens_confirm_when_target_running(self):
        # R3b-1: [v] Evaluate relocated into the producer Bring & Validate lane
        # (mode 2).  The Operate estate poll still captures the live target
        # (_target_obj); the lane consumes it.
        app, _, _ = make_app(target=SERVING_TARGET,
                             gpus=list(SERVING_TARGET.gpus),
                             surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")          # Operate — poll captures the target
            await _settle(pilot)
            await pilot.press("3")          # Bring & Validate lane (where [v] lives)
            await _settle(pilot)
            await pilot.press("v")          # ▸ Evaluate
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.kind == "evaluate"
            assert app.screen._plan.cmd == ["bash", "scripts/c3t"]
            assert app.screen._plan.requires_reconcile is False
            assert app.screen._plan.requires_confirm is True

    @pytest.mark.asyncio
    async def test_evaluate_works_lane_native_without_visiting_operate(self):
        """R3b-1 HIGH fix: the lane-NATIVE path — enter the lane via key 3 WITHOUT
        first visiting Operate (key 2) — must still find the live target, because
        lane entry now primes load_estate() itself.  Before the fix _target_obj was
        set ONLY by the Operate poll, so this path hit the 'nothing to evaluate'
        notify even with a model serving."""
        app, _, _ = make_app(target=SERVING_TARGET,
                             gpus=list(SERVING_TARGET.gpus),
                             surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")          # straight to the lane — NO Operate visit
            await _settle(pilot)
            await pilot.press("v")          # ▸ Evaluate
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)  # target was primed
            assert app.screen._plan.kind == "evaluate"

    @pytest.mark.asyncio
    async def test_evaluate_hands_off_the_shared_serving_target_by_identity(self):
        """The app's stored target IS the SAME ServingTarget the poll detected,
        and the hand-off carries that exact object (design §4/§6.6)."""
        app, _, _ = make_app(target=SERVING_TARGET, gpus=list(SERVING_TARGET.gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            assert app._target_obj is SERVING_TARGET
            handoff = app._data.evaluate_handoff(app._target_obj)
            assert handoff.target is SERVING_TARGET     # same dataclass instance
            # And it is the shared-core dataclass, not a cockpit-local type.
            from club3090_tui_core.detect import ServingTarget as CoreTarget
            assert isinstance(handoff.target, CoreTarget)

    @pytest.mark.asyncio
    async def test_evaluate_confirm_launches_mock_only_scoped_to_target(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(target=SERVING_TARGET, gpus=list(SERVING_TARGET.gpus),
                             write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")          # Operate — capture the live target
            await _settle(pilot)
            await pilot.press("3")          # lane — [v] Evaluate lives here (R3b-1)
            await _settle(pilot)
            await pilot.press("v")
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            screen.query_one("#confirm-ok-btn", Button).press()
            await _settle(pilot)
            # c3t launched via the MOCKED write runner — never live.
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/c3t"]
            assert wr.started[0]["run_type"] == "evaluate"

    @pytest.mark.asyncio
    async def test_evaluate_no_target_notifies_and_does_not_launch(self):
        wr = FakeWriteRunner()
        # No serving target (empty url) → nothing to evaluate.  [v] lives in the
        # producer lane (R3b-1): poll Operate (no target), enter the lane, press v.
        app, _, _ = make_app(target=ServingTarget(), write_runner=wr,
                             surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("3")
            await _settle(pilot)
            await pilot.press("v")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []          # never launched


class TestPromoteHookWired:
    """Hook 2 — Run → ▸ Promote to catalog (scaffold preview + gated write)."""

    @pytest.mark.asyncio
    async def test_promote_without_byo_notifies(self):
        # R3b-1: [P] promote relocated into the producer Bring & Validate lane
        # (mode 2, ⑤ Promote).  Exercise the no-BYO branch in the lane.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("3")          # enter the lane (where [P] lives)
            await _settle(pilot)
            # No BYO fit-check yet → nothing to promote.
            await pilot.press("P")
            await pilot.pause()
            assert not isinstance(app.screen, PromoteScaffoldScreen)

    @pytest.mark.asyncio
    async def test_promote_previews_scaffold_after_byo_check(self):
        # R3b-1: [P] promote lives in the lane (mode 2).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Run a BYO fit-check (① Bring / Run · BYO share byo_check) — fills facts.
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            assert app._last_byo is not None
            await pilot.press("3")          # enter the lane
            await _settle(pilot)
            await pilot.press("P")          # ⑤ Promote to catalog
            await pilot.pause()
            assert isinstance(app.screen, PromoteScaffoldScreen)
            body = str(app.screen.query_one("#promote-body", Static).render())
            assert "schema_version: 1" in body
            assert "_entry(" in body
            assert "incubating" in body
            assert "scripts/tests/*.sh" in body   # the gated guard suite

    @pytest.mark.asyncio
    async def test_promote_stage_write_is_gated_mock_only(self):
        """Staging the write opens the standard confirm gate; the plan is
        mock-only and writes nothing into scripts/ (no auto-fire)."""
        wr = FakeWriteRunner()
        # [P] promote is producer-gated (R3a).
        app, _, _ = make_app(write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("3")          # enter the lane (R3b-1)
            await _settle(pilot)
            await pilot.press("P")
            await pilot.pause()
            assert isinstance(app.screen, PromoteScaffoldScreen)
            # Stage the gated write — routes through ConfirmActionScreen.
            app.screen.query_one("#promote-stage-btn", Button).press()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.kind == "promote_catalog"
            assert app.screen._plan.requires_confirm is True
            # Nothing executed yet — the write is mock-only and never auto-fired.
            assert wr.started == []

    @pytest.mark.asyncio
    async def test_promote_write_dispatch_reaches_only_mock_runner(self):
        """Even when explicitly dispatched, the promote write reaches ONLY the
        mocked write runner (never a live scripts/ write — conftest blocks it)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            sc = app._data.promote_scaffold(byo=app._last_byo)
            app.dispatch_action(sc.write_plan)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "scripts/tests/*.sh" in " ".join(wr.started[0]["cmd"])


# ===========================================================================
# PHASE R / R3b-1 — the Bring & Validate producer lane (ordered ①→⑤ pipeline)
# ===========================================================================


GENERATED_COMPOSE_YAML = (
    "# Profile (at-a-glance):\n"
    "#   Model:     Qwen3.6-27B (BYO)\n"
    "#   Status:    🧪 Experimental\n"
    "services:\n"
    "  vllm:\n"
    "    image: vllm/vllm-openai:v0.22.0\n"
    "    command: [--model, /models/byo, --tensor-parallel-size, '2']\n"
)


class TestLaneStructureR3b1:
    """The producer lane reads as the ordered ① → ⑤ pipeline."""

    @pytest.mark.asyncio
    async def test_lane_renders_ordered_stages(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            pane_ids = [p.id for p in tc.query(TabPane)]
            assert pane_ids == [
                "tab-bring", "tab-serve", "tab-run", "tab-evidence", "tab-promote"
            ], pane_ids
            # The reused panes are wired into the stage layout (③ Gate / ④ Measure)
            # and the new lane stages exist.
            app.query_one("#lane-bring-pane", LaneBringPane)
            app.query_one("#lane-serve-pane", LaneServePane)
            app.query_one("#validate-run-pane", ValidateRunPane)        # ③ Gate
            app.query_one("#validate-evidence-pane", ValidateEvidencePane)  # ④ Measure
            app.query_one("#lane-promote-pane", LanePromotePane)

    @pytest.mark.asyncio
    async def test_lane_label_is_bring_and_validate(self):
        from club3090_cockpit.app import MODES
        assert MODES[2][0] == "Bring & Validate"
        assert MODES[2][1] == "3"        # key 3, no renumber (still index 2)

    @pytest.mark.asyncio
    async def test_bring_stage_reuses_byo_check(self):
        """① Bring renders the byo_check verdict (route / sibling) like Run · BYO."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("3")
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            card = str(app.query_one("#lane-bring-result-card", Static).render())
            # The shared verdict text (route block) is rendered into the lane card.
            assert "Route" in card or "eligible" in card


class TestLaneServeR3b1:
    """② Serve — the critical new link: generate compose → preview → reconcile-gated."""

    @pytest.mark.asyncio
    async def test_generate_compose_shells_right_argv_and_reads_back(self, tmp_path):
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            res = await app._data.generate_compose("vllm/dual")
            # Shelled generate-compose.sh with --profile <slug> --out <tmp>.
            gen_calls = [c for c in runner.calls if "generate-compose.sh" in " ".join(c)]
            assert len(gen_calls) == 1
            call = gen_calls[0]
            assert call[:4] == ["bash", "scripts/generate-compose.sh", "--profile", "vllm/dual"]
            assert "--out" in call
            assert "--accept-degraded" not in call
            # Read the generated YAML back verbatim.
            assert res["error"] == ""
            assert res["compose_yaml"] == GENERATED_COMPOSE_YAML
            assert res["compose_path"]

    @pytest.mark.asyncio
    async def test_generate_compose_accept_degraded_passes_flag(self, tmp_path):
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await app._data.generate_compose("vllm/dual", accept_degraded=True)
            gen = [c for c in runner.calls if "generate-compose.sh" in " ".join(c)][0]
            assert "--accept-degraded" in gen

    @pytest.mark.asyncio
    async def test_generate_compose_failure_surfaces_error(self, tmp_path):
        # A runner that returns rc!=0 for generate-compose (drift-guard / refuse).
        class _FailRunner(FakeRunner):
            async def run(self, cmd, *, cwd, timeout=30.0):
                self.calls.append(list(cmd))
                if "generate-compose.sh" in " ".join(cmd):
                    return RunResult(returncode=2, stdout="", stderr="refuse: drift-guard failed")
                return await super().run(cmd, cwd=cwd, timeout=timeout)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=_FailRunner())
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            res = await app._data.generate_compose("vllm/dual")
            assert res["compose_yaml"] == ""
            assert "drift-guard" in res["error"]

    @pytest.mark.asyncio
    async def test_serve_untested_without_bring_notifies(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            # No ① Bring fit-check yet → ② Serve no-ops (no preview modal).
            app.action_serve_untested()
            await pilot.pause()
            assert not isinstance(app.screen, UntestedComposePreviewScreen)

    @pytest.mark.asyncio
    async def test_serve_untested_previews_compose_badged_untested(self, tmp_path):
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # ① Bring first (fills _last_byo).
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            # ② Serve — generate + preview.
            app.action_serve_untested()
            await _settle(pilot)
            assert isinstance(app.screen, UntestedComposePreviewScreen)
            title = str(app.screen.query_one(".untested-title", Label).render())
            assert "untested" in title           # badged 👤 untested
            body = str(app.screen.query_one("#untested-body", Static).render())
            # The compose is shown VERBATIM (a distinctive line from the YAML).
            assert "vllm/vllm-openai:v0.22.0" in body

    @pytest.mark.asyncio
    async def test_serve_untested_via_g_key_opens_preview(self, tmp_path):
        """R4 (folds R3b-1 LOW item e): the [g] binding (serve_untested) on the
        producer ② Serve stage, after a ① Bring fit-check, opens the untested
        compose preview — the same path action_serve_untested takes, exercised
        through the actual keypress (the Binding is 'g')."""
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # ① Bring fit-check first (mocked — fills _last_byo).
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("3")          # enter the Bring & Validate lane
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            # [g] = serve_untested — must be enabled here (producer · ② Serve).
            assert app.check_action("serve_untested", ()) is True
            await pilot.press("g")
            await _settle(pilot)
            assert isinstance(app.screen, UntestedComposePreviewScreen)

    @pytest.mark.asyncio
    async def test_serve_untested_serve_goes_through_reconcile_gate(self, tmp_path):
        """The preview's Serve hands serve_generated to the reconcile-gated
        ConfirmActionScreen — it is NOT auto-fired, and the plan claims the GPU."""
        wr = FakeWriteRunner()
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer",
                             runner=runner, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            app.action_serve_untested()
            await _settle(pilot)
            assert isinstance(app.screen, UntestedComposePreviewScreen)
            # Confirm the preview → opens the reconcile-gated confirm (NOT a serve).
            app.screen.query_one("#untested-serve-btn", Button).press()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            plan = app.screen._plan
            assert plan.kind == "serve"
            assert plan.requires_reconcile is True     # claims the GPU → gated
            assert plan.requires_confirm is True
            assert "docker" in plan.cmd and "compose" in plan.cmd
            # Nothing served yet — the reconcile gate has NOT been committed.
            assert wr.started == []

    @pytest.mark.asyncio
    async def test_serve_generated_plan_is_gpu_claiming(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve_generated("/tmp/some-generated.yml")
            assert plan.kind == "serve"
            assert plan.requires_reconcile is True
            assert plan.requires_confirm is True
            # The exact argv: `docker compose -f <path> up -d` (how serve_generated
            # builds it — a verbatim generated-compose launch, not a switch.sh slug).
            assert plan.cmd == [
                "docker", "compose", "-f", "/tmp/some-generated.yml", "up", "-d",
            ]


class TestLaneRelocationR3b1:
    """[P] Promote + [v] Evaluate relocated OUT of consumer modes INTO the lane."""

    @pytest.mark.asyncio
    async def test_promote_reachable_in_lane_not_in_run(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # In Run (mode 0): [P] is disabled (relocated out).
            assert app._active_mode == 0
            assert app.check_action("promote_catalog", ()) is False
            # In the lane (mode 2): [P] is enabled.
            await pilot.press("3")
            await _settle(pilot)
            assert app.check_action("promote_catalog", ()) is True

    @pytest.mark.asyncio
    async def test_evaluate_reachable_in_lane_not_in_operate(self):
        app, _, _ = make_app(surface="producer", target=SERVING_TARGET,
                             gpus=list(SERVING_TARGET.gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")          # Operate
            await _settle(pilot)
            # In Operate (mode 1): [v] is disabled (relocated out).
            assert app.check_action("evaluate_target", ()) is False
            # In the lane (mode 2): [v] is enabled.
            await pilot.press("3")
            await _settle(pilot)
            assert app.check_action("evaluate_target", ()) is True

    @pytest.mark.asyncio
    async def test_serve_untested_is_producer_only_and_lane_scoped(self):
        # Hidden on consumer.
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert "serve_untested" in CockpitApp._PRODUCER_ONLY
            assert app.check_action("serve_untested", ()) is False


class TestLaneHelpSurfaceThreadedR3b1:
    """Consumer help OMITS the producer lane; producer help INCLUDES it."""

    @pytest.mark.asyncio
    async def test_consumer_help_omits_lane(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            text = app.screen.help_text
            assert "Bring & Validate" not in text
            assert "Promote" not in text
            assert "Evaluate" not in text
            # The consumer mode line stops at Operate — the real rendered mode-3
            # token (NOT the trivially-absent literal "[3]") is absent on consumer.
            assert "3[/cyan]  Bring & Validate" not in text

    @pytest.mark.asyncio
    async def test_producer_help_includes_lane(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            text = app.screen.help_text
            assert "Bring & Validate" in text
            assert "Promote" in text
            assert "Evaluate" in text
            # The producer mode line carries the real rendered mode-3 token.
            assert "3[/cyan]  Bring & Validate" in text

    @pytest.mark.asyncio
    async def test_help_screen_threads_surface(self):
        # action_help passes the app surface into HelpScreen.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            assert app.screen._surface == "producer"


class TestOptimizeHookWired:
    """Hook 3 — ▸ Optimize for my card: DORMANT v0.10.0 seam (no-op)."""

    @pytest.mark.asyncio
    async def test_optimize_shows_not_available_message(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # From Run · Catalog with a selected slug.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("O")          # ▸ Optimize for my card
            await _settle(pilot)
            assert isinstance(app.screen, OptimizeScreen)
            body = str(app.screen.query_one("#optimize-body", Static).render())
            assert "optimizer not available (v0.10.0)" in body

    @pytest.mark.asyncio
    async def test_optimize_is_a_noop_no_write(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("O")
            await _settle(pilot)
            # A no-op seam: a modal, but never a write / launch.
            assert isinstance(app.screen, OptimizeScreen)
            assert wr.started == []

    @pytest.mark.asyncio
    async def test_optimize_available_from_run_staged_slug_fallback(self):
        """Optimize is reachable from Run; when no catalog row resolves it falls
        back to the last staged serve slug (Serve folded into Run, R1)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Stage a slug via the serve confirm flow, then dismiss the modal.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")      # → stages + opens confirm modal
            await pilot.pause()
            assert app._staged_entry is not None
            await pilot.press("escape")     # cancel the confirm modal
            await pilot.pause()
            await pilot.press("O")          # ▸ Optimize from Run
            await _settle(pilot)
            assert isinstance(app.screen, OptimizeScreen)
            body = str(app.screen.query_one("#optimize-body", Static).render())
            assert "v0.10.0" in body


class TestPhase5NoLiveEffect:
    """Belt-and-suspenders: the three hooks never write live / auto-fire."""

    @pytest.mark.asyncio
    async def test_browsing_all_three_hooks_touches_only_fakes(self):
        wr = FakeWriteRunner()
        app, runner, _ = make_app(target=SERVING_TARGET, gpus=list(SERVING_TARGET.gpus),
                                  write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Optimize (Run) — dormant no-op.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("O")
            await pilot.pause()
            await pilot.press("escape")
            # Promote (Run) after a BYO check — preview only.
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("P")
            await pilot.pause()
            await pilot.press("escape")
            # Evaluate (Operate) — confirm modal staged, not committed.
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("v")
            await pilot.pause()
            await pilot.press("escape")
            # Nothing was launched / written; no c3t, no guard suite, no scripts/.
            assert wr.started == []
            assert all("scripts/c3t" not in " ".join(c) for c in runner.calls)


# ===========================================================================
# Keyboard-hotkey UX fixes (keybindings-ux branch)
# ===========================================================================
#
# Bug-class coverage:
#   (a) typing a hotkey letter into a filter Input types into the field and
#       fires NO action / mode-switch / quit
#   (b) check_action enables the right keys per mode/sub-tab
#   (c) each modal: Esc closes, Confirm commits on Enter, q/digit keys do NOT
#       act while the modal is open
#   (d) sub-tab cycle key switches tabs
#   (e) mode-switch moves focus to the mode's primary interactive widget


from textual.widgets import Footer  # noqa: E402


class TestFilterInputSafety:
    """(a) Typing hotkey letters into a filter Input never fires actions."""

    @pytest.mark.asyncio
    async def test_typing_q_into_catalog_filter_types_not_quit(self):
        """Pressing 'q' while catalog-filter is focused must type into the field,
        not quit the app."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Open the filter — brings up the Input.
            await pilot.press("slash")
            await pilot.pause()
            inp = app.query_one("#catalog-filter", Input)
            assert "visible" in inp.classes  # filter is open
            assert app.focused is inp         # Input has focus
            # Typing 'q' must go into the input, NOT quit the app.
            await pilot.press("q")
            await pilot.pause()
            assert app.is_running, "app must not have quit"
            assert inp.value == "q"

    @pytest.mark.asyncio
    async def test_typing_e_into_catalog_filter_types_not_explain(self):
        """Pressing 'e' in the filter Input must NOT open the ExplainScreen."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("slash")
            await pilot.pause()
            inp = app.query_one("#catalog-filter", Input)
            await pilot.press("e")
            await pilot.pause()
            assert not isinstance(app.screen, ExplainScreen), "ExplainScreen must not open"
            assert inp.value == "e"

    @pytest.mark.asyncio
    async def test_typing_qwen36_into_filter_stays_in_input_no_action(self):
        """Typing a multi-char query containing hotkeys must all land in the
        input and fire no actions."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("slash")
            await pilot.pause()
            inp = app.query_one("#catalog-filter", Input)
            # Type each character — includes q,e,s,w,c,p,o which are all hotkeys.
            for ch in "qwen36":
                await pilot.press(ch)
            await pilot.pause()
            assert app.is_running, "app must not have quit after typing 'q'"
            assert inp.value == "qwen36"
            # No mode switch must have happened (still in Run = mode 0).
            assert app._active_mode == 0

    @pytest.mark.asyncio
    async def test_check_action_hides_context_keys_while_input_focused(self):
        """check_action returns False for all context keys while an Input is
        focused, so the footer never shows misleading hints during typing."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("slash")
            await pilot.pause()
            assert isinstance(app.focused, Input)
            # Context keys must be hidden while typing.
            for action in ("filter_catalog", "explain", "set_default", "container_logs",
                           "estate_off", "power_cap_toggle", "evaluate_target"):
                result = app.check_action(action, ())
                assert result is False, (
                    f"check_action({action!r}) should return False while Input focused, "
                    f"got {result!r}"
                )
            # Always-on keys must still be enabled.
            for action in ("quit", "help", "refresh", "mode_run", "mode_operate"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"check_action({action!r}) should return True (always-on), got {result!r}"
                )


class TestCheckActionPerModeSubtab:
    """(b) check_action enables the right bindings per mode/sub-tab."""

    @pytest.mark.asyncio
    async def test_run_catalog_enables_explain_and_filter(self):
        # R3b-1: [P] promote relocated OUT of Run into the lane (mode 2), so on
        # Run · Catalog it is now context-False; explain + filter stay True.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            assert app.check_action("explain", ()) is True
            assert app.check_action("filter_catalog", ()) is True
            # promote is a lane (mode 2) key now — disabled in Run.
            assert app.check_action("promote_catalog", ()) is False

    @pytest.mark.asyncio
    async def test_run_catalog_disables_estate_keys(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            for action in ("container_logs", "estate_off", "power_cap_toggle",
                           "prune_images", "evaluate_target", "context_t"):
                result = app.check_action(action, ())
                assert result is False, (
                    f"Run must not enable estate action {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_orchestration_enables_orch_keys(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            assert app._active_mode == 1
            # On orchestration tab (default first tab)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-orchestration"
            await pilot.pause()
            for action in ("estate_off", "power_cap_toggle", "power_cap_sweep", "prune_images"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"Operate·Orchestration must enable {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_orchestration_disables_containers_keys(self):
        """On the Orchestration sub-tab, container-specific keys are disabled."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-orchestration"
            await pilot.pause()
            for action in ("container_logs", "container_stop", "container_rm"):
                result = app.check_action(action, ())
                assert result is False, (
                    f"Operate·Orchestration must disable container action {action!r}, "
                    f"got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_containers_enables_containers_keys(self):
        """On the Containers sub-tab, container-specific keys are enabled."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-containers"
            await pilot.pause()
            for action in ("container_logs", "container_stop", "container_rm", "context_t"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"Operate·Containers must enable {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_run_catalog_enables_filter_and_validate_drops_old_bmk_keys(self):
        """Fold 3: [/] filter lives on Run · Catalog; the old Benchmarks sort
        ([t]) is gone, so context_t is disabled in Validate mode."""
        # Validate (mode 2) is the producer lane (R3a).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Run · Catalog (default mode/tab): filter enabled.
            assert app._active_mode == 0
            assert app.check_action("filter_catalog", ()) is True
            # Validate mode: no benchmarks tab → context_t (sort) disabled,
            # and filter_catalog is off (not a Run · Catalog context).
            await pilot.press("3")
            await _settle(pilot)
            assert app._active_mode == 2
            assert app.check_action("context_t", ()) is False
            assert app.check_action("filter_catalog", ()) is False

    @pytest.mark.asyncio
    async def test_validate_evidence_enables_s_key(self):
        # Validate (mode 2) is the producer lane (R3a).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            tc.active = "tab-evidence"
            await pilot.pause()
            # s_key (submit) is enabled in Validate mode regardless of subtab.
            assert app.check_action("s_key", ()) is True

    @pytest.mark.asyncio
    async def test_estate_mode_disables_explain_and_filter(self):
        """Outside Run · Catalog there is no catalog — explain and filter must be
        off (e.g. in Operate mode)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # Operate
            await pilot.pause()
            assert app._active_mode == 1
            assert app.check_action("explain", ()) is False
            assert app.check_action("filter_catalog", ()) is False

    @pytest.mark.asyncio
    async def test_always_on_keys_active_in_every_mode(self):
        """quit/help/refresh/mode-switch must be True in every mode.

        Run on the producer surface so all three mode switches (incl.
        mode_validate, which the R3a surface gate hides on consumer) are
        genuinely always-on across every mode."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            for mode_key in ("1", "2", "3"):
                await pilot.press(mode_key)
                await pilot.pause()
                for action in ("quit", "help", "refresh",
                               "mode_run", "mode_operate", "mode_validate"):
                    result = app.check_action(action, ())
                    assert result is True, (
                        f"Always-on {action!r} must be True in mode {mode_key}, got {result!r}"
                    )


class TestModalKeyCapture:
    """(c) Modals fully capture keys — q/digits don't act while a modal is open;
    Esc closes, Enter commits ConfirmActionScreen."""

    @pytest.mark.asyncio
    async def test_q_does_not_quit_while_help_modal_open(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("question_mark")  # open help
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            # Pressing 'q' must NOT quit the app (the HelpScreen is a ModalScreen
            # so app-level bindings don't fire).
            await pilot.press("q")
            await pilot.pause()
            # Either still on the help screen OR the q was consumed by the modal.
            # Either way the app must still be running.
            assert app.is_running

    @pytest.mark.asyncio
    async def test_esc_closes_help_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpScreen)

    @pytest.mark.asyncio
    async def test_esc_closes_explain_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("e")
            await _settle(pilot)
            assert isinstance(app.screen, ExplainScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ExplainScreen)

    @pytest.mark.asyncio
    async def test_mode_digit_does_not_switch_while_confirm_modal_open(self):
        """Pressing '2' while a ConfirmActionScreen is active must NOT switch
        the mode (ModalScreen intercepts app-level bindings)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            mode_before = app._active_mode
            await pilot.press("2")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen), \
                "confirm modal must still be on screen after pressing '2'"
            assert app._active_mode == mode_before, \
                "mode must not have changed while confirm modal was open"

    @pytest.mark.asyncio
    async def test_enter_commits_confirm_modal(self):
        """Pressing Enter on a clear-gate ConfirmActionScreen must commit (same
        as clicking the Confirm button)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            assert screen._reconcile is not None and screen._reconcile.safe
            # Enter must commit (Confirm button is enabled on a safe gate).
            await pilot.press("enter")
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/switch.sh", "vllm/dual"]

    @pytest.mark.asyncio
    async def test_esc_cancels_confirm_modal_no_write(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []  # nothing written

    @pytest.mark.asyncio
    async def test_q_does_not_quit_while_confirm_modal_open(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            await pilot.press("q")
            await pilot.pause()
            assert app.is_running

    @pytest.mark.asyncio
    async def test_esc_closes_evidence_report_modal(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_repo(root)
            # Validate (mode 2) is the producer lane (R3a).
            app, _, _ = make_app(repo_root=root, surface="producer")
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("3")
                await _settle(pilot)
                app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
                await pilot.pause()
                app.query_one("#evidence-table", DataTable).move_cursor(row=0)
                await pilot.press("enter")
                await _settle(pilot)
                assert isinstance(app.screen, EvidenceReportScreen)
                await pilot.press("escape")
                await pilot.pause()
                assert not isinstance(app.screen, EvidenceReportScreen)

    @pytest.mark.asyncio
    async def test_esc_closes_optimize_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("O")
            await _settle(pilot)
            assert isinstance(app.screen, OptimizeScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, OptimizeScreen)


class TestSubtabCycling:
    """(d) Sub-tab cycle key switches tabs in modes that have sub-tabs."""

    @pytest.mark.asyncio
    async def test_right_bracket_cycles_run_subtab(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0  # Run
            tc = app.query_one("#run-tabs", TabbedContent)
            assert tc.active == "tab-catalog"  # default first tab
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-byo"

    @pytest.mark.asyncio
    async def test_right_bracket_wraps_around_on_last_tab(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#run-tabs", TabbedContent)
            # Go to last tab first, then cycle forward → should wrap to first.
            tc.active = "tab-byo"
            await pilot.pause()
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-catalog"

    @pytest.mark.asyncio
    async def test_left_bracket_cycles_backwards(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#run-tabs", TabbedContent)
            tc.active = "tab-byo"
            await pilot.pause()
            await pilot.press("left_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-catalog"

    @pytest.mark.asyncio
    async def test_subtab_key_cycles_estate_tabs(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            first = tc.active
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active != first

    @pytest.mark.asyncio
    async def test_subtab_key_cycles_validate_tabs(self):
        # Validate (mode 2) is the producer lane (R3a).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            first = tc.active
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active != first

    @pytest.mark.asyncio
    async def test_subtab_key_active_in_all_three_modes(self):
        """After R1 all three modes (Run · Operate · Validate) have sub-tabs, so
        the cycle keys are active (check_action True) in each.

        Run on the producer surface so the Validate (mode 2) lane is reachable
        (R3a gates it on consumer)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            for key, mode in (("1", 0), ("2", 1), ("3", 2)):
                await pilot.press(key)
                await pilot.pause()
                assert app._active_mode == mode
                assert app.check_action("next_subtab", ()) is True
                assert app.check_action("prev_subtab", ()) is True


class TestModeSwitchFocus:
    """(e) Mode switch moves focus to the mode's primary interactive widget."""

    @pytest.mark.asyncio
    async def test_switch_to_run_focuses_catalog_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Start at Run (mode 0) already — go away and come back.
            await pilot.press("2")
            await pilot.pause()
            await pilot.press("1")
            await pilot.pause()
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "catalog-table"

    @pytest.mark.asyncio
    async def test_switch_to_estate_focuses_scene_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            # Operate/Orchestration tab is default → scene-table should be focused.
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "scene-table"

    @pytest.mark.asyncio
    async def test_switch_to_validate_lands_on_bring_stage(self):
        # R3b-1: the producer lane's first stage is ① Bring.  Focus is deliberately
        # NOT forced into its HF-repo Input (an Input would swallow the global
        # 1/2/3 + [ ] keys); the lane lands on ① Bring with focus on the tab bar so
        # those keys keep routing to the app.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            assert app._active_mode == 2
            assert app.query_one("#validate-tabs", TabbedContent).active == "tab-bring"
            # The bring input is NOT auto-focused (so digit/bracket keys work).
            assert not (isinstance(app.focused, Input)
                        and app.focused.id == "lane-bring-url-input")

    @pytest.mark.asyncio
    async def test_tab_change_on_validate_refocuses_relevant_table(self):
        """Cycling the lane's stages moves focus to the relevant widget for the
        newly active stage (R3b-1 — ① Bring → ② Serve → ③ Gate → ④ Measure → ⑤)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            # Default stage is ① Bring (no DataTable focus — its input is not
            # auto-focused so global keys still route to the app).
            assert tc.active == "tab-bring"
            # Activate ③ Gate directly → run-ladder-table is focused.
            tc.active = "tab-run"
            await pilot.pause()
            await pilot.pause()  # extra cycle for call_after_refresh
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "run-ladder-table"
            # Cycle forward → ④ Measure (Evidence) → evidence-table.
            await pilot.press("right_square_bracket")
            await pilot.pause()
            await pilot.pause()
            assert tc.active == "tab-evidence"
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "evidence-table"


class TestContainerAutoLoad:
    """Operate·Containers drill detail loads on USER navigation (cursor move) or
    an explicit [l]/[t] — but the tab is CALM on entry (#3, Batch 1): NO forced
    selection / auto-load of the first row's detail."""

    @pytest.mark.asyncio
    async def test_entering_containers_does_not_autoload_config(self):
        """#3: switching to the Containers tab must NOT auto-fill Config — the
        tab is calm on entry (no forced selection / auto-load)."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            cfg = str(app.query_one("#drill-config", Static).render())
            assert "vllm-qwen36-27b-dual" not in cfg  # NOT auto-loaded on entry

    @pytest.mark.asyncio
    async def test_entering_containers_does_not_autoload_logs(self):
        """#3: entering Containers must NOT auto-read docker logs/top — no
        subprocess read fires until the user navigates or presses [l]/[t]."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            assert not any("docker logs" in " ".join(c) for c in runner.calls)
            assert not any("docker top" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_explicit_logs_key_loads_after_calm_entry(self):
        """#3: the user CAN still load logs explicitly with [l] after the calm
        entry — the read fires on the explicit key."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            assert not any("docker logs" in " ".join(c) for c in runner.calls)
            app.query_one("#containers-table", DataTable).move_cursor(row=0)
            await pilot.pause()
            app.action_container_logs()
            await _settle(pilot)
            assert any("docker logs" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_drill_tab_switch_to_top_reads_top(self):
        """Switching the drill tab Logs→Top reads docker top for the selected
        container (no [t])."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            app.query_one("#drill-tabs", TabbedContent).active = "drill-tab-stats"
            await _settle(pilot)
            assert any("docker top" in " ".join(c) for c in runner.calls)
            assert "PID" in str(app.query_one("#drill-stats", Static).render())

    @pytest.mark.asyncio
    async def test_highlight_change_updates_config(self):
        """Highlighting a different container updates Config immediately (the
        local read is not debounced)."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_TWO)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            tbl = app.query_one("#containers-table", DataTable)
            tbl.focus()
            await pilot.pause()
            await pilot.press("down")  # row 0 (qwen) → row 1 (gemma) — fires RowHighlighted
            await pilot.pause()
            assert tbl.cursor_row == 1, f"cursor did not move (row={tbl.cursor_row})"
            cfg = str(app.query_one("#drill-config", Static).render())
            assert "vllm-gemma-4-31b-dual" in cfg

    @pytest.mark.asyncio
    async def test_r_refresh_on_containers_does_not_rejump_load(self):
        """NH1: pressing [r] (refresh) WHILE on the Containers tab repopulates
        the table → cursor resets to row 0.  That programmatic row-0 highlight
        must NOT auto-load the drill (no docker logs/top off the user's prior
        selection — the [r]-re-jump footgun); a SUBSEQUENT real user move DOES."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_TWO)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            tbl = app.query_one("#containers-table", DataTable)
            tbl.focus()
            await pilot.pause()
            # User moves to row 1 (gemma) — a genuine highlight → loads config.
            await pilot.press("down")
            await pilot.pause()
            assert "vllm-gemma-4-31b-dual" in str(
                app.query_one("#drill-config", Static).render()
            )
            # [r]-refresh on the tab: cursor snaps to row 0, but the programmatic
            # echo must NOT spawn a drill read.
            runner.calls.clear()
            await pilot.press("r")
            await _settle(pilot)
            assert not any("docker logs" in " ".join(c) for c in runner.calls), (
                "[r]-refresh auto-loaded docker logs (re-jump footgun)"
            )
            assert not any("docker top" in " ".join(c) for c in runner.calls), (
                "[r]-refresh auto-loaded docker top (re-jump footgun)"
            )
            # A subsequent real user move STILL loads the drill detail.
            runner.calls.clear()
            await pilot.press("down")
            await pilot.pause()
            assert "vllm-gemma-4-31b-dual" in str(
                app.query_one("#drill-config", Static).render()
            )


class TestEscClosesFilter:
    """Esc closes an open filter Input + refocuses the table (it was a dead key
    inside the filter before) — and never quits the app."""

    @pytest.mark.asyncio
    async def test_esc_closes_catalog_filter(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("slash")
            await pilot.pause()
            inp = app.query_one("#catalog-filter", Input)
            assert "visible" in inp.classes and app.focused is inp
            await pilot.press("escape")
            await pilot.pause()
            assert "visible" not in inp.classes
            assert app.focused is app.query_one("#catalog-table", DataTable)

    @pytest.mark.asyncio
    async def test_no_bmk_filter_in_validate_after_fold(self):
        """Fold 3: the Benchmarks filter is gone — `/` in Validate opens no
        filter Input and Esc remains a harmless no-op (the app stays running)."""
        # Validate (mode 2) is the producer lane (R3a).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")  # Validate
            await _settle(pilot)
            await pilot.press("slash")
            await pilot.pause()
            assert not app.query("#bmk-filter")
            assert not app.query("#bmk-table")
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running

    @pytest.mark.asyncio
    async def test_esc_on_main_screen_does_not_quit(self):
        """Esc with no filter + no modal is a harmless no-op (never quits)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running


class TestSurfaceScaffold:
    """R0/R3a — the surface flag + `--contribute` indicator + the producer gate.

    R0 wired the surface flag with an EMPTY _PRODUCER_ONLY (no-op gate); R3a
    POPULATES it ({"mode_validate", "promote_catalog"}) so the consumer/producer
    split is real. These tests prove (a) the flag defaults to consumer, (b)
    producer surfaces the CONTRIBUTE indicator, and (c) the gate LOGIC hides a
    producer-only action on consumer / shows it on producer — including for
    _ALWAYS_ON actions (the gate is checked BEFORE _ALWAYS_ON, so it can hide the
    producer Bring & Validate MODE switch).

    The gate-logic tests patch the *class* attr `_PRODUCER_ONLY` via monkeypatch
    BEFORE the app mounts (auto-restored after), rather than mutating a live
    instance attr mid-test — the latter raced under accumulated full-suite
    asyncio state and flaked.
    """

    @pytest.mark.asyncio
    async def test_default_surface_is_consumer_no_indicator(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "consumer"
            assert "CONTRIBUTE" not in str(app.sub_title)

    @pytest.mark.asyncio
    async def test_producer_surface_shows_contribute_indicator(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "producer"
            assert "CONTRIBUTE" in str(app.sub_title)

    @pytest.mark.asyncio
    async def test_invalid_surface_falls_back_to_consumer(self):
        app, _, _ = make_app(surface="bogus")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "consumer"

    @pytest.mark.asyncio
    async def test_shipped_producer_set_is_mode_validate_and_promote(self):
        # R3b-1: the shipped _PRODUCER_ONLY gates the producer lane (mode_validate)
        # + the relocated-into-the-lane [P] promote + [v] evaluate + ② serve_untested.
        # R3b-2: + [m] measure_vs_bar (④ Measure) + [F] full_report (③ Gate battery).
        assert CockpitApp._PRODUCER_ONLY == frozenset({
            "mode_validate", "promote_catalog", "evaluate_target", "serve_untested",
            "measure_vs_bar", "full_report",
        })

    @pytest.mark.asyncio
    async def test_shipped_producer_set_hidden_on_consumer(self):
        # On the consumer surface the shipped producer-only actions are gated off.
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("mode_validate", ()) is False
            assert app.check_action("promote_catalog", ()) is False
            # a normal Run/mode-0 context key is unaffected by the surface gate
            assert app.check_action("explain", ()) is True

    @pytest.mark.asyncio
    async def test_shipped_producer_set_shown_on_producer(self):
        # On the producer surface they fall through to their normal context
        # result: mode_validate is always-on (True).  R3b-1: promote_catalog is a
        # mode-2 (lane) context key now, so enter the lane before asserting True.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("mode_validate", ()) is True
            await pilot.press("3")          # enter the lane (mode 2)
            await _settle(pilot)
            assert app.check_action("promote_catalog", ()) is True

    @pytest.mark.asyncio
    async def test_producer_gate_hides_on_consumer(self, monkeypatch):
        # Patch the CLASS attr before mount (stable, race-free) to prove the gate
        # fires on the consumer surface.
        monkeypatch.setattr(CockpitApp, "_PRODUCER_ONLY", frozenset({"explain"}))
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("explain", ()) is False  # surface-gated off

    @pytest.mark.asyncio
    async def test_producer_gate_passes_on_producer(self, monkeypatch):
        monkeypatch.setattr(CockpitApp, "_PRODUCER_ONLY", frozenset({"explain"}))
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # producer is not surface-gated → falls through to the normal context
            # check (explain is a mode-0 Run key; default mode is 0 → True)
            assert app.check_action("explain", ()) is True

    @pytest.mark.asyncio
    async def test_producer_gate_beats_always_on_on_consumer(self, monkeypatch):
        # The gate is checked BEFORE _ALWAYS_ON, so an _ALWAYS_ON action (here
        # "help") placed in _PRODUCER_ONLY is still hidden on consumer. This is the
        # property R3 relies on to hide the producer Bring & Validate MODE switch.
        assert "help" in CockpitApp._ALWAYS_ON  # guard: "help" is always-on
        monkeypatch.setattr(CockpitApp, "_PRODUCER_ONLY", frozenset({"help"}))
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("help", ()) is False  # gate beats _ALWAYS_ON

    @pytest.mark.asyncio
    async def test_always_on_action_shows_on_producer_when_gated(self, monkeypatch):
        # Same injection, producer surface: gate is skipped → _ALWAYS_ON wins.
        monkeypatch.setattr(CockpitApp, "_PRODUCER_ONLY", frozenset({"help"}))
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("help", ()) is True


def _mode_switcher_item_count(app) -> int:
    """How many mode rows the ModeSwitcher VISIBLY renders (its mode-N Labels).

    Match only the numbered ``mode-<N>`` row ids — NOT the ``mode-action-hint``
    Label (which also starts with ``mode-``).  R4: all three mode Labels are
    always mounted; the consumer surface HIDES the producer-only third via the
    ``mode-hidden`` class (so the runtime Contribute toggle can reveal it without
    an async re-mount), so count only the rows NOT carrying that class."""
    ms = app.query_one("#mode-switcher", ModeSwitcher)
    return len([
        lbl for lbl in ms.query(Label)
        if (lbl.id or "").startswith("mode-") and (lbl.id or "")[len("mode-"):].isdigit()
        and not lbl.has_class("mode-hidden")
    ])


class TestProducerLaneGatedR3a:
    """R3a — the producer Bring & Validate lane (mode 2 / key 3) + [P] promote are
    PRODUCER-only: hidden + unreachable on the consumer surface, reachable on
    producer. The ModeSwitcher is surface-aware (2 rows consumer, 3 producer)."""

    @pytest.mark.asyncio
    async def test_consumer_cannot_reach_validate_via_key_3(self):
        """On consumer: mode_validate is gated off and pressing 3 does NOT switch
        to the producer lane (stays in Run, mode 0)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            assert app.check_action("mode_validate", ()) is False
            await pilot.press("3")
            await _settle(pilot)
            # Did NOT enter the producer Validate lane.
            assert app._active_mode == 0
            assert "active" not in app.query_one("#panel-validate").classes
            assert "active" in app.query_one("#panel-run").classes

    @pytest.mark.asyncio
    async def test_consumer_action_mode_validate_guard_is_noop(self):
        """Belt-and-suspenders: even a direct programmatic call to
        action_mode_validate does not switch a consumer into the producer lane."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.action_mode_validate()
            await _settle(pilot)
            assert app._active_mode == 0
            assert "active" not in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_consumer_promote_is_gated_off(self):
        """On consumer: [P] promote_catalog is gated off (producer activity)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0  # Run, where [P] would otherwise be live
            assert app.check_action("promote_catalog", ()) is False

    @pytest.mark.asyncio
    async def test_consumer_mode_switcher_shows_two_modes(self):
        """The consumer ModeSwitcher renders Run + Operate only (no Validate row)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert _mode_switcher_item_count(app) == 2

    @pytest.mark.asyncio
    async def test_producer_can_reach_validate_via_key_3(self):
        """On producer: pressing 3 enters the Validate lane (mode 2)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("mode_validate", ()) is True
            await pilot.press("3")
            await _settle(pilot)
            assert app._active_mode == 2
            assert "active" in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_producer_promote_is_reachable(self):
        """On producer: [P] promote_catalog is reachable in the lane (R3b-1 mode 2)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("3")          # enter the lane (mode 2)
            await _settle(pilot)
            assert app._active_mode == 2
            assert app.check_action("promote_catalog", ()) is True

    @pytest.mark.asyncio
    async def test_producer_mode_switcher_shows_three_modes(self):
        """The producer ModeSwitcher renders all three modes (incl. Validate)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert _mode_switcher_item_count(app) == 3

    @pytest.mark.asyncio
    async def test_consumer_share_back_not_over_gated(self):
        """The consumer share-back (rig_report / submit_bench / report_problem) is
        NOT producer-gated and stays check_action-True on consumer in its modes."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            for a in ("rig_report", "submit_bench", "report_problem"):
                assert a not in app._PRODUCER_ONLY
            # Run (mode 0): rig_report + report_problem live.
            assert app.check_action("rig_report", ()) is True
            assert app.check_action("report_problem", ()) is True
            # Operate (mode 1): all three live (submit_bench is Operate-only).
            await pilot.press("2")
            await _settle(pilot)
            assert app.check_action("rig_report", ()) is True
            assert app.check_action("submit_bench", ()) is True
            assert app.check_action("report_problem", ()) is True


class TestResolveSurface:
    """R0/R4 — resolve_surface(argv, env): CLI/env opt-in + persisted fallback.

    Each test points C3_CONFIG_DIR at a fresh tmp dir so the resolver's R4
    persisted-setting fallback NEVER reads the real ~/.config (and there is no
    persisted file in an empty tmp dir → the pre-R4 behaviour is preserved)."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("C3_CONFIG_DIR", str(tmp_path))

    def test_default_is_consumer(self):
        assert resolve_surface(["c3"], {}) == "consumer"

    def test_contribute_flag_opts_in(self):
        assert resolve_surface(["c3", "--contribute"], {}) == "producer"

    def test_env_producer_opts_in(self):
        assert resolve_surface(["c3"], {"C3_SURFACE": "producer"}) == "producer"

    def test_env_is_case_and_space_insensitive(self):
        assert resolve_surface(["c3"], {"C3_SURFACE": "  Producer "}) == "producer"

    def test_env_other_value_is_consumer(self):
        assert resolve_surface(["c3"], {"C3_SURFACE": "1"}) == "consumer"
        assert resolve_surface(["c3"], {"C3_SURFACE": ""}) == "consumer"


class TestSurfacePersistence:
    """R4 — config_path / load_surface_setting / save_surface_setting + the
    resolve_surface precedence (explicit flag/env > persisted > default).

    All tests inject C3_CONFIG_DIR=tmp_path so NOTHING touches the real
    ~/.config (the persistence MUST be test-injectable)."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("C3_CONFIG_DIR", str(tmp_path))
        self._cfg_dir = tmp_path

    def test_config_path_honors_env_override(self):
        p = config_path()
        assert p == self._cfg_dir / "c3-surface.json"

    def test_load_missing_file_returns_none(self):
        assert load_surface_setting() is None

    def test_save_then_load_roundtrips(self):
        save_surface_setting("producer")
        assert load_surface_setting() == "producer"
        save_surface_setting("consumer")
        assert load_surface_setting() == "consumer"

    def test_save_ignores_invalid_surface(self):
        save_surface_setting("bogus")
        assert load_surface_setting() is None

    def test_load_corrupt_file_returns_none(self):
        config_path().write_text("{not json", encoding="utf-8")
        assert load_surface_setting() is None

    def test_load_unrecognised_surface_returns_none(self):
        config_path().write_text('{"surface": "wat"}', encoding="utf-8")
        assert load_surface_setting() is None

    def test_resolve_reads_persisted_when_no_explicit(self):
        save_surface_setting("producer")
        # No flag / env → the persisted producer setting wins (precedence 2).
        assert resolve_surface(["c3"], {}) == "producer"

    def test_explicit_flag_beats_persisted(self):
        save_surface_setting("consumer")
        # Persisted is consumer, but the explicit --contribute flag wins.
        assert resolve_surface(["c3", "--contribute"], {}) == "producer"

    def test_explicit_env_beats_persisted(self):
        save_surface_setting("consumer")
        assert resolve_surface(["c3"], {"C3_SURFACE": "producer"}) == "producer"

    def test_no_persisted_falls_to_consumer(self):
        # No flag, no env, no persisted file → consumer default.
        assert resolve_surface(["c3"], {}) == "consumer"


class TestContributeDoor:
    """R4 — the in-app Contribute DOOR: [C] toggles consumer ↔ producer at runtime
    (ModeSwitcher 2 ↔ 3 modes, producer gating un/gates, the in-lane edge handled)
    AND persists the choice (test-injectable config dir)."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("C3_CONFIG_DIR", str(tmp_path))

    @pytest.mark.asyncio
    async def test_toggle_is_always_on(self):
        # The door is the consumer's opt-in — it must NOT be producer-gated.
        assert "toggle_contribute" in CockpitApp._ALWAYS_ON
        assert "toggle_contribute" not in CockpitApp._PRODUCER_ONLY
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("toggle_contribute", ()) is True

    @pytest.mark.asyncio
    async def test_toggle_consumer_to_producer_unlocks_lane(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "consumer"
            assert _mode_switcher_item_count(app) == 2
            assert app.check_action("mode_validate", ()) is False
            await pilot.press("C")
            await _settle(pilot)
            assert app._surface == "producer"
            assert _mode_switcher_item_count(app) == 3
            assert app.check_action("mode_validate", ()) is True
            assert "CONTRIBUTE" in str(app.sub_title)
            # No forced switch — still in Run (mode 0).
            assert app._active_mode == 0

    @pytest.mark.asyncio
    async def test_toggle_persists_for_next_launch(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("C")
            await _settle(pilot)
            assert load_surface_setting() == "producer"
            # resolve_surface (next launch, no flag/env) reads the persisted value.
            assert resolve_surface(["c3"], {}) == "producer"

    @pytest.mark.asyncio
    async def test_toggle_back_to_consumer_regates_and_persists(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("C")          # producer → consumer (from Run)
            await _settle(pilot)
            assert app._surface == "consumer"
            assert _mode_switcher_item_count(app) == 2
            assert app.check_action("mode_validate", ()) is False
            assert "CONTRIBUTE" not in str(app.sub_title)
            assert load_surface_setting() == "consumer"

    @pytest.mark.asyncio
    async def test_toggle_off_while_in_lane_switches_to_run(self):
        # EDGE: toggling producer → consumer while IN the producer lane (mode 2,
        # now hidden) must move the user back to a consumer-visible mode (Run).
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("3")          # enter the lane (mode 2)
            await _settle(pilot)
            assert app._active_mode == 2
            await pilot.press("C")          # toggle OFF while stranded in the lane
            await _settle(pilot)
            assert app._surface == "consumer"
            assert app._active_mode == 0    # rescued to Run
            assert "active" in app.query_one("#panel-run").classes
            assert "active" not in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_toggle_round_trip_back_to_consumer(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("C")          # → producer
            await _settle(pilot)
            await pilot.press("C")          # → consumer
            await _settle(pilot)
            assert app._surface == "consumer"
            assert _mode_switcher_item_count(app) == 2
            assert app.check_action("mode_validate", ()) is False


# ===========================================================================
# UX Batch 2 — the "perception loop": live surfaces + honest failure
# ===========================================================================


def _count_load_estate(monkeypatch):
    """Wrap CockpitApp.load_estate so a test can count how many times it fired.
    Returns the counter dict (``{"n": int}``)."""
    import club3090_cockpit.app as appmod

    calls = {"n": 0}
    orig = appmod.CockpitApp.load_estate

    def counting(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(appmod.CockpitApp, "load_estate", counting)
    return calls


class TestBatch2A1RepollAfterEveryWrite:
    """A1 — every SUCCESSFUL GPU-mutating write re-polls the estate; a REFUSED
    write does not."""

    @pytest.mark.asyncio
    async def test_scene_switch_repolls_estate(self, monkeypatch):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        calls = _count_load_estate(monkeypatch)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")          # Operate entry → 1 poll
            await _settle(pilot)
            before = calls["n"]
            plan = app._data.scene_switch("dual-qwen")
            assert plan.kind == "scene"
            app.dispatch_action(plan)
            await _settle(pilot)
            assert calls["n"] > before      # the scene WRITE re-polled
            assert len(wr.started) == 1     # gate intact — write went through

    @pytest.mark.asyncio
    async def test_estate_down_repolls_estate(self, monkeypatch):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        calls = _count_load_estate(monkeypatch)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            before = calls["n"]
            plan = app._data.estate_down()
            assert plan.kind == "estate_down"
            app.dispatch_action(plan)
            await _settle(pilot)
            assert calls["n"] > before
            assert len(wr.started) == 1

    @pytest.mark.asyncio
    async def test_container_rm_repolls_estate(self, monkeypatch):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        calls = _count_load_estate(monkeypatch)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            before = calls["n"]
            plan = app._data.container_rm("vllm-qwen36-27b-dual")
            assert plan.kind == "container_rm"
            app.dispatch_action(plan)
            await _settle(pilot)
            assert calls["n"] > before
            assert len(wr.started) == 1

    @pytest.mark.asyncio
    async def test_serve_repolls_estate_deferred(self, monkeypatch):
        """A serve arms the DEFERRED re-poll: load_estate fires again after the
        dispatch (the immediate poll inside the pending-serve watcher)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        calls = _count_load_estate(monkeypatch)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            before = calls["n"]
            plan = app._data.serve("vllm/dual")
            app.dispatch_action(plan)
            await _settle(pilot)
            assert calls["n"] > before      # the deferred watcher re-polled
            assert len(wr.started) == 1

    @pytest.mark.asyncio
    async def test_refused_write_does_not_repoll(self, monkeypatch):
        """A REFUSED write (unsafe gate, not forced) must NOT re-poll the estate."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(
            responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus), write_runner=wr
        )
        calls = _count_load_estate(monkeypatch)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            before = calls["n"]
            plan = app._data.serve("vllm/dual")  # not forced → refused
            app.dispatch_action(plan)
            await _settle(pilot)
            assert wr.started == []          # refused at the gate
            assert calls["n"] == before      # NO re-poll on a refused write


class TestBatch2A3PeriodicRefresh:
    """A3 — the periodic refresh interval polls ONLY in Operate (mode 1), never
    in Run / Validate."""

    @pytest.mark.asyncio
    async def test_interval_polls_in_operate(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")           # Operate
            await _settle(pilot)
            assert app._active_mode == 1
            calls = {"n": 0}
            orig = app.load_estate
            app.load_estate = lambda _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o())[1]
            app._periodic_estate_refresh()   # fire the gated tick directly
            assert calls["n"] == 1           # polled in Operate

    @pytest.mark.asyncio
    async def test_interval_does_not_poll_in_run(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0     # Run
            calls = {"n": 0}
            orig = app.load_estate
            app.load_estate = lambda _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o())[1]
            app._periodic_estate_refresh()
            assert calls["n"] == 0           # NOT polled in Run

    @pytest.mark.asyncio
    async def test_interval_does_not_poll_in_validate(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")           # Validate (producer lane)
            await _settle(pilot)
            assert app._active_mode == 2
            calls = {"n": 0}
            orig = app.load_estate
            app.load_estate = lambda _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o())[1]
            app._periodic_estate_refresh()
            assert calls["n"] == 0           # NOT polled in Validate

    @pytest.mark.asyncio
    async def test_rail_shows_as_of_stamp(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "as of" in rail           # freshness stamp rendered


class TestBatch2A2DockerFailureRenders:
    """A2 + N2 — a docker / nvidia-smi failure on the READ path sets state.error
    and renders (no crash, not false-idle)."""

    @pytest.mark.asyncio
    async def test_docker_read_failure_sets_state_error(self):
        """estate_state catches the docker ps raise on the READ path → partial
        EstateState with .error set (does NOT crash the worker)."""
        responses = fake_responses(
            **{"docker ps": RunResult(returncode=1, stdout="", stderr="docker daemon not running")}
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            state = await app._data.estate_state(variants=app._variants or None)
            assert state.error                       # error recorded
            assert "docker unreachable" in state.error
            assert state.containers == []            # partial — no containers

    @pytest.mark.asyncio
    async def test_docker_failure_renders_red_strip_not_false_idle(self):
        responses = fake_responses(
            **{"docker ps": RunResult(returncode=1, stdout="", stderr="daemon down")}
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            # The orch pane shows the distinct docker-unreachable strip.
            strip = app.query_one("#estate-error-strip", Static)
            assert "visible" in strip.classes
            assert "docker unreachable" in str(strip.render())
            # The Containers table says "docker unreachable", NOT the calm idle.
            tbl = app.query_one("#containers-table", DataTable)
            blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "docker unreachable" in blob
            assert "no stack containers" not in blob
            # The rail is honest too.
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "docker unreachable" in rail

    @pytest.mark.asyncio
    async def test_no_gpus_renders_nvidia_smi_message(self):
        """When nvidia-smi returns nothing, the GPU card says so — distinct from
        the calm idle, never a blank/'not present' that hides the failure."""
        app, _, _ = make_app(
            gpus=[], target=ServingTarget(gpus=[]),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "nvidia-smi returned nothing" in bar

    @pytest.mark.asyncio
    async def test_healthy_estate_has_no_error_strip(self):
        """No false positives: a healthy estate keeps the error strip hidden."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            strip = app.query_one("#estate-error-strip", Static)
            assert "visible" not in strip.classes


class TestBatch2A10ServeLiveTerminalState:
    """A10 — the serve LivePane resolves to ✓ serving / still booting / ✗ via the
    deferred re-poll, not an inert 'boot log streams here'."""

    @pytest.mark.asyncio
    async def test_serve_live_shows_watching_not_inert_placeholder(self):
        """On dispatch the pane says it's WATCHING (honest) — never the old inert
        '(boot log streams here)'."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            live = app.query_one("#serve-live")
            emitted: list[str] = []
            orig = live.append_line
            live.append_line = lambda line, _o=orig, _e=emitted: (_e.append(line), _o(line))[1]
            plan = app._data.serve("vllm/dual")
            app.dispatch_action(plan)
            await _settle(pilot)
            joined = " ".join(emitted)
            assert "watching for it to come up" in joined
            assert "boot log streams here" not in joined

    @pytest.mark.asyncio
    async def test_serve_live_shows_serving_when_slug_matches(self):
        """When the estate's matched_slug becomes the served slug, the pane stamps
        '✓ serving <model> · :<port>'."""
        wr = FakeWriteRunner()
        tgt = ServingTarget(
            url="http://localhost:8010", model="qwen3.6-27b", host_port=8010,
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )
        app, _, _ = make_app(target=tgt, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            live = app.query_one("#serve-live")
            emitted: list[str] = []
            orig = live.append_line
            live.append_line = lambda line, _o=orig, _e=emitted: (_e.append(line), _o(line))[1]
            plan = app._data.serve("vllm/dual")
            app.dispatch_action(plan)
            await _settle(pilot)
            joined = " ".join(emitted)
            assert "serving" in joined
            assert "qwen3.6-27b" in joined
            assert ":8010" in joined
            # The pending-serve watcher resolved + stopped.
            assert app._pending_serve_slug == ""

    @pytest.mark.asyncio
    async def test_serve_live_still_booting_on_timeout(self):
        """If the slug never matches, the deferred watcher times out → an honest
        'still booting — press r to refresh' (driven directly, no real wait)."""
        wr = FakeWriteRunner()
        # A target that does NOT match vllm/dual (no port 8010) → never resolves.
        tgt = ServingTarget(gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)])
        app, _, _ = make_app(target=tgt, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            live = app.query_one("#serve-live")
            emitted: list[str] = []
            orig = live.append_line
            live.append_line = lambda line, _o=orig, _e=emitted: (_e.append(line), _o(line))[1]
            plan = app._data.serve("vllm/dual")
            app.dispatch_action(plan)
            await _settle(pilot)
            assert app._pending_serve_slug == "vllm/dual"   # still pending
            # Drive the watcher past its attempt budget directly (no 30s wait).
            app._pending_serve_attempts = app._SERVE_REPOLL_MAX_ATTEMPTS
            app._poll_pending_serve()
            await _settle(pilot)
            assert any("still booting" in ln for ln in emitted)
            assert app._pending_serve_slug == ""            # watcher stopped

    @pytest.mark.asyncio
    async def test_serve_failed_stamps_cross_and_captures(self):
        """serve_failed stamps the ✗ terminal line + captures the failure context
        for [!]."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            live = app.query_one("#serve-live")
            emitted: list[str] = []
            orig = live.append_line
            live.append_line = lambda line, _o=orig, _e=emitted: (_e.append(line), _o(line))[1]
            plan = app._data.serve("vllm/dual")
            app._staged_entry = None
            app.serve_failed(plan, "boot crashed: CUDA OOM")
            await _settle(pilot)
            assert any("did not come up" in ln for ln in emitted)
            assert any("report" in ln for ln in emitted)
            assert app._problem_slug == "vllm/dual"
            assert app._pending_serve_slug == ""


class TestBatch2N3CatalogServingMarker:
    """N3 — the Run catalog marks the live-serving slug + clears when none."""

    @pytest.mark.asyncio
    async def test_catalog_marks_serving_slug(self):
        # A target on port 8010 → matched_slug vllm/dual.
        tgt = ServingTarget(
            url="http://localhost:8010", model="qwen3.6-27b", host_port=8010,
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )
        app, _, _ = make_app(target=tgt)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)            # catalog loads
            await pilot.press("2")          # Operate poll → sets serving slug
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._serving_slug == "vllm/dual"
            tbl = app.query_one("#catalog-table", DataTable)
            blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "serving" in blob        # the ● serving badge

    @pytest.mark.asyncio
    async def test_catalog_marker_clears_when_nothing_serving(self):
        app, _, _ = make_app(target=ServingTarget())   # nothing matches
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            # Force-set a stale marker, then a poll with no match must clear it.
            pane.set_serving_slug("vllm/dual")
            assert pane._serving_slug == "vllm/dual"
            await pilot.press("2")
            await _settle(pilot)
            assert pane._serving_slug == ""


def _hook_serve_live(app):
    """Tap the serve LivePane's append_line, returning the emitted-lines list."""
    live = app.query_one("#serve-live")
    emitted: list[str] = []
    orig = live.append_line
    live.append_line = lambda line, _o=orig, _e=emitted: (_e.append(line), _o(line))[1]
    return emitted


class TestBatch2MustFix1GeneratedServe:
    """MUST-FIX 1 — a generated/BYO serve (`docker compose -f <path> up -d`) has
    NO registry slug.  It must NOT get a bogus `-d` pending slug, must NOT false-✓
    a stale staged catalog slug, and reaches an honest terminal state when its
    container appears."""

    @pytest.mark.asyncio
    async def test_generated_serve_does_not_yield_dash_d_slug(self):
        """A generated serve plan never sets `_pending_serve_slug == '-d'`."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve_generated("/tmp/c3-genc-x.yml")
            # The argv really ends in `-d` — proving the old bug would have fired.
            assert plan.cmd[-1] == "-d"
            assert app._serve_slug_for(plan) == ""        # NOT "-d", NOT a slug
            app.dispatch_action(plan)
            await _settle(pilot)
            assert app._pending_serve_slug == ""          # no bogus slug
            assert app._pending_serve_slug != "-d"
            assert app._pending_serve_generated is True   # generated lane armed

    @pytest.mark.asyncio
    async def test_generated_serve_ignores_stale_staged_catalog_slug(self):
        """A stale `_staged_entry` from a PRIOR catalog serve must NOT drive the
        generated serve's slug (which could false-✓ a wrong model)."""
        wr = FakeWriteRunner()

        class _Stale:
            slug = "vllm/dual"

        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app._staged_entry = _Stale()
            plan = app._data.serve_generated("/tmp/c3-genc-y.yml")
            # Even with a stale staged entry, the generated lane yields no slug.
            assert app._serve_slug_for(plan) == ""
            # The app's serve-generated entry point clears the stale staged entry.
            app._serve_generated_compose("/tmp/c3-genc-y.yml")
            await _settle(pilot)
            assert app._staged_entry is None

    @pytest.mark.asyncio
    async def test_generated_serve_no_false_serving_when_slug_estate_matches(self):
        """Even if the estate registry-matches the stale catalog slug, the
        generated serve must NOT stamp '✓ serving <that model>' off it — the
        generated lane resolves ONLY from a new container appearing."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            emitted = _hook_serve_live(app)
            # Arm the generated lane with a baseline that ALREADY contains the
            # only container, and a state that registry-matches "vllm/dual".
            app._pending_serve_generated = True
            app._pending_serve_slug = ""
            app._pending_serve_baseline = {"vllm-qwen36-27b-dual"}
            tgt = ServingTarget(model="qwen3.6-27b", host_port=8010)
            state = EstateState(
                target=tgt,
                matched_slug="vllm/dual",     # estate matches the stale slug…
                containers=[ContainerInfo(name="vllm-qwen36-27b-dual", kind="engine",
                                          host_port=8010, slug="vllm/dual")],
            )
            app._resolve_pending_serve(state)
            await _settle(pilot)
            # NO terminal stamp — the only container was already in the baseline.
            assert not any("serving" in ln or "launched" in ln for ln in emitted)
            assert app._pending_serve_generated is True   # still pending

    @pytest.mark.asyncio
    async def test_generated_serve_reaches_terminal_when_container_appears(self):
        """When a NEW container (not in the launch baseline) appears, the generated
        serve stamps '✓ launched …' and stops the watcher."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            emitted = _hook_serve_live(app)
            app._pending_serve_generated = True
            app._pending_serve_slug = ""
            app._pending_serve_baseline = set()           # nothing before launch
            tgt = ServingTarget(model="", host_port=0)
            state = EstateState(
                target=tgt,
                matched_slug="",                          # BYO — no registry match
                containers=[ContainerInfo(name="c3-byo-generated", kind="engine",
                                          host_port=9099)],
            )
            app._resolve_pending_serve(state)
            await _settle(pilot)
            joined = " ".join(emitted)
            assert "launched" in joined
            assert "c3-byo-generated" in joined
            assert ":9099" in joined
            assert app._pending_serve_generated is False  # watcher resolved
            assert app._pending_serve_slug == ""

    @pytest.mark.asyncio
    async def test_generated_serve_times_out_honestly(self):
        """If no new container ever appears, the generated watcher times out to an
        honest 'still booting' line (not a hang)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            emitted = _hook_serve_live(app)
            app._pending_serve_generated = True
            app._pending_serve_slug = ""
            app._pending_serve_baseline = set()
            app._pending_serve_attempts = app._SERVE_REPOLL_MAX_ATTEMPTS
            app._poll_pending_serve()
            await _settle(pilot)
            assert any("still booting" in ln for ln in emitted)
            assert app._pending_serve_generated is False


class TestBatch2MustFix2RailAsOf:
    """MUST-FIX 2 — the rail as-of stamp reflects REAL elapsed time, re-rendered
    from cached state on a lightweight tick (not stuck on 'just now')."""

    @pytest.mark.asyncio
    async def test_as_of_shows_seconds_when_clock_advances(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            assert app._last_estate_state is not None      # cached for re-render
            # Pretend the last poll was 40s ago, then run the lightweight re-stamp.
            import time as _time
            app._last_estate_poll_mono = _time.monotonic() - 40
            app._refresh_rail_as_of()
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "ago" in rail
            assert "just now" not in rail

    @pytest.mark.asyncio
    async def test_as_of_shows_minutes_when_clock_advances_far(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            import time as _time
            app._last_estate_poll_mono = _time.monotonic() - 200
            app._refresh_rail_as_of()
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "m ago" in rail

    @pytest.mark.asyncio
    async def test_as_of_refresh_is_gated_to_operate(self):
        """The as-of re-stamp is a pure cached read but must stay Operate-gated
        (it's a no-op outside Operate)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0                   # Run
            # No cached state + not Operate → pure no-op, never raises.
            app._refresh_rail_as_of()


class TestBatch2MustFix3ErrorLabel:
    """MUST-FIX 3 — a detect-failure (docker fine) must NOT be mislabeled
    'docker unreachable' on the rail / Containers."""

    @pytest.mark.asyncio
    async def test_detect_failure_not_called_docker_unreachable(self):
        """Force the services.py detect-failure path → state.error is
        'detect failed: …' and the rail/Containers render THAT, not 'docker
        unreachable'."""
        async def _boom():
            raise RuntimeError("endpoint probe blew up")

        app, _, _ = make_app()
        # Swap the detect seam to raise → estate_state sets 'detect failed: …'.
        app._data._detect_endpoint = _boom
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            state = await app._data.estate_state(variants=app._variants or None)
            assert state.error.startswith("detect failed:")
            assert "docker unreachable" not in state.error
            # Render the rail + Containers off THIS state and assert the honest text.
            app.query_one("#rail-status", RailStatus).update_from_state(state, as_of="")
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "detect failed" in rail
            assert "docker unreachable" not in rail
            app.query_one("#operate-containers-pane", OperateContainersPane).populate(
                state.containers, state.error
            )
            tbl = app.query_one("#containers-table", DataTable)
            blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "detect failed" in blob
            assert "docker unreachable" not in blob

    @pytest.mark.asyncio
    async def test_docker_failure_still_labeled_docker(self):
        """Regression guard: a REAL docker failure still reads 'docker
        unreachable' (the headline = the part before ' — ')."""
        responses = fake_responses(
            **{"docker ps": RunResult(returncode=1, stdout="", stderr="daemon down")}
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "docker unreachable" in rail


class TestBatch2NH4NvidiaSmiError:
    """NH4 — a pure nvidia-smi failure sets state.error (no silent GPU-less rail)."""

    @pytest.mark.asyncio
    async def test_nvidia_smi_failure_sets_state_error(self):
        async def _no_gpus():
            raise RuntimeError("nvidia-smi: command not found")

        # Target with no gpus so estate_state falls through to _get_gpu_info.
        app, _, _ = make_app(gpus=[], target=ServingTarget(gpus=[]))
        app._data._get_gpu_info = _no_gpus
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            state = await app._data.estate_state(variants=app._variants or None)
            assert state.error                              # cue recorded
            assert "nvidia-smi" in state.error
            assert state.gpus == []

    @pytest.mark.asyncio
    async def test_nvidia_smi_error_does_not_clobber_docker_error(self):
        """A docker error is more specific — nvidia-smi failure must not overwrite
        it (the `if not state.error` guard)."""
        async def _no_gpus():
            raise RuntimeError("nvidia-smi gone")

        responses = fake_responses(
            **{"docker ps": RunResult(returncode=1, stdout="", stderr="daemon down")}
        )
        app, _, _ = make_app(responses=responses, gpus=[], target=ServingTarget(gpus=[]))
        app._data._get_gpu_info = _no_gpus
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            state = await app._data.estate_state(variants=app._variants or None)
            assert "docker unreachable" in state.error      # docker wins
            assert "nvidia-smi" not in state.error


class TestBatch2NH5WatcherArmFailure:
    """NH5 — if set_interval raises, the pending-serve state is cleared and the
    LivePane gets an honest 'could not arm watcher' line (no hang)."""

    @pytest.mark.asyncio
    async def test_watcher_arm_failure_clears_pending_and_stamps(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            emitted = _hook_serve_live(app)
            # Make set_interval raise just for the watcher-arm call.
            def _boom(*a, **k):
                raise RuntimeError("timer subsystem down")
            app.set_interval = _boom
            plan = app._data.serve("vllm/dual")
            app._start_pending_serve_watch(plan)
            await _settle(pilot)
            assert app._pending_serve_slug == ""            # cleared, no hang
            assert app._pending_serve_generated is False
            assert app._pending_serve_timer is None
            assert any("could not arm watcher" in ln for ln in emitted)


# ===========================================================================
# UX Batch 3 — honesty + serving actions (live-rig audit)
# ===========================================================================


class TestA6CatalogLiveFreeVram:
    """A6: the catalog fit-gate is computed against the TOTAL card; downgrade the
    DISPLAYED glyph against the LIVE per-GPU free-VRAM so a "fits-clean" row that
    would OOM right now is not shown as a clean live verdict.  Pure post-process
    of the verdict — kv-calc is NOT re-run."""

    @pytest.mark.asyncio
    async def test_row_downgraded_when_est_exceeds_live_free(self):
        # vllm/dual's per-card vram_est is 19.881 G (from FIT_ALL_JSON); make GPU0
        # hold a big tenant so only ~5 G is free → the dual row can't fit BOTH
        # cards now → it must NOT render the clean "●" glyph.
        gpus = [
            GpuInfo(index=0, mem_used_mib=19000, mem_total_mib=24576),  # ~5.4 G free
            GpuInfo(index=1, mem_used_mib=1000, mem_total_mib=24576),
        ]
        target = ServingTarget(gpus=gpus)
        app, _, _ = make_app(gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")  # Operate poll feeds the live free-VRAM to Run
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            # The fit verdict itself is still fits-clean (kv-calc unchanged)…
            entry = next(e for e in pane._entries if e.slug == "vllm/dual")
            assert entry.fit.verdict == "fits-clean"
            # …but the DISPLAYED glyph is downgraded (not the clean ● live).
            from club3090_cockpit.data import downgrade_fit_glyph
            glyph, note = downgrade_fit_glyph(entry.fit, entry.row, app._live_free_gb_by_index(app._last_estate_state))
            assert glyph in ("⚠", "✗")
            assert note  # carries a reason
            # And the live data path is actually populated.
            assert pane._free_gb_by_index is not None

    @pytest.mark.asyncio
    async def test_column_labelled_vs_empty_card_when_no_live_data(self):
        # No GPUs read → free-VRAM unknown → the fit column must read "vs empty
        # card" so "fits-clean" is never mistaken for a live verdict.
        app, _, _ = make_app(gpus=[], target=ServingTarget(gpus=[]))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._free_gb_by_index is None
            status = str(app.query_one("#catalog-status", Label).render())
            assert "vs empty card" in status

    @pytest.mark.asyncio
    async def test_clean_when_live_free_is_ample(self):
        # Both cards nearly empty → 19.881 G fits → glyph stays the clean ●.
        gpus = [
            GpuInfo(index=0, mem_used_mib=500, mem_total_mib=24576),
            GpuInfo(index=1, mem_used_mib=500, mem_total_mib=24576),
        ]
        app, _, _ = make_app(gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            entry = next(e for e in pane._entries if e.slug == "vllm/dual")
            from club3090_cockpit.data import downgrade_fit_glyph
            glyph, note = downgrade_fit_glyph(entry.fit, entry.row, pane._free_gb_by_index)
            assert glyph == "●"
            assert note == ""

    @pytest.mark.asyncio
    async def test_serving_model_own_row_not_false_wont_fit(self):
        """MUST-FIX 1: nvidia-smi mem_used INCLUDES the running model's own
        allocation, so free = total − used already nets out the serving row's
        ~20 G.  Comparing the row's est against that residual free falsely stamps
        "✗ won't fit now" while it is PROVABLY serving.  The serving slug's OWN
        catalog row is EXEMPT from the live-VRAM downgrade → its base glyph (●)."""
        # vllm/dual is the matched serving slug; both cards hold ~20 G (the running
        # model) so only ~4 G is free — the exact "free < est" trap the bug hit.
        gpus = [
            GpuInfo(index=0, mem_used_mib=20000, mem_total_mib=24576),  # ~4.5 G free
            GpuInfo(index=1, mem_used_mib=20000, mem_total_mib=24576),
        ]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")            # Operate poll → sets serving slug + free-VRAM
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._serving_slug == "vllm/dual"   # it IS the serving row
            assert pane._free_gb_by_index is not None   # live free-VRAM is low
            tbl = app.query_one("#catalog-table", DataTable)
            # Find the serving row + read its fit cell (column index 2).
            serving_fit = None
            for r in range(tbl.row_count):
                row = [str(c) for c in tbl.get_row_at(r)]
                if "serving" in row[0]:
                    serving_fit = row[2]
                    break
            assert serving_fit is not None
            # NOT downgraded — no ✗/⚠ and no "won't fit"/"tight" reason on its own row.
            assert "✗" not in serving_fit and "⚠" not in serving_fit
            assert "won't fit" not in serving_fit and "tight" not in serving_fit
            assert "●" in serving_fit                   # stays the base fits-clean glyph

    @pytest.mark.asyncio
    async def test_non_serving_row_still_downgraded_when_serving_exempt(self):
        """Sibling non-serving rows are still honestly downgraded when live free is
        low — the exemption is scoped to the serving row ONLY, not a blanket skip."""
        gpus = [
            GpuInfo(index=0, mem_used_mib=20000, mem_total_mib=24576),  # ~4.5 G free
            GpuInfo(index=1, mem_used_mib=20000, mem_total_mib=24576),
        ]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            # The serving exemption holds, but a non-serving fits-clean sibling that
            # genuinely can't fit the residual free IS downgraded.  (vllm/dual is the
            # only fits-clean slug here; assert the exemption + downgrade machinery
            # agree by checking the helper directly on a non-serving copy.)
            from club3090_cockpit.data import downgrade_fit_glyph
            entry = next(e for e in pane._entries if e.slug == "vllm/dual")
            glyph, note = downgrade_fit_glyph(entry.fit, entry.row, pane._free_gb_by_index)
            assert glyph in ("⚠", "✗") and note   # the raw helper still downgrades


class TestA7ServingPanelProbedConfig:
    """A7: the serving panel shows the ACTUAL probed running config (ctx + image),
    not the catalog slug's claim, and BADGES a divergence when the probed ctx
    differs from the matched slug's claimed ctx.  The probe is a READ (mocked)."""

    @pytest.mark.asyncio
    async def test_serving_panel_shows_probed_ctx_and_image(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        # Probe reports the catalog-claimed 262144 → NO divergence; image shown.
        probe = ServedProbe(max_model_len=262144, image="vllm/vllm-openai:v0.22.0")
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, probe_served=probe)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            assert "running" in line               # probed ctx is labelled running
            assert "262K" in line                   # 262144 → "262K" (÷1000, matches catalog)
            assert "vllm/vllm-openai:v0.22.0" in line
            assert "config differs from catalog slug" not in line  # no divergence

    @pytest.mark.asyncio
    async def test_serving_panel_badges_divergence(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        # Probe reports 131072, but the slug (vllm/dual) claims 262K → divergence.
        probe = ServedProbe(max_model_len=131072, image="vllm/vllm-openai:v0.22.0")
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, probe_served=probe)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            assert app._target_slug == "vllm/dual"
            line = str(app.query_one("#serving-line", Static).render())
            assert "131K" in line                          # 131072 → "131K" (÷1000)
            assert "config differs from catalog slug" in line
            assert "vllm/dual" in line

    @pytest.mark.asyncio
    async def test_serving_panel_falls_back_to_claim_when_no_probe_ctx(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        # Probe returned NO ctx (e.g. llama.cpp) → fall back to the catalog claim,
        # clearly labelled "(per catalog slug)" — never presented as measured.
        probe = ServedProbe(max_model_len=None, image="ghcr.io/ggml-org/llama.cpp:server-cuda")
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, probe_served=probe)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            assert "per catalog slug" in line
            assert "ghcr.io/ggml-org/llama.cpp:server-cuda" in line

    @pytest.mark.asyncio
    async def test_no_false_badge_when_fit_ceiling_exceeds_configured_ctx(self):
        """MUST-FIX 2: the badge must compare the probe against the slug's
        CONFIGURED ctx (registry max_ctx = 262144), NOT the kv-calc CAPACITY
        ceiling (fit.max_ctx).  Here fit.max_ctx is 295000 (the 295K ceiling) but
        the slug is configured 262144 — an honest 262144 serve must NOT badge."""
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        # fit.max_ctx (capacity ceiling) DIFFERS from configured_ctx (262144).
        fit_all_295k = json.dumps(
            {
                "card": "rtx-3090",
                "card_vram_gb": 24.0,
                "variants": {
                    "vllm/dual": {"verdict": "fits-clean", "vram_est_gb": 19.881, "band_gb": 1.5, "max_ctx": 295000},
                    "ik-llama/iq4ks-mtp": {"verdict": "skip"},
                },
            }
        )
        responses = fake_responses(
            **{"docker ps": ok(DOCKER_PS_ENGINE), "kv-calc.py --fit-all": ok(fit_all_295k)}
        )
        # Probe reports exactly the CONFIGURED 262144 — within tolerance of config.
        probe = ServedProbe(max_model_len=262144, image="vllm/vllm-openai:v0.22.0")
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, probe_served=probe)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            # Confirm the fit ceiling really is the 295K (the value the OLD code
            # wrongly compared against — proving the fix is load-bearing).
            entry = app._catalog_entry_for("vllm/dual")
            assert entry is not None and entry.fit.max_ctx == 295000
            assert entry.configured_ctx == 262144
            line = str(app.query_one("#serving-line", Static).render())
            assert "262K" in line
            assert "config differs from catalog slug" not in line  # NO false badge

    @pytest.mark.asyncio
    async def test_fallback_path_no_false_badge_before_enrich(self):
        """MUST-FIX 2 (b): the fallback path (no enriched catalog entry yet, so no
        numeric configured_ctx from fit) must NOT false-badge when probed ==
        configured.  The configured int comes from the registry VariantRow
        (configured_ctx=262144); probe 262144 → exact match → no badge."""
        from club3090_cockpit.data import EstateState as _ES

        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        probe = ServedProbe(max_model_len=262144, image="vllm/vllm-openai:v0.22.0")
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, probe_served=probe)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Drive the OperateOrchPane.populate directly with the EXACT-INT
            # configured ctx (262144) but NO display label and NO fit-derived
            # numeric — the pre-enrich fallback shape.
            from club3090_cockpit.data import ServedProbe as _SP
            state = _ES(
                target=target,
                matched_slug="vllm/dual",
                served=_SP(max_model_len=262144, image="vllm/vllm-openai:v0.22.0"),
            )
            pane = app.query_one("#operate-orch-pane", OperateOrchPane)
            pane.populate(state, catalog_ctx_label="262K", catalog_ctx=262144)
            await pilot.pause()
            line = str(app.query_one("#serving-line", Static).render())
            assert "262K" in line
            assert "config differs from catalog slug" not in line  # no false-fire

    @pytest.mark.asyncio
    async def test_running_ctx_label_matches_catalog_ctx_label(self):
        """MUST-FIX 3: the serving-panel running-ctx label and the catalog ctx
        label must read IDENTICALLY for the same int (one ÷1000 K-convention)."""
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        probe = ServedProbe(max_model_len=262144, image="vllm/vllm-openai:v0.22.0")
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, probe_served=probe)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            # Serving panel running-ctx label.
            serving_line = str(app.query_one("#serving-line", Static).render())
            assert "262K (running)" in serving_line.replace("[dim]", "").replace("[/dim]", "")
            # Catalog row ctx label for the SAME int.
            tbl = app.query_one("#catalog-table", DataTable)
            cat_blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "262K" in cat_blob
            # And the OLD ÷1024 form ("256K") appears NOWHERE — the convention is unified.
            assert "256K" not in serving_line and "256K" not in cat_blob


class TestA4TargetedServingVerbs:
    """A4: targeted stop / restart / switch on the #serving-line — NOT just [o]
    stop-ALL.  Stop/restart resolve the serving container by matched slug and open
    the SAME confirm gate (never auto-fired); they no-op with no model serving."""

    @pytest.mark.asyncio
    async def test_serving_stop_resolves_container_and_opens_gate(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        wr = FakeWriteRunner()
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await pilot.pause()
            await pilot.press("k")     # stop THIS model
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            plan = app.screen._plan
            # Resolves the matched-slug container, NOT a stop-all.
            assert plan.cmd[:2] == ["docker", "stop"]
            assert "vllm-qwen36-27b-dual" in plan.cmd
            assert wr.started == []     # nothing auto-fired

    @pytest.mark.asyncio
    async def test_serving_restart_opens_gate(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await pilot.pause()
            await pilot.press("b")     # restart serving
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd[:2] == ["docker", "restart"]

    @pytest.mark.asyncio
    async def test_serving_stop_noops_with_no_model(self):
        # Nothing serving (empty docker ps, no target) → [k] must NOT open the gate.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await pilot.pause()
            await pilot.press("k")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)

    @pytest.mark.asyncio
    async def test_serving_switch_jumps_to_run_catalog(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await pilot.pause()
            await pilot.press("n")     # switch → Run · Catalog (navigation, no write)
            await pilot.pause()
            assert "active" in app.query_one("#panel-run").classes
            assert app.query_one("#run-tabs", TabbedContent).active == "tab-catalog"
            assert not isinstance(app.screen, ConfirmActionScreen)  # no write here


class TestDoctorRerunAndRemediation:
    """#4: a Doctor-resident key re-runs the three READ-only diagnose reads on
    demand.  N7: a surfaced issue OFFERS the obvious remediation pointer."""

    @pytest.mark.asyncio
    async def test_doctor_rerun_triggers_doctor_read(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            # Clear the call log, then press [y] — it must re-run the diagnose reads.
            runner.calls.clear()
            await pilot.press("y")
            await _settle(pilot)
            assert any("diagnose-estate.sh --json" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_doctor_health_offers_remediation_when_unreachable(self):
        # An unreachable endpoint surfaces the obvious next action (serve a model),
        # not just the symptom.
        from club3090_cockpit.data import DoctorRead
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#doctor-pane", DoctorPane)
            pane._render_health(DoctorRead(reachable=False))
            body = str(app.query_one("#doctor-health-body", Static).render())
            assert "not reachable" in body
            assert "fix:" in body and "Run · Catalog" in body


class TestA9GateLadderOutcome:
    """A9: each ③ Gate ladder row shows its last-run outcome glyph (·/⟳/✓/✗/⚠),
    cached per kind, so the producer can answer 'have I cleared the gate?'."""

    @pytest.mark.asyncio
    async def test_ladder_row_shows_passed_glyph_from_last_run(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")           # Bring & Validate lane
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            pane = app.query_one("#validate-run-pane", ValidateRunPane)
            pane.set_run_outcome("verify-full", "passed")
            pane.set_run_outcome("bench", "failed")
            await pilot.pause()
            assert pane._outcomes["verify-full"] == "passed"
            table = app.query_one("#run-ladder-table", DataTable)
            # Row 0 is verify-full (first _RUN_LADDER entry) → ✓ in the "last" col.
            cell0 = str(table.get_cell_at((0, 0)))
            assert "✓" in cell0
            # bench is the 3rd ladder row → ✗.
            cell2 = str(table.get_cell_at((2, 0)))
            assert "✗" in cell2

    @pytest.mark.asyncio
    async def test_ladder_default_outcome_is_unrun(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            table = app.query_one("#run-ladder-table", DataTable)
            cell0 = str(table.get_cell_at((0, 0)))
            assert "·" in cell0              # unrun

    @pytest.mark.asyncio
    async def test_ladder_resolves_only_after_done_event(self):
        """MUST-FIX 4: run_validation returns the state RIGHT AFTER spawning
        (verdict=='', exit_code=None) — the real verdict is written only when the
        detached reader finishes and sets state.done.  The launch must AWAIT that
        before reading the verdict, so the ladder row flips ⟳→✓ ONLY after done is
        set, never staying stuck at ⟳ on a completed run."""

        class _AsyncState:
            def __init__(self):
                self.verdict = ""
                self.exit_code = None
                self.done = asyncio.Event()

        class _AsyncWriteRunner:
            """start_raw returns a still-running state (done NOT set); the test
            sets verdict + signals done out-of-band to mimic _read_output."""

            def __init__(self):
                self.state = _AsyncState()
                self.started: list[dict[str, Any]] = []

            def set_callbacks(self, on_event=None, on_line=None, on_complete=None):
                pass

            async def start_raw(self, cmd, env, run_type, parser):
                self.started.append({"cmd": cmd, "run_type": run_type})
                return self.state

        wr = _AsyncWriteRunner()
        app, _, _ = make_app(write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            pane = app.query_one("#validate-run-pane", ValidateRunPane)

            # Launch — the worker spawns then AWAITS done.wait(); it must NOT resolve.
            app.run_validation_launch("verify-full")
            # Let the worker reach the await (it's parked on done.wait()).
            for _ in range(5):
                await pilot.pause()
            assert wr.started, "start_raw should have been called"
            # Still in flight → the row stays at ⟳ (running), NOT a fabricated pass.
            assert pane._outcomes["verify-full"] == "running"

            # The detached reader finishes: writes the verdict, then sets done.
            wr.state.verdict = "passed"
            wr.state.done.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # NOW the row resolves to ✓ (passed) — only after done fired.
            assert pane._outcomes["verify-full"] == "passed"

    @pytest.mark.asyncio
    async def test_ladder_resolves_failed_after_done_event(self):
        """MUST-FIX 4 (failed leg): a real run that fails flips ⟳→✗ after done."""

        class _AsyncState:
            def __init__(self):
                self.verdict = ""
                self.exit_code = None
                self.done = asyncio.Event()

        class _AsyncWriteRunner:
            def __init__(self):
                self.state = _AsyncState()
                self.started: list[dict[str, Any]] = []

            def set_callbacks(self, on_event=None, on_line=None, on_complete=None):
                pass

            async def start_raw(self, cmd, env, run_type, parser):
                self.started.append({"cmd": cmd})
                return self.state

        wr = _AsyncWriteRunner()
        app, _, _ = make_app(write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            pane = app.query_one("#validate-run-pane", ValidateRunPane)
            app.run_validation_launch("verify-full")
            for _ in range(5):
                await pilot.pause()
            assert pane._outcomes["verify-full"] == "running"
            wr.state.verdict = "failed"
            wr.state.done.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert pane._outcomes["verify-full"] == "failed"

    def test_run_verdict_never_fabricates_a_pass(self):
        """MUST-FIX 4 unit: _run_verdict maps verdict=='failed'→failed,
        exit_code==0→passed, and a bare dict / None state (no verdict, no
        exit_code) → 'running' — NEVER a fabricated pass."""
        from club3090_cockpit.app import CockpitApp

        class _S:
            def __init__(self, verdict="", exit_code=None):
                self.verdict = verdict
                self.exit_code = exit_code

        assert CockpitApp._run_verdict(_S(verdict="failed")) == "failed"
        assert CockpitApp._run_verdict(_S(verdict="passed")) == "passed"
        assert CockpitApp._run_verdict(_S(exit_code=0)) == "passed"
        assert CockpitApp._run_verdict(_S(exit_code=1)) == "failed"
        # The mock-phase shapes: a bare dict and a None state carry no verdict →
        # 'running' (never a fabricated pass).
        assert CockpitApp._run_verdict({"mock_state": True}) == "running"
        assert CockpitApp._run_verdict(None) == "running"


# ===========================================================================
# UX Batch 4a — navigation & discoverability (A5 · #5 · #8 · N6 · A11)
# ===========================================================================


class TestA5HelpTeachesHiddenKeys:
    """A5 — the help overlay teaches the otherwise-undiscoverable keys: the
    sub-tab cycle ([ ]) and the Contribute door (C).  Both must appear on the
    CONSUMER help (C is always-on — a consumer needs it to opt in)."""

    @pytest.mark.asyncio
    async def test_consumer_help_lists_subtab_cycle_keys(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            text = app.screen.help_text
            # The bracket keys (the only no-mouse sub-tab move) are taught.
            assert "previous / next sub-tab" in text
            assert "[" in text and "]" in text
            # A Navigation section heading anchors them.
            assert "Navigation" in text

    @pytest.mark.asyncio
    async def test_consumer_help_lists_contribute_key(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            text = app.screen.help_text
            # The [C] Contribute door is taught on CONSUMER (it is the opt-in).
            assert "toggle Contribute" in text
            assert "[cyan]C[/cyan]" in text

    @pytest.mark.asyncio
    async def test_consumer_help_lists_rail_toggle_key(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            text = app.screen.help_text
            assert "toggle the left rail" in text

    @pytest.mark.asyncio
    async def test_help_keys_are_real_bindings(self):
        """Every cyan key the help advertises must be a real app Binding /
        action so the help can't drift into teaching a dead key."""
        # The bindings the help references by action.
        bound_actions = {b.action for b in CockpitApp.BINDINGS}
        for action in ("toggle_contribute", "toggle_rail",
                       "prev_subtab", "next_subtab", "help", "refresh"):
            assert action in bound_actions, action


class TestHashFiveFooterTabFocus:
    """#5 — the footer's action keys are reachable via the Tab focus chain."""

    @pytest.mark.asyncio
    async def test_footer_keys_are_focusable_and_in_chain(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ff = app.query_one(FocusableFooter)
            keys = list(ff.query(FooterKey))
            assert keys, "footer has key items"
            assert all(k.can_focus for k in keys)
            chain = app.screen.focus_chain
            assert any(isinstance(w, FooterKey) for w in chain), \
                "at least one FooterKey participates in the Tab focus chain"

    @pytest.mark.asyncio
    async def test_focus_next_lands_on_a_footer_key(self):
        """Walking the focus chain (what Tab does) reaches a footer key."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # focus_next is exactly the action Screen binds to Tab.
            landed = app.screen.focus_next()
            # If the first focusable wasn't a footer key, keep walking until we
            # either hit one or exhaust the chain (no intervening pause → no
            # recompose churn).
            seen = []
            for _ in range(len(app.screen.focus_chain) + 1):
                cur = app.screen.focused
                seen.append(cur)
                if isinstance(cur, FooterKey):
                    break
                landed = app.screen.focus_next()
            assert any(isinstance(w, FooterKey) for w in seen), \
                "Tab traversal reaches a focusable footer key"

    @pytest.mark.asyncio
    async def test_enter_on_focused_footer_key_activates_it(self):
        """A focused footer key activates on Enter (keyboard analogue of the
        stock mouse-down activation) — the footer's on_key routes through
        simulate_key, firing the SAME action the binding would.  Asserts the
        EFFECT: 'r' = refresh, which re-reads the catalog, so the data layer's
        load_catalog_rows is called exactly once more (mirrors TestA11's
        one-start fidelity — a broken on_key would no longer pass).

        Drives the production FocusableFooter.on_key handler directly with a
        synthesized Enter event: a nested ``simulate_key`` posted from *inside*
        Textual's own key-event processing (what ``pilot.press('enter')`` would
        trigger) is swallowed by the focused FooterKey in the headless harness,
        so we invoke the handler from the test coroutine to exercise the same
        code path without the harness's re-entrancy quirk."""
        from textual import events
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ff = app.query_one(FocusableFooter)
            # Spy the data layer's catalog read so we can count refresh re-reads.
            calls = {"n": 0}
            orig = app._data.load_catalog_rows

            async def counting_load_catalog_rows():
                calls["n"] += 1
                return await orig()

            app._data.load_catalog_rows = counting_load_catalog_rows
            # Focus the 'r' (refresh) footer key — the Tab-reachable affordance.
            chain = app.screen.focus_chain
            rkey = next(k for k in chain
                        if isinstance(k, FooterKey) and k.key == "r")
            rkey.focus()
            await pilot.pause()
            assert app.focused is rkey
            before = calls["n"]
            # Enter on the focused key → footer on_key → simulate_key('r') →
            # action_refresh → exactly one catalog read.
            ff.on_key(events.Key("enter", None))
            await _settle(pilot)
            assert calls["n"] == before + 1, (
                "Enter on the focused 'r' footer key fires refresh, re-reading "
                "the catalog exactly once"
            )

    @pytest.mark.asyncio
    async def test_tab_does_not_break_table_focus(self):
        """The catalog DataTable stays focusable + in the chain — #5 only ADDS
        footer keys to the chain, it doesn't strip the existing widgets."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            chain = app.screen.focus_chain
            assert any(isinstance(w, DataTable) for w in chain)

    @pytest.mark.asyncio
    async def test_focused_footer_key_survives_a_real_binding_change(self):
        """The bindings_changed focus-held early-return: while a FooterKey holds
        focus, a GENUINE displayed-binding change (a surface flip, which adds the
        [3] Bring & Validate mode key to the footer) must NOT recompose the
        footer — the focused key keeps its identity + focus, and the footer
        recomposes only once focus leaves it."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ff = app.query_one(FocusableFooter)
            # Focus the '1' mode key (present on BOTH surfaces, so its identity is
            # a clean witness for "did the footer get rebuilt?").
            onekey = next(k for k in ff.query(FooterKey) if k.key == "1")
            onekey.focus()
            await pilot.pause()
            assert app.focused is onekey
            sig_before = ff._last_binding_sig
            id_before = id(onekey)
            # Sanity: '3' (Bring & Validate) is NOT shown on the consumer surface.
            assert "3" not in {k.key for k in ff.query(FooterKey)}
            # Trigger a REAL displayed-binding change without re-homing focus:
            # flipping to the producer surface UN-gates [3] (mode_validate), which
            # IS show=True, so the footer signature genuinely changes.  Fires
            # bindings_changed while the '1' FooterKey still holds focus.
            app._surface = "producer"
            app.refresh_bindings()
            await pilot.pause()
            # Focus-held branch took the early return: footer NOT rebuilt — same
            # FooterKey object, focus retained, snapshot signature untouched, and
            # the new [3] key has NOT been composed in yet (recompose suppressed).
            assert app.focused is onekey
            assert id(app.focused) == id_before
            assert ff._last_binding_sig == sig_before
            assert app.query_one(FocusableFooter) is ff   # no crash / re-creation
            assert "3" not in {k.key for k in ff.query(FooterKey)}
            # Move focus off the footer onto the table, then a binding change DOES
            # recompose (the suppression only holds while a FooterKey is focused).
            table = next(w for w in app.screen.focus_chain
                         if isinstance(w, DataTable))
            table.focus()
            await pilot.pause()
            assert not isinstance(app.focused, FooterKey)
            app.refresh_bindings()
            await _settle(pilot)
            # Footer recomposed for the new (producer) signature — the stale
            # snapshot was replaced + [3] is now composed in, now that no footer
            # key pinned it.
            assert ff._last_binding_sig != sig_before
            assert "3" in {k.key for k in ff.query(FooterKey)}


class TestHashEightRailToggle:
    """#8 — collapse / restore the left rail; mode keys still work hidden."""

    @pytest.mark.asyncio
    async def test_dot_toggles_left_rail_hidden_and_back(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            rail = app.query_one("#left-rail")
            assert "rail-hidden" not in rail.classes
            await pilot.press("full_stop")
            await pilot.pause()
            assert "rail-hidden" in rail.classes
            await pilot.press("full_stop")
            await pilot.pause()
            assert "rail-hidden" not in rail.classes

    @pytest.mark.asyncio
    async def test_mode_keys_work_while_rail_hidden(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("full_stop")   # hide rail
            await pilot.pause()
            await pilot.press("2")           # Operate
            await _settle(pilot)
            assert app._active_mode == 1
            await pilot.press("1")           # back to Run
            await _settle(pilot)
            assert app._active_mode == 0
            # Rail is still hidden — the toggle is independent of mode switching.
            assert "rail-hidden" in app.query_one("#left-rail").classes

    @pytest.mark.asyncio
    async def test_hiding_rail_does_not_strand_focus(self):
        """Hiding the rail re-homes focus to the active mode's primary widget so
        no hidden widget is left holding focus."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("full_stop")
            await pilot.pause()
            focused = app.focused
            # Focus must NOT be inside the now-hidden rail subtree.
            if focused is not None:
                node = focused
                in_rail = False
                while node is not None:
                    if getattr(node, "id", "") == "left-rail":
                        in_rail = True
                        break
                    node = node.parent
                assert not in_rail


class TestN6CommandPalette:
    """N6 — Textual command palette wired to the app's actions, surface-gated."""

    def test_palette_provider_registered(self):
        assert CockpitCommands in CockpitApp.COMMANDS

    def test_palette_commands_resolve_to_real_actions(self):
        """Drift guard: every _PALETTE_COMMANDS verb must map to a real
        action_<name> method — a future rename would otherwise ship a
        fuzzy-findable no-op."""
        for action, title, _help in _PALETTE_COMMANDS:
            assert hasattr(CockpitApp, f"action_{action}"), (
                f"palette command '{title}' → action_{action} has no method"
            )

    @pytest.mark.asyncio
    async def test_palette_lists_core_actions_on_consumer(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            prov = CockpitCommands(app.screen)
            names = [a for a, _, _ in prov._available()]
            # A solid consumer core set.
            for action in ("mode_run", "mode_operate", "primary_action",
                           "rig_report", "toggle_rail"):
                assert action in names, action

    @pytest.mark.asyncio
    async def test_palette_gates_producer_actions_on_consumer(self):
        """A producer-only action is NOT offered on the consumer surface (the
        palette respects the same surface gate as check_action)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            prov = CockpitCommands(app.screen)
            names = [a for a, _, _ in prov._available()]
            for action in _PALETTE_PRODUCER_ONLY:
                assert action not in names, action
            # …but every producer-only palette action IS a real _PRODUCER_ONLY
            # action (the two sets agree — no drift).
            assert _PALETTE_PRODUCER_ONLY <= set(CockpitApp._PRODUCER_ONLY)

    @pytest.mark.asyncio
    async def test_palette_exposes_producer_actions_on_producer(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            prov = CockpitCommands(app.screen)
            names = [a for a, _, _ in prov._available()]
            assert "promote_catalog" in names
            assert "serve_untested" in names

    @pytest.mark.asyncio
    async def test_palette_search_runs_a_core_action(self):
        """Selecting a palette hit invokes the SAME action method the binding
        would — fuzzy-search 'Operate mode' then run it → mode 1."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            prov = CockpitCommands(app.screen)
            cb = None
            async for hit in prov.search("Operate mode"):
                cb = hit.command
                break
            assert cb is not None
            assert app._active_mode == 0
            cb()
            await _settle(pilot)
            assert app._active_mode == 1

    @pytest.mark.asyncio
    async def test_palette_discover_yields_consumer_set(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            prov = CockpitCommands(app.screen)
            titles = []
            async for hit in prov.discover():
                titles.append(hit.text if hasattr(hit, "text") else str(hit))
            assert titles, "discover surfaces a default set"
            # No producer-only title leaks onto consumer discover.
            joined = " ".join(titles)
            assert "Promote to catalog" not in joined


class TestA11ConfirmModalDiscoverableBindings:
    """A11 — Enter/Force are discoverable BINDINGS (show=True) in the modal
    footer; Force visibility follows plan safety; behaviour is UNCHANGED."""

    @pytest.mark.asyncio
    async def test_modal_has_enter_force_escape_bindings(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            actions = {b.action: b for b in app.screen.BINDINGS}
            assert "confirm" in actions and actions["confirm"].show is True
            assert "force" in actions and actions["force"].show is True
            assert "cancel" in actions  # Esc still cancels

    @pytest.mark.asyncio
    async def test_safe_gate_shows_confirm_hides_force(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            sc = app.screen
            assert sc._reconcile.safe is True
            # Confirm visible (gate safe), Force hidden (forcing meaningless).
            assert sc.check_action("confirm", ()) is True
            assert sc.check_action("force", ()) is False
            footer_keys = {k.key for k in sc.query(FooterKey)}
            assert "enter" in footer_keys      # Enter Confirm advertised
            assert "f" not in footer_keys      # Force hidden on a safe gate

    @pytest.mark.asyncio
    async def test_unsafe_gate_hides_confirm_shows_force(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            sc = app.screen
            assert sc._reconcile.safe is False
            assert sc.check_action("confirm", ()) is False
            assert sc.check_action("force", ()) is True
            footer_keys = {k.key for k in sc.query(FooterKey)}
            assert "f" in footer_keys          # Force advertised on unsafe gate
            assert "enter" not in footer_keys  # Confirm hidden when unsafe

    @pytest.mark.asyncio
    async def test_enter_still_commits_through_the_same_gate(self):
        """Enter (now a Binding) commits through the SAME reconcile gate — the
        mocked write runner sees exactly one start, no live process."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            await pilot.press("enter")
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/switch.sh", "vllm/dual"]

    @pytest.mark.asyncio
    async def test_f_still_forces_through_the_same_gate(self):
        """f (now a Binding) forces through the SAME gate — --force inserted,
        still via the mocked runner."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            await pilot.press("f")
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "--force" in wr.started[0]["cmd"]

    @pytest.mark.asyncio
    async def test_f_is_inert_on_a_safe_gate(self):
        """On a SAFE gate the Force binding is gated off (check_action False) —
        pressing f must NOT force-launch (behaviour unchanged from the old
        on_key guard that checked the disabled button)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            assert app.screen.check_action("force", ()) is False
            await pilot.press("f")
            await _settle(pilot)
            # f did nothing (no forced launch); the safe path is Enter/Confirm.
            assert all("--force" not in c["cmd"] for c in wr.started)

    @pytest.mark.asyncio
    async def test_enter_is_inert_on_an_unsafe_gate(self):
        """SAFETY: on an UNSAFE gate the Confirm binding is gated off, and Enter
        must NOT fall through to Force and force-launch the teardown.  Pressing
        Enter starts no write at all; the explicit `f` key is the only force
        path and it STILL forces (the Force path is unchanged)."""
        wr = FakeWriteRunner()
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            sc = app.screen
            assert sc._reconcile.safe is False
            assert sc.check_action("confirm", ()) is False
            # Enter on an unsafe gate is INERT — no write started at all (and in
            # particular no --force teardown the gate guards).
            await pilot.press("enter")
            await _settle(pilot)
            assert wr.started == []
            # The explicit `f` key STILL forces — the Force path is unchanged.
            await pilot.press("f")
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "--force" in wr.started[0]["cmd"]


# ===========================================================================
# UX Batch 4b — preview & input ergonomics
#   #9/A8 Catalog preview · #11 Scene preview · N8 evidence/gate preview ·
#   #6/A12 profile-template Select + rig default + unknown-profile ·
#   N9 producer ①→② hand-off.  All previews are LOCAL reads (no I/O).
# ===========================================================================

from club3090_cockpit.app import (  # noqa: E402
    profile_templates,
    default_profile_template,
)
from club3090_tui_core import VariantRow  # noqa: E402

# A registry variant carrying a status_note caveat (⚠️ Production w/ caveats) so
# the Catalog preview can be asserted to surface the caveat text inline.
REGISTRY_JSON_CAVEAT = json.dumps(
    {
        "defaults": [],
        "profiles": {},
        "variants": [
            {
                "slug": "vllm/dual",
                "switch_engine": "vllm",
                "launch_engine": "vllm",
                "compose_dir": "models/qwen3.6-27b/vllm/compose/dual/autoround-int4",
                "file": "fp8-mtp.yml",
                "port": 8010,
                "model": "qwen3.6-27b",
                "engine": "vllm-stable",
                "kvcalc_key": "qwen3.6-27b:dual",
                "container": "vllm_qwen36_27b",
                "compose_path": "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
                "status": "caveats",
                "ctx_label": "262K",
                "configured_ctx": 262144,
                "status_note": "Cliff-2b at >50K accumulated ctx",
                "source": "curated",
            },
            {
                "slug": "vllm/single",
                "switch_engine": "vllm",
                "launch_engine": "vllm",
                "compose_dir": "models/qwen3.6-27b/vllm/compose/single/autoround-int4",
                "file": "base.yml",
                "port": 8011,
                "model": "qwen3.6-27b",
                "engine": "vllm-stable",
                "kvcalc_key": "qwen3.6-27b:single",
                "container": "vllm_qwen36_27b_single",
                "compose_path": "models/qwen3.6-27b/vllm/compose/single/autoround-int4/base.yml",
                "status": "production",
                "ctx_label": "120K",
                "configured_ctx": 120000,
                "status_note": "",
                "source": "curated",
            },
        ],
    }
)


class TestProfileTemplateDerivation:
    """#6/A12 — the pure profile-template derivation + rig-topology default."""

    def _mk(self, slug, engine, path):
        return VariantRow(
            slug=slug, switch_engine=engine, launch_engine=engine,
            compose_dir=path.rsplit("/", 1)[0], file=path.rsplit("/", 1)[1],
            port=8000, model="q", engine=engine, kvcalc_key="k", container="c",
            compose_path=path, status="production", ctx_label="262K", status_note="",
        )

    def test_every_slug_is_selectable_none_collapsed(self):
        # BLOCKER: the dropdown must emit ONE option per registry SLUG — the old
        # (engine, topology) collapse made siblings unreachable.  THREE vllm dual
        # slugs sharing (vllm-stable, dual) must ALL survive (real-registry shape).
        rows = [
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
            self._mk("vllm/qwen-27b-dual-max", "vllm-stable", "models/q/vllm/compose/dual/aq/int8.yml"),
            self._mk("vllm/qwen-27b-dual-fast", "vllm-stable", "models/q/vllm/compose/dual/aq/fast.yml"),
            self._mk("vllm/minimal", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
            self._mk("ik-llama/iq4ks-mtp", "ik-llama", "models/q/ik-llama/compose/single/u/mtp.yml"),
            self._mk("llamacpp/default", "llama-cpp-local", "models/q/llama-cpp/compose/single/u/default.yml"),
        ]
        opts = profile_templates(rows)
        values = [o.slug for o in opts]
        # NONE collapsed away — every distinct slug is a selectable option.
        for slug in (
            "vllm/dual", "vllm/qwen-27b-dual-max", "vllm/qwen-27b-dual-fast",
            "vllm/minimal", "ik-llama/iq4ks-mtp", "llamacpp/default",
        ):
            assert slug in values, slug
        assert len(values) == len(set(values)) == 6  # one option per slug, deduped
        # labels carry the engine/topology so a user can read what they pick.
        assert any("vllm-stable / dual" in o.label for o in opts)
        # topology is carried THROUGH on the option (not re-derived from the label).
        by_slug = {o.slug: o for o in opts}
        assert by_slug["vllm/dual"].topology == "dual"
        assert by_slug["vllm/minimal"].topology == "single"

    def test_rig_default_is_canonical_not_alphabetical_gemma(self):
        # BLOCKER: the rig-topology default must be the CANONICAL vllm slug for the
        # topology — NOT the alphabetically-first survivor (a Gemma/beellama slug).
        rows = [
            # alphabetically-first dual slug is a beellama/gemma one — the OLD
            # default_profile_template would have picked it.
            self._mk("beellama/gemma-q8-dflash-dual", "beellama-local", "models/q/beellama/compose/dual/q8/dflash.yml"),
            self._mk("vllm/qwen-27b-dual-max", "vllm-stable", "models/q/vllm/compose/dual/aq/int8.yml"),
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
            # alphabetically-first single slug is a beellama one.
            self._mk("beellama/dflash", "beellama-local", "models/q/beellama/compose/single/u/dflash.yml"),
            self._mk("vllm/single", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
        ]
        opts = profile_templates(rows)
        # 2 cards → the canonical vllm/dual, NOT beellama/gemma-q8-dflash-dual.
        assert default_profile_template(opts, 2) == "vllm/dual"
        # 1 card → the canonical vllm/single, NOT beellama/dflash.
        assert default_profile_template(opts, 1) == "vllm/single"

    def test_rig_default_falls_back_when_no_canonical_vllm_slug(self):
        # When no literal vllm/<topo> exists (the real registry has vllm/minimal,
        # not vllm/single, for single-card), prefer any vllm-prefixed slug of that
        # topology before falling back — never an arbitrary beellama/gemma slug.
        rows = [
            self._mk("beellama/dflash", "beellama-local", "models/q/beellama/compose/single/u/dflash.yml"),
            self._mk("vllm/minimal", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
        ]
        opts = profile_templates(rows)
        assert default_profile_template(opts, 1) == "vllm/minimal"

    def test_rig_topology_default_single_vs_dual(self):
        rows = [
            self._mk("vllm/single", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
        ]
        opts = profile_templates(rows)
        assert default_profile_template(opts, 1) == "vllm/single"
        assert default_profile_template(opts, 2) == "vllm/dual"


class TestCatalogPreview:
    """#9/A8 — Run · Catalog row-highlight renders an inline preview strip."""

    @pytest.mark.asyncio
    async def test_highlight_renders_status_note_fit_ctx(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            t = app.query_one("#catalog-table", DataTable)
            t.move_cursor(row=0)  # vllm/dual (the caveat row)
            await pilot.pause()
            prev = str(app.query_one("#catalog-preview", Static).render())
            # The caveat status_note appears inline (no longer Explain-only).
            assert "Cliff-2b at >50K" in prev
            # fit verdict + ctx are in the preview too.
            assert "fits-clean" in prev
            assert "262K" in prev
            assert "vllm/dual" in prev

    @pytest.mark.asyncio
    async def test_preview_updates_on_cursor_move(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            t = app.query_one("#catalog-table", DataTable)
            t.move_cursor(row=0)
            await pilot.pause()
            first = str(app.query_one("#catalog-preview", Static).render())
            assert "vllm/dual" in first
            t.move_cursor(row=1)  # vllm/single — different slug, no caveat
            await pilot.pause()
            second = str(app.query_one("#catalog-preview", Static).render())
            assert "vllm/single" in second
            assert "Cliff-2b" not in second  # the caveat is row-0's, not row-1's

    @pytest.mark.asyncio
    async def test_vs_empty_card_not_doubled_when_free_unknown(self):
        # N3 — with live free-VRAM UNKNOWN, the fit line must show "vs empty card"
        # exactly ONCE (the trailing "({fit_basis})"), not doubled by also
        # appending the downgrade note ("… vs empty card (vs empty card)").
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses, gpus=[], target=ServingTarget(gpus=[]))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.query_one("#catalog-pane", CatalogPane)._free_gb_by_index is None
            t = app.query_one("#catalog-table", DataTable)
            t.move_cursor(row=0)  # vllm/dual — fits-clean (no live downgrade)
            await pilot.pause()
            prev = str(app.query_one("#catalog-preview", Static).render())
            assert "fits-clean" in prev
            assert prev.count("vs empty card") == 1  # basis only, not doubled


class TestScenePreview:
    """#11 — Operate · Orchestration scene-row highlight renders a preview."""

    @pytest.mark.asyncio
    async def test_scene_highlight_shows_description_and_services(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=0)  # "27b" scene
            await pilot.pause()
            prev = str(app.query_one("#scene-preview", Static).render())
            assert "27b" in prev
            assert "Qwen" in prev                       # description
            assert "vllm-qwen36-27b-dual" in prev       # service it starts
            assert "8010" in prev                       # port

    @pytest.mark.asyncio
    async def test_scene_preview_updates_on_cursor_move(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=1)  # "off" scene
            await pilot.pause()
            prev = str(app.query_one("#scene-preview", Static).render())
            assert "off" in prev
            assert "Stop all" in prev


class TestEvidenceAndGatePreview:
    """N8 — ④ Measure evidence-tag + ③ Gate validation-step highlight previews."""

    @pytest.mark.asyncio
    async def test_evidence_highlight_shows_artifacts(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
            await pilot.pause()
            app.query_one("#evidence-table", DataTable).move_cursor(row=0)
            await pilot.pause()
            prev = str(app.query_one("#evidence-preview", Static).render())
            assert "vllm-dual-test" in prev      # the tag
            assert "REPORT.md" in prev           # artifact labels
            assert "_internal.json" in prev

    @pytest.mark.asyncio
    async def test_gate_step_highlight_shows_blurb(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            app.query_one("#run-ladder-table", DataTable).move_cursor(row=0)
            await pilot.pause()
            prev = str(app.query_one("#run-step-preview", Static).render())
            assert "verify-full" in prev                   # the step label
            assert "functional smoke" in prev              # its blurb (what it checks)

    @pytest.mark.asyncio
    async def test_gate_step_preview_updates_on_cursor_move(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            tbl = app.query_one("#run-ladder-table", DataTable)
            tbl.move_cursor(row=1)  # verify-stress
            await pilot.pause()
            prev = str(app.query_one("#run-step-preview", Static).render())
            assert "verify-stress" in prev
            assert "boundary matrix" in prev


class TestProfileSelectInputErgonomics:
    """#6/A12 — the BYO/① Bring profile input is a registry-derived Select that
    defaults to the rig topology + reports unknown profiles precisely."""

    @pytest.mark.asyncio
    async def test_byo_profile_is_select_with_registry_templates(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            sel = app.query_one("#byo-profile-input", Select)
            # The dropdown is fed from the SAME registry-derived templates.
            template_values = {o.slug for o in profile_templates(app._variants)}
            assert template_values == {"vllm/dual", "vllm/single"}
            # The selected value is a known registry slug, not free text.
            assert sel.value in template_values

    @pytest.mark.asyncio
    async def test_byo_defaults_to_single_on_one_card(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1)]
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Enter Operate so an estate poll learns the (1-card) GPU count, then
            # the dropdown re-defaults to a single-card template.
            await pilot.press("2")
            await _settle(pilot)
            sel = app.query_one("#byo-profile-input", Select)
            assert sel.value == "vllm/single"

    @pytest.mark.asyncio
    async def test_byo_defaults_to_dual_on_two_cards(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            sel = app.query_one("#byo-profile-input", Select)
            assert sel.value == "vllm/dual"

    @pytest.mark.asyncio
    async def test_manual_pick_survives_rig_default_reapply(self):
        # NICE-TO-HAVE 2 — a profile the user picked BEFORE the first estate poll
        # must NOT be clobbered by the rig-default reapply (estate-poll re-default).
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus), surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            sel = app.query_one("#byo-profile-input", Select)
            # User manually picks the non-default (single) template.
            sel.value = "vllm/single"
            await pilot.pause()
            assert app._profile_user_touched is True
            # Enter Operate → first estate poll fires reapply_default=True.
            await pilot.press("2")
            await _settle(pilot)
            # The manual pick STANDS — the reapply did NOT clobber it back to dual.
            assert app.query_one("#byo-profile-input", Select).value == "vllm/single"

    @pytest.mark.asyncio
    async def test_unknown_profile_reports_known_list(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # A legacy / free-text profile-like that isn't a known registry slug.
            app.run_byo_check("org/Model", "vllm/legacy-gone")
            await _settle(pilot)
            card = str(app.query_one("#byo-result-card", Static).render())
            assert "unknown profile vllm/legacy-gone" in card
            assert "known:" in card
            assert "vllm/dual" in card  # a real slug is listed

    @pytest.mark.asyncio
    async def test_known_selected_value_still_runs_byo_check(self):
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # The selected (known) value flows straight to pull.sh --dry-run.
            app.run_byo_check("org/Model", "vllm/dual")
            await _settle(pilot)
            pull = next(c for c in runner.calls if "pull.sh" in " ".join(c))
            assert "--dry-run" in pull
            assert "vllm/dual" in pull


class TestProducerLaneHandoff:
    """N9 — a successful ① Bring fit-check pre-arms ② Serve (no re-entry)."""

    @pytest.mark.asyncio
    async def test_serve_prearmed_after_fit_check(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            # Run ① Bring's fit-check (PULL_JSON resolves Route C → sibling vllm/dual).
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            # ② Serve is pre-armed with the resolved target WITHOUT re-navigating.
            body = str(app.query_one("#lane-serve-pane", LaneServePane).query_one(
                "#lane-serve-body", Static
            ).render())
            assert "armed from ① Bring" in body
            assert "vllm/dual" in body                       # resolved catalog target
            assert "unsloth/Qwen3-27B-abliterated" in body   # the brought repo

    @pytest.mark.asyncio
    async def test_bring_result_points_forward_to_serve(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            card = str(app.query_one("#lane-bring-pane", LaneBringPane).query_one(
                "#lane-bring-result-card", Static
            ).render())
            assert "→ ② Serve" in card
            assert "vllm/dual" in card

    @pytest.mark.asyncio
    async def test_serve_tab_rearms_from_cached_byo(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            # Navigate AWAY then back to ② Serve — it re-arms from the cached result.
            app.query_one("#validate-tabs", TabbedContent).active = "tab-bring"
            await pilot.pause()
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            body = str(app.query_one("#lane-serve-pane", LaneServePane).query_one(
                "#lane-serve-body", Static
            ).render())
            assert "armed from ① Bring" in body
            assert "vllm/dual" in body

    @pytest.mark.asyncio
    async def test_failed_rebring_clears_stale_armed(self):
        # N9 — after a valid ① Bring arms ② Serve with vllm/dual, a FAILED re-Bring
        # (unknown profile) must clear the stale "● armed …" so ② Serve restores the
        # "run ① Bring first" placeholder, consistent with the error _last_byo.
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            body = str(app.query_one("#lane-serve-pane", LaneServePane).query_one(
                "#lane-serve-body", Static
            ).render())
            assert "armed from ① Bring" in body  # armed by the valid Bring
            # Re-Bring with a typo'd / unknown profile → early-return error path.
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/typo-gone")
            await _settle(pilot)
            body = str(app.query_one("#lane-serve-pane", LaneServePane).query_one(
                "#lane-serve-body", Static
            ).render())
            assert "armed from ① Bring" not in body   # stale arm cleared
            assert "Run ① Bring first" in body        # placeholder restored
