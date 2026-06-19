"""Headless tests for the CockpitApp (Phase 3 — wired).

Verifies:
  1. The app mounts without error (no TTY, no GPU, no Docker, no live script).
     The data layer is a CockpitData backed by a FakeRunner + fake detect +
     a FakeWriteRunner, so NO subprocess is ever spawned.
  2. All four modes are reachable via digit-key bindings; nav nodes exist.
  3. Discover · Catalog populates from real enriched entries (fit glyph, TPS,
     8pk, source) and filters live.
  4. BYO renders the swap_path route from byo_check.
  5. Serve stages a plan and ⏎ opens the reconcile-gated confirm modal.
  6. Estate · Orchestration + Containers populate from estate_state.
  7. EVERY write path goes through the reconcile gate, and NO test ever
     executes a live write — the FakeWriteRunner records start_raw calls and
     never spawns a process; an unsafe gate refuses to even reach it.

The whole service layer is dependency-injected; tests never touch the real
RealRunner / SubprocessRunner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane, Label

from club3090_tui_core.detect import GpuInfo, ServingTarget

from club3090_cockpit.app import (
    CockpitApp,
    CatalogPane,
    ConfirmActionScreen,
    ExplainScreen,
    HelpScreen,
    ModeSwitcher,
    ByoPane,
    ServePane,
    EstateOrchPane,
    EstateContainersPane,
    ValidateRunPane,
    ValidateDoctorPane,
    ValidateBenchmarksPane,
    ValidateEvidencePane,
    EvidenceReportScreen,
    RailStatus,
)
from club3090_cockpit.services import CockpitData, RunResult


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


def ok(stdout: str) -> RunResult:
    return RunResult(returncode=0, stdout=stdout, stderr="")


def make_detect(target: ServingTarget):
    async def _detect() -> ServingTarget:
        return target
    return _detect


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
) -> tuple[CockpitApp, FakeRunner, FakeWriteRunner]:
    """Build a CockpitApp wired to a fully-faked CockpitData.

    Returns (app, read_runner, write_runner) so tests can assert on calls.
    ``repo_root`` overrides the fake root for the filesystem-backed reads
    (benchmarks explorer / evidence list) — seed it with BENCHMARKS.md and a
    results/rebench/ tree for those panes.
    """
    root = repo_root or FAKE_REPO_ROOT
    runner = FakeRunner(responses or fake_responses())
    gpus = gpus if gpus is not None else [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
    target = target if target is not None else ServingTarget(gpus=gpus)
    write_runner = write_runner or FakeWriteRunner()
    data = CockpitData(
        root,
        runner=runner,
        detect_endpoint_fn=make_detect(target),
        get_gpu_info_fn=make_gpu_info(gpus),
        write_runner=write_runner,
    )
    app = CockpitApp(repo_root=root, data=data)
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


PANEL_IDS = ["panel-discover", "panel-serve", "panel-estate", "panel-validate"]


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
    async def test_discover_panel_visible_on_start(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert "active" in app.query_one("#panel-discover").classes

    @pytest.mark.asyncio
    async def test_other_panels_hidden_on_start(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for pid in ["panel-serve", "panel-estate", "panel-validate"]:
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
    async def test_switch_to_serve_mode(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            assert "active" in app.query_one("#panel-serve").classes
            assert "active" not in app.query_one("#panel-discover").classes

    @pytest.mark.asyncio
    async def test_switch_to_estate_mode(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            assert "active" in app.query_one("#panel-estate").classes

    @pytest.mark.asyncio
    async def test_switch_to_validate_mode(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            assert "active" in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_switch_back_to_discover(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await pilot.press("1")
            assert "active" in app.query_one("#panel-discover").classes
            assert "active" not in app.query_one("#panel-serve").classes

    @pytest.mark.asyncio
    async def test_all_four_modes_cycle(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for key, expected_active in [("1", 0), ("2", 1), ("3", 2), ("4", 3), ("1", 0)]:
                await pilot.press(key)
                await pilot.pause()
                active = [pid for pid in PANEL_IDS if "active" in app.query_one(f"#{pid}").classes]
                assert len(active) == 1, f"after {key!r}: {active}"
                assert active[0] == PANEL_IDS[expected_active]


class TestNavNodesExist:
    @pytest.mark.asyncio
    async def test_discover_tabs_exist(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#discover-tabs", TabbedContent)
            app.query_one("#tab-catalog", TabPane)
            app.query_one("#tab-byo", TabPane)

    @pytest.mark.asyncio
    async def test_estate_tabs_exist(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#estate-tabs", TabbedContent)
            app.query_one("#tab-orchestration", TabPane)
            app.query_one("#tab-containers", TabPane)

    @pytest.mark.asyncio
    async def test_validate_tabs_exist(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#validate-tabs", TabbedContent)
            app.query_one("#tab-run", TabPane)
            app.query_one("#tab-doctor", TabPane)
            app.query_one("#tab-benchmarks", TabPane)
            app.query_one("#tab-evidence", TabPane)

    @pytest.mark.asyncio
    async def test_catalog_datatable_has_columns(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            table = app.query_one("#catalog-table", DataTable)
            col_labels = [str(c.label) for c in table.columns.values()]
            for expected in ("slug", "engine", "fit", "ctx", "TPS", "8pk", "status", "source"):
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
# Discover · Catalog (now wired to real enriched entries)
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
# Discover · BYO (wired to byo_check)
# ===========================================================================


class TestByoWired:
    @pytest.mark.asyncio
    async def test_byo_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#byo-panel", ByoPane)
            app.query_one("#byo-url-input", Input)
            app.query_one("#byo-profile-input", Input)
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
# Serve (plan staging + reconcile-gated confirm modal)
# ===========================================================================


class TestServeWired:
    @pytest.mark.asyncio
    async def test_serve_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#serve-panel", ServePane)
            app.query_one("#serve-plan-box")
            app.query_one("#serve-launch-btn", Button)
            app.query_one("#serve-cancel-btn", Button)
            app.query_one("#serve-live")  # core LivePane for the boot stream

    @pytest.mark.asyncio
    async def test_enter_in_catalog_stages_plan_and_jumps_to_serve(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert "active" in app.query_one("#panel-serve").classes
            assert app._staged_entry is not None
            assert app._staged_entry.slug == "vllm/dual"
            detail = str(app.query_one("#serve-plan-detail", Static).render())
            assert "vllm/dual" in detail

    @pytest.mark.asyncio
    async def test_serve_enter_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")       # stage + jump to Serve
            await pilot.pause()
            await pilot.press("enter")       # commit → confirm modal
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)

    @pytest.mark.asyncio
    async def test_serve_plan_is_not_force(self):
        """The serve plan built by ⏎ is the GATED switch.sh <slug> — NOT --force."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.serve("vllm/dual")
            assert "--force" not in plan.cmd
            assert plan.requires_reconcile is True


# ===========================================================================
# Estate · Orchestration + Containers (wired to estate_state)
# ===========================================================================


class TestEstateWired:
    @pytest.mark.asyncio
    async def test_estate_orch_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-orch-pane", EstateOrchPane)
            app.query_one("#gpu0-card")
            app.query_one("#gpu1-card")
            app.query_one("#doctor-line")
            app.query_one("#scene-table", DataTable)
            app.query_one("#services-strip")

    @pytest.mark.asyncio
    async def test_estate_scene_table_populates_from_gpu_mode(self):
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
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
            await pilot.press("3")
            await _settle(pilot)
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "18.0 / 24.0 GiB" in bar
            assert "71%" in bar

    @pytest.mark.asyncio
    async def test_estate_doctor_line_serving(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            line = str(app.query_one("#doctor-line", Static).render())
            assert "serving" in line.lower()

    @pytest.mark.asyncio
    async def test_estate_containers_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-containers-pane", EstateContainersPane)
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
            await pilot.press("3")
            await _settle(pilot)
            pane = app.query_one("#estate-containers-pane", EstateContainersPane)
            names = [c.name for c in pane._containers]
            assert "vllm-qwen36-27b-dual" in names
            assert "open-webui" not in names  # not an engine prefix

    @pytest.mark.asyncio
    async def test_estate_scene_switch_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#scene-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)


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
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-containers-pane", EstateContainersPane).query_one(
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
            await pilot.press("3")
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
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-containers-pane", EstateContainersPane).query_one(
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
            await pilot.press("3")
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
            # stage → confirm → commit
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
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
    async def test_validate_doctor_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#validate-doctor-pane", ValidateDoctorPane)
            app.query_one("#doctor-card-health")
            app.query_one("#doctor-card-estate")
            app.query_one("#doctor-card-profile")

    @pytest.mark.asyncio
    async def test_validate_doctor_health_line_goes_live_on_estate_poll(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")  # estate poll feeds the doctor pane too
            await _settle(pilot)
            body = str(app.query_one("#doctor-health-body", Static).render())
            assert "serving" in body.lower()

    @pytest.mark.asyncio
    async def test_validate_benchmarks_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#validate-benchmarks-pane", ValidateBenchmarksPane)
            t = app.query_one("#bmk-table", DataTable)
            # Real explorer — empty until the Validate poll fills it (no mock rows).
            assert t.row_count == 0

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
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await pilot.press("enter")


# ===========================================================================
# Validate · Doctor (wired to doctor() — health + estate + profile cards)
# ===========================================================================


class TestValidateDoctorWired:
    @pytest.mark.asyncio
    async def test_doctor_cards_populate_from_doctor_read(self):
        """Entering Validate runs the full Doctor read → estate + profile cards
        fill from diagnose-estate.sh --json + diagnose-profile.sh (text)."""
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            estate = str(app.query_one("#doctor-estate-body", Static).render())
            assert "GREEN" in estate and "2/2" in estate  # 2/2 instances fit
            assert any("diagnose-estate.sh --json" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_doctor_profile_triage_after_estate_target(self):
        """When a running engine is detected (matched slug), Doctor triages it
        via diagnose-profile.sh and renders the 6 steps + verdict."""
        # A detect target on port 8010 matches vllm/dual in the registry → the
        # estate poll captures the slug, which the doctor read then triages.
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses, gpus=gpus, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")            # estate poll → captures target slug
            await _settle(pilot)
            assert app._target_slug == "vllm/dual"
            await pilot.press("4")            # validate → doctor read uses the slug
            await _settle(pilot)
            profile = str(app.query_one("#doctor-profile-body", Static).render())
            assert "GREEN" in profile
            assert any("diagnose-profile.sh" in " ".join(c) for c in runner.calls)


# ===========================================================================
# Validate · Benchmarks (wired to benchmarks_explorer — filter + sort)
# ===========================================================================


class TestValidateBenchmarksWired:
    @pytest.mark.asyncio
    async def test_benchmarks_populate_from_explorer(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            t = app.query_one("#bmk-table", DataTable)
            assert t.row_count == 1  # the one BENCHMARKS.md scrape row
            pane = app.query_one("#validate-benchmarks-pane", ValidateBenchmarksPane)
            assert pane._rows[0].model == "qwen3.6-27b"
            assert pane._rows[0].quality_8pk == "109/150"

    @pytest.mark.asyncio
    async def test_benchmarks_filter_narrows(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            pane = app.query_one("#validate-benchmarks-pane", ValidateBenchmarksPane)
            pane.set_filter("gemma")             # no gemma rows
            assert app.query_one("#bmk-table", DataTable).row_count == 0
            pane.set_filter("qwen")
            assert app.query_one("#bmk-table", DataTable).row_count == 1

    @pytest.mark.asyncio
    async def test_benchmarks_sort_cycles(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            pane = app.query_one("#validate-benchmarks-pane", ValidateBenchmarksPane)
            assert pane._sort == "tps"
            pane.cycle_sort()
            assert pane._sort == "8pk"

    @pytest.mark.asyncio
    async def test_benchmarks_explorer_empty_root_surfaces_message(self):
        """No corpus + no BENCHMARKS.md → an honest 'no data' status, not a crash."""
        app, _, _ = make_app()  # FAKE_REPO_ROOT has neither
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            status = str(app.query_one("#bmk-status", Label).render()).lower()
            assert "no benchmark data" in status or "no benchmark" in status


# ===========================================================================
# Validate · Evidence (wired to evidence_list / evidence_report / submit)
# ===========================================================================


class TestValidateEvidenceWired:
    @pytest.mark.asyncio
    async def test_evidence_list_populates_from_rebench_dir(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            t = app.query_one("#evidence-table", DataTable)
            assert t.row_count == 1
            pane = app.query_one("#validate-evidence-pane", ValidateEvidencePane)
            assert pane._tags[0].tag == "vllm-dual-test"
            assert pane._tags[0].date == "2026-06-18"

    @pytest.mark.asyncio
    async def test_evidence_enter_opens_report_modal(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
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
        app, _, _ = make_app(repo_root=tmp_path, write_runner=wr)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
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
# Validate · Run (launch a validation step — confirm-gated, MOCKED stream)
# ===========================================================================


class TestValidateRunWired:
    @pytest.mark.asyncio
    async def test_run_enter_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
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
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
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

        app, _, _ = make_app(write_runner=wr)
        # Swap the detect to one that screams if the reconcile gate runs on confirm.
        app._data._detect_endpoint = detect_should_not_be_called
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
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
            await pilot.press("3")
            await _settle(pilot)
            strip = str(app.query_one("#powercap-strip", Static).render())
            assert "GPU0" in strip and "230W" in strip
            assert "capped" in strip  # GPU0 limit 230 < default 370

    @pytest.mark.asyncio
    async def test_power_cap_toggle_opens_confirm_off(self):
        """GPU0 is capped → [c] stages a 'power-cap off' (uncap) confirm."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
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
            await pilot.press("3")
            await _settle(pilot)
            await pilot.press("w")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert "power-cap-sweep" in " ".join(app.screen._plan.cmd)

    @pytest.mark.asyncio
    async def test_prune_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
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
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-tabs", TabbedContent).active = "tab-containers"
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
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-tabs", TabbedContent).active = "tab-containers"
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
        app, runner, _ = make_app(repo_root=tmp_path, write_runner=wr)
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            # Browse every Validate tab + Estate extras — pure reads, no writes.
            await pilot.press("4")
            await _settle(pilot)
            for tab in ("tab-doctor", "tab-benchmarks", "tab-evidence", "tab-run"):
                app.query_one("#validate-tabs", TabbedContent).active = tab
                await pilot.pause()
            await pilot.press("3")
            await _settle(pilot)
            # Nothing was written; submit-bench / prune / power-cap never auto-fired.
            assert wr.started == []
            assert all("--auto-submit" not in " ".join(c) for c in runner.calls)
            assert all("power-cap on" not in " ".join(c) and "power-cap off" not in " ".join(c)
                       for c in runner.calls)


# ===========================================================================
# PHASE 5 — the three v2 hooks (app wiring)
# ===========================================================================


from club3090_cockpit.app import PromoteScaffoldScreen, OptimizeScreen  # noqa: E402


SERVING_TARGET = ServingTarget(
    url="http://localhost:8010",
    model="qwen3.6-27b",
    container="vllm-qwen36-27b-dual",
    slug="vllm/dual",
    gpus=[GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=22000)],
)


class TestEvaluateHookWired:
    """Hook 1 — Estate → ▸ Evaluate (c3t hand-off, confirm-gated, MOCK-ONLY)."""

    @pytest.mark.asyncio
    async def test_evaluate_opens_confirm_when_target_running(self):
        app, _, _ = make_app(target=SERVING_TARGET,
                             gpus=list(SERVING_TARGET.gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")          # Estate — poll captures the target
            await _settle(pilot)
            await pilot.press("v")          # ▸ Evaluate
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.kind == "evaluate"
            assert app.screen._plan.cmd == ["bash", "scripts/c3t"]
            assert app.screen._plan.requires_reconcile is False
            assert app.screen._plan.requires_confirm is True

    @pytest.mark.asyncio
    async def test_evaluate_hands_off_the_shared_serving_target_by_identity(self):
        """The app's stored target IS the SAME ServingTarget the poll detected,
        and the hand-off carries that exact object (design §4/§6.6)."""
        app, _, _ = make_app(target=SERVING_TARGET, gpus=list(SERVING_TARGET.gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
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
                             write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
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
        # No serving target (empty url) → nothing to evaluate.
        app, _, _ = make_app(target=ServingTarget(), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            await pilot.press("v")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)
            assert wr.started == []          # never launched


class TestPromoteHookWired:
    """Hook 2 — Discover → ▸ Promote to catalog (scaffold preview + gated write)."""

    @pytest.mark.asyncio
    async def test_promote_without_byo_notifies(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # No BYO fit-check yet → nothing to promote.
            await pilot.press("P")
            await pilot.pause()
            assert not isinstance(app.screen, PromoteScaffoldScreen)

    @pytest.mark.asyncio
    async def test_promote_previews_scaffold_after_byo_check(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Run a BYO fit-check (Discover · BYO) — fills the arch facts.
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            assert app._last_byo is not None
            await pilot.press("P")          # ▸ Promote to catalog
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
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
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


class TestOptimizeHookWired:
    """Hook 3 — ▸ Optimize for my card: DORMANT v0.10.0 seam (no-op)."""

    @pytest.mark.asyncio
    async def test_optimize_shows_not_available_message(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # From Discover · Catalog with a selected slug.
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
    async def test_optimize_available_from_serve_staged_slug(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Stage a slug into Serve, then open the optimizer from there.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")      # → stages + jumps to Serve
            await pilot.pause()
            await pilot.press("O")
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
            # Optimize (Discover) — dormant no-op.
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("O")
            await pilot.pause()
            await pilot.press("escape")
            # Promote (Discover) after a BYO check — preview only.
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("P")
            await pilot.pause()
            await pilot.press("escape")
            # Evaluate (Estate) — confirm modal staged, not committed.
            await pilot.press("3")
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
            # No mode switch must have happened (still in Discover = mode 0).
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
            for action in ("quit", "help", "refresh", "mode_discover", "mode_serve"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"check_action({action!r}) should return True (always-on), got {result!r}"
                )


class TestCheckActionPerModeSubtab:
    """(b) check_action enables the right bindings per mode/sub-tab."""

    @pytest.mark.asyncio
    async def test_discover_catalog_enables_explain_and_filter(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            assert app.check_action("explain", ()) is True
            assert app.check_action("filter_catalog", ()) is True
            assert app.check_action("promote_catalog", ()) is True

    @pytest.mark.asyncio
    async def test_discover_catalog_disables_estate_keys(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            for action in ("container_logs", "estate_off", "power_cap_toggle",
                           "prune_images", "evaluate_target", "context_t"):
                result = app.check_action(action, ())
                assert result is False, (
                    f"Discover must not enable estate action {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_orchestration_enables_orch_keys(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            assert app._active_mode == 2
            # On orchestration tab (default first tab)
            tc = app.query_one("#estate-tabs", TabbedContent)
            tc.active = "tab-orchestration"
            await pilot.pause()
            for action in ("estate_off", "power_cap_toggle", "power_cap_sweep", "prune_images"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"Estate·Orchestration must enable {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_orchestration_disables_containers_keys(self):
        """On the Orchestration sub-tab, container-specific keys are disabled."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            tc = app.query_one("#estate-tabs", TabbedContent)
            tc.active = "tab-orchestration"
            await pilot.pause()
            for action in ("container_logs", "container_stop", "container_rm"):
                result = app.check_action(action, ())
                assert result is False, (
                    f"Estate·Orchestration must disable container action {action!r}, "
                    f"got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_containers_enables_containers_keys(self):
        """On the Containers sub-tab, container-specific keys are enabled."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            tc = app.query_one("#estate-tabs", TabbedContent)
            tc.active = "tab-containers"
            await pilot.pause()
            for action in ("container_logs", "container_stop", "container_rm", "context_t"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"Estate·Containers must enable {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_validate_benchmarks_enables_filter_and_sort(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            tc.active = "tab-benchmarks"
            await pilot.pause()
            assert app.check_action("filter_catalog", ()) is True
            assert app.check_action("context_t", ()) is True

    @pytest.mark.asyncio
    async def test_validate_evidence_enables_s_key(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            tc.active = "tab-evidence"
            await pilot.pause()
            # s_key (submit) is enabled in Validate mode regardless of subtab.
            assert app.check_action("s_key", ()) is True

    @pytest.mark.asyncio
    async def test_serve_mode_disables_explain_and_filter(self):
        """In Serve mode there is no catalog — explain and filter must be off."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await pilot.pause()
            assert app._active_mode == 1
            assert app.check_action("explain", ()) is False
            assert app.check_action("filter_catalog", ()) is False

    @pytest.mark.asyncio
    async def test_always_on_keys_active_in_every_mode(self):
        """quit/help/refresh/mode-switch must be True in every mode."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for mode_key in ("1", "2", "3", "4"):
                await pilot.press(mode_key)
                await pilot.pause()
                for action in ("quit", "help", "refresh",
                               "mode_discover", "mode_serve", "mode_estate", "mode_validate"):
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
            app, _, _ = make_app(repo_root=root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("4")
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
    async def test_right_bracket_cycles_discover_subtab(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0  # Discover
            tc = app.query_one("#discover-tabs", TabbedContent)
            assert tc.active == "tab-catalog"  # default first tab
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-byo"

    @pytest.mark.asyncio
    async def test_right_bracket_wraps_around_on_last_tab(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#discover-tabs", TabbedContent)
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
            tc = app.query_one("#discover-tabs", TabbedContent)
            tc.active = "tab-byo"
            await pilot.pause()
            await pilot.press("left_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-catalog"

    @pytest.mark.asyncio
    async def test_subtab_key_cycles_estate_tabs(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            tc = app.query_one("#estate-tabs", TabbedContent)
            first = tc.active
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active != first

    @pytest.mark.asyncio
    async def test_subtab_key_cycles_validate_tabs(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            first = tc.active
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active != first

    @pytest.mark.asyncio
    async def test_subtab_key_inactive_in_serve_mode(self):
        """Serve mode has no sub-tabs — cycle key must be disabled (check_action
        returns False) so it doesn't appear in the footer or do anything."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await pilot.pause()
            assert app._active_mode == 1  # Serve
            assert app.check_action("next_subtab", ()) is False
            assert app.check_action("prev_subtab", ()) is False


class TestModeSwitchFocus:
    """(e) Mode switch moves focus to the mode's primary interactive widget."""

    @pytest.mark.asyncio
    async def test_switch_to_discover_focuses_catalog_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Start at Discover (mode 0) already — go away and come back.
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
            await pilot.press("3")
            await _settle(pilot)
            # Estate/Orchestration tab is default → scene-table should be focused.
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "scene-table"

    @pytest.mark.asyncio
    async def test_switch_to_validate_focuses_run_ladder(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            # Validate/Run tab is default → run-ladder-table should be focused.
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "run-ladder-table"

    @pytest.mark.asyncio
    async def test_tab_change_on_validate_refocuses_relevant_table(self):
        """Using the sub-tab cycle key on Validate must move focus to the
        relevant DataTable for the newly active tab."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            # Default tab is Run → run-ladder-table.
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "run-ladder-table"
            # Cycle forward once → Doctor tab (no table, focus stays somewhere).
            await pilot.press("right_square_bracket")
            await pilot.pause()
            await pilot.pause()  # extra cycle for call_after_refresh
            tc = app.query_one("#validate-tabs", TabbedContent)
            assert tc.active == "tab-doctor"
            # Cycle forward again → Benchmarks tab → bmk-table.
            await pilot.press("right_square_bracket")
            await pilot.pause()
            await pilot.pause()  # extra cycle for call_after_refresh
            assert tc.active == "tab-benchmarks"
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "bmk-table"


class TestContainerAutoLoad:
    """Estate·Containers drill detail auto-loads on selection (lazydocker-style)
    — no [l]/[t] keypress needed."""

    @pytest.mark.asyncio
    async def test_entering_containers_autoloads_config(self):
        """Switching to the Containers tab auto-fills the Config drill tab for the
        highlighted container — no keypress."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            cfg = str(app.query_one("#drill-config", Static).render())
            assert "vllm-qwen36-27b-dual" in cfg  # config loaded with no [t]

    @pytest.mark.asyncio
    async def test_entering_containers_autoloads_logs(self):
        """Default drill tab is Logs — entering Containers auto-reads docker logs
        for the selected container (no [l])."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            assert any("docker logs" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_drill_tab_switch_to_top_reads_top(self):
        """Switching the drill tab Logs→Top reads docker top for the selected
        container (no [t])."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-tabs", TabbedContent).active = "tab-containers"
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
            await pilot.press("3")
            await _settle(pilot)
            app.query_one("#estate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            tbl = app.query_one("#containers-table", DataTable)
            tbl.focus()
            await pilot.pause()
            await pilot.press("down")  # row 0 (qwen) → row 1 (gemma) — fires RowHighlighted
            await pilot.pause()
            assert tbl.cursor_row == 1, f"cursor did not move (row={tbl.cursor_row})"
            cfg = str(app.query_one("#drill-config", Static).render())
            assert "vllm-gemma-4-31b-dual" in cfg


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
    async def test_esc_closes_bmk_filter(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("4")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-benchmarks"
            await _settle(pilot)
            await pilot.press("slash")
            await pilot.pause()
            assert app.query("#bmk-filter")
            await pilot.press("escape")
            await pilot.pause()
            assert not app.query("#bmk-filter")
            assert app.focused is app.query_one("#bmk-table", DataTable)

    @pytest.mark.asyncio
    async def test_esc_on_main_screen_does_not_quit(self):
        """Esc with no filter + no modal is a harmless no-op (never quits)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running
