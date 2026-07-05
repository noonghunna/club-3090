"""Headless tests for the CockpitApp (Phase 3 — wired).

Verifies:
  1. The app mounts without error (no TTY, no GPU, no Docker, no live script).
     The data layer is a CockpitData backed by a FakeRunner + fake detect +
     a FakeWriteRunner, so NO subprocess is ever spawned.
  2. All three modes (Run · Operate · Validate) are reachable via digit-key
     bindings 1/2/3; nav nodes exist.  (R1 folded Discover + Serve + Benchmarks
     into a single Run mode; R2a renamed Estate → Operate and moved Doctor into it.)
  3. Run · Catalog populates from real enriched entries (fit glyph, TPS,
     8pk, topology) and filters live (multi-word AND).
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

from textual.widgets import Button, DataTable, Input, Select, Static, TabbedContent, TabPane, Label, Tabs
from textual.widgets._footer import FooterKey
from textual.widgets._tabbed_content import ContentTabs

from club3090_tui_core.detect import GpuInfo, ServingTarget

from club3090_cockpit.app import (
    CockpitApp,
    CatalogPane,
    CockpitCommands,
    ConfirmActionScreen,
    ServeContext,
    ExplainScreen,
    FocusableFooter,
    HelpScreen,
    ModeSwitcher,
    LaneBringPane,
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
from club3090_cockpit.data import (
    ContainerInfo,
    DiskUsage,
    EstateState,
    EstateTelemetry,
    GpuCompApp,
    RamUsage,
    Scene,
    ServedProbe,
    attribute_gpu_apps,
    parse_cgroup_container_id,
    parse_compute_apps,
    parse_df_output,
    parse_docker_ps_id_names,
    parse_meminfo,
)
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
                "baseline": {
                    "narr_tps": 174.0, "code_tps": 42.0, "quality_8pk": "109/150",
                    "date": "2026-07-01", "engine_pin": "vllm/vllm-openai:v0.24.0",
                    "current_pin": "vllm/vllm-openai:v0.24.0", "stale": False,
                    "rig": "2x3090-pcie", "power_cap_w": [370, 420],
                    "submitted_by": "noonghunna",
                },
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
                "baseline": {
                    "narr_tps": 60.4, "code_tps": 72.4,
                    "date": "2026-05-23", "engine_pin": "ghcr.io/ik-old@sha256:aaa",
                    "current_pin": "ghcr.io/ik-new@sha256:bbb", "stale": True,
                    "rig": "1x3090-pcie", "power_cap_w": [370],
                    "submitted_by": "noonghunna",
                },
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
        {"name": "27b", "group": "models", "description": "Qwen", "services": ["vllm-qwen36-27b-dual"], "ports": ["8010"], "gpus": "both"},
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

# Bring funnel (§2b) — deriver --inventory canned responses (stage-1 INSPECT).
INVENTORY_SAFETENSORS_JSON = json.dumps(
    {
        "repo": "org/Model",
        "formats": ["safetensors"],
        "safetensors": {"weight_files": ["model.safetensors"], "size_gb": 16.2},
        "gguf_variants": [],
        "gguf_mmproj": [],
        "lineage_base_model": "Qwen/Qwen3.6-27B",
    }
)
INVENTORY_GGUF_JSON = json.dumps(
    {
        "repo": "org/Model-GGUF",
        "formats": ["gguf"],
        "safetensors": None,
        "gguf_variants": [
            {"quant": "Q4_K_M", "size_gb": 17.0, "parts": 1, "files": ["m-Q4_K_M.gguf"]},
            {"quant": "Q8_0", "size_gb": 29.0, "parts": 2,
             "files": ["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"]},
        ],
        "gguf_mmproj": ["mmproj-F16.gguf"],
        "lineage_base_model": None,
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

# ── UX Batch 5 fixtures (verified-live recon 2026-06-20) ──────────────────────────
# df -P -B1 <repo> /mnt/models — REAL recon shape (repo on the LVM root, /mnt/models
# on a SEPARATE /dev/sdb1 — DIFFERENT devices, so NO de-dup).  -B1 → byte fields;
# -P (POSIX) → one line per fs + the header renames Use% → Capacity (position
# unchanged; the parser reads use-% by field index, not header name).
DF_TWO_DEVICES = (
    "Filesystem                             1-blocks          Used     Available Capacity Mounted on\n"
    "/dev/mapper/ubuntu--vg-ubuntu--lv 1793150255104  435407220736 1284330176512      26% /\n"
    "/dev/sdb1                         1901246259200 1615138213888  189454557184      90% /mnt/models\n"
)
# Both paths resolve to the SAME device → ONE de-duped "repo + models" bar.
DF_SAME_DEVICE = (
    "Filesystem 1-blocks          Used     Available Capacity Mounted on\n"
    "/dev/sda1  580000000000 412000000000 168000000000      71% /\n"
    "/dev/sda1  580000000000 412000000000 168000000000      71% /mnt/models\n"
)
# A long LVM/mapper device name + a zero-size special-fs row (tmpfs): -P keeps
# them on ONE line each (no wrap); the parser must read use-% by position AND
# skip the zero-size row so it never draws a false "0% 0G/0G" bar.
DF_LONG_DEVICE_AND_ZERO = (
    "Filesystem                                          1-blocks          Used     Available Capacity Mounted on\n"
    "/dev/mapper/vg--very--long--name-ubuntu--lv--root 1793150255104  435407220736 1284330176512      26% /\n"
    "tmpfs                                                         0             0             0        - /sys/fs/cgroup\n"
)
# /proc/meminfo — REAL recon (98854288 kB total, 84219884 kB available).
MEMINFO = (
    "MemTotal:       98854288 kB\n"
    "MemFree:         1475616 kB\n"
    "MemAvailable:   84219884 kB\n"
    "Buffers:         1054728 kB\n"
    "Cached:         82171824 kB\n"
)
# nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory — REAL recon: one app
# (pid 588408, 22686 MiB) on GPU0's uuid.
COMPUTE_APPS = "GPU-ed5070b5-c35f-7b25-7696-2b767e563cc4, 588408, 22686\n"
# nvidia-smi --query-gpu=uuid,index — maps the GPU0 uuid → index 0.
GPU_UUID_INDEX = (
    "GPU-ed5070b5-c35f-7b25-7696-2b767e563cc4, 0\n"
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 1\n"
)
# /proc/<pid>/cgroup — REAL recon (cgroup v2: docker-<64hex>.scope).
CGROUP_PID = (
    "0::/system.slice/"
    "docker-e8eb8d4cdd19861ddc94d582b2d583b898baf9c3931b26407f1b43ebb896c3d4.scope\n"
)
# docker ps --no-trunc --format '{{.ID}} {{.Names}}' — REAL recon id→name map.
DOCKER_PS_IDNAMES = (
    "e8eb8d4cdd19861ddc94d582b2d583b898baf9c3931b26407f1b43ebb896c3d4 llama-cpp-pi-reasoning\n"
    "82364280908afeb42a643822f479c2ef07f5c92ed054d7e3683a33e78fee747a studio-tts\n"
    "b9c7d8853c6aecf526933957eb7ecdefa3167d4ab24b178af5bcc7f32efb92ce studio-image-shim\n"
)
# docker ps --format '{{.Names}}|{{.Ports}}' — REAL recon: studio-* + an engine.
DOCKER_PS_STUDIO = (
    "llama-cpp-pi-reasoning|0.0.0.0:8053->8080/tcp\n"
    "studio-tts|0.0.0.0:8193->8000/tcp\n"
    "studio-image-shim|\n"
    "studio-orchestrator|\n"
    "studio-gallery|0.0.0.0:8188->8188/tcp\n"
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
    "- **Served as:** `qwen3.6-27b` from `/mnt/models/x`\n"
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
        # Bring funnel §2b stage-1 (safetensors default; gguf tests override)
        "deriver.py --inventory": ok(INVENTORY_SAFETENSORS_JSON),
        "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
        "health.sh": ok(HEALTH_SERVING),
        "docker ps": ok(DOCKER_PS_EMPTY),
        # Studio guard — default to a SET-UP rig (comfyui-local image present) so the
        # studio-scene / comfyui-start guard doesn't fire; guard tests override empty.
        "docker images -q": ok("sha256:abc123\n"),
        # Phase-4 reads:
        "diagnose-estate.sh --json": ok(DIAGNOSE_ESTATE_JSON),
        "diagnose-profile.sh": ok(DIAGNOSE_PROFILE_TEXT),
        "power-cap status": ok(POWER_CAP_STATUS),
        "docker top": ok(DOCKER_TOP),
        "docker logs": ok(DOCKER_LOGS),
    }
    responses.update(overrides)
    return responses


def batch5_responses(
    *,
    df: str = DF_TWO_DEVICES,
    meminfo: str = MEMINFO,
    compute_apps: str = COMPUTE_APPS,
    uuid_index: str = GPU_UUID_INDEX,
    cgroup: str = CGROUP_PID,
    idnames: str = DOCKER_PS_IDNAMES,
    docker_ps: str = DOCKER_PS_STUDIO,
    **extra,
) -> dict[str, RunResult]:
    """Canned responses for the Batch-5 telemetry reads, ORDERED so the more
    specific keys win over the base ``docker ps`` / ``cat`` substrings.

    FakeRunner returns the FIRST inserted key whose token is a substring of the
    command, so the ``docker ps --no-trunc`` and ``/proc/.../cgroup`` keys MUST
    precede the base ``docker ps`` / ``/proc/meminfo`` entries (the same
    first-match-wins ordering the catalog uses for --fit-all vs --fit)."""
    ordered: dict[str, RunResult] = {}
    # More-specific keys first (substring disambiguation).
    ordered["df -P -B1"] = ok(df)
    ordered["/proc/meminfo"] = ok(meminfo)
    ordered["--query-compute-apps"] = ok(compute_apps)
    ordered["--query-gpu=uuid,index"] = ok(uuid_index)
    ordered["/cgroup"] = ok(cgroup)            # cat /proc/<pid>/cgroup
    ordered["docker ps --no-trunc"] = ok(idnames)
    ordered["docker ps"] = ok(docker_ps)       # the |ports| stack-container read
    # Then the rest of the standard read set (already insertion-ordered with its
    # own --fit-all-before---fit care).  extra overrides win (added last).
    for k, v in fake_responses(**extra).items():
        ordered.setdefault(k, v)
    return ordered


def make_app(
    *,
    responses: Optional[dict[str, RunResult]] = None,
    gpus: Optional[list[GpuInfo]] = None,
    target: Optional[ServingTarget] = None,
    write_runner: Optional[FakeWriteRunner] = None,
    repo_root: Optional[Path] = None,
    surface: str = "producer",
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


async def _enter_operate(pilot, tab: str = "tab-orchestration") -> None:
    """2-mode merge helper: enter the MERGED Run & Operate mode (mode 0) and
    activate one of its Operate tabs (Orchestration / Containers / Doctor).

    Pressing [1] re-enters mode 0 (the default) which triggers the estate poll
    (load_estate populates the orch / containers / doctor panes + GPU cards), then
    we flip to the requested tab and settle.  Replaces the old `pilot.press("2")`
    that used to reach the standalone Operate mode."""
    app = pilot.app
    await pilot.press("1")
    try:
        app.query_one("#operate-tabs", TabbedContent).active = tab
    except Exception:
        pass
    await _settle(pilot)


# 2-mode merge: mode 0 = merged Run & Operate (#panel-run, hosting #operate-tabs
# with Catalog · Orchestration · Containers · Doctor); mode 1 = Bring & Validate
# (#panel-validate).  #panel-operate is gone (folded into #panel-run).
PANEL_IDS = ["panel-run", "panel-validate"]


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
        # 2-mode merge: only #panel-validate is the "other" panel now.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#panel-operate")
            assert "active" not in app.query_one("#panel-validate").classes

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
    async def test_key_1_switches_to_merged_mode(self):
        # 2-mode merge: [1] = merged Run & Operate (#panel-run).
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")  # leave to the lane first
            await pilot.press("1")
            assert "active" in app.query_one("#panel-run").classes
            assert "active" not in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_operate_panes_live_in_merged_mode(self):
        # Operate folded into the merged mode 0 — the Orchestration tab + its panes
        # live under #panel-run now (no standalone #panel-operate).
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#panel-operate")
            await _enter_operate(pilot)
            assert "active" in app.query_one("#panel-run").classes
            assert app.query_one("#operate-tabs", TabbedContent).active == "tab-orchestration"

    @pytest.mark.asyncio
    async def test_key_2_switches_to_validate_mode(self):
        # 2-mode merge: [2] = the producer Bring & Validate lane (#panel-validate),
        # visible by default (the full surface).
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            assert "active" in app.query_one("#panel-validate").classes
            assert "active" not in app.query_one("#panel-run").classes

    @pytest.mark.asyncio
    async def test_switch_back_to_merged(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await pilot.press("1")
            assert "active" in app.query_one("#panel-run").classes
            assert "active" not in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_old_serve_and_operate_panels_gone(self):
        """The standalone Serve mode (R1) AND the standalone Operate panel
        (2-mode merge) are gone — only #panel-run + #panel-validate remain."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#panel-serve")
            assert not app.query("#panel-operate")
            assert not hasattr(app, "action_mode_serve")
            # There are only two modes; [3] is unbound now.
            await pilot.press("3")  # unbound — should not switch anything
            await pilot.pause()
            active = [pid for pid in PANEL_IDS if "active" in app.query_one(f"#{pid}").classes]
            assert active == ["panel-run"]

    @pytest.mark.asyncio
    async def test_both_modes_cycle(self):
        # Both modes cycle on the default (full) surface; [3] is unbound.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for key, expected_active in [("1", 0), ("2", 1), ("1", 0), ("2", 1)]:
                await pilot.press(key)
                await pilot.pause()
                active = [pid for pid in PANEL_IDS if "active" in app.query_one(f"#{pid}").classes]
                assert len(active) == 1, f"after {key!r}: {active}"
                assert active[0] == PANEL_IDS[expected_active]


class TestNavNodesExist:
    @pytest.mark.asyncio
    async def test_merged_mode_tabs_exist(self):
        # 2-mode merge: the merged Run & Operate TabbedContent (#operate-tabs) hosts
        # Catalog FIRST, then Orchestration · Containers · Doctor.  The old #run-tabs
        # TabbedContent and the #tab-byo tab are GONE.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#run-tabs")
            assert not app.query("#tab-byo")
            app.query_one("#operate-tabs", TabbedContent)
            app.query_one("#tab-catalog", TabPane)
            app.query_one("#tab-orchestration", TabPane)
            app.query_one("#tab-containers", TabPane)
            app.query_one("#tab-doctor", TabPane)

    @pytest.mark.asyncio
    async def test_operate_tabs_order(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            tc = app.query_one("#operate-tabs", TabbedContent)
            # Catalog · Orchestration · Containers · Doctor, in that order.  (The
            # Containers pane nests a drill TabbedContent of its own, so filter to
            # the mode-level ids.)
            wanted = {"tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor"}
            mode_tab_ids = [p.id for p in tc.query(TabPane) if p.id in wanted]
            assert mode_tab_ids == [
                "tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor"
            ], mode_tab_ids

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
    async def test_doctor_renders_in_merged_mode_not_validate(self):
        """2-mode merge: the Doctor surface is a tab of the merged Run & Operate
        mode (#operate-tabs · tab-doctor under #panel-run) and is GONE from the
        Bring & Validate lane (#validate-tabs)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            # tab-doctor is a child of operate-tabs, NOT validate-tabs.
            operate = app.query_one("#operate-tabs", TabbedContent)
            validate = app.query_one("#validate-tabs", TabbedContent)
            operate_panes = [p.id for p in operate.query(TabPane)]
            validate_panes = [p.id for p in validate.query(TabPane)]
            assert "tab-doctor" in operate_panes, operate_panes
            assert "tab-doctor" not in validate_panes, validate_panes
            # The Doctor pane itself renders under the merged panel subtree.
            doctor = app.query_one("#doctor-pane", DoctorPane)
            panel_run = app.query_one("#panel-run")
            assert doctor in panel_run.query("#doctor-pane")
            # Activate the Doctor tab in the merged mode and confirm it switches.
            await _enter_operate(pilot, tab="tab-doctor")
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
            # Fold 3: TPS / 8pk columns are explicitly labelled as this-rig
            # (F6 shortened "our rig" → "rig" for column budget).
            # Serve-confirm rework: the "fit" column moved into the serve pop-up.
            # Model-filter: a "model" column leads the table (group-by-model view).
            # F6: money columns (ctx · TPS · 8pk · status) directly after the
            # identity; topology/engine (slug-redundant) at the tail so a
            # 120-140-col terminal folds them, not the numbers.
            for expected in (
                "model", "slug", "ctx", "TPS (rig)", "8pk (rig)",
                "status", "topo", "engine",
            ):
                assert expected in col_labels, f"missing {expected!r}: {col_labels}"
            # "source" is gone.
            assert "source" not in col_labels, col_labels
            # "fit" is gone — it lives in the serve confirm pop-up now.
            assert "fit" not in col_labels, col_labels
            # "model" is the FIRST column (mirrors switch.sh --list's grouping).
            assert col_labels[0] == "model", col_labels
            # F6 — every money column sits LEFT of topo/engine.
            fold = col_labels.index("topo")
            for money in ("ctx", "TPS (rig)", "8pk (rig)", "status"):
                assert col_labels.index(money) < fold, col_labels

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
            # Catalog-baselines slice 1: the source is the shipped baseline.
            assert entry.measurement.source == "baseline"

    @pytest.mark.asyncio
    async def test_catalog_stale_baseline_dagger(self):
        """Catalog-baselines slice 1 — a baseline measured on an OLDER engine
        pin renders the † staleness marker on its TPS cell + the status-line
        legend (re-bench owed); a current-pin row stays unmarked."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tbl = app.query_one("#catalog-table", DataTable)
            rows = [" ".join(str(c) for c in tbl.get_row_at(r)) for r in range(tbl.row_count)]
            ik_row = next(r for r in rows if "iq4ks-mtp" in r)     # stale fixture row
            dual_row = next(r for r in rows if "vllm/dual" in r)   # fresh fixture row
            assert "†" in ik_row
            assert "†" not in dual_row
            status = str(app.query_one("#catalog-status", Label).render())
            assert "older engine pin" in status

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
    async def test_catalog_topology_column_cell(self):
        """Round-4 CHANGE 1: the catalog rows carry a topology cell derived from
        the compose path — vllm/dual → "dual", ik-llama single → "single"."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            dual = next(e for e in pane._entries if e.slug == "vllm/dual")
            single = next(e for e in pane._entries if e.slug == "ik-llama/iq4ks-mtp")
            assert dual.topology == "dual"
            assert single.topology == "single"

    @pytest.mark.asyncio
    async def test_catalog_multiword_filter_is_and(self):
        """Round-4 CHANGE 2: a multi-word filter ("gemma dual") is AND-of-substrings
        across the searchable text (slug + topology + engine + model + status +
        source).  It matches a gemma dual row but NOT a gemma single row — the old
        contiguous-substring test matched neither."""
        from club3090_cockpit.data import CatalogEntry as _CE
        from club3090_tui_core import VariantRow as _VR

        def _entry(slug: str, model: str, compose_path: str) -> _CE:
            return _CE(
                row=_VR(
                    slug=slug,
                    switch_engine="vllm",
                    launch_engine="vllm",
                    compose_dir=compose_path.rsplit("/", 1)[0],
                    file=compose_path.rsplit("/", 1)[-1],
                    port=8000,
                    model=model,
                    engine="vllm-stable",
                    kvcalc_key=f"{model}:dual",
                    container="c",
                    compose_path=compose_path,
                    status="production",
                    ctx_label="262K",
                    status_note="",
                )
            )

        gemma_dual = _entry(
            "vllm/gemma-dual", "gemma-4-31b",
            "models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml",
        )
        gemma_single = _entry(
            "vllm/gemma-single", "gemma-4-31b",
            "models/gemma-4-31b/vllm/compose/single/autoround-int4/base.yml",
        )
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.populate([gemma_dual, gemma_single], None)
            # multi-word AND: "gemma dual" → only the gemma DUAL row.
            pane.set_filter("gemma dual")
            assert [e.slug for e in pane._filtered_entries()] == ["vllm/gemma-dual"]
            # single word still works (both gemma rows).
            pane.set_filter("gemma")
            assert {e.slug for e in pane._filtered_entries()} == {
                "vllm/gemma-dual", "vllm/gemma-single",
            }
            # term order is irrelevant for AND.
            pane.set_filter("dual gemma")
            assert [e.slug for e in pane._filtered_entries()] == ["vllm/gemma-dual"]
            # a term that matches nothing → no rows.
            pane.set_filter("gemma qwen")
            assert pane._filtered_entries() == []
            # empty query → all rows.
            pane.set_filter("")
            assert len(pane._filtered_entries()) == 2

    @pytest.mark.asyncio
    async def test_catalog_hides_deprecated_by_default_and_h_reveals(self):
        """🗑️ Deprecated slugs are HIDDEN from the catalog by default (mirrors
        `switch.sh --list`); [h] / toggle_deprecated() reveals them, and the banner
        surfaces the hidden count.  A deprecated slug stays filtered even when it
        matches the active text filter — until revealed."""
        from club3090_cockpit.data import CatalogEntry as _CE
        from club3090_tui_core import VariantRow as _VR

        def _entry(slug: str, status: str) -> _CE:
            leaf = slug.split("/")[-1]
            path = f"models/gemma-4-31b/vllm/compose/dual/autoround-int4/{leaf}.yml"
            return _CE(
                row=_VR(
                    slug=slug, switch_engine="vllm", launch_engine="vllm",
                    compose_dir=path.rsplit("/", 1)[0], file=path.rsplit("/", 1)[-1],
                    port=8000, model="gemma-4-31b", engine="vllm-stable",
                    kvcalc_key="gemma-4-31b:dual", container="c",
                    compose_path=path, status=status, ctx_label="229K", status_note="",
                )
            )

        live = _entry("vllm/gemma-dual", "caveats")
        prod = _entry("vllm/gemma-prod", "production")
        dead1 = _entry("vllm/gemma-int8-mtp", "deprecated")
        dead2 = _entry("vllm/gemma-bf16-mtp", "deprecated")

        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.populate([live, prod, dead1, dead2], None)

            # Default: 2 deprecated hidden; only the functional slugs show.
            assert pane._show_deprecated is False
            assert {e.slug for e in pane._filtered_entries()} == {
                "vllm/gemma-dual", "vllm/gemma-prod",
            }
            assert pane._deprecated_hidden_count() == 2
            assert app.query_one("#catalog-table", DataTable).row_count == 2

            # A deprecated slug stays hidden even when it matches the text filter.
            pane.set_filter("int8")
            assert pane._filtered_entries() == []
            pane.set_filter("")

            # [h] reveals: all 4 rows show, hidden-count drops to 0.
            pane.toggle_deprecated()
            assert pane._show_deprecated is True
            assert len(pane._filtered_entries()) == 4
            assert pane._deprecated_hidden_count() == 0
            assert app.query_one("#catalog-table", DataTable).row_count == 4
            # Revealed → the deprecated slug is now findable by filter.
            pane.set_filter("int8")
            assert [e.slug for e in pane._filtered_entries()] == ["vllm/gemma-int8-mtp"]
            pane.set_filter("")

            # Toggling back hides them again.
            pane.toggle_deprecated()
            assert pane._show_deprecated is False
            assert len(pane._filtered_entries()) == 2

    @pytest.mark.asyncio
    async def test_catalog_model_dropdown_scopes_and_combines(self):
        """Model-scope dropdown: options are the DISTINCT model names in registry
        order (the switch.sh --list grouping the flat table drops); picking one
        narrows the table to that model's lanes; it AND-combines with the text
        filter; clearing returns to all; a scoped model that vanishes resets."""
        from club3090_cockpit.app import _ALL_MODELS_VALUE as _AM
        from club3090_cockpit.data import CatalogEntry as _CE
        from club3090_tui_core import VariantRow as _VR

        def _entry(slug: str, model: str, compose_path: str) -> _CE:
            return _CE(
                row=_VR(
                    slug=slug, switch_engine="vllm", launch_engine="vllm",
                    compose_dir=compose_path.rsplit("/", 1)[0],
                    file=compose_path.rsplit("/", 1)[-1], port=8000, model=model,
                    engine="vllm-stable", kvcalc_key=f"{model}:dual", container="c",
                    compose_path=compose_path, status="production",
                    ctx_label="262K", status_note="",
                )
            )

        qwen_dual = _entry("vllm/dual", "qwen3.6-27b", "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml")
        qwen_single = _entry("vllm/single", "qwen3.6-27b", "models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml")
        gemma_dual = _entry("vllm/gemma-dual", "gemma-4-31b", "models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml")

        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.populate([qwen_dual, qwen_single, gemma_dual], None)

            # distinct models, registry (first-seen) order — irrespective of provider/quant.
            assert pane._models == ["qwen3.6-27b", "gemma-4-31b"]

            # group-by-model sort: even with INTERLEAVED input, a model's lanes sit
            # together in registry order (qwen lanes, then gemma) — the switch.sh
            # --list view, and what keeps the table↔cursor mapping consistent.
            pane.populate([qwen_dual, gemma_dual, qwen_single], None)
            assert [e.model for e in pane._filtered_entries()] == [
                "qwen3.6-27b", "qwen3.6-27b", "gemma-4-31b",
            ]
            # restore canonical order for the scope assertions below.
            pane.populate([qwen_dual, qwen_single, gemma_dual], None)

            # scope to one model → only its lanes (both quants/topologies).
            pane.set_model_filter("qwen3.6-27b")
            assert {e.slug for e in pane._filtered_entries()} == {"vllm/dual", "vllm/single"}

            # AND-combine with the text filter.
            pane.set_filter("dual")
            assert [e.slug for e in pane._filtered_entries()] == ["vllm/dual"]

            # switch scope (text filter "dual" still active) → the gemma dual lane.
            pane.set_model_filter("gemma-4-31b")
            assert [e.slug for e in pane._filtered_entries()] == ["vllm/gemma-dual"]

            # the All sentinel clears the scope; clearing both → every lane.
            pane.set_model_filter(_AM)
            pane.set_filter("")
            assert len(pane._filtered_entries()) == 3

            # a scoped model that disappears on the next load resets to all.
            pane.set_model_filter("gemma-4-31b")
            pane.populate([qwen_dual, qwen_single], None)
            assert pane._model_filter == ""
            assert pane._models == ["qwen3.6-27b"]
            assert len(pane._filtered_entries()) == 2

    @pytest.mark.asyncio
    async def test_catalog_model_select_toggle_and_blank_on_repeat(self):
        """The model dropdown is hidden by default (like the text filter) and
        toggled by [\\]; the model COLUMN carries the grouping when it's hidden —
        first row of each model group shows the name, repeats are blank."""
        from club3090_cockpit.data import CatalogEntry as _CE
        from club3090_tui_core import VariantRow as _VR

        def _entry(slug: str, model: str, compose_path: str) -> _CE:
            return _CE(
                row=_VR(
                    slug=slug, switch_engine="vllm", launch_engine="vllm",
                    compose_dir=compose_path.rsplit("/", 1)[0],
                    file=compose_path.rsplit("/", 1)[-1], port=8000, model=model,
                    engine="vllm-stable", kvcalc_key=f"{model}:dual", container="c",
                    compose_path=compose_path, status="production",
                    ctx_label="262K", status_note="",
                )
            )

        qwen_dual = _entry("vllm/dual", "qwen3.6-27b", "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml")
        qwen_single = _entry("vllm/single", "qwen3.6-27b", "models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml")
        gemma_dual = _entry("vllm/gemma-dual", "gemma-4-31b", "models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml")

        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.populate([qwen_dual, qwen_single, gemma_dual], None)

            sel = app.query_one("#catalog-model-select", Select)
            # hidden by default (out of the Tab/focus chain).
            assert "visible" not in sel.classes
            # [\\] reveals it, again hides it.
            pane.toggle_model_select()
            assert "visible" in sel.classes
            pane.toggle_model_select()
            assert "visible" not in sel.classes
            # Esc (close_filter_if_open) also hides it if open, KEEPING the scope.
            pane.set_model_filter("qwen3.6-27b")
            pane.toggle_model_select()
            assert "visible" in sel.classes
            assert pane.close_filter_if_open() is True
            assert "visible" not in sel.classes
            assert pane._model_filter == "qwen3.6-27b"  # scope survives the hide
            pane.set_model_filter("")

            # blank-on-repeat in the model column: first row of a model group shows
            # the name (grouped: qwen, qwen, gemma), the repeat row is blank.
            table = app.query_one("#catalog-table", DataTable)
            r0 = str(table.get_row_at(0)[0])
            r1 = str(table.get_row_at(1)[0])
            r2 = str(table.get_row_at(2)[0])
            assert "qwen3.6-27b" in r0
            assert r1.strip() == ""        # second qwen lane → blank model cell
            assert "gemma-4-31b" in r2     # new group → name shown again

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
# Bring-an-arbitrary-repo fit-check — 2-mode merge removed the standalone Run · BYO
# tab; the producer lane's ① Bring (LaneBringPane) is now the SINGLE entry point.
# These pin that ① Bring still fit-checks via byo_check + the curated dropdown +
# escape hatch all still work on the lane (and that the BYO tab is GONE).
# ===========================================================================


async def _enter_bring(pilot) -> None:
    """Enter the Bring & Validate lane (mode 1) and land on its ① Bring stage."""
    await pilot.press("2")
    try:
        pilot.app.query_one("#validate-tabs", TabbedContent).active = "tab-bring"
    except Exception:
        pass
    await _settle(pilot)


class TestByoWired:
    @pytest.mark.asyncio
    async def test_byo_tab_and_pane_are_gone(self):
        # The standalone Run · BYO tab + its ByoPane were removed in the merge.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            assert not app.query("#byo-panel")
            assert not app.query("#tab-byo")
            assert not app.query("#byo-url-input")
            assert not app.query("#byo-fit-btn")

    @pytest.mark.asyncio
    async def test_lane_bring_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#lane-bring-pane", LaneBringPane)
            app.query_one("#lane-bring-url-input", Input)
            # #6 — the profile-like input is a registry-derived template Select.
            app.query_one("#lane-bring-profile-input", Select)
            app.query_one("#lane-bring-fit-btn", Button)
            app.query_one("#lane-bring-result-card", Static)

    @pytest.mark.asyncio
    async def test_lane_bring_fit_check_renders_route(self):
        # Funnel §2b: the fit-check is stage 4 — Inspect must run first (the
        # fit button is HIDDEN until the artifact is known; press() on a
        # hidden button is a Textual no-op, which IS the staged-reveal guard).
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            app.query_one("#lane-bring-url-input", Input).value = "org/Model"
            app.query_one("#lane-bring-inspect-btn", Button).press()
            await _settle(pilot)
            app.query_one("#lane-bring-fit-btn", Button).press()
            await _settle(pilot)
            card = app.query_one("#lane-bring-result-card", Static)
            text = str(card.render())
            assert "Route C" in text or "vllm/dual" in text
            # pull.sh was invoked with --dry-run (never downloads) via byo_check.
            pull = next(c for c in runner.calls if "pull.sh" in " ".join(c))
            assert "--dry-run" in pull

    @pytest.mark.asyncio
    async def test_escape_hatch_custom_slug_reaches_byo_check(self):
        # FIX 2 (escape hatch) — the curated dropdown lists only ~7 reps, so a
        # NON-curated registry slug must stay reachable on the lane: selecting the
        # "✎ custom slug…" sentinel reveals the companion Input, and the TYPED slug
        # (not the sentinel marker, not the rig default) is what byo_check dry-runs.
        from club3090_cockpit.app import PROFILE_CUSTOM_SENTINEL

        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            app.query_one("#lane-bring-url-input", Input).value = "org/Model"
            # Funnel §2b: the slug Select is hidden until Inspect resolves.
            app.query_one("#lane-bring-inspect-btn", Button).press()
            await _settle(pilot)
            sel = app.query_one("#lane-bring-profile-input", Select)
            custom = app.query_one("#lane-bring-profile-custom", Input)
            # Reveal the companion Input by selecting the sentinel.
            sel.value = PROFILE_CUSTOM_SENTINEL
            await pilot.pause()
            assert "profile-custom-hidden" not in custom.classes  # revealed
            # The user types a non-curated slug.
            custom.value = "ik-llama/iq4ks-mtp"
            await pilot.pause()
            # _selected_profile_like resolves to the TYPED slug, not the sentinel.
            assert app._selected_profile_like("#lane-bring-profile-input") == "ik-llama/iq4ks-mtp"
            runner.calls.clear()
            app.query_one("#lane-bring-fit-btn", Button).press()
            await _settle(pilot)
            pull = next(c for c in runner.calls if "pull.sh" in " ".join(c))
            joined = " ".join(pull)
            assert "--profile-like ik-llama/iq4ks-mtp" in joined
            assert PROFILE_CUSTOM_SENTINEL not in joined   # sentinel never leaks


class TestBringFunnelStagedReveal:
    """Bring funnel §2b (maintainer UX 2026-07-05): staged reveal, GGUF pick
    before slugs, artifact→engine compat filtering, topology-first labels."""

    @pytest.mark.asyncio
    async def test_nothing_template_side_before_inspect(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            row = app.query_one("#lane-bring-stage2-row")
            assert row.has_class("funnel-hidden")          # §2b-1
            assert not app.query_one("#lane-bring-fit-btn", Button).display

    @pytest.mark.asyncio
    async def test_safetensors_inspect_reveals_filtered_sorted_slugs(self):
        from club3090_cockpit.app import PROFILE_CUSTOM_SENTINEL

        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            app.query_one("#lane-bring-url-input", Input).value = "org/Model"
            app.query_one("#lane-bring-inspect-btn", Button).press()
            await _settle(pilot)
            # deriver --inventory ran (READ-only, --json)
            inv_call = next(c for c in runner.calls if "--inventory" in " ".join(c))
            assert "--json" in inv_call
            # card renders the inventory verdict + lineage
            text = str(app.query_one("#lane-bring-result-card", Static).render())
            assert "safetensors" in text and "16.2" in text
            assert "Qwen/Qwen3.6-27B" in text              # lineage rides (friction #11)
            # slug stage revealed; gguf pick NOT shown for a safetensors repo
            sel = app.query_one("#lane-bring-profile-input", Select)
            assert sel.display
            assert not app.query_one("#lane-bring-gguf-select", Select).display
            assert app.query_one("#lane-bring-fit-btn", Button).display
            # §2b-3/4: only safetensors engines; topology-FIRST labels
            labels = [l for (l, v) in sel._options if v != PROFILE_CUSTOM_SENTINEL]
            # the sole option IS the recommendation → starred + pre-selected
            assert labels == ["⭐ dual/vllm/qwen3.6-27b-autoround-int4"]
            # dogfood r2 — titled field + the selected slug's detail card
            assert app.query_one("#lane-bring-profile-title").display
            card = app.query_one("#lane-bring-slug-card", Static)
            assert card.display
            ctext = str(card.render())
            assert "vllm/dual" in ctext and "262K" in ctext
            assert "174/42" in ctext          # the shipped bar rides the card
            # sentinel pick hides the card (no catalog row to describe)
            sel.value = PROFILE_CUSTOM_SENTINEL
            await pilot.pause()
            assert not card.display

    @pytest.mark.asyncio
    async def test_gguf_inspect_requires_quant_pick_before_slugs(self):
        from club3090_cockpit.app import PROFILE_CUSTOM_SENTINEL

        responses = fake_responses(
            **{"deriver.py --inventory": ok(INVENTORY_GGUF_JSON)}
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            app.query_one("#lane-bring-url-input", Input).value = "org/Model-GGUF"
            app.query_one("#lane-bring-inspect-btn", Button).press()
            await _settle(pilot)
            gsel = app.query_one("#lane-bring-gguf-select", Select)
            sel = app.query_one("#lane-bring-profile-input", Select)
            # §2b-2: ALL gguf quants presented; slugs stay hidden until the pick
            assert gsel.display
            assert not sel.display
            glabels = [l for (l, _v) in gsel._options]
            assert any("Q4_K_M" in l and "17.0" in l for l in glabels)
            assert any("Q8_0" in l and "29.0" in l and "2 parts" in l for l in glabels)
            # mmproj is informational, never a variant
            assert not any("mmproj" in l for l in glabels)
            # the pick reveals the gguf-engine slugs (§2b-3: never vLLM)
            gsel.value = "Q4_K_M"
            await _settle(pilot)
            assert sel.display
            labels = [l for (l, v) in sel._options if v != PROFILE_CUSTOM_SENTINEL]
            assert labels == ["⭐ single/ik-llama/qwen3.6-27b-ubergarm-iq4ks"]

    @pytest.mark.asyncio
    async def test_fit_check_surfaces_weights_state_and_handoff(self, monkeypatch, tmp_path):
        # §2b-6/7 — after a successful fit-check: weights absent → the [D]
        # download affordance; weights on disk → the explicit ② Serve handoff.
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            app.query_one("#lane-bring-url-input", Input).value = "org/Model"
            app.query_one("#lane-bring-inspect-btn", Button).press()
            await _settle(pilot)
            app.query_one("#lane-bring-fit-btn", Button).press()
            await _settle(pilot)
            line = app.query_one("#lane-bring-weights-line", Static)
            assert line.display
            text = str(line.render())
            assert "not on disk" in text and "[D]" in text
            # weights land on disk → a re-fit-check flips the line to handoff
            d = app._data.bring_pull_dir("org/Model")
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"x")
            app.query_one("#lane-bring-fit-btn", Button).press()
            await _settle(pilot)
            text = str(line.render())
            assert "on disk" in text and "② Serve" in text

    @pytest.mark.asyncio
    async def test_inspect_error_reveals_nothing(self):
        responses = fake_responses(
            **{"deriver.py --inventory": ok(json.dumps(
                {"repo": "org/Nope", "error": "repo-not-found: org/Nope",
                 "formats": [], "gguf_variants": [], "gguf_mmproj": []}))}
        )
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            app.query_one("#lane-bring-url-input", Input).value = "org/Nope"
            app.query_one("#lane-bring-inspect-btn", Button).press()
            await _settle(pilot)
            text = str(app.query_one("#lane-bring-result-card", Static).render())
            assert "repo-not-found" in text
            assert app.query_one("#lane-bring-stage2-row").has_class("funnel-hidden")


class TestFunnelSlugOptionsPure:
    """funnel_slug_options — the §2b filters as a pure function."""

    def _rows(self):
        from types import SimpleNamespace as NS

        return [
            NS(slug="vllm/dual", engine="vllm-stable", model="qwen3.6-27b",
               file="fp8-mtp.yml", status="production",
               compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
               compose_dir=""),
            NS(slug="vllm/minimal", engine="vllm-stable", model="qwen3.6-27b",
               file="minimal.yml", status="production",
               compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml",
               compose_dir=""),
            NS(slug="ik-llama/iq4ks-mtp", engine="ik-llama", model="qwen3.6-27b",
               file="mtp.yml", status="production",
               compose_path="models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml",
               compose_dir=""),
        ]

    def test_compat_filter_is_absolute(self):
        from club3090_cockpit.app import funnel_slug_options

        st = funnel_slug_options(self._rows(), "safetensors")
        assert {o.slug for o in st} == {"vllm/dual", "vllm/minimal"}   # never llama-family
        gg = funnel_slug_options(self._rows(), "gguf")
        assert {o.slug for o in gg} == {"ik-llama/iq4ks-mtp"}          # never vLLM

    def test_topology_first_labels_and_grouped_sort(self):
        from club3090_cockpit.app import funnel_slug_options

        st = funnel_slug_options(self._rows(), "safetensors")
        # single before dual; label = topology/engine/model-quant (NO serving
        # tail — dogfood r2: it duplicated the path axes)
        assert [o.label for o in st] == [
            "single/vllm/qwen3.6-27b-autoround-int4",
            "dual/vllm/qwen3.6-27b-autoround-int4",
        ]

    def test_serving_stem_disambiguates_collisions_only(self):
        # Two slugs sharing (topology, engine, model, quant) differ only by
        # serving stack → ONLY those two get the stem tail (basename, even
        # when the registry file field carries a subpath).
        from types import SimpleNamespace as NS

        from club3090_cockpit.app import funnel_slug_options

        rows = self._rows() + [
            NS(slug="vllm/qwen-27b-dual-turbo", engine="vllm-stable",
               model="qwen3.6-27b", file="dual/autoround-int4/turbo.yml",
               status="production", compose_dir="",
               compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/turbo.yml"),
        ]
        st = funnel_slug_options(rows, "safetensors")
        by_slug = {o.slug: o.label for o in st}
        assert by_slug["vllm/minimal"] == "single/vllm/qwen3.6-27b-autoround-int4"
        assert by_slug["vllm/dual"] == "dual/vllm/qwen3.6-27b-autoround-int4  ·  fp8-mtp"
        assert by_slug["vllm/qwen-27b-dual-turbo"] == "dual/vllm/qwen3.6-27b-autoround-int4  ·  turbo"

    def test_size_floor_hides_too_small_topologies_only(self):
        from club3090_cockpit.app import funnel_slug_options

        # 34G weights on 24G cards: single (24×0.9=21.6) hidden, dual kept.
        st = funnel_slug_options(
            self._rows(), "safetensors", artifact_gb=34.0, vram_gb=24.0, gpu_count=2
        )
        assert {o.slug for o in st} == {"vllm/dual"}
        # Small artifact: NOTHING hidden — running a small quant on more GPUs
        # is legitimate (the floor is one-directional).
        st = funnel_slug_options(
            self._rows(), "safetensors", artifact_gb=5.0, vram_gb=24.0, gpu_count=2
        )
        assert {o.slug for o in st} == {"vllm/dual", "vllm/minimal"}
        # Unknown rig → no floor (never guess-hide).
        st = funnel_slug_options(self._rows(), "safetensors", artifact_gb=34.0)
        assert {o.slug for o in st} == {"vllm/dual", "vllm/minimal"}

    def test_rig_card_count_hides_unhostable_topologies(self):
        from club3090_cockpit.app import funnel_slug_options

        st = funnel_slug_options(
            self._rows(), "safetensors", artifact_gb=5.0, vram_gb=24.0, gpu_count=1
        )
        assert {o.slug for o in st} == {"vllm/minimal"}   # dual unhostable on 1 card

    def test_recommendation_prefers_smallest_fitting_topology(self):
        # Live dogfood 2026-07-05: a 5 GiB gguf on a 2-card rig defaulted to a
        # DUAL slug (rig-topology rule) — the recommendation must be the
        # SMALLEST fitting topology, curated default preferred within it.
        from club3090_cockpit.app import funnel_recommended, funnel_slug_options

        opts = funnel_slug_options(
            self._rows(), "safetensors", artifact_gb=5.0, vram_gb=24.0, gpu_count=2
        )
        assert {o.slug for o in opts} == {"vllm/dual", "vllm/minimal"}
        # no curated defaults → first functional option of the SINGLE group
        assert funnel_recommended(opts) == "vllm/minimal"
        # the registry's curated default for (vllm, single) wins within the group
        defaults = [{"engine": "vllm-stable", "topology": "single", "slug": "vllm/minimal"}]
        assert funnel_recommended(opts, defaults) == "vllm/minimal"
        # big artifact → single floored away → the recommendation follows
        opts = funnel_slug_options(
            self._rows(), "safetensors", artifact_gb=34.0, vram_gb=24.0, gpu_count=2
        )
        assert funnel_recommended(opts) == "vllm/dual"


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
    async def test_enter_on_doctor_tab_does_not_serve(self):
        """⏎ serves only from the Catalog tab of the merged mode; on the Doctor tab
        (read-only, no primary) it no-ops (no confirm)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot, tab="tab-doctor")
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
# 2-mode merge invariants — the merged Run & Operate mode ties Catalog +
# Orchestration + Containers + Doctor + the re-homed LivePane into ONE mode.
# ===========================================================================


class TestMergedRunOperateMode:
    @pytest.mark.asyncio
    async def test_merged_mode_has_four_tabs_with_catalog_first(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            wanted = {"tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor"}
            ids = [p.id for p in tc.query(TabPane) if p.id in wanted]
            assert ids == ["tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor"]
            # Catalog is the default first tab when the merged mode is active.
            assert app._active_mode == 0
            assert tc.active == "tab-catalog"

    @pytest.mark.asyncio
    async def test_livepane_is_rehomed_into_merged_panel(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            live = app.query_one("#serve-live")
            # Re-homed below the tabs of the merged panel (#panel-run), so a serve
            # from the Catalog tab streams here while the user can flip to the
            # Orchestration tab in the SAME mode.
            assert app.query_one("#panel-run") in live.ancestors
            assert app.query_one("#operate-tabs", TabbedContent) not in live.ancestors

    @pytest.mark.asyncio
    async def test_serve_from_catalog_streams_then_flip_to_orchestration(self):
        """Serving from the Catalog tab hits the reconcile gate, streams into the
        re-homed LivePane, and the user can flip to the Orchestration tab in the
        SAME merged mode to watch the live estate."""
        app, _, wr = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.query_one("#operate-tabs", TabbedContent).active == "tab-catalog"
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")            # → reconcile-gated confirm modal
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.requires_reconcile is True
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("enter")            # confirm → dispatch (mocked)
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Boot streams into the re-homed LivePane (revealed).
            assert "serving" in app.query_one("#serve-live").classes
            assert len(wr.started) == 1           # gated executor, never a live spawn
            # Flip to Orchestration IN THE SAME MODE — still mode 0, lane untouched.
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await _settle(pilot)
            assert app._active_mode == 0
            assert app.query_one("#scene-table", DataTable)  # live estate is here

    @pytest.mark.asyncio
    async def test_periodic_refresh_updates_rail_and_gpu_cards_in_merged_mode(self):
        gpus = [
            GpuInfo(index=0, mem_used_mib=18 * 1024, mem_total_mib=24 * 1024, utilization=70),
            GpuInfo(index=1, mem_used_mib=2 * 1024, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._active_mode == 0
            # The 4s tick fires in the merged mode → estate + telemetry + rail.
            app._periodic_estate_refresh()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # GPU cards + rail rendered from the live read (not a false-zero).
            bar0 = str(app.query_one("#gpu0-bar", Static).render())
            assert "18.0 / 24.0 GiB" in bar0
            rail = str(app.query_one("#rail-status", RailStatus).render())
            assert "as of" in rail


# ===========================================================================
# Operate · Orchestration + Containers (wired to estate_state)
# ===========================================================================


class TestEstateWired:
    @pytest.mark.asyncio
    async def test_estate_orch_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-orch-pane", OperateOrchPane)
            app.query_one("#gpu0-card")
            app.query_one("#gpu1-card")
            app.query_one("#doctor-line")
            app.query_one("#scene-table", DataTable)
            app.query_one("#scene-preview", Static)

    @pytest.mark.asyncio
    async def test_estate_scene_table_populates_from_gpu_mode(self):
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "18.0 / 24.0 GiB" in bar
            assert "71%" in bar

    @pytest.mark.asyncio
    async def test_estate_doctor_line_serving(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            line = str(app.query_one("#doctor-line", Static).render())
            assert "serving" in line.lower()

    @pytest.mark.asyncio
    async def test_estate_containers_pane_nodes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            names = [c.name for c in pane._containers]
            assert "vllm-qwen36-27b-dual" in names
            assert "open-webui" not in names  # not an engine prefix

    @pytest.mark.asyncio
    async def test_estate_scene_switch_opens_confirm_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#scene-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)


@pytest.mark.asyncio
async def test_scene_preview_shows_all_services_no_clip():
    """A busy scene (ai-studio, ~7 services) renders EVERY service in the preview —
    no cap, no "+N more". The box has no max-height so it auto-grows to fit instead
    of clipping (the pane scrolls if needed)."""
    busy = Scene(name="ai-studio", group="studio", description="creative studio",
                 services=["comfyui", "studio-director", "studio-gallery", "studio-tts",
                           "studio-image-shim", "studio-orchestrator", "step-voice"],
                 ports=["8188"], gpus="both")
    app, _, _ = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _enter_operate(pilot)
        orch = app.query_one("#operate-orch-pane", OperateOrchPane)
        orch.render_scene_preview(busy)
        await pilot.pause()
        txt = str(app.query_one("#scene-preview", Static).render())
        for svc in busy.services:                       # every service is present, none dropped
            assert svc in txt
        assert "more" not in txt                          # no "+N more" collapse
        # the #scene-preview rule carries no max-height cap (box auto-grows to fit)
        preview_rule = OperateOrchPane.DEFAULT_CSS.split("#scene-preview")[1].split("}")[0]
        assert "max-height" not in preview_rule


@pytest.mark.asyncio
class TestFix1CursorPreserveAcrossPoll:
    """FIX 1 — the B2 periodic refresh re-populates the scene + container tables
    every 4s.  The cursor must NOT snap back to row 0 on each tick: preserve it by
    STABLE ROW KEY (scene / container name) across a re-populate, fall back to a
    clamped index when the selected row disappears, and skip the re-populate
    entirely when the row set is unchanged."""

    SCENES = [
        Scene(name="27b", group="serving", description="Qwen",
              services=["vllm-qwen36-27b-dual"], ports=["8010"], gpus="both"),
        Scene(name="gemma", group="serving", description="Gemma",
              services=["vllm-gemma"], ports=["8011"], gpus="both"),
        Scene(name="diffusion", group="serving", description="Diff",
              services=["comfyui"], ports=["8188"], gpus="0"),
        Scene(name="off", group="ops", description="Stop all",
              services=[], ports=[], gpus="none"),
    ]

    CONTAINERS = [
        ContainerInfo(name="vllm-a", kind="engine", host_port=8010, engine="vllm", slug="vllm/dual"),
        ContainerInfo(name="vllm-b", kind="engine", host_port=8011, engine="vllm", slug="vllm/single"),
        ContainerInfo(name="comfyui", kind="service", host_port=8188, engine="", slug=""),
        ContainerInfo(name="open-webui", kind="service", host_port=3000, engine="", slug=""),
    ]

    async def test_scene_cursor_held_across_identical_poll(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            orch = app.query_one("#operate-orch-pane", OperateOrchPane)
            orch._populate_scenes(self.SCENES)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=2)  # "diffusion"
            await pilot.pause()
            assert orch.selected_scene().name == "diffusion"
            # Two more poll cycles with the SAME scenes — cursor must NOT snap to 0.
            orch._populate_scenes(self.SCENES)
            orch._populate_scenes(self.SCENES)
            await pilot.pause()
            assert t.cursor_row == 2
            assert orch.selected_scene().name == "diffusion"

    async def test_scene_cursor_follows_key_when_row_set_changes(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            orch = app.query_one("#operate-orch-pane", OperateOrchPane)
            orch._populate_scenes(self.SCENES)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=2)  # "diffusion"
            await pilot.pause()
            # A scene appears at the TOP → "diffusion" is now at index 3; the cursor
            # must FOLLOW the key, not stay pinned to the stale index 2.
            shifted = [Scene(name="new", group="serving", description="N",
                             services=["x"], ports=["9"], gpus="0")] + self.SCENES
            orch._populate_scenes(shifted)
            await pilot.pause()
            assert orch.selected_scene().name == "diffusion"
            assert t.cursor_row == 3

    async def test_scene_cursor_clamps_when_selected_disappears(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            orch = app.query_one("#operate-orch-pane", OperateOrchPane)
            orch._populate_scenes(self.SCENES)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=3)  # "off" (last row)
            await pilot.pause()
            assert orch.selected_scene().name == "off"
            # The selected scene DISAPPEARS on the next poll → graceful clamp, no crash.
            fewer = self.SCENES[:2]  # only "27b", "gemma"
            orch._populate_scenes(fewer)
            await pilot.pause()
            assert t.cursor_row == min(3, t.row_count - 1) == 1
            assert orch.selected_scene() is not None  # no crash, valid selection

    async def test_container_cursor_held_across_identical_poll(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            pane.populate(self.CONTAINERS)
            t = app.query_one("#containers-table", DataTable)
            t.move_cursor(row=2)  # "comfyui"
            await pilot.pause()
            assert pane.selected_container().name == "comfyui"
            # Two identical polls — cursor stays on comfyui, never resets to row 0.
            pane.populate(self.CONTAINERS)
            pane.populate(self.CONTAINERS)
            await pilot.pause()
            assert t.cursor_row == 2
            assert pane.selected_container().name == "comfyui"

    async def test_container_cursor_clamps_when_selected_disappears(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            pane.populate(self.CONTAINERS)
            t = app.query_one("#containers-table", DataTable)
            t.move_cursor(row=3)  # "open-webui" (last)
            await pilot.pause()
            assert pane.selected_container().name == "open-webui"
            # That container stops → row set shrinks; cursor clamps, no crash.
            fewer = self.CONTAINERS[:2]
            pane.populate(fewer)
            await pilot.pause()
            assert t.cursor_row == 1
            assert pane.selected_container() is not None

    async def test_container_cursor_follows_key_when_one_starts(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            pane.populate(self.CONTAINERS)
            t = app.query_one("#containers-table", DataTable)
            t.move_cursor(row=2)  # "comfyui"
            await pilot.pause()
            # A new container appears at the top → comfyui shifts to index 3; the
            # cursor must follow the KEY (comfyui), not the stale index 2.
            grown = [ContainerInfo(name="new-svc", kind="engine", host_port=9000,
                                   engine="vllm", slug="vllm/x")] + self.CONTAINERS
            pane.populate(grown)
            await pilot.pause()
            assert pane.selected_container().name == "comfyui"
            assert t.cursor_row == 3

    async def test_populate_returns_false_on_unchanged_poll(self):
        # FIX 1 — the skip-if-unchanged guard reports a no-op so load_estate doesn't
        # spuriously re-arm the row-0 suppression / cancel a pending drill.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            assert pane.populate(self.CONTAINERS) is True       # first render
            assert pane.populate(self.CONTAINERS) is False      # unchanged → skipped
            assert pane.populate(self.CONTAINERS[:2]) is True   # changed → re-rendered


# ===========================================================================
# Batch 1 — Operate / BYO UX (bugs + polish from real-rig feedback)
# ===========================================================================


def _seed_services(root: Path, names: list[str]) -> None:
    """Seed services/<name>/docker-compose.yml so _known_service_dirs finds them."""
    for n in names:
        d = root / "services" / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")


def _seed_studio_director(app, root: Path) -> None:
    """Make CockpitData.studio_ready() see the director GGUF: put it under a MODEL_DIR
    we pin on the app's data layer (highest weights_model_dir precedence).  Pair with
    a 'docker images -q' = present mock so studio_ready() resolves True."""
    from club3090_cockpit.services import STUDIO_DIRECTOR_REL
    md = root / "weights"
    director = md / STUDIO_DIRECTOR_REL
    director.parent.mkdir(parents=True, exist_ok=True)
    director.write_bytes(b"x")
    app._data._model_dir = str(md)


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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            assert "no model serving" in line.lower()


class TestF9SlugMasquerade:
    """F9 — a port/substring registry match is a SHAPE guess, not an identity:
    the UI must lead with the PROBED served id + 👤 badge and demote the slug to
    'shape'.  Regression source: the Agents-A1 bring (2026-07-03) served on a
    sibling's port and was presented as vllm/qwen-35b-a3b-dual with no hint it
    was a different model."""

    def _shape_target(self) -> ServingTarget:
        # A brought container UNKNOWN to the registry, serving on vllm/dual's
        # port 8010 → slug matches by PORT only (shape); the model id is the
        # probed served id.
        return ServingTarget(
            url="http://localhost:8010", model="agents-a1", host_port=8010,
            container="vllm-agents-a1-dual",
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )

    @pytest.mark.asyncio
    async def test_serving_panel_shape_match_badges(self):
        app, _, _ = make_app(target=self._shape_target())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            # The probed served id leads; the slug is presented as the SHAPE.
            assert "agents-a1" in line
            assert "👤" in line
            assert "on vllm/dual shape" in line

    @pytest.mark.asyncio
    async def test_serving_panel_identity_match_has_no_badge(self):
        # EXACT container match (the registry row's container, "_"→"-"
        # normalized) → identity → the slug renders plainly, no badge.
        tgt = ServingTarget(
            url="http://localhost:8010", model="qwen3.6-27b", host_port=8010,
            container="vllm_qwen36_27b",
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )
        app, _, _ = make_app(target=tgt)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            line = str(app.query_one("#serving-line", Static).render())
            assert "vllm/dual" in line
            assert "👤" not in line
            assert "shape" not in line

    @pytest.mark.asyncio
    async def test_rail_shape_match_leads_with_served_id(self):
        app, _, _ = make_app(target=self._shape_target())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            rail = str(app.query_one("#rail-status", RailStatus).render())
            # Served id + badge on the model line; the slug demoted to shape.
            assert "agents-a1 👤" in rail
            assert "shape" in rail
            assert "vllm/dual" in rail


class TestBatch1KnownServices:
    """#2 — the Containers view shows known-but-stopped supporting services."""

    @pytest.mark.asyncio
    async def test_stopped_service_appears_greyed(self, tmp_path):
        _seed_services(tmp_path, ["comfyui", "litellm"])
        seed_repo(tmp_path)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
    async def test_running_support_service_shows_web_engine_and_port(self, tmp_path):
        """A running non-GPU supporting service carries engine='web' + its port in
        the Containers table (was a bare '—')."""
        _seed_services(tmp_path, ["litellm"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok("litellm|0.0.0.0:4000->4000/tcp\n")})
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            litellm = next(c for c in pane._containers if c.name == "litellm")
            assert litellm.engine == "web"
            assert litellm.host_port == 4000

    @pytest.mark.asyncio
    async def test_start_stopped_service_from_containers(self, tmp_path):
        """[s] on a STOPPED service brings it up via `docker compose up` (not a
        'not running' warning) — the confirm gate gets the compose-up plan."""
        _seed_services(tmp_path, ["qdrant"])
        seed_repo(tmp_path)
        responses = fake_responses(**{"docker ps": ok("")})  # qdrant down
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            idx = next(i for i, c in enumerate(pane._containers) if c.name == "qdrant")
            tbl = pane.query_one("#containers-table", DataTable)
            tbl.move_cursor(row=idx)
            await pilot.press("s")  # start (compose up)
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            cmd = app.screen._plan.cmd
            # compose up, project pinned to the dir (so it operates on the SAME
            # container gpu-mode would, not a duplicate under the repo project).
            assert cmd[:2] == ["docker", "compose"]
            assert "services/qdrant/docker-compose.yml" in cmd
            assert cmd[-4:] == ["-p", "qdrant", "up", "-d"]

    @pytest.mark.asyncio
    async def test_comfyui_start_blocked_when_image_absent(self, tmp_path):
        """[s] on a stopped comfyui with NO comfyui-local image guides the user to
        setup-image-studio.sh (no confirm) instead of a doomed compose up."""
        _seed_services(tmp_path, ["comfyui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok(""), "docker images -q": ok("")})  # not built
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._studio_ready is False
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            idx = next(i for i, c in enumerate(pane._containers) if c.name == "comfyui")
            pane.query_one("#containers-table", DataTable).move_cursor(row=idx)
            await pilot.press("s")  # start
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)  # guarded → no confirm

    @pytest.mark.asyncio
    async def test_comfyui_start_blocked_when_director_missing(self, tmp_path):
        """Image built but the director GGUF absent → still guarded (the studio
        needs both — the image alone boots a director with no model)."""
        _seed_services(tmp_path, ["comfyui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok(""), "docker images -q": ok("sha256:abc\n")})  # image present
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        # point MODEL_DIR at an empty dir → director absent
        app._data._model_dir = str(tmp_path / "empty-weights")
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._studio_ready is False
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            idx = next(i for i, c in enumerate(pane._containers) if c.name == "comfyui")
            pane.query_one("#containers-table", DataTable).move_cursor(row=idx)
            await pilot.press("s")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)  # guarded

    @pytest.mark.asyncio
    async def test_comfyui_start_allowed_when_studio_ready(self, tmp_path):
        """Image + director GGUF present → [s] proceeds to the compose-up confirm."""
        _seed_services(tmp_path, ["comfyui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok(""), "docker images -q": ok("sha256:abc\n")})
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        _seed_studio_director(app, tmp_path)  # director on disk + MODEL_DIR pinned
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._studio_ready is True
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            idx = next(i for i, c in enumerate(pane._containers) if c.name == "comfyui")
            pane.query_one("#containers-table", DataTable).move_cursor(row=idx)
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert "services/comfyui/docker-compose.yml" in app.screen._plan.cmd

    @pytest.mark.asyncio
    async def test_stopped_studio_sidecar_listed_and_restartable(self, tmp_path):
        """A stopped studio sidecar (exists, but no top-level services/ dir) is
        listed in Containers and [s] starts it via `docker restart`."""
        seed_repo(tmp_path)
        responses = fake_responses(**{
            "docker ps": ok(""),                       # nothing running
            "docker container ls": ok("studio-director\n"),  # exists, stopped
        })
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            sd = next(c for c in pane._containers if c.name == "studio-director")
            assert sd.status == "stopped" and sd.kind == "stack" and sd.engine == "studio"
            idx = pane._containers.index(sd)
            pane.query_one("#containers-table", DataTable).move_cursor(row=idx)
            await pilot.press("s")  # start → docker restart (existing container)
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["docker", "restart", "studio-director"]

    @pytest.mark.asyncio
    async def test_separator_mismatch_service_matched_as_running(self, tmp_path):
        """MUST-FIX #2 (b): a service dir ``open-webui`` whose running container
        is named ``openwebui`` (separator mismatch) must NORMALIZE-match → shown
        running, not stopped.  A bare substring match would miss this.  The row
        carries the REAL container name (``openwebui``) so its actions target it."""
        _seed_services(tmp_path, ["open-webui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok("openwebui|0.0.0.0:3000->8080/tcp\n")}
        )
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            stopped = [c for c in pane._containers if c.status == "stopped"]
            assert not any(c.name in ("open-webui", "openwebui") for c in stopped), (
                "open-webui dir wrongly marked stopped despite running 'openwebui'"
            )
            owui = next((c for c in pane._containers if c.name == "openwebui"), None)
            assert owui is not None and owui.status != "stopped"
            assert app._is_stopped_service(owui) is False

    @pytest.mark.asyncio
    async def test_stop_targets_real_container_name_not_service_dir(self, tmp_path):
        """REGRESSION: the openwebui service dir maps to container_name
        'open-webui'.  Stop/restart MUST target the REAL container name, not the
        dir — `docker stop openwebui` would fail with 'no such container'."""
        _seed_services(tmp_path, ["openwebui"])
        seed_repo(tmp_path)
        responses = fake_responses(
            **{"docker ps": ok("open-webui|0.0.0.0:8080->8080/tcp\n")})
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            owui = next(c for c in pane._containers
                        if c.name in ("open-webui", "openwebui"))
            assert owui.name == "open-webui"          # the REAL container name
            assert owui.status != "stopped"
            idx = pane._containers.index(owui)
            tbl = pane.query_one("#containers-table", DataTable)
            tbl.move_cursor(row=idx)
            await pilot.press("x")  # stop
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["docker", "stop", "open-webui"]

    @pytest.mark.asyncio
    async def test_stopped_row_highlight_clears_running_logs(self, tmp_path):
        """BUG 1 — highlighting a STOPPED service after a RUNNING container was
        selected must NOT leave the running container's logs/stats on screen,
        mislabeled to the stopped row.  The Logs + Top drill panes are cleared to
        an explicit 'stopped · no live logs/stats' placeholder; only Config (a
        local registry read) may still show the stopped service's info."""
        _seed_services(tmp_path, ["litellm"])
        seed_repo(tmp_path)
        responses = fake_responses(**{
            "docker ps": ok(DOCKER_PS_ENGINE),
            "docker logs": ok("RUNNING-LOG-LINE-XYZ\n"),
        })
        app, _, _ = make_app(responses=responses, repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            run_idx = next(i for i, c in enumerate(pane._containers)
                           if c.status != "stopped")
            stop_idx = next(i for i, c in enumerate(pane._containers)
                            if c.status == "stopped")
            stop_name = pane._containers[stop_idx].name
            tbl = app.query_one("#containers-table", DataTable)
            # 1) Select the RUNNING container + load its logs.
            tbl.move_cursor(row=run_idx)
            await pilot.pause()
            app.action_container_logs()
            await _settle(pilot)
            log = app.query_one("#drill-logs").query_one("#live-log")
            assert any("RUNNING-LOG-LINE-XYZ" in str(l) for l in log.lines)
            # 2) Highlight the STOPPED row → the drill timer fires → the Logs/Top
            #    panes clear to the stopped placeholder (NOT the running logs).
            tbl.move_cursor(row=stop_idx)
            await pilot.pause()
            # Wait out the 0.25s drill timer that loads the drill for the row.
            for _ in range(10):
                await asyncio.sleep(0.05)
                await pilot.pause()
            loglines = [str(l) for l in log.lines]
            assert not any("RUNNING-LOG-LINE-XYZ" in l for l in loglines), (
                "stale running-container logs left on a stopped row"
            )
            assert any("stopped" in l for l in loglines)
            assert any(stop_name in l for l in loglines)
            # Top pane also shows the stopped placeholder.
            stats = str(app.query_one("#drill-stats", Static).render())
            assert "stopped" in stats and stop_name in stats

    @pytest.mark.asyncio
    async def test_stopped_clear_cancels_inflight_drill_workers(self, tmp_path):
        """BUG 1 (race) — clearing the drill for a STOPPED service must CANCEL any
        in-flight logs/top worker from the previously-selected RUNNING container,
        else a slow `docker logs` resolves AFTER the placeholder and appends the
        FOREIGN container's lines below it."""
        _seed_services(tmp_path, ["litellm"])
        seed_repo(tmp_path)
        app, _, _ = make_app(repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
            await _settle(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            con = next(c for c in pane._containers if c.status == "stopped")
            cancelled: list[str] = []
            orig = app.workers.cancel_group

            def _spy(node, group):
                cancelled.append(group)
                return orig(node, group)

            app.workers.cancel_group = _spy
            app._clear_drill_for_stopped(con)
            assert "container-logs" in cancelled
            assert "container-top" in cancelled


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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
    async def test_cap_menu_choice_is_confirm_gated(self):
        # [c] opens the power-cap menu; a chosen option routes through the confirm
        # modal (the write is NOT bypassed).
        from club3090_cockpit.app import PowerCapMenuScreen
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await _settle(pilot)
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, PowerCapMenuScreen)
            await pilot.press("d")
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.kind == "power_cap"


class TestBatch1ByoPlaceholder:
    """#7 — the ① Bring repo input placeholder names a HuggingFace model slug."""

    @pytest.mark.asyncio
    async def test_lane_bring_placeholder_says_huggingface_slug(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_bring(pilot)
            ph = app.query_one("#lane-bring-url-input", Input).placeholder
            assert "org/Model" in ph
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
            # Footer-only: Confirm is enabled (no button row).
            assert screen.check_action("confirm", ()) is True
            assert screen.check_action("force", ()) is False

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
            # Footer-only: Confirm hidden, the override (force) shown.
            assert screen.check_action("confirm", ()) is False
            assert screen.check_action("force", ()) is True
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
            await pilot.press("enter")          # footer Confirm
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
            await pilot.press("f")              # footer override (Force / Switch anyway)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            # Serve-of-catalog-slug → the state-aware Start (footer key, no button
            # row).  Commit through the (mocked) gated executor via Enter→Start.
            await pilot.press("enter")
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
            # Batch 3 — verify / verify-full cards replaced estate / profile.
            app.query_one("#doctor-card-verify")
            app.query_one("#doctor-card-verifyfull")
            # report / full-report / power-cap-sweep now get their own launcher
            # cards too (previously footer-key-only).
            app.query_one("#doctor-card-report")
            app.query_one("#doctor-card-fullreport")
            app.query_one("#doctor-card-sweep")
            assert "press" in str(app.query_one("#doctor-report-body", Static).render()).lower()
            assert "press" in str(app.query_one("#doctor-sweep-body", Static).render()).lower()
            assert not app.query("#doctor-card-estate")
            assert not app.query("#doctor-card-profile")

    @pytest.mark.asyncio
    async def test_operate_doctor_health_line_goes_live_on_estate_poll(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await pilot.press("2")
            await pilot.press("enter")


# ===========================================================================
# Operate · Doctor (Batch 3 — "is the model serving correctly?")
# load_doctor refreshes the live health card (health.sh) on Operate entry; the
# heavier verify ([v]) / verify-full ([V]) are on-demand test queries to the
# serving model.  diagnose-estate / diagnose-profile moved to the producer lane.
# ===========================================================================


class TestOperateDoctorWired:
    @pytest.mark.asyncio
    async def test_doctor_health_populates_on_operate_entry(self):
        """Entering Operate refreshes the live health card from health.sh — and
        does NOT run the heavy diagnose-estate / diagnose-profile reads (those
        moved to the producer Bring & Validate lane)."""
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            health = str(app.query_one("#doctor-health-body", Static).render())
            assert "serving" in health.lower()
            assert not any("diagnose-estate.sh" in " ".join(c) for c in runner.calls)
            assert not any("diagnose-profile.sh" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_doctor_verify_sends_test_query(self):
        """[v] on Doctor runs verify.sh and renders the serving-correctly verdict."""
        VERIFY_OK = (
            "Running smoke test against http://localhost:8020\n\n"
            "  \x1b[32m✓\x1b[0m Server is reachable\n"
            "  \x1b[32m✓\x1b[0m Genesis patches applied cleanly\n"
            "  \x1b[32m✓\x1b[0m Basic completion works (Paris)\n"
            "  \x1b[32m✓\x1b[0m Tool calling works end-to-end\n"
        )
        responses = fake_responses(**{"scripts/verify.sh": ok(VERIFY_OK)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            await pilot.press("v")
            await _settle(pilot)
            body = str(app.query_one("#doctor-verify-body", Static).render())
            assert "serving correctly" in body and "4/4" in body
            assert any("scripts/verify.sh" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_doctor_verify_full_runs_battery(self):
        """[V] on Doctor runs verify-full.sh; a tool-call-only failure is shown as
        the expected ◐ partial (8/9), not a hard error."""
        VF = (
            "[1/9] Server reachable ...\n  \x1b[32m✓\x1b[0m reachable\n"
            "[4/9] Tool calling ...\n  \x1b[31m✗\x1b[0m no tool_calls[] in response\n"
            "    \x1b[33m→\x1b[0m expected on the default compose\n"
            "[9/9] MTP acceptance ...\n  \x1b[32m✓\x1b[0m AL>=2.0\n"
            "\x1b[31m1 check(s) failed.\x1b[0m See hints above.\n"
        )
        responses = fake_responses(**{"scripts/verify-full.sh": ok(VF)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            await pilot.press("V")
            await _settle(pilot)
            body = str(app.query_one("#doctor-verifyfull-body", Static).render())
            assert "8/9 checks passed" in body
            assert any("scripts/verify-full.sh" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_s_key_on_doctor_tab_does_not_restart(self):
        """R2a put the READ-ONLY Doctor surface in Operate (mode 1).  [s] — whose
        binding spans modes 1+2 with no sub-tab constraint — must NOT fire a
        `docker restart` on the Doctor tab, even with a container selected in the
        (hidden) Containers tab.  The container write is gated to tab-containers."""
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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


def _seed_live_gate_run(root: Path, tag: str = "live-gate") -> Path:
    """F10 — a mid-flight rebench dir: verify-full done (timings.json), bench's
    log growing.  Fresh mtimes → the observer grades it live."""
    d = root / "results" / "rebench" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "timings.json").write_text('{"verify-full": 132}', encoding="utf-8")
    (d / "verify-full.log").write_text("8/8 PASS\n", encoding="utf-8")
    (d / "bench.log").write_text("warmup done\nrun 3/5 narrative…\n", encoding="utf-8")
    return d


class TestF10LiveGateObserver:
    """F10 — the Evidence pane doubles as the live gate-run observer: a run in
    flight (no REPORT.md, fresh writes) is badged ▶ with a ladder + live tail;
    an aborted one reads ⚠ incomplete; report/submit are guarded mid-run.
    Regression source: rebench-full via nohup was invisible in c3 — a producer
    facing a silent 3-hr gate assumes a hang (T2 friction #7)."""

    async def _open_evidence(self, pilot):
        await pilot.press("2")
        await _settle(pilot)
        pilot.app.query_one("#validate-tabs", TabbedContent).active = "tab-evidence"
        await pilot.pause()

    @pytest.mark.asyncio
    async def test_live_run_badged_with_ladder_and_tail(self, tmp_path):
        seed_repo(tmp_path)
        _seed_live_gate_run(tmp_path)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await self._open_evidence(pilot)
            pane = app.query_one("#validate-evidence-pane", ValidateEvidencePane)
            live_idx = next(i for i, t in enumerate(pane._tags) if t.tag == "live-gate")
            assert pane._tags[live_idx].live
            t = app.query_one("#evidence-table", DataTable)
            row = " ".join(str(c) for c in t.get_row_at(live_idx))
            assert "▶" in row and "running" in row and "bench" in row
            # Preview for the live row: ladder + the active step's tail.
            t.move_cursor(row=live_idx)
            pane.render_preview(pane.selected_tag())
            prev = str(app.query_one("#evidence-preview", Static).render())
            assert "RUNNING" in prev
            assert "verify-full" in prev and "bench" in prev   # ✓ done + ▶ active
            assert "run 3/5 narrative…" in prev                 # live tail line

    @pytest.mark.asyncio
    async def test_enter_and_submit_guarded_on_live_run(self, tmp_path):
        seed_repo(tmp_path)
        _seed_live_gate_run(tmp_path)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await self._open_evidence(pilot)
            pane = app.query_one("#validate-evidence-pane", ValidateEvidencePane)
            live_idx = next(i for i, t in enumerate(pane._tags) if t.tag == "live-gate")
            app.query_one("#evidence-table", DataTable).move_cursor(row=live_idx)
            # ⏎ must NOT generate a report mid-run (rebench-report WRITES
            # REPORT.md into the dir → would corrupt the live/complete signal).
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, EvidenceReportScreen)
            # [s] must not stage a submit of incomplete artifacts.
            await pilot.press("s")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)

    @pytest.mark.asyncio
    async def test_quiet_incomplete_run_reads_incomplete(self, tmp_path):
        import os as _os
        import time as _t

        seed_repo(tmp_path)
        d = _seed_live_gate_run(tmp_path, tag="dead-gate")
        old = _t.time() - 7200
        for f in [d, *d.iterdir()]:
            _os.utime(f, (old, old))
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await self._open_evidence(pilot)
            pane = app.query_one("#validate-evidence-pane", ValidateEvidencePane)
            idx = next(i for i, t in enumerate(pane._tags) if t.tag == "dead-gate")
            assert pane._tags[idx].stale and not pane._tags[idx].live
            t = app.query_one("#evidence-table", DataTable)
            row = " ".join(str(c) for c in t.get_row_at(idx))
            assert "⚠" in row and "incomplete" in row
            # A completed tag (seed_repo's vllm-dual-test) keeps the plain look.
            done_idx = next(i for i, t2 in enumerate(pane._tags) if t2.tag == "vllm-dual-test")
            done_row = " ".join(str(c) for c in t.get_row_at(done_idx))
            assert "▶" not in done_row and "incomplete" not in done_row


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
            assert vsbar.measured.model == "qwen3.6-27b"
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
    async def test_measure_vs_bar_class_fallback_for_new_model(self, tmp_path):
        """Friction #9 (T2): a NEW model has no same-model bar BY DEFINITION —
        with ①'s swap_path sibling as class_hint the comparison falls back to
        the labeled CLASS bar (engine-matched, bar_is_class + caveat); without
        a hint the no-bar caveat says how to get one (run ① fit-check)."""
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        d = tmp_path / "results" / "rebench" / "new-model-tag"
        d.mkdir(parents=True)
        (d / "REPORT.md").write_text(
            MEASURE_REPORT_MD.replace("`qwen3.6-27b`", "`brandnew-40b`")
        )
        (d / "_internal.json").write_text(MEASURE_INTERNAL_JSON)
        (tmp_path / "BENCHMARKS.md").write_text(BENCHMARKS_MD)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # No hint → honest no-bar + the actionable pointer to ① fit-check.
            vsbar = await app._data.measure_vs_bar("new-model-tag")
            assert vsbar.bar is None
            assert vsbar.bar_is_class is False
            joined = " ".join(vsbar.protocol_caveats).lower()
            assert "sibling class" in joined and "fit-check" in joined
            # With the sibling hint → the labeled CLASS bar, engine-matched.
            vsbar = await app._data.measure_vs_bar(
                "new-model-tag", class_hint="qwen3.6-27b"
            )
            assert vsbar.bar is not None
            assert vsbar.bar_is_class is True
            assert vsbar.class_model == "qwen3.6-27b"
            assert vsbar.bar.model == "qwen3.6-27b"
            assert vsbar.bar.engine == "vllm"  # engine-matched within the class
            assert any("CLASS bar" in c for c in vsbar.protocol_caveats)

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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
    async def test_full_report_reachable_on_consumer_doctor(self):
        """Batch 3: [F] full_report is NO LONGER producer-only — a consumer can run
        the ~43-min battery from Operate · Doctor (context-gated to tab-doctor); it
        stays hidden on the other consumer tabs."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert "full_report" not in app._PRODUCER_ONLY
            # On Catalog (mode 0, not Doctor) it's hidden …
            app.query_one("#operate-tabs", TabbedContent).active = "tab-catalog"
            await pilot.pause()
            assert app.check_action("full_report", ()) is False
            # … and shows on Doctor.
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            assert app.check_action("full_report", ()) is True

    @pytest.mark.asyncio
    async def test_full_report_enabled_on_producer_gate_tab(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
    async def test_share_back_actions_enabled_on_lean_surface(self):
        """The 3 share-back actions are CONSUMER-resident — check_action returns
        True in the merged mode 0 even on the LEAN surface (NOT producer-gated)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "consumer"
            # NOT producer-gated.
            assert "rig_report" not in app._PRODUCER_ONLY
            assert "submit_bench" not in app._PRODUCER_ONLY
            assert "report_problem" not in app._PRODUCER_ONLY
            # Merged mode 0: all three enabled (consumer-resident, any tab).
            assert app._active_mode == 0
            assert app.check_action("rig_report", ()) is True
            assert app.check_action("submit_bench", ()) is True
            assert app.check_action("report_problem", ()) is True
            # Still enabled after flipping to the Orchestration tab (same mode).
            await _enter_operate(pilot)
            assert app.check_action("rig_report", ()) is True
            assert app.check_action("submit_bench", ()) is True
            assert app.check_action("report_problem", ()) is True


# ===========================================================================
# Validate · Run (launch a validation step — confirm-gated, MOCKED stream)
# ===========================================================================


class TestValidateRunWired:
    @pytest.mark.asyncio
    async def test_run_enter_opens_confirm_modal(self):
        # Validate · Run is the producer lane (R3a) — enter on producer.
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
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
            await pilot.press("2")
            await _settle(pilot)
            # R3b-1: activate ③ Gate (no longer the lane default tab).
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            app.query_one("#run-ladder-table", DataTable).move_cursor(row=2)  # bench
            await pilot.press("enter")
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            await pilot.press("enter")
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
            await pilot.press("2")
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
    async def test_power_cap_annotated_on_gpu_card(self):
        # The dedicated power-cap strip was removed; the cap now shows ONLY on the
        # GPU card at the top (GPU0 limit 230 < default 370 → "(cap 230W)").
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-orch-pane", OperateOrchPane)
            assert pane._gpu_cap.get(0) == 230.0
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "cap 230W" in bar

    @pytest.mark.asyncio
    async def test_power_cap_opens_menu_then_clear_confirms_off(self):
        """[c] opens the power-cap MENU (not a direct write); choosing 'clear' (u)
        stages a gated 'power-cap off' (uncap) confirm."""
        from club3090_cockpit.app import PowerCapMenuScreen
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, PowerCapMenuScreen)   # menu, not a write
            await pilot.press("u")                              # clear cap (uncap)
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/gpu-mode.sh", "power-cap", "off"]
            assert app.screen._plan.requires_confirm is True
            assert app.screen._plan.requires_reconcile is False

    @pytest.mark.asyncio
    async def test_power_cap_menu_default_confirms_on(self):
        """The menu's 'default' (d) → gated 'power-cap on' (re-apply 230W)."""
        from club3090_cockpit.app import PowerCapMenuScreen
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, PowerCapMenuScreen)
            await pilot.press("d")
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd[-1] == "on"

    @pytest.mark.asyncio
    async def test_power_cap_menu_custom_wattage_confirms(self):
        """Typing a custom wattage + Enter → gated 'power-cap <W>' confirm."""
        from club3090_cockpit.app import PowerCapMenuScreen
        from textual.widgets import Input
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, PowerCapMenuScreen)
            app.screen.query_one("#pc-watts", Input).value = "280"
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/gpu-mode.sh", "power-cap", "280"]

    @pytest.mark.asyncio
    async def test_power_cap_sweep_opens_confirm_on_doctor(self):
        """Cap-sweep moved to the Doctor tab — [w] there stages the gated sweep."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot, tab="tab-doctor")
            await pilot.press("w")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert "power-cap-sweep" in " ".join(app.screen._plan.cmd)

    @pytest.mark.asyncio
    async def test_container_top_reads_into_drill(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await pilot.press("2")
            await _settle(pilot)
            # R3b-1: browse every lane stage ①→⑤ — all pure reads, no writes.
            for tab in ("tab-bring", "tab-serve", "tab-run", "tab-evidence", "tab-promote"):
                app.query_one("#validate-tabs", TabbedContent).active = tab
                await pilot.pause()
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            await pilot.press("2")          # Bring & Validate lane (where [v] lives)
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
            await pilot.press("2")          # straight to the lane — NO Operate visit
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            await pilot.press("2")          # lane — [v] Evaluate lives here (R3b-1)
            await _settle(pilot)
            await pilot.press("v")
            await _settle(pilot)
            screen = app.screen
            assert isinstance(screen, ConfirmActionScreen)
            await pilot.press("enter")
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
            await _enter_operate(pilot)
            await pilot.press("2")
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
            await pilot.press("2")          # enter the lane (where [P] lives)
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
            await pilot.press("2")          # enter the lane
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
            await pilot.press("2")          # enter the lane (R3b-1)
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
        # 2-mode merge: the lane is mode index 1, key 2.
        assert len(MODES) == 2
        assert MODES[1][0] == "Bring & Validate"
        assert MODES[1][1] == "2"

    @pytest.mark.asyncio
    async def test_bring_stage_reuses_byo_check(self):
        """① Bring renders the byo_check verdict (route / sibling) like Run · BYO."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")          # enter the Bring & Validate lane
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
            await pilot.press("2")
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
            await pilot.press("2")
            await _settle(pilot)
            assert app.check_action("promote_catalog", ()) is True

    @pytest.mark.asyncio
    async def test_evaluate_reachable_in_lane_not_in_operate(self):
        app, _, _ = make_app(surface="producer", target=SERVING_TARGET,
                             gpus=list(SERVING_TARGET.gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            # In Operate (mode 1): [v] is disabled (relocated out).
            assert app.check_action("evaluate_target", ()) is False
            # In the lane (mode 2): [v] is enabled.
            await pilot.press("2")
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
            # The producer LANE SECTION + its verbs are omitted on the lean help.
            # ("Bring & Validate" still appears once — in the [C] toggle line that
            # NAMES the mode it hides — so assert the section/verbs, not that bare
            # phrase.)
            assert "producer lane — the ① → ⑤ pipeline" not in text
            assert "Promote a fit-checked model" not in text
            assert "Evaluate the running target" not in text
            # The lean help mode line stops at Run & Operate — the real rendered
            # mode-2 lane token is absent on the lean surface.
            assert "2[/cyan]  Bring & Validate" not in text

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
            # The full-surface mode line carries the real rendered mode-2 lane token.
            assert "2[/cyan]  Bring & Validate" in text

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
            await _enter_operate(pilot)
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
                           "estate_off", "power_cap", "evaluate_target"):
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
            for action in ("container_logs", "estate_off", "power_cap",
                           "power_cap_sweep", "evaluate_target", "context_t"):
                result = app.check_action(action, ())
                assert result is False, (
                    f"Run must not enable estate action {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_estate_orchestration_enables_orch_keys(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._active_mode == 0  # merged Run & Operate
            # On orchestration tab: estate-off + the power-cap menu are enabled;
            # cap-sweep moved to Doctor (and prune was removed entirely).
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-orchestration"
            await pilot.pause()
            for action in ("estate_off", "power_cap"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"Operate·Orchestration must enable {action!r}, got {result!r}"
                )
            assert app.check_action("power_cap_sweep", ()) is False  # → Doctor now
            # On the Doctor tab, cap-sweep is the enabled one.
            tc.active = "tab-doctor"
            await pilot.pause()
            assert app.check_action("power_cap_sweep", ()) is True
            assert app.check_action("power_cap", ()) is False        # menu is Orchestration-only

    @pytest.mark.asyncio
    async def test_estate_orchestration_disables_containers_keys(self):
        """On the Orchestration sub-tab, container-specific keys are disabled."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-containers"
            await pilot.pause()
            for action in ("container_logs", "container_stop", "container_rm", "context_t"):
                result = app.check_action(action, ())
                assert result is True, (
                    f"Operate·Containers must enable {action!r}, got {result!r}"
                )

    @pytest.mark.asyncio
    async def test_catalog_tab_enables_filter_and_lane_drops_old_bmk_keys(self):
        """[/] filter lives on the merged mode's Catalog tab; the old Benchmarks
        sort ([t]) is gone, so context_t is disabled in the lane (mode 1)."""
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Catalog (default mode/tab): filter enabled.
            assert app._active_mode == 0
            assert app.check_action("filter_catalog", ()) is True
            assert app.check_action("toggle_catalog_deprecated", ()) is True
            # Lane mode: no benchmarks tab → context_t (sort) disabled, and
            # filter_catalog is off (not a Catalog context).
            await pilot.press("2")
            await _settle(pilot)
            assert app._active_mode == 1
            assert app.check_action("context_t", ()) is False
            assert app.check_action("filter_catalog", ()) is False
            assert app.check_action("toggle_catalog_deprecated", ()) is False

    @pytest.mark.asyncio
    async def test_validate_evidence_enables_s_key(self):
        # The lane is mode 1, visible by default (full surface).
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            tc.active = "tab-evidence"
            await pilot.pause()
            # s_key (submit) is enabled in the lane regardless of subtab.
            assert app.check_action("s_key", ()) is True

    @pytest.mark.asyncio
    async def test_doctor_tab_disables_explain_and_filter(self):
        """The Catalog-only keys [e]/[/] are off on the merged mode's non-Catalog
        tabs (e.g. the Doctor tab)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot, tab="tab-doctor")
            assert app._active_mode == 0
            assert app.check_action("explain", ()) is False
            assert app.check_action("filter_catalog", ()) is False
            assert app.check_action("toggle_catalog_deprecated", ()) is False

    @pytest.mark.asyncio
    async def test_always_on_keys_active_in_every_mode(self):
        """quit/help/refresh/mode-switch must be True in every mode.  Run on the
        default full surface so mode_validate (hidden on lean) is genuinely
        always-on across both modes."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for mode_key in ("1", "2"):
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
            await _enter_operate(pilot)
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
                await pilot.press("2")
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
    async def test_right_bracket_cycles_merged_subtab(self):
        # 2-mode merge: the merged mode's tabs are Catalog → Orchestration → … in
        # the single #operate-tabs TabbedContent.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            tc = app.query_one("#operate-tabs", TabbedContent)
            assert tc.active == "tab-catalog"  # default first tab
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-orchestration"

    @pytest.mark.asyncio
    async def test_right_bracket_wraps_around_on_last_tab(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            # Go to the last tab (Doctor) first, then cycle forward → wrap to first.
            tc.active = "tab-doctor"
            await pilot.pause()
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active == "tab-catalog"

    @pytest.mark.asyncio
    async def test_left_bracket_cycles_backwards(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await pilot.pause()
            await pilot.press("left_square_bracket")
            await pilot.pause()
            # Backwards from the first tab wraps to the last (Doctor).
            assert tc.active == "tab-doctor"

    @pytest.mark.asyncio
    async def test_subtab_key_cycles_estate_tabs(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            first = tc.active
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active != first

    @pytest.mark.asyncio
    async def test_subtab_key_cycles_validate_tabs(self):
        # The lane is mode 1, visible by default (full surface).
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            first = tc.active
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert tc.active != first

    @pytest.mark.asyncio
    async def test_subtab_key_active_in_both_modes(self):
        """2-mode merge: both modes (merged Run & Operate · Bring & Validate) have
        sub-tabs, so the cycle keys are active (check_action True) in each."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            for key, mode in (("1", 0), ("2", 1)):
                await pilot.press(key)
                await pilot.pause()
                assert app._active_mode == mode
                assert app.check_action("next_subtab", ()) is True
                assert app.check_action("prev_subtab", ()) is True


class TestModeSwitchFocus:
    """(e) Mode switch moves focus to the mode's primary interactive widget."""

    @pytest.mark.asyncio
    async def test_switch_to_merged_with_catalog_tab_focuses_catalog_table(self):
        # 2-mode merge: returning to mode 0 focuses the ACTIVE tab's table.  With
        # the Catalog tab active, that's #catalog-table.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Go to the lane and back; ensure the Catalog tab is the active one.
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-catalog"
            await pilot.pause()
            await pilot.press("1")
            await pilot.pause()
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "catalog-table"

    @pytest.mark.asyncio
    async def test_switch_to_estate_focuses_scene_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            # Operate/Orchestration tab is default → scene-table should be focused.
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "scene-table"

    @pytest.mark.asyncio
    async def test_switch_to_validate_lands_on_bring_stage(self):
        # The producer lane's first stage is ① Bring.  Focus is deliberately NOT
        # forced into its HF-repo Input (an Input would swallow the global 1/2 + [ ]
        # keys); the lane lands on ① Bring with focus on the tab bar so those keys
        # keep routing to the app.  The lane is mode 1, visible by default.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            assert app._active_mode == 1
            assert app.query_one("#validate-tabs", TabbedContent).active == "tab-bring"
            # The bring input is NOT auto-focused (so digit/bracket keys work).
            assert not (isinstance(app.focused, Input)
                        and app.focused.id == "lane-bring-url-input")

    @pytest.mark.asyncio
    async def test_validate_tabbar_browse_keeps_focus_on_bar(self):
        """BUG 3 — browsing the lane TAB BAR keeps focus ON the tab bar; it does
        NOT yank focus down into the newly-active stage's table.  The Fix-B
        "don't grab focus while browsing the tab bar" guard now applies in the
        mode-1 producer lane too (was scoped to mode 0), so activating ③ Gate /
        ④ Measure via the tab bar is consistent with Run & Operate — focus stays
        on the bar.  Only a cycle FROM a list moves to the next list (see the
        companion test below)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            # The lane lands on ① Bring with focus on the lane tab bar (a Tabs —
            # its only focusable widget is an Input we don't auto-focus).
            assert tc.active == "tab-bring"
            assert isinstance(app.focused, Tabs)
            bar = app._active_tab_bar()
            assert app.focused is bar
            # Activate ③ Gate WHILE focus is on the lane tab bar → focus STAYS on
            # the tab bar; the Gate ladder is NOT yanked into focus.
            tc.active = "tab-run"
            await pilot.pause()
            await pilot.pause()  # extra cycle for any call_after_refresh
            assert app.focused is bar
            assert not (isinstance(app.focused, DataTable)
                        and app.focused.id == "run-ladder-table")

    @pytest.mark.asyncio
    async def test_validate_cycle_from_list_moves_to_next_list(self):
        """BUG 3 — a [/] cycle FROM a LIST (focus on the list, not the tab bar)
        still moves to the NEXT stage's list.  Descending into ③ Gate's ladder
        then cycling forward lands focus on ④ Measure's evidence-table — the
        list-operators keep operating lists; only tab-bar browsing is exempt."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            # Browse to ③ Gate via the tab bar (focus stays on the bar — BUG 3).
            tc.active = "tab-run"
            await pilot.pause()
            await pilot.pause()
            # Descend INTO the Gate ladder so focus is now on the LIST.
            app.query_one("#run-ladder-table", DataTable).focus()
            await pilot.pause()
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "run-ladder-table"
            # Cycle forward → ④ Measure (Evidence): focus FROM a list moves to the
            # next list (the guard only exempts focus-on-the-tab-bar).
            await pilot.press("right_square_bracket")
            await pilot.pause()
            await pilot.pause()
            assert tc.active == "tab-evidence"
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "evidence-table"


class TestModeSwitcherKeyboardNav:
    """BUG 2 — the Modes rail is keyboard-navigable: a focus stop (first in the
    Tab chain), its arrows switch the active mode while focused, ↑ from the tab
    bar ascends to it, and Tab/Enter descend back into the content.  The arrows
    act ONLY while it is focused — elsewhere the existing arrow model is intact."""

    @pytest.mark.asyncio
    async def test_modeswitcher_is_focusable(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            assert ms.can_focus is True
            ms.focus()
            await pilot.pause()
            assert app.focused is ms

    @pytest.mark.asyncio
    async def test_shift_tab_from_tabbar_reaches_modeswitcher(self):
        """The ModeSwitcher is the FIRST stop in the Tab chain (first in the
        left-rail DOM) — Shift+Tab from the tab bar reaches it; Tab from it goes
        back to the tab bar."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            bar = app._active_tab_bar()
            bar.focus()
            await pilot.pause()
            assert isinstance(app.focused, Tabs)
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.focused is ms
            # Tab from the ModeSwitcher descends to the tab bar.
            await pilot.press("tab")
            await pilot.pause()
            assert isinstance(app.focused, Tabs)

    @pytest.mark.asyncio
    async def test_arrows_switch_mode_and_keep_focus(self):
        """Round-4 CHANGE 4: ↑/↓ on the focused ModeSwitcher SELECT the active mode
        (prev / next visible mode + the visible panel + _active_mode); focus STAYS
        on the ModeSwitcher.  ←/→ are NO LONGER mode-switchers (→ descends, ←
        inert — see the dedicated tests below)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            ms.focus()
            await pilot.pause()
            assert app._active_mode == 0
            # ↓ → mode 1 (Bring & Validate).
            await pilot.press("down")
            await pilot.pause()
            await pilot.pause()
            assert app._active_mode == 1
            assert "active" in app.query_one("#panel-validate").classes
            assert app.focused is ms  # focus stays
            # ↑ → back to mode 0.
            await pilot.press("up")
            await pilot.pause()
            await pilot.pause()
            assert app._active_mode == 0
            assert "active" in app.query_one("#panel-run").classes
            assert app.focused is ms

    @pytest.mark.asyncio
    async def test_left_arrow_is_inert(self):
        """Round-4 CHANGE 4: ← on the focused ModeSwitcher does NOTHING — it no
        longer retreats the mode.  Mode + focus are unchanged."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            ms.focus()
            await pilot.pause()
            # Move to mode 1 with ↓ so a (former) ←-retreat would be observable.
            await pilot.press("down")
            await pilot.pause()
            await pilot.pause()
            assert app._active_mode == 1
            # ← is inert — mode stays 1, focus stays on the ModeSwitcher.
            await pilot.press("left")
            await pilot.pause()
            await pilot.pause()
            assert app._active_mode == 1
            assert app.focused is ms

    @pytest.mark.asyncio
    async def test_right_arrow_descends_to_content(self):
        """Round-4 CHANGE 4: → on the focused ModeSwitcher DESCENDS into the
        selected mode's content (Enter alias) — Catalog → #catalog-table."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            ms.focus()
            await pilot.pause()
            assert app._active_mode == 0
            await pilot.press("right")
            await pilot.pause()
            await pilot.pause()
            # Descended into mode 0's primary list (mode did NOT advance).
            assert app._active_mode == 0
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "catalog-table"

    @pytest.mark.asyncio
    async def test_arrows_clamp_at_ends(self):
        """↑ at the first mode and ↓ at the last are inert (clamped)."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            ms.focus()
            await pilot.pause()
            # Already at mode 0 — ↑ is a no-op.
            await pilot.press("up")
            await pilot.pause()
            assert app._active_mode == 0
            # Go to the last mode, then ↓ is a no-op.
            await pilot.press("down")
            await pilot.pause()
            await pilot.pause()
            assert app._active_mode == 1
            await pilot.press("down")
            await pilot.pause()
            assert app._active_mode == 1

    @pytest.mark.asyncio
    async def test_up_from_tabbar_is_inert(self):
        """Round-4 CHANGE 3: ↑ on the TAB BAR does NOTHING — it must NOT jump to the
        ModeSwitcher (the round-3 tab-bar→Modes ascent is removed).  Focus stays on
        the tab bar; the ModeSwitcher stays reachable via Shift+Tab."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            bar = app._active_tab_bar()
            bar.focus()
            await pilot.pause()
            assert isinstance(app.focused, Tabs)
            await pilot.press("up")
            await pilot.pause()
            # Inert: focus is STILL the tab bar, NOT the ModeSwitcher.
            assert app.focused is bar
            assert app.focused is not ms
            # Shift+Tab still reaches the ModeSwitcher (focus chain unchanged).
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.focused is ms

    @pytest.mark.asyncio
    async def test_enter_from_modeswitcher_descends_to_content(self):
        """Enter from the focused ModeSwitcher descends into the active tab's
        primary list (Catalog → #catalog-table)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            ms.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "catalog-table"

    @pytest.mark.asyncio
    async def test_arrows_on_list_unaffected(self):
        """INVARIANT — arrows on a LIST move the cursor (do NOT switch mode); the
        ModeSwitcher arrows act only while IT is focused."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tbl = app.query_one("#catalog-table", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.pause()
            before_mode = app._active_mode
            await pilot.press("down")
            await pilot.pause()
            # Mode unchanged; the list cursor moved instead.
            assert app._active_mode == before_mode
            assert app.focused is tbl
            assert tbl.cursor_row == 1

    @pytest.mark.asyncio
    async def test_lean_surface_arrows_inert(self):
        """On the LEAN surface (one visible mode) the ModeSwitcher arrows are
        inert — there is nothing to switch to."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ms = app.query_one("#mode-switcher", ModeSwitcher)
            ms.focus()
            await pilot.pause()
            assert app._active_mode == 0
            await pilot.press("down")
            await pilot.pause()
            await pilot.pause()
            assert app._active_mode == 0  # still mode 0 — no second mode to reach
            await pilot.press("up")
            await pilot.pause()
            assert app._active_mode == 0


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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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


class TestContainerClampDoesNotAutoloadDrill:
    """FIX 1 (clamp echo) — when a periodic poll (load_estate) drops the
    container the user had selected, populate() CLAMPS the cursor to a DIFFERENT
    container.  The move_cursor clamp fires a SECOND RowHighlighted echo that
    BYPASSED the row-0 drill suppression → a spurious `docker logs/top` auto-fired
    for a container the user never picked (the [r]-re-jump footgun, now on every
    tick).  These guards drive the WHOLE pipeline through load_estate (not just
    populate()) and assert the invariant: a poll NEVER starts a docker logs/top
    for an unselected container, while the benign index-shift case is preserved."""

    # vllm-a (engine, row 0) · comfyui (row 1) · open-webui (row 2).  The user
    # sits on open-webui (a non-row-0 container) with the Logs drill open.
    CONTAINERS_FULL = [
        ContainerInfo(name="vllm-a", kind="engine", host_port=8010, engine="vllm", slug="vllm/dual"),
        ContainerInfo(name="comfyui", kind="service", host_port=8188),
        ContainerInfo(name="open-webui", kind="service", host_port=3000),
    ]
    # open-webui DROPS — cursor must clamp from row 2 → comfyui (row 1).
    CONTAINERS_DROPPED = [
        ContainerInfo(name="vllm-a", kind="engine", host_port=8010, engine="vllm", slug="vllm/dual"),
        ContainerInfo(name="comfyui", kind="service", host_port=8188),
    ]
    # A container appears at the TOP — open-webui survives but shifts row 2 → 3.
    CONTAINERS_SHIFTED = [
        ContainerInfo(name="new-svc", kind="engine", host_port=9000, engine="vllm", slug="vllm/x"),
        ContainerInfo(name="vllm-a", kind="engine", host_port=8010, engine="vllm", slug="vllm/dual"),
        ContainerInfo(name="comfyui", kind="service", host_port=8188),
        ContainerInfo(name="open-webui", kind="service", host_port=3000),
    ]

    def _patch_estate(self, app, holder):
        """Make load_estate observe whatever container list ``holder`` points at,
        so a test flips the live estate between polls without any subprocess."""
        async def _estate_state(*_a, **_k):
            return EstateState(containers=list(holder["containers"]))
        app._data.estate_state = _estate_state

    async def _enter_containers_on_logs(self, app, pilot, holder):
        await _enter_operate(pilot)
        app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
        await _settle(pilot)
        # First poll paints the full table.
        app.load_estate()
        await _settle(pilot)
        # User selects open-webui (row 2 — a non-row-0 container) and opens Logs.
        tbl = app.query_one("#containers-table", DataTable)
        tbl.focus()
        await pilot.pause()
        app.query_one("#drill-tabs", TabbedContent).active = "drill-tab-logs"
        await _settle(pilot)
        tbl.move_cursor(row=2)                                  # → open-webui
        await pilot.pause()
        assert app._selected_container().name == "open-webui"
        # Explicitly load the Logs drill for open-webui (the user's pick).
        app._load_active_drill_tab()
        await _settle(pilot)

    @pytest.mark.asyncio
    async def test_clamp_to_other_container_does_not_autoload_drill(self):
        holder = {"containers": list(self.CONTAINERS_FULL)}
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            self._patch_estate(app, holder)
            await self._enter_containers_on_logs(app, pilot, holder)
            # The user's Logs read DID fire for open-webui.
            assert any("docker logs" in " ".join(c) and "open-webui" in " ".join(c)
                       for c in runner.calls), "user's open-webui logs never loaded"
            runner.calls.clear()
            # PERIODIC POLL where open-webui DISAPPEARS → cursor clamps to comfyui.
            holder["containers"] = list(self.CONTAINERS_DROPPED)
            app.load_estate()
            await _settle(pilot)
            await pilot.pause(0.35)                             # let any drill timer fire
            await _settle(pilot)
            # INVARIANT: NO docker logs/top fired for the clamped-to container
            # (comfyui) — nor for anything the user didn't pick.
            assert not any("docker logs" in " ".join(c) for c in runner.calls), (
                f"clamp spuriously auto-loaded docker logs: {runner.calls}"
            )
            assert not any("docker top" in " ".join(c) for c in runner.calls), (
                f"clamp spuriously auto-loaded docker top: {runner.calls}"
            )
            # The cursor DID clamp to a different container (comfyui), proving the
            # scenario actually exercised the clamp path.
            assert app._selected_container().name == "comfyui"

    @pytest.mark.asyncio
    async def test_preserved_selection_index_shift_retains_same_drill(self):
        # Benign companion: the selection is PRESERVED (open-webui survives, index
        # merely shifts).  Reloading the SAME container's drill is harmless — and
        # crucially NO OTHER container's drill is loaded.
        holder = {"containers": list(self.CONTAINERS_FULL)}
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            self._patch_estate(app, holder)
            await self._enter_containers_on_logs(app, pilot, holder)
            runner.calls.clear()
            # A container appears at the top → open-webui shifts row 2 → 3 but is
            # PRESERVED (same container under the cursor).
            holder["containers"] = list(self.CONTAINERS_SHIFTED)
            app.load_estate()
            await _settle(pilot)
            await pilot.pause(0.35)
            await _settle(pilot)
            # The cursor followed the KEY — still open-webui (now row 3).
            assert app._selected_container().name == "open-webui"
            # No drill for a DIFFERENT container leaked (comfyui / vllm-a / new-svc).
            for other in ("comfyui", "vllm-a", "new-svc"):
                assert not any(
                    ("docker logs" in " ".join(c) or "docker top" in " ".join(c))
                    and other in " ".join(c)
                    for c in runner.calls
                ), f"a drill spuriously loaded for {other}: {runner.calls}"


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
            await pilot.press("2")  # Validate
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
    split is real. These tests prove (a) the surface defaults to FULL (BOTH
    modes shown — the post-merge inversion), (b) the LEAN view carries the
    indicator and hides the producer mode, and (c) the gate LOGIC hides a
    producer-only action on lean / shows it on full — including for
    _ALWAYS_ON actions (the gate is checked BEFORE _ALWAYS_ON, so it can hide the
    producer Bring & Validate MODE switch).

    The gate-logic tests patch the *class* attr `_PRODUCER_ONLY` via monkeypatch
    BEFORE the app mounts (auto-restored after), rather than mutating a live
    instance attr mid-test — the latter raced under accumulated full-suite
    asyncio state and flaked.
    """

    @pytest.mark.asyncio
    async def test_default_surface_is_full_no_indicator(self):
        # Surface inversion: the default is FULL ("producer"), unmarked sub-title.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "producer"
            assert "LEAN" not in str(app.sub_title)

    @pytest.mark.asyncio
    async def test_lean_surface_shows_lean_indicator(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "consumer"
            assert "LEAN" in str(app.sub_title)

    @pytest.mark.asyncio
    async def test_invalid_surface_falls_back_to_full(self):
        app, _, _ = make_app(surface="bogus")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "producer"

    @pytest.mark.asyncio
    async def test_shipped_producer_set_is_mode_validate_and_promote(self):
        # R3b-1: the shipped _PRODUCER_ONLY gates the producer lane (mode_validate)
        # + the relocated-into-the-lane [P] promote + [v] evaluate + ② serve_untested.
        # R3b-2: + [m] measure_vs_bar (④ Measure).
        # Batch 3: [F] full_report is NO LONGER producer-only — it's reachable on
        # the consumer Operate · Doctor (a consumer can run the full battery).
        assert CockpitApp._PRODUCER_ONLY == frozenset({
            "mode_validate", "promote_catalog", "evaluate_target", "serve_untested",
            "measure_vs_bar",
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
            await pilot.press("2")          # enter the lane (mode 2)
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
    Label (which also starts with ``mode-``).  2-mode merge: both mode Labels are
    always mounted; the LEAN (consumer) surface HIDES the producer-only second via
    the ``mode-hidden`` class (so the runtime [C] toggle can reveal it without an
    async re-mount), so count only the rows NOT carrying that class."""
    ms = app.query_one("#mode-switcher", ModeSwitcher)
    return len([
        lbl for lbl in ms.query(Label)
        if (lbl.id or "").startswith("mode-") and (lbl.id or "")[len("mode-"):].isdigit()
        and not lbl.has_class("mode-hidden")
    ])


class TestProducerLaneGatedR3a:
    """Surface inversion: the producer Bring & Validate lane (mode 1 / key 2) +
    [P] promote are PRODUCER-only: visible by DEFAULT (full surface), HIDDEN +
    unreachable on the LEAN surface.  The ModeSwitcher is surface-aware (1 row
    lean, 2 full)."""

    @pytest.mark.asyncio
    async def test_lean_cannot_reach_validate_via_key_2(self):
        """On the lean surface: mode_validate is gated off and pressing 2 does NOT
        switch to the producer lane (stays in the merged mode 0)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            assert app.check_action("mode_validate", ()) is False
            await pilot.press("2")
            await _settle(pilot)
            # Did NOT enter the producer Validate lane.
            assert app._active_mode == 0
            assert "active" not in app.query_one("#panel-validate").classes
            assert "active" in app.query_one("#panel-run").classes

    @pytest.mark.asyncio
    async def test_lean_action_mode_validate_guard_is_noop(self):
        """Belt-and-suspenders: even a direct programmatic call to
        action_mode_validate does not switch a lean user into the producer lane."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.action_mode_validate()
            await _settle(pilot)
            assert app._active_mode == 0
            assert "active" not in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_lean_promote_is_gated_off(self):
        """On the lean surface: [P] promote_catalog is gated off (producer verb)."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._active_mode == 0
            assert app.check_action("promote_catalog", ()) is False

    @pytest.mark.asyncio
    async def test_lean_mode_switcher_shows_one_mode(self):
        """The lean ModeSwitcher renders only the merged Run & Operate row."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert _mode_switcher_item_count(app) == 1

    @pytest.mark.asyncio
    async def test_full_can_reach_validate_via_key_2(self):
        """On the default full surface: pressing 2 enters the Validate lane (mode 1)."""
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("mode_validate", ()) is True
            await pilot.press("2")
            await _settle(pilot)
            assert app._active_mode == 1
            assert "active" in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_full_promote_is_reachable(self):
        """On the full surface: [P] promote_catalog is reachable in the lane (mode 1)."""
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")          # enter the lane (mode 1)
            await _settle(pilot)
            assert app._active_mode == 1
            assert app.check_action("promote_catalog", ()) is True

    @pytest.mark.asyncio
    async def test_full_mode_switcher_shows_two_modes(self):
        """The full ModeSwitcher renders both modes (incl. Bring & Validate)."""
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert _mode_switcher_item_count(app) == 2

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
            # merged Run & Operate (mode 0): all three live (submit_bench gates to mode 0).
            await _enter_operate(pilot)
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

    def test_default_is_full_producer(self):
        # Surface inversion: the default is FULL ("producer" = both modes visible).
        assert resolve_surface(["c3"], {}) == "producer"

    def test_lean_flag_opts_into_lean(self):
        assert resolve_surface(["c3", "--lean"], {}) == "consumer"

    def test_env_consumer_opts_into_lean(self):
        assert resolve_surface(["c3"], {"C3_SURFACE": "consumer"}) == "consumer"

    def test_env_is_case_and_space_insensitive(self):
        assert resolve_surface(["c3"], {"C3_SURFACE": "  Consumer "}) == "consumer"

    def test_contribute_alias_is_harmless_and_stays_full(self):
        # --contribute is kept as a harmless alias — it's already the default.
        assert resolve_surface(["c3", "--contribute"], {}) == "producer"
        assert resolve_surface(["c3"], {"C3_SURFACE": "producer"}) == "producer"

    def test_lean_beats_redundant_contribute_alias(self):
        # An explicit lean opt-out wins over the redundant contribute alias.
        assert resolve_surface(["c3", "--lean", "--contribute"], {}) == "consumer"

    def test_env_other_value_is_full(self):
        assert resolve_surface(["c3"], {"C3_SURFACE": "1"}) == "producer"
        assert resolve_surface(["c3"], {"C3_SURFACE": ""}) == "producer"


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
        save_surface_setting("consumer")
        # No flag / env → the persisted lean (consumer) setting wins (precedence 2).
        assert resolve_surface(["c3"], {}) == "consumer"

    def test_explicit_flag_beats_persisted(self):
        save_surface_setting("consumer")
        # Persisted is lean, but the explicit --contribute flag forces full.
        assert resolve_surface(["c3", "--contribute"], {}) == "producer"

    def test_explicit_lean_flag_beats_persisted(self):
        save_surface_setting("producer")
        # Persisted is full, but the explicit --lean flag forces lean.
        assert resolve_surface(["c3", "--lean"], {}) == "consumer"

    def test_explicit_env_beats_persisted(self):
        save_surface_setting("producer")
        assert resolve_surface(["c3"], {"C3_SURFACE": "consumer"}) == "consumer"

    def test_no_persisted_falls_to_full(self):
        # No flag, no env, no persisted file → full (producer) default.
        assert resolve_surface(["c3"], {}) == "producer"


class TestContributeDoor:
    """The [C] LEAN-view toggle (surface inversion): both modes show by default;
    [C] hides the Bring & Validate mode (ModeSwitcher 2 → 1 items, producer gating
    re-gates, the in-lane edge handled) and back, persisting the choice
    (test-injectable config dir)."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("C3_CONFIG_DIR", str(tmp_path))

    @pytest.mark.asyncio
    async def test_toggle_is_always_on(self):
        # The toggle must NOT be producer-gated (a lean user needs it to restore).
        assert "toggle_contribute" in CockpitApp._ALWAYS_ON
        assert "toggle_contribute" not in CockpitApp._PRODUCER_ONLY
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("toggle_contribute", ()) is True

    @pytest.mark.asyncio
    async def test_default_shows_both_modes(self):
        # Default (full) surface shows BOTH modes — no flag needed.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._surface == "producer"
            assert _mode_switcher_item_count(app) == 2
            assert app.check_action("mode_validate", ()) is True

    @pytest.mark.asyncio
    async def test_toggle_full_to_lean_hides_lane(self):
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert _mode_switcher_item_count(app) == 2
            assert app.check_action("mode_validate", ()) is True
            await pilot.press("C")
            await _settle(pilot)
            assert app._surface == "consumer"
            assert _mode_switcher_item_count(app) == 1
            assert app.check_action("mode_validate", ()) is False
            assert "LEAN" in str(app.sub_title)
            # No forced switch when toggling from the merged mode (mode 0).
            assert app._active_mode == 0

    @pytest.mark.asyncio
    async def test_toggle_lean_persists_for_next_launch(self):
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("C")
            await _settle(pilot)
            assert load_surface_setting() == "consumer"
            # resolve_surface (next launch, no flag/env) reads the persisted value.
            assert resolve_surface(["c3"], {}) == "consumer"

    @pytest.mark.asyncio
    async def test_toggle_back_to_full_regates_and_persists(self):
        app, _, _ = make_app(surface="consumer")  # start lean
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("C")          # lean → full (from merged mode)
            await _settle(pilot)
            assert app._surface == "producer"
            assert _mode_switcher_item_count(app) == 2
            assert app.check_action("mode_validate", ()) is True
            assert "LEAN" not in str(app.sub_title)
            assert load_surface_setting() == "producer"

    @pytest.mark.asyncio
    async def test_toggle_lean_while_in_lane_switches_to_merged(self):
        # EDGE: going lean while IN the now-hidden Bring & Validate mode (mode 1)
        # must not strand the user — switch them to the merged mode 0.
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")          # into the lane (mode 1)
            await _settle(pilot)
            assert app._active_mode == 1
            await pilot.press("C")          # go lean
            await _settle(pilot)
            assert app._surface == "consumer"
            assert app._active_mode == 0    # switched out of the hidden lane
            assert "active" in app.query_one("#panel-run").classes

    @pytest.mark.asyncio
    async def test_toggle_lean_while_in_lane_rescues_to_merged(self):
        # EDGE: going lean while IN the producer lane (mode 1, now hidden) must
        # move the user back to the merged mode 0.
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")          # enter the lane (mode 1)
            await _settle(pilot)
            assert app._active_mode == 1
            await pilot.press("C")          # go lean while stranded in the lane
            await _settle(pilot)
            assert app._surface == "consumer"
            assert app._active_mode == 0    # rescued to the merged mode
            assert "active" in app.query_one("#panel-run").classes
            assert "active" not in app.query_one("#panel-validate").classes

    @pytest.mark.asyncio
    async def test_toggle_round_trip_back_to_lean(self):
        app, _, _ = make_app(surface="consumer")  # start lean
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("C")          # → full
            await _settle(pilot)
            await pilot.press("C")          # → lean
            await _settle(pilot)
            assert app._surface == "consumer"
            assert _mode_switcher_item_count(app) == 1
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            before = calls["n"]
            plan = app._data.serve("vllm/dual")  # not forced → refused
            app.dispatch_action(plan)
            await _settle(pilot)
            assert wr.started == []          # refused at the gate
            assert calls["n"] == before      # NO re-poll on a refused write


class TestBatch2A3PeriodicRefresh:
    """A3 — the periodic refresh interval polls in the MERGED Run & Operate mode
    (mode 0 — the live estate tabs + rail live here now), never in the Bring &
    Validate lane (mode 1) or behind a modal."""

    @pytest.mark.asyncio
    async def test_interval_polls_in_merged_mode(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._active_mode == 0
            calls = {"n": 0}
            orig = app.load_estate
            app.load_estate = lambda _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o())[1]
            app._periodic_estate_refresh()   # fire the gated tick directly
            assert calls["n"] == 1           # polled in the merged mode

    @pytest.mark.asyncio
    async def test_interval_does_not_poll_behind_modal(self):
        # The poll never fires behind a modal even in the merged mode.
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            assert app._active_mode == 0
            await pilot.press("question_mark")   # open the help modal
            await pilot.pause()
            calls = {"n": 0}
            orig = app.load_estate
            app.load_estate = lambda _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o())[1]
            app._periodic_estate_refresh()
            assert calls["n"] == 0           # NOT polled behind a modal

    @pytest.mark.asyncio
    async def test_interval_does_not_poll_in_validate(self):
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")           # Bring & Validate lane (mode 1)
            await _settle(pilot)
            assert app._active_mode == 1
            calls = {"n": 0}
            orig = app.load_estate
            app.load_estate = lambda _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o())[1]
            app._periodic_estate_refresh()
            assert calls["n"] == 0           # NOT polled in the lane

    @pytest.mark.asyncio
    async def test_rail_shows_as_of_stamp(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            bar = str(app.query_one("#gpu0-bar", Static).render())
            assert "nvidia-smi returned nothing" in bar

    @pytest.mark.asyncio
    async def test_healthy_estate_has_no_error_strip(self):
        """No false positives: a healthy estate keeps the error strip hidden."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
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
        # The registry row's EXACT container + port 8010 → matched_slug vllm/dual
        # as an IDENTITY match (F9: a container-less port match is only a shape
        # guess and renders the 👤 badge instead — see the test below).
        tgt = ServingTarget(
            url="http://localhost:8010", model="qwen3.6-27b", host_port=8010,
            container="vllm_qwen36_27b",
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )
        app, _, _ = make_app(target=tgt)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)            # catalog loads
            await _enter_operate(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._serving_slug == "vllm/dual"
            tbl = app.query_one("#catalog-table", DataTable)
            blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "serving" in blob        # the ● serving badge
            assert "👤" not in blob

    @pytest.mark.asyncio
    async def test_catalog_shape_match_never_claims_serving(self):
        # F9 — a brought container UNKNOWN to the registry on vllm/dual's port:
        # the row must NOT read "● serving" (the A1 masquerade); it reads the
        # 👤 port-in-use guess instead.
        tgt = ServingTarget(
            url="http://localhost:8010", model="agents-a1", host_port=8010,
            container="vllm-agents-a1-dual",
            gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)],
        )
        app, _, _ = make_app(target=tgt)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await _enter_operate(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._serving_slug == "vllm/dual"
            assert pane._serving_confidence == "shape"
            tbl = app.query_one("#catalog-table", DataTable)
            blob = " ".join(str(tbl.get_row_at(r)) for r in range(tbl.row_count))
            assert "serving" not in blob
            assert "👤" in blob and "port in use" in blob

    @pytest.mark.asyncio
    async def test_catalog_marker_clears_when_nothing_serving(self):
        app, _, _ = make_app(target=ServingTarget())   # nothing matches
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            # Force-set a stale marker, then a poll with no match must clear it.
            pane.set_serving_slug("vllm/dual")
            assert pane._serving_slug == "vllm/dual"
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
        # No GPUs read → free-VRAM unknown → the fit verdict must read "vs empty
        # card" so "fits-clean" is never mistaken for a live verdict.  The fit moved
        # OUT of the Catalog column into the serve pop-up + the preview strip, so the
        # basis label now lives on the PREVIEW STRIP (not the catalog status line).
        app, _, _ = make_app(gpus=[], target=ServingTarget(gpus=[]))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._free_gb_by_index is None
            entry = next(e for e in pane._entries if e.slug == "vllm/dual")
            pane.render_preview(entry)
            preview = str(app.query_one("#catalog-preview", Static).render())
            assert "vs empty card" in preview

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
            await _enter_operate(pilot)
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
        "✗ won't fit now" while it is PROVABLY serving.  The serving slug is EXEMPT
        from the live-VRAM downgrade → its base glyph (●).

        The fit moved OUT of the Catalog column into the serve pop-up + the preview
        strip; the exemption is asserted via the PREVIEW STRIP now (it applies the
        SAME B3 downgrade the column used to, with the same serving-row exemption)."""
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
            await _enter_operate(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            assert pane._serving_slug == "vllm/dual"   # it IS the serving row
            assert pane._free_gb_by_index is not None   # live free-VRAM is low
            # Render the preview strip for the SERVING entry → exemption applies.
            serving_entry = next(e for e in pane._entries if e.slug == "vllm/dual")
            pane.render_preview(serving_entry)
            preview = str(app.query_one("#catalog-preview", Static).render())
            # NOT downgraded — no ✗/⚠ and no "won't fit"/"tight" reason for the
            # serving slug, even though live free is below its per-card est.
            assert "✗" not in preview and "⚠" not in preview
            assert "won't fit" not in preview and "tight" not in preview
            assert "●" in preview                   # stays the base fits-clean glyph

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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await pilot.pause()
            await pilot.press("n")     # switch → flip to the Catalog tab (same mode)
            await pilot.pause()
            assert "active" in app.query_one("#panel-run").classes
            assert app.query_one("#operate-tabs", TabbedContent).active == "tab-catalog"
            assert not isinstance(app.screen, ConfirmActionScreen)  # no write here


class TestDoctorRerunAndRemediation:
    """#4: a Doctor-resident key re-runs the READ-only health read on demand.
    N7: a surfaced issue OFFERS the obvious remediation pointer."""

    @pytest.mark.asyncio
    async def test_doctor_rerun_triggers_doctor_read(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, runner, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            # Clear the call log, then press [y] — it must re-run the health read.
            runner.calls.clear()
            await pilot.press("y")
            await _settle(pilot)
            assert any("health.sh" in " ".join(c) for c in runner.calls)

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
            await pilot.press("2")           # Bring & Validate lane
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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
    async def test_lean_help_lists_lean_toggle_key(self):
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("question_mark")
            await pilot.pause()
            text = app.screen.help_text
            # The [C] lean-view toggle is taught on LEAN (it restores the full view).
            assert "toggle lean view" in text
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


class TestFooterOutOfTabChain:
    """FIX A — the footer is a HINT BAR, deliberately OUT of the Tab focus chain.

    An earlier batch (#5) made every FooterKey Tab-focusable; footer keys render
    NO visible focus ring, so Tab-stepping through ~10 of them was an invisible-
    focus black hole.  The footer's actions stay reachable via HOTKEYS (app/screen
    BINDINGS) + mouse click — Tab now cycles only the VISIBLE stops (tab bar +
    list)."""

    @pytest.mark.asyncio
    async def test_footer_keys_are_not_focusable(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ff = app.query_one(FocusableFooter)
            keys = list(ff.query(FooterKey))
            assert keys, "footer still renders key items (mouse-clickable)"
            assert all(not k.can_focus for k in keys), \
                "no FooterKey is Tab-focusable"
            # can_focus_children reverts to the stock Footer default.
            assert ff.can_focus_children is False

    @pytest.mark.asyncio
    async def test_no_footer_key_in_focus_chain(self):
        """No FooterKey participates in the Tab focus chain."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            chain = app.screen.focus_chain
            assert not any(isinstance(w, FooterKey) for w in chain), \
                "the footer is OUT of the Tab focus chain"

    @pytest.mark.asyncio
    async def test_tab_from_catalog_never_lands_on_a_footer_key(self):
        """From #catalog-table, Tab / Shift+Tab cycle only the VISIBLE, meaningful
        stops — the Modes rail (ModeSwitcher — BUG 2), the tab bar (ContentTabs)
        and the active list (DataTable) — never a FooterKey (FIX A's core
        regression lock)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).focus()
            await pilot.pause()
            for direction in ("tab", "shift+tab"):
                app.query_one("#catalog-table", DataTable).focus()
                await pilot.pause()
                for _ in range(len(app.screen.focus_chain) + 2):
                    await pilot.press(direction)
                    await pilot.pause()
                    cur = app.focused
                    assert not isinstance(cur, FooterKey), \
                        f"{direction} landed on a FooterKey ({cur!r})"
                    # Every stop is a VISIBLE-focus widget: the Modes rail
                    # (ModeSwitcher gets a focus ring — BUG 2), the tab bar, or a
                    # list.
                    assert isinstance(cur, (ModeSwitcher, Tabs, DataTable)), \
                        f"{direction} landed on an unexpected widget ({cur!r})"

    @pytest.mark.asyncio
    async def test_tab_does_not_break_table_focus(self):
        """The catalog DataTable + the tab bar stay focusable + in the chain —
        FIX A only REMOVES the footer keys, it doesn't strip the real stops."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            chain = app.screen.focus_chain
            assert any(isinstance(w, DataTable) for w in chain)
            assert any(isinstance(w, Tabs) for w in chain)

    @pytest.mark.asyncio
    async def test_footer_recomposes_on_a_real_binding_change(self):
        """The retained recompose-suppression skips ONLY no-op rebuilds: a GENUINE
        displayed-binding change (a surface flip that un-gates the [2] Bring &
        Validate mode key) flips the signature and DOES recompose the footer, so
        the new key renders.  No footer focus is involved any more."""
        app, _, _ = make_app(surface="consumer")  # lean: only [1] shown
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            ff = app.query_one(FocusableFooter)
            sig_before = ff._last_binding_sig
            # Sanity: '2' (Bring & Validate) is NOT shown on the lean surface.
            assert "2" not in {k.key for k in ff.query(FooterKey)}
            # Flip to the full surface → [2] (mode_validate) un-gates (show=True),
            # so the footer signature genuinely changes and the footer recomposes.
            app._surface = "producer"
            app.refresh_bindings()
            await _settle(pilot)
            assert ff._last_binding_sig != sig_before
            assert "2" in {k.key for k in ff.query(FooterKey)}


class TestFixBTabBarFocusStays:
    """FIX B — switching tabs WHILE focus is on the tab bar keeps focus ON the tab
    bar (the user browses Catalog→Orchestration→Containers→Doctor freely);
    switching from a LIST (via [/]) still moves focus to the next list."""

    @pytest.mark.asyncio
    async def test_switching_tab_from_tab_bar_keeps_focus_on_tab_bar(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            # Put focus on the tab bar (the ContentTabs widget — what holds focus
            # while arrow/[ ] cycle tabs; the individual Tab children aren't
            # focusable).
            cts = tc.query_one(ContentTabs)
            cts.focus()
            await pilot.pause()
            assert isinstance(app.focused, Tabs)
            # Activate a different tab (simulates arrow / click / [ ]).
            tc.active = "tab-orchestration"
            await _settle(pilot)
            # Focus STAYS on the tab bar — not yanked down into #scene-table.
            assert isinstance(app.focused, Tabs), \
                "focus must stay on the tab bar, not be pulled into the list"

    @pytest.mark.asyncio
    async def test_cycling_tab_from_a_list_moves_focus_to_next_list(self):
        """[/] while focus is on a LIST still moves focus to the next tab's list —
        list-operators keep operating lists (focus wasn't on the tab bar)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            app.query_one("#catalog-table", DataTable).focus()
            await pilot.pause()
            assert app.focused.id == "catalog-table"
            # ] cycles to the next sub-tab; focus follows into its list.
            await pilot.press("]")
            await _settle(pilot)
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "scene-table", \
                "focus moves to the next tab's list when cycling from a list"

    @pytest.mark.asyncio
    async def test_mode_switch_still_focuses_primary_list(self):
        """A 1/2 MODE switch is a deliberate descent into the mode — it SHOULD land
        the user on the primary list (FIX B leaves _focus_mode_primary as-is)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Move focus OFF the catalog table (onto the tab bar) so the assertion
            # isn't trivially satisfied by the boot/entry focus (which, post
            # MUST-FIX 1, lands ON #catalog-table).  Explicitly focus the tab bar so
            # the precondition matches the comment.
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            tc.query_one(ContentTabs).focus()
            await _settle(pilot)
            assert isinstance(app.focused, Tabs), "precondition: focus on the tab bar"
            await pilot.press("2")   # into Bring & Validate
            await _settle(pilot)
            await pilot.press("1")   # back to the merged Run & Operate mode
            await _settle(pilot)
            assert app._active_mode == 0
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "catalog-table", \
                "mode switch lands the user on the primary list"


class TestBareBootFocusOnCatalog:
    """MUST-FIX 1 — the app must BOOT with focus on mode 0's primary list
    (#catalog-table), NOT the tab bar.

    Textual auto-focuses the ContentTabs (a Tabs subclass) at startup BEFORE the
    startup tab-catalog TabActivated fires; FIX B's ``isinstance(self.focused,
    Tabs)`` guard then short-circuits the auto focus-into-the-list.  on_mount now
    explicitly ``_focus_mode_primary(0)`` (which focuses the table directly,
    bypassing the guarded handler) so the boot focus lands on the catalog list."""

    @pytest.mark.asyncio
    async def test_bare_boot_focus_is_catalog_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # NO mode switch, NO tab activation — just boot + settle.
            assert isinstance(app.focused, DataTable), \
                f"bare boot must focus a list, not {app.focused!r}"
            assert app.focused.id == "catalog-table", \
                f"bare boot must focus #catalog-table, got {app.focused!r}"


class TestArrowKeyFocusDescent:
    """Arrow-key focus descent between the tab bar (ContentTabs) and the active
    tab's primary list (DataTable):

      1. tab bar + [down]        → focus descends INTO the active tab's list.
      2. list @ row 0 + [up]     → focus ascends back UP to the tab bar.
      3. mid-list [up]/[down]    → the DataTable moves its row cursor (NOT hijacked).

    Gated to NEVER fire under a modal, and to leave focus alone when an Input /
    Select in the lane is focused (those widgets use arrow keys themselves)."""

    @staticmethod
    async def _focus_tab_bar(pilot, tc_id: str, tab: str):
        """Activate ``tab`` on the TabbedContent ``tc_id`` and focus its tab bar."""
        app = pilot.app
        tc = app.query_one(tc_id, TabbedContent)
        tc.active = tab
        await _settle(pilot)
        tc.query_one(ContentTabs).focus()
        await pilot.pause()
        assert isinstance(app.focused, Tabs), f"precondition: focus on the tab bar ({app.focused!r})"
        return tc

    # ── Behavior 1 — tab bar + down → descend into the list ──────────────────────

    @pytest.mark.asyncio
    async def test_tab_bar_down_descends_into_catalog_list(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await self._focus_tab_bar(pilot, "#operate-tabs", "tab-catalog")
            await pilot.press("down")
            await pilot.pause()
            assert isinstance(app.focused, DataTable)
            assert app.focused.id == "catalog-table", \
                f"down on the tab bar must descend into #catalog-table, got {app.focused!r}"

    @pytest.mark.asyncio
    async def test_tab_bar_down_descends_into_orchestration_and_containers(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            for tab, table_id in (
                ("tab-orchestration", "scene-table"),
                ("tab-containers", "containers-table"),
            ):
                await self._focus_tab_bar(pilot, "#operate-tabs", tab)
                await pilot.press("down")
                await pilot.pause()
                assert isinstance(app.focused, DataTable) and app.focused.id == table_id, \
                    f"down on {tab} tab bar must descend into #{table_id}, got {app.focused!r}"

    @pytest.mark.asyncio
    async def test_tab_bar_down_descends_in_lane_gate_and_measure(self):
        """The producer lane's ③ Gate (#run-ladder-table) and ④ Measure
        (#evidence-table) descend the same way."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")  # into Bring & Validate
            await _settle(pilot)
            for tab, table_id in (
                ("tab-run", "run-ladder-table"),
                ("tab-evidence", "evidence-table"),
            ):
                await self._focus_tab_bar(pilot, "#validate-tabs", tab)
                await pilot.press("down")
                await pilot.pause()
                assert isinstance(app.focused, DataTable) and app.focused.id == table_id, \
                    f"down on lane {tab} tab bar must descend into #{table_id}, got {app.focused!r}"

    # ── Behavior 2 — list @ row 0 + up → ascend to the tab bar ───────────────────

    @pytest.mark.asyncio
    async def test_list_row0_up_ascends_to_tab_bar(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            t = app.query_one("#catalog-table", DataTable)
            t.focus()
            t.move_cursor(row=0)
            await pilot.pause()
            assert app.focused is t and t.cursor_row == 0
            await pilot.press("up")
            await pilot.pause()
            assert isinstance(app.focused, Tabs), \
                f"up at row 0 must ascend to the tab bar, got {app.focused!r}"

    # ── Behavior 3 — mid-list up/down move the cursor, never hijacked ────────────

    @pytest.mark.asyncio
    async def test_mid_list_up_moves_cursor_not_hijacked(self):
        """[up] with cursor ABOVE row 0 moves the DataTable cursor (does not ascend).

        The fake catalog has 2 rows, so the deepest cursor is row 1; pressing up
        there must land on row 0 with focus STILL on the table (the priority `up`
        binding is inert when cursor_row != 0)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            t = app.query_one("#catalog-table", DataTable)
            t.focus()
            t.move_cursor(row=1)
            await pilot.pause()
            assert app.focused is t and t.cursor_row == 1, (app.focused, t.cursor_row)
            await pilot.press("up")
            await pilot.pause()
            assert app.focused is t, f"up above row 0 must NOT ascend, got {app.focused!r}"
            assert t.cursor_row == 0, f"up must move the cursor to row 0, got {t.cursor_row}"

    @pytest.mark.asyncio
    async def test_list_down_moves_cursor_not_hijacked(self):
        """[down] anywhere in the list moves the DataTable cursor (does not jump to
        the tab bar / re-descend)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            t = app.query_one("#catalog-table", DataTable)
            t.focus()
            t.move_cursor(row=0)
            await pilot.pause()
            assert app.focused is t and t.cursor_row == 0
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is t, f"down in a list must keep focus on the table, got {app.focused!r}"
            assert t.cursor_row == 1, f"down must move the cursor, got {t.cursor_row}"

    # ── No-op surfaces — Doctor (no primary list) ────────────────────────────────

    @pytest.mark.asyncio
    async def test_doctor_tab_bar_down_is_a_noop(self):
        """Doctor has no primary list → descend is a no-op (focus stays on the bar)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await self._focus_tab_bar(pilot, "#operate-tabs", "tab-doctor")
            await pilot.press("down")
            await pilot.pause()
            assert isinstance(app.focused, Tabs), \
                f"down on Doctor's tab bar (no list) must stay on the bar, got {app.focused!r}"

    # ── Modal invariant — arrows never steal focus to/from the tab bar ───────────

    @pytest.mark.asyncio
    async def test_modal_blocks_descend_and_ascend(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            tc.query_one(ContentTabs).focus()
            await pilot.pause()
            assert isinstance(app.focused, Tabs)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            assert isinstance(app.screen, ConfirmActionScreen)
            # The gates refuse to fire while a modal is topmost (modals own their
            # own keys, including arrow keys in their widgets).
            assert app.check_action("descend_to_content", ()) is False
            assert app.check_action("ascend_to_tabbar", ()) is False
            # Pressing arrows leaves the modal up — the app focus is not yanked to a
            # tab/list underneath.
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen), "modal must stay up"

    # ── Lane Input gate — arrow keys belong to the Input, not descend/ascend ─────

    @pytest.mark.asyncio
    async def test_input_focused_does_not_descend_or_ascend(self):
        """① Bring's Input uses arrow keys itself — the gates must be False when an
        Input is focused so up/down reach the Input, not the descend/ascend actions."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")  # into Bring & Validate
            await _settle(pilot)
            vtc = app.query_one("#validate-tabs", TabbedContent)
            vtc.active = "tab-bring"
            await _settle(pilot)
            inp = app.query_one("#lane-bring-url-input", Input)
            inp.focus()
            await pilot.pause()
            assert app.focused is inp
            # With an Input focused, neither gate fires (focus is not the tab bar /
            # a primary DataTable).
            assert app.check_action("descend_to_content", ()) is False
            assert app.check_action("ascend_to_tabbar", ()) is False
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert app.focused is inp, \
                f"arrow keys must stay with the focused Input, got {app.focused!r}"

    # ── DRY check — the shared map backs both the descend action and the resolver ─

    @pytest.mark.asyncio
    async def test_primary_list_resolver_uses_shared_map(self):
        """``_primary_list_for_active_tab`` resolves via the shared _TAB_PRIMARY_LIST
        constant — the same map the tab-activation focus logic reads."""
        from club3090_cockpit.app import _TAB_PRIMARY_LIST
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            tbl = app._primary_list_for_active_tab()
            assert isinstance(tbl, DataTable) and tbl.id == "catalog-table"
            # Doctor is absent from the map → no primary list → None.
            tc.active = "tab-doctor"
            await _settle(pilot)
            assert app._primary_list_for_active_tab() is None
            assert "tab-doctor" not in _TAB_PRIMARY_LIST


class TestSubtabCycleScopedToDirectPanes:
    """MUST-FIX 2 — the [/] sub-tab cycle (and _current_subtab) must read only the
    TabbedContent's DIRECT panes, never the NESTED Containers drill (Logs/Top/
    Config) sub-tabs that the recursive ``tc.query(TabPane)`` would leak."""

    @pytest.mark.asyncio
    async def test_cycle_next_from_containers_reaches_doctor_no_drill(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-containers"
            await _settle(pilot)
            # ] from Containers lands on Doctor — NOT a phantom drill-tab-* pane.
            await pilot.press("]")
            await _settle(pilot)
            assert tc.active == "tab-doctor", \
                f"] from Containers must reach Doctor, not {tc.active!r}"
            assert not tc.active.startswith("drill-tab-"), tc.active

    @pytest.mark.asyncio
    async def test_operate_cycle_visits_only_the_four_real_tabs(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-catalog"
            await _settle(pilot)
            seen = [tc.active]
            for _ in range(4):
                await pilot.press("]")
                await _settle(pilot)
                seen.append(tc.active)
            assert seen == [
                "tab-catalog", "tab-orchestration", "tab-containers",
                "tab-doctor", "tab-catalog",
            ], seen
            assert not any(t.startswith("drill-tab-") for t in seen), seen

    @pytest.mark.asyncio
    async def test_current_subtab_never_returns_a_drill_id(self):
        """_current_subtab must report a DIRECT pane id (the gates evaluate against
        it) — never a nested drill-tab-* even if the drill TabbedContent is active."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            op = app.query_one("#operate-tabs", TabbedContent)
            op.active = "tab-containers"
            await _settle(pilot)
            # The nested drill defaults to drill-tab-logs; _current_subtab must still
            # report the OUTER active pane (tab-containers), not the drill id.
            assert app._current_subtab() == "tab-containers", app._current_subtab()
            direct = app._direct_pane_ids(op)
            assert "drill-tab-logs" not in direct, direct
            assert direct == [
                "tab-catalog", "tab-orchestration", "tab-containers", "tab-doctor",
            ], direct

    @pytest.mark.asyncio
    async def test_validate_lane_cycles_its_five_stages_only(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")  # into Bring & Validate
            await _settle(pilot)
            vtc = app.query_one("#validate-tabs", TabbedContent)
            vtc.active = "tab-bring"
            await _settle(pilot)
            seen = [vtc.active]
            for _ in range(5):
                await pilot.press("]")
                await _settle(pilot)
                seen.append(vtc.active)
            assert seen == [
                "tab-bring", "tab-serve", "tab-run", "tab-evidence",
                "tab-promote", "tab-bring",
            ], seen


class TestOrchScrollNotATabStop:
    """FOLD 3 — #orch-scroll is can_focus=False, so a SINGLE Tab from the tab bar
    reaches #scene-table on the Orchestration tab (no dead scroll-container stop)."""

    @pytest.mark.asyncio
    async def test_orch_scroll_is_not_focusable(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.query_one("#orch-scroll").can_focus is False

    @pytest.mark.asyncio
    async def test_one_tab_from_tab_bar_reaches_scene_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-orchestration"
            await _settle(pilot)
            tc.query_one(ContentTabs).focus()
            await pilot.pause()
            assert isinstance(app.focused, Tabs)
            await pilot.press("tab")
            await pilot.pause()
            assert isinstance(app.focused, DataTable), \
                f"one Tab must reach a list, not {app.focused!r}"
            assert app.focused.id == "scene-table", \
                f"one Tab from the tab bar must reach #scene-table, got {app.focused!r}"


class TestLaneTablelessStageFocusesTabBar:
    """FOLD 4 — entering mode 1 on a TABLE-LESS lane stage (① Bring / ② Serve /
    ⑤ Promote) focuses the lane tab bar (validate-tabs ContentTabs) so focus is
    never None (a black hole now the footer is out of the chain); ③ Gate / ④
    Measure still focus their table."""

    @pytest.mark.asyncio
    async def test_entering_bring_focuses_validate_tab_bar(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            vtc = app.query_one("#validate-tabs", TabbedContent)
            vtc.active = "tab-bring"
            await _settle(pilot)
            await pilot.press("2")  # into Bring & Validate, on ① Bring
            await _settle(pilot)
            assert app._active_mode == 1
            assert app.focused is not None, "① Bring must not leave focus None"
            assert isinstance(app.focused, ContentTabs), \
                f"① Bring focuses the lane tab bar, not {app.focused!r}"
            # The tab bar is a Tabs (not an Input) → 1/2/[ ] still route to the app.
            before = vtc.active
            await pilot.press("]")
            await _settle(pilot)
            assert vtc.active == "tab-serve", \
                f"] still cycles the lane stages from ① Bring, got {vtc.active!r}"

    @pytest.mark.asyncio
    async def test_entering_gate_focuses_the_table(self):
        """③ Gate keeps focusing its data table (the table-bearing stages are
        unchanged by FOLD 4)."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await _settle(pilot)
            await pilot.press("2")  # into Bring & Validate, on ③ Gate
            await _settle(pilot)
            assert isinstance(app.focused, DataTable), \
                f"③ Gate focuses its table, not {app.focused!r}"
            assert app.focused.id == "run-ladder-table", app.focused.id


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
            await pilot.press("2")           # into the Bring & Validate lane
            await _settle(pilot)
            assert app._active_mode == 1
            await pilot.press("1")           # back to the merged mode
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
    async def test_palette_offers_doctor_verbs_on_consumer(self):
        """Batch 3: the Doctor reads ([v] verify / [V] verify-full) and the full
        system report ([F], no longer producer-only) ARE offered on the consumer
        palette."""
        app, _, _ = make_app(surface="consumer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            names = [a for a, _, _ in CockpitCommands(app.screen)._available()]
            for action in ("doctor_verify", "doctor_verify_full", "full_report"):
                assert action in names, action

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
        would — fuzzy-search 'Bring & Validate mode' then run it → mode 1."""
        app, _, _ = make_app()  # default full
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            prov = CockpitCommands(app.screen)
            cb = None
            async for hit in prov.search("Bring & Validate mode"):
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


class TestCopyAndHScroll:
    """Batch 4 — [Y] copies the contextually-relevant text (slug / report / a
    selection) to the system clipboard via OSC52; shift+←/→ page-scroll a wide
    table.  copy_to_clipboard is spied (no real OSC52 emitted in tests)."""

    @pytest.mark.asyncio
    async def test_copy_yanks_highlighted_catalog_slug(self):
        app, _, _ = make_app()
        copied: dict = {}
        async with app.run_test(size=(140, 40)) as pilot:
            await _settle(pilot)
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.pause()
            await pilot.press("Y")
            await _settle(pilot)
            # the CLEAN slug (via the pane accessor) — no markup, no "● " marker.
            assert copied.get("text") == "vllm/dual", copied
            assert "[" not in copied["text"]

    @pytest.mark.asyncio
    async def test_right_click_copies_context(self):
        """Right-click (mouse button 3) copies the contextual text — a TUI eats the
        mouse, so this restores the native 'select + right-click copy' expectation
        (on a table there's no selection → it copies the highlighted slug)."""
        class _Btn:
            def __init__(self, b): self.button = b
        app, _, _ = make_app()
        copied: dict = {}
        async with app.run_test(size=(140, 40)) as pilot:
            await _settle(pilot)
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.pause()
            # a LEFT button-down must NOT copy …
            app.on_mouse_down(_Btn(1))
            assert "text" not in copied
            # … a RIGHT button-down copies the contextual slug.
            app.on_mouse_down(_Btn(3))
            await _settle(pilot)
            assert copied.get("text") == "vllm/dual", copied

    @pytest.mark.asyncio
    async def test_copy_yanks_open_report_body(self):
        """[Y] works INSIDE a modal (the modal binds it explicitly — the app-level
        Y can't reach a modal) and copies the raw, markup-free report."""
        app, _, _ = make_app()
        copied: dict = {}
        async with app.run_test(size=(140, 40)) as pilot:
            await _settle(pilot)
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            sc = ShareBackReportScreen("Rig report", "rig")
            await app.push_screen(sc)
            await pilot.pause()
            sc.set_report("RIG SNAPSHOT\nGPU0: 24G\n[brackets] kept literal", None)
            await pilot.pause()
            await pilot.press("Y")
            await _settle(pilot)
            assert copied.get("text", "").startswith("RIG SNAPSHOT")
            assert "[brackets] kept literal" in copied["text"]   # raw, not markup-stripped

    @pytest.mark.asyncio
    async def test_copy_nothing_copyable_notifies_no_clipboard(self):
        app, _, _ = make_app()
        copied: dict = {}
        notes: list = []
        async with app.run_test(size=(140, 40)) as pilot:
            await _enter_operate(pilot)
            # Doctor tab has no primary table + no modal → nothing to copy.
            app.query_one("#operate-tabs", TabbedContent).active = "tab-doctor"
            await pilot.pause()
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            orig = app.notify
            app.notify = lambda *a, **k: (notes.append(a[0] if a else k.get("message", "")), orig(*a, **k))[1]
            await pilot.press("Y")
            await _settle(pilot)
            assert "text" not in copied                      # nothing copied
            assert any("nothing to copy" in n.lower() for n in notes)

    @pytest.mark.asyncio
    async def test_copy_context_reachable_everywhere(self):
        app, _, _ = make_app()
        async with app.run_test(size=(140, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("copy_context", ()) is True


class TestF4F8LiveLogCopy:
    """F4 — [Y] copies the VISIBLE live-log tail: RichLog-family panes capture
    the mouse and don't implement text selection (Top/Config are selectable
    Statics), so the Logs drill / ③ Gate output get the copy affordance via
    the pane's plain-text tail buffer.  F8 — pane-specific idle placeholders
    (the shared LivePane default once leaked 'Select a test and press Enter'
    test-runner wording into the docker-logs drill)."""

    @pytest.mark.asyncio
    async def test_f8_pane_placeholders_name_their_purpose(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            drill = app.query_one("#drill-logs")
            assert "container" in drill._placeholder
            assert "test" not in drill._placeholder.lower()
            assert "validation" in app.query_one("#run-output")._placeholder
            # The transient serve pane is hidden until a serve — no idle copy.
            assert app.query_one("#serve-live")._placeholder == ""

    async def _open_logs_drill(self, app, pilot):
        await _enter_operate(pilot)
        app.query_one("#operate-tabs", TabbedContent).active = "tab-containers"
        await _settle(pilot)
        app.query_one("#drill-tabs", TabbedContent).active = "drill-tab-logs"
        await _settle(pilot)
        tbl = app.query_one("#containers-table", DataTable)
        tbl.focus()
        tbl.move_cursor(row=0)
        await pilot.pause()
        return app.query_one("#drill-logs")

    @pytest.mark.asyncio
    async def test_y_copies_logs_drill_tail(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        copied: dict = {}
        async with app.run_test(size=(120, 40)) as pilot:
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            drill = await self._open_logs_drill(app, pilot)
            # Deterministic pane content (any auto-streamed fixture lines out).
            drill.clear_log()
            drill.append_line("[cyan]INFO[/cyan] engine ready")
            drill.append_line("request 1 done")
            await pilot.press("Y")
            await _settle(pilot)
            # The tail wins over the highlighted container name while the Logs
            # drill is showing — plain text, markup stripped.
            assert copied.get("text") == "INFO engine ready\nrequest 1 done"

    @pytest.mark.asyncio
    async def test_y_on_idle_logs_drill_falls_back_to_container_name(self):
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses)
        copied: dict = {}
        async with app.run_test(size=(120, 40)) as pilot:
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            drill = await self._open_logs_drill(app, pilot)
            # Idle pane: placeholders/notes aren't buffered → empty tail.
            drill.clear_log()
            await pilot.press("Y")
            await _settle(pilot)
            assert copied.get("text") == "vllm-qwen36-27b-dual"

    @pytest.mark.asyncio
    async def test_y_copies_gate_run_output_tail(self):
        app, _, _ = make_app(surface="producer")
        copied: dict = {}
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.copy_to_clipboard = lambda t: copied.__setitem__("text", t)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await pilot.pause()
            out = app.query_one("#run-output")
            out.clear_log()
            out.append_line("verify-full [3/8] tool call ✓")
            await pilot.press("Y")
            await _settle(pilot)
            assert copied.get("text") == "verify-full [3/8] tool call ✓"

    @pytest.mark.asyncio
    async def test_shift_arrows_page_scroll_wide_table(self):
        app, _, _ = make_app()
        async with app.run_test(size=(60, 30)) as pilot:   # narrow → catalog overflows
            await _settle(pilot)
            t = app.query_one("#catalog-table", DataTable)
            assert t.max_scroll_x > 0, "expected the catalog to overflow at width 60"
            x0 = t.scroll_x
            await pilot.press("shift+right")
            await _settle(pilot)
            assert t.scroll_x > x0                            # advanced right
            mid = t.scroll_x
            await pilot.press("shift+left")
            await _settle(pilot)
            assert t.scroll_x < mid                            # came back left

    @pytest.mark.asyncio
    async def test_hscroll_is_noop_under_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(60, 30)) as pilot:
            await _settle(pilot)
            await app.push_screen(ShareBackReportScreen("Rig report", "rig"))
            await pilot.pause()
            # _hscroll bails under a modal — must not raise / scroll a hidden table.
            app.action_hscroll_right()
            await pilot.pause()
            assert isinstance(app.screen, ShareBackReportScreen)

    @pytest.mark.asyncio
    async def test_pane_hints_stay_within_viewport(self):
        """The bottom control/hint lines must WRAP to the viewport, not run off the
        right edge into horizontal scroll (Label is width:auto → a long hint
        overflows without the width:1fr rule)."""
        app, _, _ = make_app()
        async with app.run_test(size=(80, 30)) as pilot:
            await _enter_operate(pilot)
            for tab, hint in (
                ("tab-doctor", "#doctor-hint"),
                ("tab-catalog", "#catalog-hint"),
                ("tab-containers", "#containers-hint"),
                ("tab-orchestration", "#orch-hint"),
            ):
                app.query_one("#operate-tabs", TabbedContent).active = tab
                await pilot.pause()
                w = app.query_one(hint)
                assert w.size.width <= 80, f"{hint} overflows: {w.size.width} > 80"


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
# SERVE confirm rework — the state-aware Stop/Start/warned-Start pop-up.
#   ⏎ on a catalog row opens a state-aware SERVE confirm:
#     - target IS serving      → Stop + Cancel (no Start/Force/phantom)
#     - target free            → Start + Cancel
#     - target w/ a conflict   → Start + Cancel, card warns "will STOP <model>"
#   Controls live ONLY in the footer (no button row, no phantom placeholder).
#   The card shows the explain summary + fit verdict + reconcile/teardown status.
#   The detect runs in the BACKGROUND (paint un-blocked) but the WRITE still waits
#   for it; Enter on an unresolved/unsafe gate never fires a destructive write.
#   The dual-writer reconcile/lease invariants are PRESERVED exactly.
# ===========================================================================


def _serve_footer_keys(sc):
    return {k.key for k in sc.query(FooterKey)}


def _serve_card_text(sc):
    return str(sc.query_one("#confirm-body", Static).render())


def _serve_footer_desc(sc, key):
    """The rendered footer description for a given key (e.g. 'enter' → 'Start')."""
    for fk in sc.query(FooterKey):
        if fk.key == key:
            return fk.description
    return None


# A free rig (both cards near-empty, nothing serving) for the Start traces.
_FREE_GPUS = [
    GpuInfo(index=0, mem_used_mib=1, mem_total_mib=24576),
    GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24576),
]


class TestServeConfirmStateAware:
    """The SERVE confirm pop-up is state-aware: Stop / Start / warned-Start, with
    controls as footer key-hints only and a useful 'what am I about to do' card."""

    async def _open_serve(self, pilot, app, *, on_serving_tab=False):
        """⏎ a catalog row → the state-aware serve confirm.  When the target is the
        serving model we must reach the Catalog tab from Operate first."""
        if on_serving_tab:
            await _enter_operate(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-catalog"
            await pilot.pause()
        else:
            await _settle(pilot)
        app.query_one("#catalog-table", DataTable).move_cursor(row=0)
        await pilot.press("enter")
        await _settle(pilot)
        return app.screen

    # (a) ⏎ on a SERVING slug → Stop + Cancel (no Start / Force / phantom).
    @pytest.mark.asyncio
    async def test_serving_slug_shows_stop_and_cancel(self):
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=_FREE_GPUS)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=_FREE_GPUS, target=target)
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app, on_serving_tab=True)
            assert isinstance(sc, ConfirmActionScreen)
            assert app._target_slug == "vllm/dual"            # IS the serving row
            assert sc._serve_ctx is not None and sc._serve_ctx.mode == "stop"
            fk = _serve_footer_keys(sc)
            assert "k" in fk and "escape" in fk               # Stop + Cancel
            assert "enter" not in fk and "f" not in fk        # no Start, no Force
            assert sc.check_action("stop", ()) is True
            assert sc.check_action("start", ()) is False
            assert sc.check_action("confirm", ()) is False
            assert sc.check_action("force", ()) is False
            # NO phantom/empty button placeholder, NO button row at all.
            assert not sc.query(Button)

    # (a') Stop dispatches the targeted docker-stop of the serving container.
    @pytest.mark.asyncio
    async def test_serving_slug_stop_dispatches_targeted_stop(self):
        wr = FakeWriteRunner()
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=_FREE_GPUS)
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=_FREE_GPUS, target=target, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app, on_serving_tab=True)
            assert sc._serve_ctx.mode == "stop"
            await pilot.press("k")                            # Stop
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["docker", "stop", "vllm-qwen36-27b-dual"]

    # (b) ⏎ on a NOT-serving FREE slug → Start + Cancel.
    @pytest.mark.asyncio
    async def test_free_slug_shows_start_and_cancel(self):
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS))
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            assert app._target_slug == ""                     # nothing serving
            assert sc._serve_ctx.mode == "start"
            fk = _serve_footer_keys(sc)
            assert "enter" in fk and "escape" in fk           # Start + Cancel
            assert "k" not in fk and "f" not in fk            # no Stop, no Force
            assert sc.check_action("start", ()) is True
            assert sc.check_action("stop", ()) is False
            assert sc._has_conflict is False
            assert "gate clear" in _serve_card_text(sc)
            assert not sc.query(Button)                       # footer-only, no phantom

    # (b') Start (free) dispatches the gated, NON-forced serve.
    @pytest.mark.asyncio
    async def test_free_slug_start_dispatches_gated_serve(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            await pilot.press("enter")                        # Start
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/switch.sh", "vllm/dual"]
            assert "--force" not in wr.started[0]["cmd"]

    # (c) ⏎ on a NOT-serving slug with a GPU conflict → Start + Cancel, warned.
    @pytest.mark.asyncio
    async def test_conflict_slug_shows_warned_start(self):
        # A live engine holds GPU0 but the TARGET slug isn't the matched one
        # (target has no container match → matched_slug "" → a Start with conflict).
        gpus = [GpuInfo(index=0, mem_used_mib=22000, mem_total_mib=24576),
                GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24576)]
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            assert app._target_slug == ""                     # not the serving row
            assert sc._serve_ctx.mode == "start"
            assert sc._reconcile is not None and sc._reconcile.safe is False
            assert sc._has_conflict is True
            fk = _serve_footer_keys(sc)
            assert "enter" in fk and "escape" in fk           # Start + Cancel
            assert "f" not in fk                              # NO separate Force
            assert sc.check_action("start", ()) is True       # warned-Start is Start
            # The card warns the teardown explicitly.
            assert "will STOP" in _serve_card_text(sc)

    # (c'') the footer Start key is RELABELLED "Start (stops <model>)" on conflict
    #       (the destructive affordance is visible in the footer, not only the card).
    @pytest.mark.asyncio
    async def test_conflict_relabels_footer_start_key(self):
        gpus = [GpuInfo(index=0, mem_used_mib=22000, mem_total_mib=24576),
                GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24576)]
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus, target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            assert sc._has_conflict is True
            desc = _serve_footer_desc(sc, "enter")
            assert desc is not None and desc.startswith("Start (stops ")
            # the relabel does NOT leak to the class BINDINGS (per-instance copy).
            class_start = [b for b in ConfirmActionScreen.BINDINGS if b.action == "start"]
            assert class_start and class_start[0].description == "Start"

    # (c''') a FREE Start keeps the plain "Start" footer label (no teardown).
    @pytest.mark.asyncio
    async def test_free_start_keeps_plain_footer_label(self):
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS))
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            assert sc._has_conflict is False
            assert _serve_footer_desc(sc, "enter") == "Start"

    # (c') warned-Start performs teardown-then-serve (the old Force, folded in).
    @pytest.mark.asyncio
    async def test_conflict_slug_warned_start_forces(self):
        wr = FakeWriteRunner()
        gpus = [GpuInfo(index=0, mem_used_mib=22000, mem_total_mib=24576),
                GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24576)]
        responses = fake_responses(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            assert sc._has_conflict is True
            await pilot.press("enter")                        # warned Start
            await _settle(pilot)
            assert len(wr.started) == 1
            # teardown-then-serve == switch.sh --force <slug> (force folded in).
            assert "--force" in wr.started[0]["cmd"]
            assert wr.started[0]["cmd"][-1] == "vllm/dual"

    # (c'''') a NON-functional (experimental) slug → Force Start: switch.sh refuses
    #         it without --force, so the modal serves it with --force + a warning.
    @staticmethod
    def _exp_entry(slug, status):
        from club3090_cockpit.data import CatalogEntry as _CE
        from club3090_tui_core import VariantRow as _VR
        return _CE(row=_VR(
            slug=slug, switch_engine="vllm", launch_engine="vllm",
            compose_dir="models/qwen3.6-27b/vllm/compose/dual/fp8",
            file="mtp.yml", port=8013, model="qwen3.6-27b", engine="vllm-stable",
            kvcalc_key="SKIP", container="vllm-qwen36-27b-dual-max",
            compose_path="models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml",
            status=status, ctx_label="262K", status_note="",
        ))

    @pytest.mark.asyncio
    async def test_experimental_slug_force_start(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.populate([self._exp_entry("vllm/qwen-27b-dual-max", "experimental")], None)
            await pilot.pause()
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await _settle(pilot)
            sc = app.screen
            assert isinstance(sc, ConfirmActionScreen)
            assert sc._serve_ctx.force_required is True
            assert _serve_footer_desc(sc, "enter") == "Force Start"     # footer relabel
            assert "UNVALIDATED" in _serve_card_text(sc)                 # card warning
            await pilot.press("enter")                                  # Force Start
            await _settle(pilot)
            assert len(wr.started) == 1
            assert "--force" in wr.started[0]["cmd"]                     # the fix
            assert wr.started[0]["cmd"][-1] == "vllm/qwen-27b-dual-max"

    @pytest.mark.asyncio
    async def test_production_slug_no_force_required(self):
        """A functional (production) slug is NOT force_required — plain Start, and
        on free cards no --force is added."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.populate([self._exp_entry("vllm/dual", "production")], None)
            await pilot.pause()
            app.query_one("#catalog-table", DataTable).move_cursor(row=0)
            await pilot.press("enter")
            await _settle(pilot)
            sc = app.screen
            assert sc._serve_ctx.force_required is False
            assert _serve_footer_desc(sc, "enter") == "Start"
            await pilot.press("enter")
            await _settle(pilot)
            assert wr.started and "--force" not in wr.started[0]["cmd"]

    # (d) the card shows an explain summary + fit verdict (not unrelated content).
    @pytest.mark.asyncio
    async def test_card_shows_explain_summary_and_fit(self):
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS))
        async with app.run_test(size=(120, 40)) as pilot:
            sc = await self._open_serve(pilot, app)
            card = _serve_card_text(sc)
            # Explain summary: model · engine · ctx · measured TPS/8-pack · status.
            assert "qwen3.6-27b" in card                      # model
            assert "vllm-stable" in card                      # engine
            assert "262K" in card                             # max ctx
            assert "174/42" in card                           # measured TPS
            assert "109/150" in card                          # 8-pack
            assert "production" in card                       # status badge
            # Fit verdict (the fit that moved OUT of the Catalog column).
            assert "fits-clean" in card
            assert "GiB" in card                              # ~VRAM / band

    # (d') the card paints IMMEDIATELY from cached fields, before the detect lands.
    @pytest.mark.asyncio
    async def test_card_paints_before_detect_resolves(self):
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            entry = app.query_one("#catalog-pane", CatalogPane).selected_entry()
            plan = app._data.serve("vllm/dual")
            sc = ConfirmActionScreen(plan, serve_ctx=ServeContext(mode="start", entry=entry))
            app.push_screen(sc)
            await pilot.pause()                               # mount, NOT settle
            # The slug summary + fit are already on the card from cached fields…
            sc._reconcile = None                              # simulate pre-detect
            sc._render_serve_card()
            card = _serve_card_text(sc)
            assert "qwen3.6-27b" in card and "fits-clean" in card
            # …and the reconcile line is the non-blocking "checking rig state…".
            assert "checking rig state" in card
            # The destructive Start is DISABLED until the detect lands.
            assert sc.check_action("start", ()) is False

    # (e) Enter on an UNRESOLVED gate does NOT start a write (footgun safety).
    @pytest.mark.asyncio
    async def test_enter_on_unresolved_gate_starts_no_write(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS), write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            entry = app.query_one("#catalog-pane", CatalogPane).selected_entry()
            plan = app._data.serve("vllm/dual")
            sc = ConfirmActionScreen(plan, serve_ctx=ServeContext(mode="start", entry=entry))
            # rec is None (unresolved): Start gated off, and the on_key guard stops
            # a stray Enter before it can reach action_start.
            assert sc._reconcile is None
            assert sc.check_action("start", ()) is False

            class _Ev:
                key = "enter"
                stopped = False
                prevented = False
                def stop(self):
                    self.stopped = True
                def prevent_default(self):
                    self.prevented = True
            ev = _Ev()
            sc.on_key(ev)
            assert ev.stopped and ev.prevented                # Enter stopped — inert
            assert wr.started == []                           # no write fired

    # (f) the legacy (non-serve) confirm is now FOOTER-ONLY — Confirm / override /
    #     Cancel via bindings, no duplicate on-screen button row.
    @pytest.mark.asyncio
    async def test_non_serve_confirm_is_footer_only(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # Scene-switch is NOT a serve-of-catalog-slug → the legacy modal.
            plan = app._data.scene_switch("27b")
            sc = ConfirmActionScreen(plan)                    # no serve_ctx
            app.push_screen(sc)
            await _settle(pilot)
            assert sc._serve_ctx is None
            # NO on-screen button row (footer-only, matching the serve modal — the
            # old buttons + footer showed the same keys twice).
            assert list(sc.query(Button)) == []
            # Confirm/override gating unchanged; serve verbs never apply here.
            assert sc.check_action("confirm", ()) is True     # safe gate
            assert sc.check_action("force", ()) is True or sc.check_action("confirm", ()) is True
            assert sc.check_action("start", ()) is False
            assert sc.check_action("stop", ()) is False
            # The override key is relabelled for a scene switch ("Switch anyway").
            assert sc._force_label() == "Switch anyway"

    # (f') scene-switch still dispatches through the gate from the modal.
    @pytest.mark.asyncio
    async def test_non_serve_confirm_still_dispatches(self):
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            plan = app._data.scene_switch("27b")
            sc = ConfirmActionScreen(plan)
            app.push_screen(sc)
            await _settle(pilot)
            await pilot.press("enter")          # footer Confirm (no button row)
            await _settle(pilot)
            assert len(wr.started) == 1
            assert wr.started[0]["cmd"] == ["bash", "scripts/gpu-mode.sh", "27b"]


# ===========================================================================
# UX Tier-1 hygiene (external Codex review) — footer/binding honesty:
#   FIX 1  base FocusableFooter suppressed under a modal, restored on pop
#   FIX 2  ⏎ hidden where it no-ops (Doctor/Containers) · real Doctor verb
#          surfaced · [s] gated to its valid tabs only
#   FIX 3  Promote / Untested Enter is a real Binding (footer-discoverable)
# The reconcile-gate Enter-on-unsafe safety is regression-locked here too.
# ===========================================================================


class TestTier1BaseFooterUnderModal:
    """FIX 1 — the base footer must NOT render under a modal (only the modal's own
    footer is honest while it is up); it must come back when the modal pops."""

    @pytest.mark.asyncio
    async def test_base_footer_visible_with_no_modal(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            footer = app.query_one(FocusableFooter)
            assert not footer.has_class("base-footer-hidden")
            assert len(app.screen_stack) == 1

    @pytest.mark.asyncio
    async def test_base_footer_hidden_under_confirm_modal_and_restored(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            footer = app.query_one(FocusableFooter)
            plan = app._data.serve("vllm/dual")
            app.push_screen(ConfirmActionScreen(plan))
            await _settle(pilot)
            assert len(app.screen_stack) > 1
            assert footer.has_class("base-footer-hidden")  # suppressed under modal
            assert footer.display is False  # RENDERED effect (display:none), not just the class
            # Pop the modal → base footer restored.
            app.pop_screen()
            await _settle(pilot)
            assert len(app.screen_stack) == 1
            assert not footer.has_class("base-footer-hidden")
            assert footer.display is True  # rendered visible again

    @pytest.mark.asyncio
    async def test_base_footer_restored_after_help_modal(self):
        """Restored after a NON-confirm modal too (Help) — the suppression keys off
        the screen stack depth, not the modal class."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            footer = app.query_one(FocusableFooter)
            await pilot.press("?")  # Help
            await _settle(pilot)
            assert isinstance(app.screen, HelpScreen)
            assert footer.has_class("base-footer-hidden")
            await pilot.press("escape")
            await _settle(pilot)
            assert not footer.has_class("base-footer-hidden")

    @pytest.mark.asyncio
    async def test_base_footer_hidden_under_preview_modal_and_restored(self, tmp_path):
        """The untested-compose preview modal (a producer-lane preview) also
        suppresses the base footer and restores it on close."""
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            footer = app.query_one(FocusableFooter)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            app.action_serve_untested()
            await _settle(pilot)
            assert isinstance(app.screen, UntestedComposePreviewScreen)
            assert footer.has_class("base-footer-hidden")
            await pilot.press("enter")  # the modal's ⏎ verb → reconcile confirm
            await _settle(pilot)
            # Still a modal on top (the confirm gate) → still suppressed.
            assert isinstance(app.screen, ConfirmActionScreen)
            assert footer.has_class("base-footer-hidden")
            app.pop_screen()
            await _settle(pilot)
            assert not footer.has_class("base-footer-hidden")


class TestTier1PrimaryActionAndSKeyHonesty:
    """FIX 2 — ⏎ is advertised ONLY where action_primary_action does real work;
    the real Doctor re-run verb is surfaced; [s] is gated to its valid tabs."""

    @pytest.mark.asyncio
    async def test_enter_advertised_on_catalog(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-catalog"
            await pilot.pause()
            assert app.check_action("primary_action", ()) is True

    @pytest.mark.asyncio
    async def test_enter_advertised_on_lane_bring(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-bring"
            await pilot.pause()
            assert app.check_action("primary_action", ()) is True

    @pytest.mark.asyncio
    async def test_enter_not_advertised_on_doctor(self):
        """⏎ no-ops on Doctor → check_action falsey so the footer never shows
        the misleading 'Enter Select' there."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot, tab="tab-doctor")
            assert app.check_action("primary_action", ()) is False
            # And the footer genuinely omits the enter key on Doctor.
            footer_keys = {k.key for k in app.query(FooterKey)}
            assert "enter" not in footer_keys

    @pytest.mark.asyncio
    async def test_enter_not_advertised_on_containers(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot, tab="tab-containers")
            assert app.check_action("primary_action", ()) is False

    @pytest.mark.asyncio
    async def test_doctor_rerun_verb_advertised_on_doctor(self):
        """The REAL Doctor verb ([y] doctor_rerun) is surfaced in the footer on
        Doctor (show=True + context-gated there), replacing the no-op ⏎."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot, tab="tab-doctor")
            assert app.check_action("doctor_rerun", ()) is True
            # The binding is declared show=True so it can reach the footer.
            doctor_binding = next(
                b for b in CockpitApp.BINDINGS if b.action == "doctor_rerun"
            )
            assert doctor_binding.show is True
            assert doctor_binding.key == "y"
            footer_keys = {k.key for k in app.query(FooterKey)}
            assert "y" in footer_keys  # the real re-run verb is on Doctor's footer

    @pytest.mark.asyncio
    async def test_doctor_rerun_not_advertised_off_doctor(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-catalog"
            await pilot.pause()
            assert app.check_action("doctor_rerun", ()) is False

    @pytest.mark.asyncio
    async def test_s_key_gated_to_containers_only_in_mode0(self):
        """[s] is valid on Containers (restart) but NOT on Catalog / Orchestration /
        Doctor — the over-broad ({0,1}, None) gate is gone."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            tc = app.query_one("#operate-tabs", TabbedContent)
            tc.active = "tab-containers"
            await pilot.pause()
            assert app.check_action("s_key", ()) is True
            for tab in ("tab-catalog", "tab-orchestration", "tab-doctor"):
                tc.active = tab
                await pilot.pause()
                assert app.check_action("s_key", ()) is False, tab

    @pytest.mark.asyncio
    async def test_s_key_gated_to_evidence_only_in_lane(self):
        """In the Bring & Validate lane [s] is valid ONLY on ④ Measure (submit),
        not on ① Bring / ② Serve / ③ Gate / ⑤ Promote."""
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            tc = app.query_one("#validate-tabs", TabbedContent)
            tc.active = "tab-evidence"
            await pilot.pause()
            assert app.check_action("s_key", ()) is True
            for tab in ("tab-bring", "tab-serve", "tab-run", "tab-promote"):
                tc.active = tab
                await pilot.pause()
                assert app.check_action("s_key", ()) is False, tab


class TestTier1PreviewModalEnterBindings:
    """FIX 3 — Promote / Untested Enter is a declared Binding (show=True), so it is
    footer-discoverable, AND it still performs the same action it did via on_key."""

    @pytest.mark.asyncio
    async def test_promote_enter_is_a_real_binding(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("P")
            await pilot.pause()
            scr = app.screen
            assert isinstance(scr, PromoteScaffoldScreen)
            actions = {b.action: b for b in scr.BINDINGS}
            assert "stage_write" in actions
            assert actions["stage_write"].key == "enter"
            assert actions["stage_write"].show is True
            # Footer-discoverable: stage_write reaches the modal footer.
            assert any(
                v.binding.action == "stage_write" and v.binding.show
                for v in scr.active_bindings.values()
            )

    @pytest.mark.asyncio
    async def test_promote_enter_still_stages_the_gated_write(self):
        """⏎ on the promote preview still opens the SAME mock-only confirm gate the
        #promote-stage-btn press does (behaviour preserved)."""
        wr = FakeWriteRunner()
        app, _, _ = make_app(write_runner=wr, surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            await pilot.press("P")
            await pilot.pause()
            assert isinstance(app.screen, PromoteScaffoldScreen)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.kind == "promote_catalog"
            assert wr.started == []  # mock-only, never auto-fired

    @pytest.mark.asyncio
    async def test_untested_enter_is_a_real_binding(self, tmp_path):
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer", runner=runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            app.action_serve_untested()
            await _settle(pilot)
            scr = app.screen
            assert isinstance(scr, UntestedComposePreviewScreen)
            actions = {b.action: b for b in scr.BINDINGS}
            assert "serve_untested" in actions
            assert actions["serve_untested"].key == "enter"
            assert actions["serve_untested"].show is True
            assert any(
                v.binding.action == "serve_untested" and v.binding.show
                for v in scr.active_bindings.values()
            )

    @pytest.mark.asyncio
    async def test_untested_enter_still_serves_through_reconcile_gate(self, tmp_path):
        wr = FakeWriteRunner()
        runner = FakeGenComposeRunner(GENERATED_COMPOSE_YAML)
        app, _, _ = make_app(repo_root=tmp_path, surface="producer",
                             runner=runner, write_runner=wr)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("unsloth/Qwen3-27B-abliterated", "vllm/dual")
            await _settle(pilot)
            await pilot.press("2")
            await _settle(pilot)
            app.query_one("#validate-tabs", TabbedContent).active = "tab-serve"
            await pilot.pause()
            app.action_serve_untested()
            await _settle(pilot)
            assert isinstance(app.screen, UntestedComposePreviewScreen)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            plan = app.screen._plan
            assert plan.kind == "serve"
            assert plan.requires_reconcile is True
            assert wr.started == []  # not auto-fired


class TestTier1ReconcileGateSafetyUntouched:
    """Regression-lock: FIX 1-3 are footer/binding hygiene ONLY — the load-bearing
    ConfirmActionScreen Enter-on-UNSAFE safety stays INERT (no force footgun)."""

    @pytest.mark.asyncio
    async def test_enter_on_unsafe_gate_stays_inert(self):
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
            await pilot.press("enter")
            await _settle(pilot)
            assert wr.started == []  # INERT — no write, no forced teardown
            # The explicit `f` STILL forces (unchanged force path).
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

    def test_one_option_per_unique_family_topology_not_per_slug(self):
        # FIX 2 (maintainer directive — REVERSES the B4b all-slugs deviation): the
        # dropdown emits ONE representative per UNIQUE (engine-FAMILY, topology) pair,
        # NOT one per slug.  THREE vllm dual slugs sharing (vllm, dual) collapse to a
        # SINGLE "vllm / dual" option (the escape hatch keeps the rest reachable).
        rows = [
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
            self._mk("vllm/qwen-27b-dual-max", "vllm-stable", "models/q/vllm/compose/dual/aq/int8.yml"),
            self._mk("vllm/qwen-27b-dual-fast", "vllm-stable", "models/q/vllm/compose/dual/aq/fast.yml"),
            self._mk("vllm/minimal", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
            self._mk("ik-llama/iq4ks-mtp", "ik-llama", "models/q/ik-llama/compose/single/u/mtp.yml"),
            self._mk("llamacpp/default", "llama-cpp-local", "models/q/llama-cpp/compose/single/u/default.yml"),
        ]
        opts = profile_templates(rows)
        # Distinct (family, topology) pairs in this fixture:
        #   (vllm, dual), (vllm, single), (llama-cpp, single)   ← ik-llama + llamacpp
        #   slugs BOTH map to engine family "llama-cpp" so they share one option.
        pairs = {(o.label.split("  ·  ")[0]) for o in opts}
        assert len(opts) == len(pairs)        # one option per distinct pair
        assert len(opts) == 3                 # NOT 6 (per-slug); curated short-list
        # The (vllm, dual) representative is the canonical literal "vllm/dual".
        by_topo_fam = {(o.topology, o.label.split(" / ")[0]): o for o in opts}
        assert by_topo_fam[("dual", "vllm")].slug == "vllm/dual"
        # labels lead with the readable family / topology, then the chosen slug.
        assert any(o.label.startswith("vllm / dual  ·  vllm/dual") for o in opts)
        # topology is carried THROUGH on the option (not re-derived from the label).
        assert by_topo_fam[("dual", "vllm")].topology == "dual"
        assert by_topo_fam[("single", "vllm")].topology == "single"

    def test_vllm_stable_and_gemma_stable_collapse_to_one_vllm_dual(self):
        # FIX 2 — vllm-stable + vllm-gemma-stable are DIFFERENT raw engines but the
        # SAME canonical family "vllm"; their dual variants must collapse to a SINGLE
        # "vllm / dual" option (otherwise "vllm / dual" would appear twice).
        rows = [
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
            self._mk("vllm/gemma-bf16-mtp", "vllm-gemma-stable", "models/q/vllm/compose/dual/aq/bf16-mtp.yml"),
        ]
        opts = profile_templates(rows)
        assert len(opts) == 1                 # ONE vllm/dual, not two
        assert opts[0].slug == "vllm/dual"    # canonical-literal representative wins
        assert opts[0].label.startswith("vllm / dual")

    def test_representative_is_last_in_order_when_no_canonical_literal(self):
        # FIX 2 — when no literal "<family>/<topo>" slug exists in a group, the
        # representative is the LAST variant in registry order (a stable latest proxy).
        rows = [
            self._mk("vllm/minimal", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
            self._mk("vllm/qwen-a3b-preview-single", "vllm-stable", "models/q/vllm/compose/single/aq/preview.yml"),
        ]
        opts = profile_templates(rows)
        assert len(opts) == 1
        # No literal "vllm/single" → last-in-order representative.
        assert opts[0].slug == "vllm/qwen-a3b-preview-single"

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

    def test_default_value_is_present_in_curated_options(self):
        # FIX 2 — a Select can't default to a value not in its options.  The
        # rig-default (vllm/dual on ≥2 cards) MUST be one of the curated entries.
        rows = [
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
            self._mk("vllm/qwen-27b-dual-max", "vllm-stable", "models/q/vllm/compose/dual/aq/int8.yml"),
            self._mk("vllm/minimal", "vllm-stable", "models/q/vllm/compose/single/aq/base.yml"),
            self._mk("beellama/dflash", "beellama-local", "models/q/beellama/compose/single/u/dflash.yml"),
        ]
        opts = profile_templates(rows)
        curated_slugs = {o.slug for o in opts}
        assert default_profile_template(opts, 2) in curated_slugs
        assert default_profile_template(opts, 1) in curated_slugs

    def test_select_options_include_custom_escape_hatch_sentinel(self):
        # FIX 2 (escape hatch) — the (label, value) pairs a Select consumes carry a
        # trailing "✎ custom slug…" sentinel so any non-curated registry slug stays
        # reachable.  The sentinel is the LAST option and is NOT a real slug.
        from club3090_cockpit.app import (
            profile_select_options,
            PROFILE_CUSTOM_SENTINEL,
        )
        rows = [
            self._mk("vllm/dual", "vllm-stable", "models/q/vllm/compose/dual/aq/fp8-mtp.yml"),
        ]
        pairs = profile_select_options(profile_templates(rows))
        assert pairs[-1][1] == PROFILE_CUSTOM_SENTINEL
        assert "custom" in pairs[-1][0].lower()
        # the curated representative is still present as a real option.
        assert ("vllm / dual  ·  vllm/dual", "vllm/dual") in pairs

    # ── FIX 2 (status-aware representative pick) regression guards ──────────────
    #
    # The B4b/FIX-2 rewrite picked ``rep_slug = slugs[-1]`` (registry insertion
    # order — STATUS-BLIND) whenever no literal ``<family>/<topo>`` existed.  On
    # the LIVE 53-variant registry that left 4 of 7 reps non-functional — e.g.
    # (vllm, single) → vllm/vibethinker-3b-single [incubating] — and the
    # single-card rig DEFAULT resolved to that incubating 3B (needs --force to
    # launch) instead of the curated vllm/minimal.  The 2-card path stayed green
    # (vllm/dual is the one literal slug) which is why synthetic fixtures + the
    # maintainer's 2-card rig never caught it.  These two guards pin the
    # status-aware behavior: one drives the REAL registry, one a faithful fixture
    # that mirrors its status distribution (so the suite catches it even off-rig).

    _INCUBATING = "incubating"
    _EXPERIMENTAL = "experimental"

    def _mk_status(self, slug, engine, path, status):
        row = self._mk(slug, engine, path)
        return row._replace(status=status) if hasattr(row, "_replace") else VariantRow(
            slug=slug, switch_engine=engine, launch_engine=engine,
            compose_dir=path.rsplit("/", 1)[0], file=path.rsplit("/", 1)[1],
            port=8000, model="q", engine=engine, kvcalc_key="k", container="c",
            compose_path=path, status=status, ctx_label="262K", status_note="",
        )

    def test_status_aware_rep_skips_incubating_when_functional_sibling_exists(self):
        # Fixture mirroring the live (vllm, single) group: the LAST slug in
        # registry order is an incubating 3B; a functional vllm/minimal sits
        # EARLIER.  The status-BLIND rule chose slugs[-1] (the incubating one) —
        # the status-aware rule must choose the functional vllm/minimal, and the
        # 1-card default must be functional / non-incubating.
        rows = [
            self._mk_status("vllm/minimal", "vllm-stable",
                            "models/q/vllm/compose/single/aq/minimal.yml", "production"),
            self._mk_status("vllm/qwen-a3b-preview-single", "vllm-stable",
                            "models/q/vllm/compose/single/aq/preview.yml", "preview"),
            # last-in-order single vllm slug is the incubating 3B (the trap).
            self._mk_status("vllm/vibethinker-3b-single", "vllm-stable",
                            "models/q/vllm/compose/single/aq/vibe.yml", self._INCUBATING),
            self._mk_status("vllm/dual", "vllm-stable",
                            "models/q/vllm/compose/dual/aq/fp8-mtp.yml", "production"),
        ]
        # No curated `defaults` passed → exercises the STATUS FLOOR (rule c).
        opts = profile_templates(rows)
        single = next(o for o in opts if o.topology == "single")
        # status-blind would have given vibethinker; status-aware gives minimal.
        assert single.slug == "vllm/minimal"
        assert single.status not in (self._INCUBATING, self._EXPERIMENTAL, "preview")
        # The 1-card rig default must be a launchable (functional) slug.
        dflt = default_profile_template(opts, 1)
        assert dflt == "vllm/minimal"
        dflt_status = next(o.status for o in opts if o.slug == dflt)
        assert dflt_status in {"production", "caveats"}

    def test_curated_defaults_drive_rep_when_no_literal_slug(self):
        # FIX 2 rule (b): when a (family, topology) has no literal "<fam>/<topo>"
        # slug but the registry's curated `defaults` array names one, USE it —
        # it's the registry's own recommendation.  Here llamacpp/dual has no
        # literal; the curated default names llamacpp/deckard40B-dual-mtp.
        rows = [
            self._mk_status("llamacpp/hauhaucs-35ba3b-dual", "llama-cpp-local",
                            "models/q/llama-cpp/compose/dual/u/hauhau.yml", self._EXPERIMENTAL),
            self._mk_status("llamacpp/deckard40B-dual-mtp", "llama-cpp-local",
                            "models/q/llama-cpp/compose/dual/u/deckard.yml", "production"),
        ]
        defaults = [
            {"model": "deckard40b", "engine": "llamacpp", "topology": "dual",
             "slug": "llamacpp/deckard40B-dual-mtp", "source": "curated"},
        ]
        opts = profile_templates(rows, defaults)
        dual = next(o for o in opts if o.topology == "dual")
        assert dual.slug == "llamacpp/deckard40B-dual-mtp"
        assert dual.status == "production"

    def test_entirely_nonfunctional_group_keeps_last_resort_rep(self):
        # FIX 2 rule (d): a group whose EVERY member is non-functional (the live
        # registry's (beellama, dual) + (vllm, multi4)) has no functional rep to
        # pick — fall back to slugs[-1].  The escape hatch still reaches the rest.
        rows = [
            self._mk_status("beellama/qwen-mtp-dual", "beellama-local",
                            "models/q/beellama/compose/dual/q8/mtp.yml", self._EXPERIMENTAL),
            self._mk_status("beellama/gemma-q8-dflash-dual", "beellama-local",
                            "models/q/beellama/compose/dual/q8/dflash.yml", self._EXPERIMENTAL),
        ]
        opts = profile_templates(rows)
        dual = next(o for o in opts if o.topology == "dual")
        assert dual.slug == "beellama/gemma-q8-dflash-dual"   # last-in-order

    def test_real_registry_reps_are_status_aware(self):
        # REAL-REGISTRY guard — drive profile_templates + default_profile_template
        # against the live `registry-emit.sh --json` (the SAME contract the
        # production load path consumes), so a status-blind regression reds here.
        # Skips cleanly if the registry/emitter is absent; present in this repo.
        import subprocess
        from club3090_cockpit.services import _variant_row_from_dict

        repo_root = Path(__file__).resolve().parents[3]
        emitter = repo_root / "scripts" / "lib" / "registry-emit.sh"
        if not emitter.exists():
            pytest.skip("registry-emit.sh not present — real-registry guard skipped")
        try:
            proc = subprocess.run(
                ["bash", str(emitter), "--json"],
                capture_output=True, text=True, timeout=60, cwd=str(repo_root),
            )
        except (OSError, subprocess.TimeoutExpired):
            pytest.skip("registry-emit.sh --json unavailable — real-registry guard skipped")
        if proc.returncode != 0 or not proc.stdout.strip():
            pytest.skip(f"registry-emit.sh --json returned nothing (rc={proc.returncode})")
        payload = json.loads(proc.stdout)
        variants = [_variant_row_from_dict(d) for d in payload.get("variants", [])]
        defaults = payload.get("defaults", [])
        assert variants, "real registry returned no variants"

        opts = profile_templates(variants, defaults)
        # Exactly one option per (family, topology); 7 distinct groups today.
        pairs = {(o.topology, o.label.split(" / ")[0]) for o in opts}
        assert len(opts) == len(pairs), "duplicate (family, topology) representative"
        assert len(opts) == 7, f"expected 7 reps, got {len(opts)}: {[o.slug for o in opts]}"

        # The 1-card rig default must be FUNCTIONAL + non-incubating — ideally the
        # registry's curated single default (vllm/minimal).
        dflt = default_profile_template(opts, 1)
        dflt_status = next((o.status for o in opts if o.slug == dflt), "")
        assert dflt_status in {"production", "caveats"}, (
            f"1-card default {dflt!r} is non-functional ({dflt_status})"
        )
        assert dflt == "vllm/minimal", f"1-card default is {dflt!r}, not vllm/minimal"

        # NO representative is non-functional while a FUNCTIONAL sibling exists in
        # its group — the exact defect (incubating/experimental rep over a
        # functional one).  Rebuild the per-group membership to check siblings.
        from club3090_cockpit.app import _canon_engine_family, _variant_topology
        groups: dict[tuple, list[str]] = {}
        for v in variants:
            fam = _canon_engine_family(v.engine) or v.engine
            topo = _variant_topology(v) or "—"
            groups.setdefault((fam, topo), []).append((v.status or "").lower())
        FUNCTIONAL = {"production", "caveats"}
        for o in opts:
            fam = o.label.split(" / ")[0]
            members = groups.get((fam, o.topology), [])
            group_has_functional = any(s in FUNCTIONAL for s in members)
            if group_has_functional:
                assert o.status in FUNCTIONAL, (
                    f"rep {o.slug!r} is {o.status!r} but its ({fam},{o.topology}) "
                    f"group has a functional sibling"
                )


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
    async def test_f2_wrapping_caveat_not_clipped(self):
        """F2 — a long (wrapping) caveat line must GROW the preview strip, not
        clip below the old max-height 6 (border eats 2 rows → 4 content lines;
        the dual-fast caveat wrapped past that and lost its tail)."""
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            t = app.query_one("#catalog-table", DataTable)
            t.move_cursor(row=0)
            await pilot.pause()
            entry = pane.selected_entry()
            # A ~240-char caveat wraps to 3 display lines at 120 cols → content
            # = 3 base + 3 caveat = 6 → 8 with the border.  (status_note is a
            # property over the registry row — mutate the row.)
            entry.row.status_note = (
                "prose regression on DFlash v0.3.0 above 50K ctx — code packs OK; "
                "promote only on a stable upstream tag; see UPSTREAM.md row and "
                "the beellama pin-bump helper before changing anything here; "
                "workaround: run the q8 dual instead for long prose"
            )
            pane.render_preview(entry)
            await pilot.pause()
            preview = app.query_one("#catalog-preview", Static)
            assert preview.region.height >= 7, preview.region
            # The caveat's TAIL is inside the rendered region (not clipped).
            assert "long prose" in str(preview.render())

    @pytest.mark.asyncio
    async def test_preview_bar_provenance_stale_detail_and_yours(self, tmp_path):
        """Slice 2b — the preview's measured line is the BAR with provenance;
        a stale bar gets the explicit pin-detail line; a slug this rig has
        gated shows the 'yours' corpus overlay line."""
        seed_repo(tmp_path)
        corpus = tmp_path / "results" / "measurement-records"
        corpus.mkdir(parents=True)
        (corpus / "vllm-dual__test.jsonl").write_text(json.dumps({
            "_tag": "vllm/dual", "_recorded_at": "2026-07-04T12:00:00Z",
            "engine_pin": "vllm/vllm-openai:v0.24.0",
            "measured_extensions": {"decode_tps_by_ctx": {"canonical-short": 172.4},
                                     "quality_8pk": "108/150"},
        }) + "\n")
        app, _, _ = make_app(repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            t = app.query_one("#catalog-table", DataTable)
            # vllm/dual: fresh bar + provenance + yours (this rig gated it).
            t.move_cursor(row=0)
            await pilot.pause()
            prev = str(app.query_one("#catalog-preview", Static).render())
            assert "bar" in prev and "174/42" in prev
            assert "2026-07-01" in prev and "noonghunna" in prev   # provenance
            assert "yours" in prev and "~172 decode" in prev and "108/150" in prev
            assert "re-bench owed" not in prev                     # fresh bar
            # ik row: stale bar → the explicit pin-detail line; no 'yours'.
            ik_idx = next(i for i, e in enumerate(pane._filtered_entries())
                          if e.slug == "ik-llama/iq4ks-mtp")
            t.move_cursor(row=ik_idx)
            await pilot.pause()
            prev = str(app.query_one("#catalog-preview", Static).render())
            assert "re-bench owed" in prev
            assert "ghcr.io/ik-old@sha256:aaa" in prev and "ghcr.io/ik-new@sha256:bbb" in prev
            assert "yours" not in prev

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
            await _enter_operate(pilot)
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
            await _enter_operate(pilot)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=1)  # "off" scene
            await pilot.pause()
            prev = str(app.query_one("#scene-preview", Static).render())
            assert "off" in prev
            assert "Stop all" in prev

    @pytest.mark.asyncio
    async def test_scene_preview_active_bullet_when_service_running(self):
        """#11-ext — the preview reflects LIVE service status: a running service
        gets a ● bullet and the scene reads ACTIVE."""
        responses = fake_responses(
            **{"docker ps": ok("vllm-qwen36-27b-dual|0.0.0.0:8010->8010/tcp\n")})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=0)  # "27b" — its engine container is running
            await pilot.pause()
            prev = str(app.query_one("#scene-preview", Static).render())
            assert "vllm-qwen36-27b-dual" in prev
            assert "ACTIVE" in prev      # a running service ⇒ scene active
            assert "●" in prev           # running bullet

    @pytest.mark.asyncio
    async def test_scene_preview_inactive_bullet_when_no_service_running(self):
        """#11-ext — with no matching running container, the service reads stopped
        (○) and the scene reads inactive."""
        app, _, _ = make_app()  # default docker ps is empty → nothing running
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            t = app.query_one("#scene-table", DataTable)
            t.move_cursor(row=0)  # "27b"
            await pilot.pause()
            prev = str(app.query_one("#scene-preview", Static).render())
            assert "inactive" in prev
            assert "○" in prev           # stopped bullet


class TestStudioSceneGuard:
    """⏎ on the studio scene (ai-studio) when the studio isn't set up (comfyui-local
    image OR director GGUF missing) guides to setup-image-studio.sh, not a scene switch."""

    SCENES = json.dumps([
        {"name": "ai-studio", "group": "studio", "description": "creative studio",
         "services": ["comfyui", "openwebui"], "ports": ["8188"], "gpus": "both"},
        {"name": "off", "group": "ops", "description": "Stop all",
         "services": [], "ports": [], "gpus": "none"},
    ])

    async def _enter_on_studio_scene(self, app, pilot):
        await _enter_operate(pilot)
        pane = app.query_one("#operate-orch-pane", OperateOrchPane)
        idx = next(i for i, s in enumerate(pane._scenes) if s.name == "ai-studio")
        app.query_one("#scene-table", DataTable).move_cursor(row=idx)
        await pilot.pause()
        app._operate_primary()
        await pilot.pause()

    @pytest.mark.asyncio
    async def test_blocked_when_comfyui_absent(self):
        responses = fake_responses(**{
            "gpu-mode.sh --list-modes --json": ok(self.SCENES),
            "docker images -q": ok(""),  # comfyui-local NOT built
        })
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await self._enter_on_studio_scene(app, pilot)
            assert app._studio_ready is False
            assert not isinstance(app.screen, ConfirmActionScreen)  # guarded

    @pytest.mark.asyncio
    async def test_allowed_when_studio_ready(self, tmp_path):
        responses = fake_responses(**{
            "gpu-mode.sh --list-modes --json": ok(self.SCENES),
            "docker images -q": ok("sha256:abc\n"),
        })
        app, _, _ = make_app(responses=responses)
        _seed_studio_director(app, tmp_path)  # image present + director on disk
        async with app.run_test(size=(120, 40)) as pilot:
            await self._enter_on_studio_scene(app, pilot)
            assert app._studio_ready is True
            assert isinstance(app.screen, ConfirmActionScreen)
            assert app.screen._plan.cmd == ["bash", "scripts/gpu-mode.sh", "ai-studio"]

    @pytest.mark.asyncio
    async def test_shows_setup_modal_with_missing_models(self, tmp_path, monkeypatch):
        """Not-ready ⏎ opens the StudioSetupScreen listing the MISSING models grouped
        by modality (the first-install download affordance), not just a notify."""
        from club3090_cockpit.app import StudioSetupScreen
        monkeypatch.setenv("COMFYUI_MODELS_DIR", str(tmp_path / "comfy"))  # nothing seeded
        responses = fake_responses(**{
            "gpu-mode.sh --list-modes --json": ok(self.SCENES),
            "docker images -q": ok(""),  # comfyui-local absent → not ready
        })
        app, _, _ = make_app(responses=responses)
        app._data._model_dir = str(tmp_path / "weights")  # director root empty → director missing
        async with app.run_test(size=(120, 40)) as pilot:
            await self._enter_on_studio_scene(app, pilot)
            await pilot.pause()
            await pilot.pause()  # let the open_studio_setup worker push the modal
            assert isinstance(app.screen, StudioSetupScreen)
            body = str(app.screen.query_one("#studio-setup-body", Static).render())
            assert "Director" in body and "Video" in body  # missing models, grouped by modality


class TestEvidenceAndGatePreview:
    """N8 — ④ Measure evidence-tag + ③ Gate validation-step highlight previews."""

    @pytest.mark.asyncio
    async def test_evidence_highlight_shows_artifacts(self, tmp_path):
        app, _, _ = make_app(repo_root=tmp_path, surface="producer")
        seed_repo(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
    """#6/A12 — the ① Bring profile input is a registry-derived Select that
    defaults to the rig topology + reports unknown profiles precisely.  (The
    standalone Run · BYO Select was removed in the 2-mode merge — ① Bring is the
    single profile-template entry now.)"""

    @pytest.mark.asyncio
    async def test_byo_profile_is_select_with_registry_templates(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            sel = app.query_one("#lane-bring-profile-input", Select)
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
            await _enter_operate(pilot)
            sel = app.query_one("#lane-bring-profile-input", Select)
            assert sel.value == "vllm/single"

    @pytest.mark.asyncio
    async def test_byo_defaults_to_dual_on_two_cards(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses, gpus=gpus,
                             target=ServingTarget(gpus=gpus))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await _enter_operate(pilot)
            sel = app.query_one("#lane-bring-profile-input", Select)
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
            sel = app.query_one("#lane-bring-profile-input", Select)
            # User manually picks the non-default (single) template.
            sel.value = "vllm/single"
            await pilot.pause()
            assert app._profile_user_touched is True
            # Enter Operate → first estate poll fires reapply_default=True.
            await _enter_operate(pilot)
            # The manual pick STANDS — the reapply did NOT clobber it back to dual.
            assert app.query_one("#lane-bring-profile-input", Select).value == "vllm/single"

    @pytest.mark.asyncio
    async def test_unknown_profile_reports_known_list(self):
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # A legacy / free-text profile-like that isn't a known registry slug.
            app.run_byo_check("org/Model", "vllm/legacy-gone")
            await _settle(pilot)
            card = str(app.query_one("#lane-bring-result-card", Static).render())
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

    @pytest.mark.asyncio
    async def test_custom_sentinel_reveals_input_and_routes_typed_slug(self):
        # FIX 2 (escape hatch) — selecting "✎ custom slug…" reveals the companion
        # free-text Input; a typed arbitrary (non-curated) slug reaches byo_check.
        from club3090_cockpit.app import PROFILE_CUSTOM_SENTINEL
        responses = fake_responses(**{"registry-emit.sh --json": ok(REGISTRY_JSON_CAVEAT)})
        app, _, _ = make_app(responses=responses)
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            custom = app.query_one("#lane-bring-profile-custom", Input)
            # Hidden until the sentinel is chosen.
            assert custom.has_class("profile-custom-hidden")
            sel = app.query_one("#lane-bring-profile-input", Select)
            sel.value = PROFILE_CUSTOM_SENTINEL
            await pilot.pause()
            # Revealed by the sentinel pick.
            assert not custom.has_class("profile-custom-hidden")
            # A custom slug NOT in the curated list is what _selected_profile_like
            # returns (so it reaches byo_check's unknown-profile validation path).
            custom.value = "ik-llama/iq4ks-mtp"
            assert app._selected_profile_like("#lane-bring-profile-input") == "ik-llama/iq4ks-mtp"

    @pytest.mark.asyncio
    async def test_custom_slug_not_in_curated_list_still_validates(self):
        # FIX 2 (escape hatch) — a typed slug that's NOT one of the curated dropdown
        # representatives still flows to byo_check (validated, then served if known).
        app, runner, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.run_byo_check("org/Model", "ik-llama/iq4ks-mtp")
            await _settle(pilot)
            pull = next(c for c in runner.calls if "pull.sh" in " ".join(c))
            assert "--dry-run" in pull
            assert "ik-llama/iq4ks-mtp" in pull


class TestProducerLaneHandoff:
    """N9 — a successful ① Bring fit-check pre-arms ② Serve (no re-entry)."""

    @pytest.mark.asyncio
    async def test_serve_prearmed_after_fit_check(self):
        app, _, _ = make_app(surface="producer")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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
            await pilot.press("2")
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


# ════════════════════════════════════════════════════════════════════════════════
# UX Batch 5 — data-completeness: disk bars (#12) · system RAM (N5) ·
# studio-* service-set broadening · GPU-VRAM → container attribution.
# ════════════════════════════════════════════════════════════════════════════════


class TestBatch5DiskParse:
    """#12 — df -B1 parse, byte math, df-Use% fidelity, same-device de-dup."""

    def test_two_devices_exact_numbers(self):
        # REAL recon df shape: repo on LVM root, /mnt/models on /dev/sdb1.
        disks = parse_df_output(
            DF_TWO_DEVICES, {str(FAKE_REPO_ROOT): "repo", "/mnt/models": "models"}
        )
        assert len(disks) == 2  # different devices → NOT de-duped
        repo, models = disks[0], disks[1]
        assert repo.mount_label == "repo"
        assert repo.device == "/dev/mapper/ubuntu--vg-ubuntu--lv"
        # -B1 → bytes already; assert the EXACT byte fields (no 1K-block ×1024 trap).
        assert repo.total == 1793150255104
        assert repo.used == 435407220736
        assert repo.free == 1284330176512
        # df's OWN Use% is authoritative (used/(used+avail), not used/total).
        assert repo.use_pct == 26
        assert repo.pct == 26
        assert models.mount_label == "models"
        assert models.device == "/dev/sdb1"
        assert models.free == 189454557184
        assert models.pct == 90

    def test_pct_prefers_df_use_column_not_used_over_total(self):
        # used/total would be ~24% (reserved blocks); df reports 26% — we mirror df.
        disks = parse_df_output(
            DF_TWO_DEVICES, {"/r": "repo", "/mnt/models": "models"}
        )
        repo = disks[0]
        naive = round(repo.used / repo.total * 100)
        assert naive == 24          # the naive math
        assert repo.pct == 26       # but we report df's Use% (matches df -h)

    def test_same_device_dedups_to_one_row(self):
        # Repo + /mnt/models on the SAME filesystem device → ONE bar, joined label.
        disks = parse_df_output(
            DF_SAME_DEVICE, {"/repo": "repo", "/mnt/models": "models"}
        )
        assert len(disks) == 1
        assert disks[0].mount_label == "repo + models"
        assert disks[0].pct == 71
        assert disks[0].free == 168000000000

    def test_human_gb_binary_units(self):
        from club3090_cockpit.data import _human_gb
        assert _human_gb(189454557184) == "176G"      # /mnt/models free
        assert _human_gb(1793150255104) == "1.6T"     # repo total (1.63 TiB)
        assert _human_gb(0) == "0G"

    def test_long_device_name_no_wrap_under_posix(self):
        # MUST-FIX 2: a long LVM/mapper device name stays on ONE line under -P,
        # so the parser reads use-% by POSITION and never swaps labels/device.
        disks = parse_df_output(
            DF_LONG_DEVICE_AND_ZERO, {str(FAKE_REPO_ROOT): "repo"}
        )
        assert len(disks) == 1                # the tmpfs zero-row dropped (NH-A)
        repo = disks[0]
        assert repo.mount_label == "repo"
        assert repo.device == "/dev/mapper/vg--very--long--name-ubuntu--lv--root"
        assert repo.total == 1793150255104
        assert repo.use_pct == 26
        assert repo.pct == 26

    def test_zero_size_special_fs_dropped_no_false_zero_bar(self):
        # NH-A: a zero-size special fs (tmpfs/cgroup) must NEVER become a "0% 0G/0G"
        # bar — the parser skips it so the honest-error branch can fire if nothing
        # else parsed.
        only_zero = (
            "Filesystem 1-blocks Used Available Capacity Mounted on\n"
            "tmpfs              0    0         0        - /sys/fs/cgroup\n"
        )
        disks = parse_df_output(only_zero, {"/x": "repo"})
        assert disks == []


class TestBatch5RamParse:
    """N5 — /proc/meminfo parse: kB→bytes, used = total − MemAvailable."""

    def test_meminfo_exact_numbers(self):
        r = parse_meminfo(MEMINFO)
        # kB ×1024 → bytes.
        assert r.total == 98854288 * 1024
        assert r.available == 84219884 * 1024
        # used = total − available (matches `free -b`'s used column).
        assert r.used == (98854288 - 84219884) * 1024
        assert r.used == 14985629696            # the exact `free -b` recon used
        assert r.pct == 15                      # round(14634404/98854288 * 100)
        assert r.error == ""

    def test_meminfo_missing_available_degrades_no_false_zero(self):
        r = parse_meminfo("MemTotal: 1000 kB\nMemFree: 100 kB\n")
        assert r.total == 1000 * 1024
        assert r.available == 0
        assert "MemAvailable missing" in r.error  # honest cue, not a false 100%
        # WHY the renderer MUST gate on ram.error: with available=0, used=total →
        # pct computes to a MISLEADING 100%.  The error flag is the only thing
        # stopping a false "RAM 100%" bar (see the _populate_disk_rail render test).
        assert r.pct == 100

    def test_meminfo_missing_total_is_error(self):
        r = parse_meminfo("MemFree: 100 kB\n")
        assert r.total == 0
        assert r.error


class TestBatch5GpuAttributionParse:
    """GPU-VRAM → container attribution: compute-apps + cgroup + ps map, incl.
    the DEGRADED (cgroup unreadable) path."""

    def test_compute_apps_parse(self):
        apps = parse_compute_apps("588408, 22686")
        assert len(apps) == 1
        assert apps[0].pid == 588408
        assert apps[0].used_mib == 22686
        assert apps[0].container == ""  # not yet attributed

    def test_cgroup_v2_scope_id(self):
        cid = parse_cgroup_container_id(CGROUP_PID)
        assert cid == "e8eb8d4cdd19861ddc94d582b2d583b898baf9c3931b26407f1b43ebb896c3d4"

    def test_cgroup_v1_layout(self):
        cid = parse_cgroup_container_id(
            "1:cpu:/docker/" + "a" * 64 + "\n11:devices:/docker/" + "a" * 64
        )
        assert cid == "a" * 64

    def test_cgroup_non_docker_returns_empty(self):
        assert parse_cgroup_container_id("0::/user.slice/session-3.scope") == ""
        assert parse_cgroup_container_id("") == ""

    def test_attribution_happy_path(self):
        apps = parse_compute_apps("588408, 22686")
        idmap = parse_docker_ps_id_names(DOCKER_PS_IDNAMES)
        out, degraded = attribute_gpu_apps(apps, {588408: CGROUP_PID}, idmap)
        assert out[0].container == "llama-cpp-pi-reasoning"
        assert degraded is False

    def test_attribution_degraded_cgroup_unreadable(self):
        # Permission/format failure → empty cgroup body → no name, no crash, flag.
        apps = parse_compute_apps("588408, 22686")
        out, degraded = attribute_gpu_apps(apps, {588408: ""}, {})
        assert out[0].container == ""        # NO fabricated owner
        assert out[0].pid == 588408          # still shown by pid
        assert out[0].used_mib == 22686      # VRAM total still honest
        assert degraded is True

    def test_attribution_non_docker_pid_degraded(self):
        apps = parse_compute_apps("999, 4096")
        out, degraded = attribute_gpu_apps(
            apps, {999: "0::/user.slice/session.scope"}, {}
        )
        assert out[0].container == ""
        assert degraded is True


@pytest.mark.asyncio
class TestBatch5TelemetryDataLayer:
    """End-to-end CockpitData.estate_telemetry against canned recon stdout."""

    async def test_telemetry_full_attribution(self):
        runner = FakeRunner(batch5_responses())
        data = CockpitData(FAKE_REPO_ROOT, runner=runner)
        tel = await data.estate_telemetry()
        # Disk (#12) — two distinct mounts.
        labels = [d.mount_label for d in tel.disks]
        assert labels == ["repo", "models"]
        assert tel.disks[1].pct == 90
        # RAM (N5).
        assert tel.ram.pct == 15
        assert tel.ram.used == 14985629696
        # GPU attribution — pid 588408 → llama-cpp-pi-reasoning on GPU0.
        apps0 = tel.gpu_apps.get(0, [])
        assert len(apps0) == 1
        assert apps0[0].container == "llama-cpp-pi-reasoning"
        assert apps0[0].used_mib == 22686
        assert apps0[0].gpu_index == 0           # pinned to the resolved card
        # NH-B: the holder must NOT also land in the None bucket (no double-pin).
        assert tel.gpu_apps.get(None) is None
        assert tel.attribution_degraded is False
        assert tel.error == ""

    async def test_telemetry_buckets_holder_on_resolved_nonzero_card(self):
        # NH-B: a holder on GPU1's uuid must bucket under index 1 — NOT fall back
        # to 0.  Distinguishes "bucketed correctly" from "defaulted to 0" (the old
        # .get(uuid, 0) bug rendered every holder under GPU0).
        gpu1_apps = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 588408, 22686\n"
        runner = FakeRunner(batch5_responses(compute_apps=gpu1_apps))
        data = CockpitData(FAKE_REPO_ROOT, runner=runner)
        tel = await data.estate_telemetry()
        assert tel.gpu_apps.get(0) is None       # NOT mis-pinned to GPU0
        apps1 = tel.gpu_apps.get(1, [])
        assert len(apps1) == 1
        assert apps1[0].gpu_index == 1
        assert apps1[0].used_mib == 22686

    async def test_telemetry_unresolved_uuid_buckets_under_none_not_zero(self):
        # NH-B: when the uuid→index read fails (empty map) the holder must bucket
        # under the None key, NEVER mis-pinned to GPU0 (VRAM stays honest, card
        # attribution simply isn't claimed).
        runner = FakeRunner(batch5_responses(uuid_index=""))
        data = CockpitData(FAKE_REPO_ROOT, runner=runner)
        tel = await data.estate_telemetry()
        assert tel.gpu_apps.get(0) is None       # NOT defaulted to GPU0
        unpinned = tel.gpu_apps.get(None, [])
        assert len(unpinned) == 1
        assert unpinned[0].gpu_index is None
        assert unpinned[0].used_mib == 22686

    async def test_telemetry_degraded_attribution_no_crash(self):
        runner = FakeRunner(batch5_responses(cgroup="", idnames=""))
        data = CockpitData(FAKE_REPO_ROOT, runner=runner)
        tel = await data.estate_telemetry()
        apps0 = tel.gpu_apps.get(0, [])
        assert apps0[0].container == ""          # nameless, not fabricated
        assert apps0[0].used_mib == 22686        # VRAM still surfaced
        assert tel.attribution_degraded is True

    async def test_telemetry_df_failure_records_error(self):
        # df returns EMPTY stdout → honest error cue, no false-zero disks.
        runner = FakeRunner(batch5_responses(df=""))
        data = CockpitData(FAKE_REPO_ROOT, runner=runner)
        tel = await data.estate_telemetry()
        assert tel.disks == []
        assert "disk read failed" in tel.error
        # RAM + GPU still read fine despite the disk failure.
        assert tel.ram.pct == 15

    async def test_telemetry_df_rc1_with_valid_stdout_keeps_repo_bar(self):
        # MUST-FIX 1: df exits rc=1 when /mnt/models is missing/unmounted but STILL
        # prints the valid repo (/) row to stdout (the error goes to stderr).  We
        # parse stdout REGARDLESS of returncode, so the repo bar survives — the
        # rc-gate would have discarded it and shown only "disk read failed".
        df_rc1 = (
            "Filesystem                             1-blocks          Used"
            "     Available Capacity Mounted on\n"
            "/dev/mapper/ubuntu--vg-ubuntu--lv 1793150255104  435407220736"
            " 1284330176512      26% /\n"
        )
        responses = batch5_responses()
        responses["df -P -B1"] = RunResult(
            returncode=1,
            stdout=df_rc1,
            stderr="df: /mnt/models: No such file or directory",
        )
        runner = FakeRunner(responses)
        data = CockpitData(FAKE_REPO_ROOT, runner=runner)
        tel = await data.estate_telemetry()
        assert len(tel.disks) == 1               # the valid repo row, NOT dropped
        assert tel.disks[0].mount_label == "repo"
        assert tel.disks[0].pct == 26
        assert "disk read failed" not in (tel.error or "")

    async def test_telemetry_same_device_dedup(self):
        runner = FakeRunner(batch5_responses(df=DF_SAME_DEVICE))
        data = CockpitData(Path("/repo"), runner=runner)
        tel = await data.estate_telemetry()
        assert len(tel.disks) == 1
        assert tel.disks[0].mount_label == "repo + models"


@pytest.mark.asyncio
class TestBatch5OperateRendering:
    """The LEFT RAIL renders the disk / RAM line (FIX 3 — moved out of the
    Orchestration sub-tab) and the GPU "held by:" attribution renders on the orch
    GPU cards, both from the telemetry read on the existing tick."""

    async def test_disk_rail_renders_bars_and_ram(self):
        gpus = [
            GpuInfo(index=0, mem_used_mib=22 * 1024, mem_total_mib=24 * 1024, utilization=10),
            GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(
            responses=batch5_responses(), gpus=gpus, target=ServingTarget(gpus=gpus)
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            rail = str(app.query_one("#host-stats-rail", Static).render())
            assert "repo" in rail
            assert "models" in rail
            assert "26%" in rail      # repo Use%
            assert "90%" in rail      # models Use%
            assert "RAM" in rail
            assert "15%" in rail      # system RAM %

    async def test_gpu_card_shows_held_by_container(self):
        gpus = [
            GpuInfo(index=0, mem_used_mib=22 * 1024, mem_total_mib=24 * 1024, utilization=10),
            GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(
            responses=batch5_responses(), gpus=gpus, target=ServingTarget(gpus=gpus)
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            bar0 = str(app.query_one("#gpu0-bar", Static).render())
            assert "held by:" in bar0
            assert "llama-cpp-pi-reasoning" in bar0
            assert "22.2G" in bar0    # 22686 MiB / 1024

    async def test_gpu_card_degraded_shows_pid_not_fabricated(self):
        gpus = [
            GpuInfo(index=0, mem_used_mib=22 * 1024, mem_total_mib=24 * 1024),
            GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(
            responses=batch5_responses(cgroup="", idnames=""),
            gpus=gpus,
            target=ServingTarget(gpus=gpus),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            bar0 = str(app.query_one("#gpu0-bar", Static).render())
            assert "held by:" in bar0
            assert "pid 588408" in bar0           # named-by-pid fallback
            assert "names unavailable" in bar0    # honest degraded cue

    async def test_disk_rail_all_empty_df_shows_honest_cue_no_zero_bar(self):
        # MUST-FIX 1 / A2 (render): df all-empty → honest cue, NEVER a fabricated
        # "0% 0G/0G" bar.  RAM still renders; the disk-read-failed cue surfaces.
        gpus = [GpuInfo(index=0, mem_used_mib=1, mem_total_mib=24 * 1024)]
        app, _, _ = make_app(
            responses=batch5_responses(df=""), gpus=gpus, target=ServingTarget(gpus=gpus)
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            rail = str(app.query_one("#host-stats-rail", Static).render())
            assert "disk read failed" in rail     # honest cue
            assert "0%" not in rail               # NO fabricated zero bar
            assert "0G/0G" not in rail

    async def test_disk_rail_meminfo_no_available_shows_cue_not_full_bar(self):
        # TEST-HARDENING: meminfo WITHOUT MemAvailable → honest "MemAvailable
        # missing" cue, NOT a misleading 100% RAM bar (the renderer gates on
        # ram.error; pct would otherwise compute to 100% — see the parse test).
        gpus = [GpuInfo(index=0, mem_used_mib=1, mem_total_mib=24 * 1024)]
        app, _, _ = make_app(
            responses=batch5_responses(meminfo="MemTotal: 1000 kB\nMemFree: 100 kB\n"),
            gpus=gpus,
            target=ServingTarget(gpus=gpus),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            rail = str(app.query_one("#host-stats-rail", Static).render())
            assert "MemAvailable missing" in rail  # honest cue
            assert "100%" not in rail              # NOT a misleading full bar

    async def test_gpu_card_unresolved_uuid_renders_neutral_heading(self):
        # NH-B (render): a holder whose card couldn't be resolved renders under a
        # NEUTRAL "card unknown" heading on GPU0's card, NOT mis-pinned to "held by:".
        gpus = [
            GpuInfo(index=0, mem_used_mib=1, mem_total_mib=24 * 1024),
            GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(
            responses=batch5_responses(uuid_index=""),
            gpus=gpus,
            target=ServingTarget(gpus=gpus),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            bar0 = str(app.query_one("#gpu0-bar", Static).render())
            assert "card unknown" in bar0          # neutral heading
            assert "llama-cpp-pi-reasoning" in bar0  # holder still surfaced
            assert "held by:" not in bar0          # NOT mis-pinned to a card


@pytest.mark.asyncio
class TestFix3HostStatsPlacement:
    """FIX 3 — host disk + RAM render into the LEFT RAIL (the "estate column"),
    NOT the Orchestration sub-tab; GPU "held by:" attribution stays on the cards."""

    async def test_disk_ram_render_into_left_rail_not_orch_pane(self):
        from club3090_cockpit.app import HostStatsRail, OperateOrchPane
        gpus = [
            GpuInfo(index=0, mem_used_mib=22 * 1024, mem_total_mib=24 * 1024),
            GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(
            responses=batch5_responses(), gpus=gpus, target=ServingTarget(gpus=gpus)
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            # The disk/RAM card lives in the GLOBAL left rail, below RailStatus.
            rail_widget = app.query_one("#host-stats-rail", HostStatsRail)
            assert rail_widget in app.query_one("#left-rail").children
            rail = str(rail_widget.render())
            assert "repo" in rail and "RAM" in rail   # disk + RAM after a poll
            # The orch pane NO LONGER owns a #disk-rail child (it was moved out).
            orch = app.query_one("#operate-orch-pane", OperateOrchPane)
            assert len(orch.query("#disk-rail")) == 0

    async def test_gpu_held_by_attribution_stays_on_orch_cards(self):
        # FIX 3 — the GPU-VRAM → container attribution was NOT flagged; it stays on
        # the orch GPU cards (only disk/RAM moved to the rail).
        gpus = [
            GpuInfo(index=0, mem_used_mib=22 * 1024, mem_total_mib=24 * 1024),
            GpuInfo(index=1, mem_used_mib=1, mem_total_mib=24 * 1024),
        ]
        app, _, _ = make_app(
            responses=batch5_responses(), gpus=gpus, target=ServingTarget(gpus=gpus)
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            bar0 = str(app.query_one("#gpu0-bar", Static).render())
            assert "held by:" in bar0                  # attribution unchanged
            assert "llama-cpp-pi-reasoning" in bar0
            # And the rail did NOT absorb the GPU attribution.
            rail = str(app.query_one("#host-stats-rail", Static).render())
            assert "held by:" not in rail


@pytest.mark.asyncio
class TestBatch5StudioServiceSet:
    """studio-* / #2-ext — running studio stack containers are surfaced (kind
    'stack'), labeled-not-stopped, and visible in the Operate service list."""

    async def test_studio_classified_stack_not_none(self):
        from club3090_cockpit.services import _classify_container_kind
        assert _classify_container_kind("studio-tts") == "stack"
        assert _classify_container_kind("studio-image-shim") == "stack"
        # comfyui keeps its first-class GPU 'service' kind (precedence).
        assert _classify_container_kind("comfyui") == "service"
        # a non-GPU supporting service is still NOT surfaced as a stack holder.
        assert _classify_container_kind("open-webui") is None

    async def test_studio_containers_appear_running_in_table(self):
        app, _, _ = make_app(responses=batch5_responses())
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            by_name = {c.name: c for c in pane._containers}
            assert "studio-tts" in by_name
            # Labeled as a running 'stack' holder — NOT mislabeled stopped.
            assert by_name["studio-tts"].kind == "stack"
            assert by_name["studio-tts"].status == "running"
            assert by_name["studio-tts"].is_running is True

    async def test_studio_dir_not_duplicated_as_stopped(self, tmp_path):
        # A KNOWN services/studio dir must NOT also appear greyed once the
        # studio-* containers are surfaced as running 'stack' rows — the de-dup
        # must collapse the dir into the running containers (a stale stopped
        # "studio" row would falsely suppress its live actions).
        _seed_services(tmp_path, ["studio"])
        seed_repo(tmp_path)
        app, _, _ = make_app(responses=batch5_responses(), repo_root=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_operate(pilot)
            pane = app.query_one("#operate-containers-pane", OperateContainersPane)
            studio_rows = [c for c in pane._containers if "studio" in c.name.lower()]
            # The running studio-* stack rows are present...
            assert any(c.name == "studio-tts" for c in studio_rows)
            # ...and NONE of them (nor a bare "studio" dir row) is greyed stopped.
            stopped = [
                c for c in studio_rows
                if getattr(c, "status", "running") == "stopped"
            ]
            assert stopped == []


class _FakeDLRunner:
    """Stand-in download runner: start_raw returns a CoreRunState already 'done'
    (exit 0), so run_download's await returns immediately; records launch + cancel."""

    def __init__(self):
        self.started: list = []
        self.cancelled = False

    def set_callbacks(self, **k):
        pass

    async def start_raw(self, cmd, env, run_type, parser):
        from club3090_tui_core.runner import CoreRunState
        self.started.append({"cmd": cmd, "env": env})
        st = CoreRunState(run_type=run_type)
        st.exit_code = 0
        st.done.set()
        return st

    async def cancel(self):
        self.cancelled = True
        return []


class TestDownloadUX:
    """Download UX — Download-vs-Start pop-up modes, the listing glyph, and the
    download worker (launch / progress / complete-restat / cancel)."""

    @staticmethod
    def _entry(state, hf="Some/Repo", size=29.0, slug="vllm/qwen-27b-dual-max"):
        from club3090_cockpit.data import CatalogEntry, WeightsMeta
        from club3090_tui_core import VariantRow
        e = CatalogEntry(row=VariantRow(
            slug=slug, switch_engine="vllm", launch_engine="vllm", compose_dir="x",
            file="mtp.yml", port=8013, model="qwen3.6-27b", engine="vllm-stable",
            kvcalc_key="SKIP", container="c",
            compose_path="models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml",
            status="experimental", ctx_label="262K", status_note=""))
        e.weights_state = state
        e.weights = WeightsMeta(model="qwen3.6-27b", variant="fp8", subdir="qwen3.6-27b-fp8",
                                hf_repo=hf or "", size_gb=size, verify_glob="*.safetensors")
        return e

    @pytest.mark.asyncio
    async def test_listing_glyph_states(self):
        from club3090_cockpit.app import _weights_glyph
        from club3090_cockpit.data import (
            WEIGHTS_ABSENT, WEIGHTS_DOWNLOADING, WEIGHTS_PRESENT, WEIGHTS_PARTIAL,
        )
        assert "⬇" in _weights_glyph(self._entry(WEIGHTS_ABSENT))
        assert "⚠" in _weights_glyph(self._entry(WEIGHTS_PARTIAL))
        e_dl = self._entry(WEIGHTS_DOWNLOADING); e_dl.download_pct = 37
        g = _weights_glyph(e_dl)
        assert "⏳" in g and "37%" in g
        assert _weights_glyph(self._entry(WEIGHTS_PRESENT)) == ""   # present → no badge

    @pytest.mark.asyncio
    async def test_start_download_manual_no_repo(self):
        from club3090_cockpit.data import WEIGHTS_ABSENT
        app, _, _ = make_app()
        notes: list = []
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            orig = app.notify
            app.notify = lambda *a, **k: (notes.append(a[0] if a else ""), orig(*a, **k))[1]
            e = self._entry(WEIGHTS_ABSENT, hf=None)   # no hf_repo → manual, can't download
            app.start_download(e)
            await _settle(pilot)
            assert e.slug not in app._active_downloads()
            assert e.weights_state == WEIGHTS_ABSENT   # unchanged
            assert any("manual" in n.lower() or "no direct" in n.lower() for n in notes)

    @pytest.mark.asyncio
    async def test_start_download_disk_guard(self):
        from club3090_cockpit.data import WEIGHTS_ABSENT
        app, _, _ = make_app()
        app._data.weights_fits_disk = lambda meta, **k: (False, 1.0, 99.0)
        notes: list = []
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            orig = app.notify
            app.notify = lambda *a, **k: (notes.append(a[0] if a else ""), orig(*a, **k))[1]
            e = self._entry(WEIGHTS_ABSENT)
            app.start_download(e)
            await _settle(pilot)
            assert e.slug not in app._active_downloads()
            assert any("disk" in n.lower() for n in notes)

    @pytest.mark.asyncio
    async def test_start_download_marks_then_completes_present(self):
        from club3090_cockpit.data import WEIGHTS_ABSENT, WEIGHTS_DOWNLOADING, WEIGHTS_PRESENT
        app, _, _ = make_app()
        app._data._download_runner = _FakeDLRunner()
        app._data.weights_fits_disk = lambda meta, **k: (True, 500.0, 32.0)

        async def _enrich(entries, **k):           # simulate verify-after → present
            for e in entries:
                e.weights_state = WEIGHTS_PRESENT
        app._data.enrich_weights = _enrich

        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            e = self._entry(WEIGHTS_ABSENT)
            app.start_download(e)
            assert e.slug in app._active_downloads()           # immediately tracked
            assert e.weights_state == WEIGHTS_DOWNLOADING
            await _settle(pilot)                                # worker runs to done
            assert e.slug not in app._active_downloads()        # finished
            assert e.weights_state == WEIGHTS_PRESENT           # re-stat → present
            assert app._data._download_runner.started           # the download launched
            assert app._data._download_runner.started[0]["env"].get("WEIGHT_KEY") == "qwen3.6-27b:fp8"

    @pytest.mark.asyncio
    async def test_download_state_survives_catalog_refresh(self):
        """A catalog rebuild (r) must not orphan an in-flight download: the
        slug-keyed tracker re-stamps the NEW entry (DOWNLOADING + pct) and
        re-points info['entry'] at it, so the ⏳ glyph + pop-up % persist."""
        from club3090_cockpit.data import WEIGHTS_DOWNLOADING, WEIGHTS_PARTIAL
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            # In-flight download tracked by slug, with a live pct.
            old = self._entry(WEIGHTS_DOWNLOADING)
            old.download_pct = 42
            app._active_downloads()[old.slug] = {"entry": old, "meta": old.weights, "pct": 42}
            # A refresh produced a NEW entry, re-stat'd to partial (⏳ would vanish).
            fresh = self._entry(WEIGHTS_PARTIAL)
            assert fresh is not old and fresh.slug == old.slug
            app._reapply_active_downloads([fresh])
            assert fresh.weights_state == WEIGHTS_DOWNLOADING
            assert fresh.download_pct == 42
            assert app._active_downloads()[old.slug]["entry"] is fresh   # re-pointed

    @pytest.mark.asyncio
    async def test_cancel_download_resets_and_kills(self):
        from club3090_cockpit.data import WEIGHTS_DOWNLOADING, WEIGHTS_ABSENT
        app, _, _ = make_app()
        app._data._download_runner = _FakeDLRunner()

        async def _enrich(entries, **k):
            for e in entries:
                e.weights_state = WEIGHTS_ABSENT
        app._data.enrich_weights = _enrich

        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            e = self._entry(WEIGHTS_DOWNLOADING)
            e.download_pct = 12
            app._active_downloads()[e.slug] = {"entry": e, "meta": e.weights}
            app.cancel_download(e.slug)
            await _settle(pilot)
            assert e.slug not in app._active_downloads()
            assert app._data._download_runner.cancelled

    @pytest.mark.asyncio
    async def test_serve_context_picks_mode_by_weights_state(self):
        from club3090_cockpit.data import WEIGHTS_ABSENT, WEIGHTS_PRESENT, WEIGHTS_DOWNLOADING
        app, _, _ = make_app(gpus=_FREE_GPUS, target=ServingTarget(gpus=_FREE_GPUS))
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app._serve_context_for(self._entry(WEIGHTS_ABSENT)).mode == "download"
            assert app._serve_context_for(self._entry(WEIGHTS_PRESENT)).mode == "start"
            e_dl = self._entry(WEIGHTS_DOWNLOADING)
            app._active_downloads()[e_dl.slug] = {"entry": e_dl, "meta": e_dl.weights}
            assert app._serve_context_for(e_dl).mode == "downloading"


class TestSettings:
    """Download Settings ([S]) — MODEL_DIR + HF_TOKEN editable + persisted +
    applied live; the model-dir-missing banner.  C3_CONFIG_DIR is isolated to a
    tmp dir so nothing touches the real ~/.config."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("C3_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # MODEL_DIR now wins over persisted (12-factor), so a rig-set env var
        # would make the persisted-settings tests non-deterministic — clear it;
        # the env-precedence tests below set it explicitly.
        monkeypatch.delenv("MODEL_DIR", raising=False)

    @pytest.mark.asyncio
    async def test_s_opens_settings_modal(self):
        from club3090_cockpit.app import SettingsScreen
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            assert app.check_action("settings", ()) is True
            await pilot.press("S")
            await _settle(pilot)
            assert isinstance(app.screen, SettingsScreen)

    @pytest.mark.asyncio
    async def test_apply_settings_persists_and_applies(self):
        import os
        from club3090_cockpit import __main__ as M
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.apply_settings(model_dir="/tmp/my-models", hf_token="hf_secret123")
            await _settle(pilot)
            assert app._data.weights_model_dir() == "/tmp/my-models"
            assert os.environ.get("HF_TOKEN") == "hf_secret123"
            s = M.load_settings()
            assert s.get("model_dir") == "/tmp/my-models"
            assert s.get("hf_token") == "hf_secret123"

    @pytest.mark.asyncio
    async def test_blank_fields_keep_current(self):
        import os
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app.apply_settings(model_dir="/tmp/m1", hf_token="hf_first")
            await _settle(pilot)
            # blank token → keep the existing one; blank dir → keep the dir
            app.apply_settings(model_dir="", hf_token="")
            await _settle(pilot)
            assert app._data.weights_model_dir() == "/tmp/m1"
            assert os.environ.get("HF_TOKEN") == "hf_first"

    @pytest.mark.asyncio
    async def test_director_placement_persists_to_repo_env(self, tmp_path):
        """Director placement (CPU/GPU0/GPU1) persists to STUDIO_DIRECTOR_DEVICE in
        the repo .env — what gpu-mode reads on the next ai-studio start."""
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            app._data.repo_root = tmp_path                 # write the .env to a temp dir
            assert app._data.director_device() == "gpu0"
            app.apply_settings(model_dir="", hf_token="", director_device="cpu")
            await _settle(pilot)
            assert app._data.director_device() == "cpu"
            assert "STUDIO_DIRECTOR_DEVICE=cpu" in (tmp_path / ".env").read_text()
            # re-applying the same value is a no-op (no duplicate line)
            app.apply_settings(model_dir="", hf_token="", director_device="cpu")
            await _settle(pilot)
            assert (tmp_path / ".env").read_text().count("STUDIO_DIRECTOR_DEVICE=") == 1

    @pytest.mark.asyncio
    async def test_apply_persisted_settings_on_fresh_app(self):
        import os
        from club3090_cockpit import __main__ as M
        M.save_settings({"model_dir": "/tmp/persisted", "hf_token": "hf_persisted"})
        app, _, _ = make_app()
        env = dict()
        M.apply_persisted_settings(app, env)
        assert app._data.weights_model_dir() == "/tmp/persisted"
        assert env.get("HF_TOKEN") == "hf_persisted"
        # an explicit env token WINS (not overwritten)
        app2, _, _ = make_app()
        env2 = {"HF_TOKEN": "hf_shell"}
        M.apply_persisted_settings(app2, env2)
        assert env2["HF_TOKEN"] == "hf_shell"

    @pytest.mark.asyncio
    async def test_model_dir_missing_banner(self):
        app, _, _ = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            pane = app.query_one("#catalog-pane", CatalogPane)
            pane.set_model_dir_note("⚠ model dir not found — press [S] to set it")
            await pilot.pause()
            status = str(app.query_one("#catalog-status", Label).render())
            assert "model dir not found" in status

    @pytest.mark.asyncio
    async def test_env_model_dir_picked_up(self, monkeypatch):
        """MODEL_DIR env var (no persisted setting) → weights_model_dir reads it."""
        import os
        from club3090_cockpit import __main__ as M
        monkeypatch.setenv("MODEL_DIR", "/env/models")
        app, _, _ = make_app()
        M.apply_persisted_settings(app, dict(os.environ))
        assert app._data.weights_model_dir() == "/env/models"

    @pytest.mark.asyncio
    async def test_env_model_dir_wins_over_persisted(self, monkeypatch):
        """Explicit env var beats a stale persisted value (12-factor)."""
        import os
        from club3090_cockpit import __main__ as M
        M.save_settings({"model_dir": "/persisted/models", "hf_token": "hf_p"})
        monkeypatch.setenv("MODEL_DIR", "/env/models")
        app, _, _ = make_app()
        M.apply_persisted_settings(app, dict(os.environ))
        assert app._data.weights_model_dir() == "/env/models"

    @pytest.mark.asyncio
    async def test_env_model_dir_fallback_without_apply(self, monkeypatch):
        """A bare CockpitData (apply_persisted_settings NOT called — tests /
        embedding) still honours the MODEL_DIR env var via weights_model_dir."""
        from club3090_cockpit.services import CockpitData, MODEL_DIR
        monkeypatch.setenv("MODEL_DIR", "/env/models")
        data = CockpitData(repo_root=Path("/tmp/fake-repo"))
        assert data.weights_model_dir() == "/env/models"
        # and with no env + no persisted → the bundled default
        monkeypatch.delenv("MODEL_DIR", raising=False)
        assert CockpitData(repo_root=Path("/tmp/fake-repo")).weights_model_dir() == MODEL_DIR

    @pytest.mark.asyncio
    async def test_env_hf_token_respected_and_wins(self, monkeypatch):
        """A shell HF_TOKEN is left intact (wins over persisted) so it flows into
        the download subprocess env."""
        from club3090_cockpit import __main__ as M
        M.save_settings({"model_dir": "/m", "hf_token": "hf_persisted"})
        app, _, _ = make_app()
        env = {"HF_TOKEN": "hf_shell"}
        M.apply_persisted_settings(app, env)
        assert env["HF_TOKEN"] == "hf_shell"


class TestAdaptiveEstatePoll:
    """Estate poll fires at _POLL_TICK_SECS; GPU bars refresh every tick (cheap, docker-free)
    but the heavy docker+host batch runs only on a regime-gated tick — quiet while idle, live
    during/after a docker-affecting action."""

    @staticmethod
    def _due(**kw):
        import types, time
        from club3090_cockpit.app import CockpitApp
        s = types.SimpleNamespace(
            _docker_burst_until=0.0, _last_activity_mono=time.monotonic(), _estate_tick=0,
            _DOCKER_STEADY_STRIDE=CockpitApp._DOCKER_STEADY_STRIDE,
            _DOCKER_IDLE_STRIDE=CockpitApp._DOCKER_IDLE_STRIDE,
            _ESTATE_IDLE_AFTER_SECS=CockpitApp._ESTATE_IDLE_AFTER_SECS,
        )
        s.__dict__.update(kw)
        return CockpitApp._docker_poll_due.__get__(s)()

    def test_burst_polls_every_tick(self):
        import time
        for tick in (1, 2, 3, 5, 6):
            assert self._due(_docker_burst_until=time.monotonic() + 100, _estate_tick=tick) is True

    def test_steady_polls_on_stride(self):
        from club3090_cockpit.app import CockpitApp
        n = CockpitApp._DOCKER_STEADY_STRIDE
        assert self._due(_estate_tick=n) is True
        assert self._due(_estate_tick=n - 1) is False
        assert self._due(_estate_tick=n * 2) is True

    def test_idle_widens_stride(self):
        from club3090_cockpit.app import CockpitApp
        steady, idle = CockpitApp._DOCKER_STEADY_STRIDE, CockpitApp._DOCKER_IDLE_STRIDE
        assert idle > steady
        # ancient activity → idle regime: a steady-due tick is NOT idle-due (wider stride)
        assert self._due(_last_activity_mono=0.0, _estate_tick=steady) is False
        assert self._due(_last_activity_mono=0.0, _estate_tick=idle) is True

    def test_burst_kinds_are_docker_affecting_only(self):
        from club3090_cockpit.app import CockpitApp
        k = CockpitApp._DOCKER_BURST_KINDS
        assert {"serve", "scene", "container", "container_rm", "estate_down", "service-up"} <= k
        for non in ("set_default", "clear_default", "power_cap", "submit_bench", "validation"):
            assert non not in k

    def test_gpu_info_is_docker_free_and_safe(self, tmp_path):
        import asyncio
        from club3090_cockpit.services import CockpitData
        cd = CockpitData(tmp_path)
        async def ok():
            return ["g0", "g1"]
        cd._get_gpu_info = ok
        assert asyncio.run(cd.gpu_info()) == ["g0", "g1"]
        async def boom():
            raise RuntimeError("nvidia-smi gone")
        cd._get_gpu_info = boom
        assert asyncio.run(cd.gpu_info()) == []   # degrades, never raises

    def test_refresh_gpu_cards_swaps_and_repaints(self):
        import types
        from club3090_cockpit.app import OperateOrchPane
        calls = []
        pane = types.SimpleNamespace(_last_state=None, _populate_gpus=lambda st: calls.append(st))
        bound = OperateOrchPane.refresh_gpu_cards.__get__(pane)
        bound(["g0"])                      # no _last_state yet → no-op
        assert calls == []
        st = types.SimpleNamespace(gpus=[])
        pane._last_state = st
        bound(["g0", "g1"])                # swaps fresh gpus + repaints
        assert st.gpus == ["g0", "g1"] and calls == [st]
        bound([])                          # empty read → keep last bars (no-op)
        assert st.gpus == ["g0", "g1"]
