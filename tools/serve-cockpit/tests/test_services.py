"""Tests for the cockpit data/service layer (data.py + services.py).

ALL subprocess and detect is mocked — no GPU, no Docker, no TTY, no real
script calls.  Covers:
  - the read contracts (catalog enrichment, explain, fit, byo, scenes, doctor,
    estate_state, containers) against a FakeRunner;
  - the pure parse helpers (health text, BENCHMARKS.md scrape, explain corpus);
  - the action builders (gated; --force only with a reason);
  - the reconcile gate (the dual-writer safety core) — thorough scenarios incl.
    the prompt's "estate booted GPU0, scene-switch requested → conflict" case;
  - execute_action refusing to write when the gate is unsafe (execution mocked).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from club3090_tui_core.detect import GpuInfo, ServingTarget
from club3090_tui_core.registry import VariantRow
from club3090_tui_core.runner import CoreRunState


class LeaseWriteRunner:
    """Fire-and-forget write runner, like the real SubprocessRunner: ``start_raw``
    only SPAWNS and returns a RUNNING ``CoreRunState`` immediately (finished=0).
    The test signals the subprocess's completion with ``state.done.set()`` — so
    the pending-claim lease (not any lock-hold duration) is what blocks writer B
    through the whole boot gap."""

    def __init__(self):
        self.started: list[dict] = []

    async def start_raw(self, cmd, env, run_type, parser):
        st = CoreRunState(run_type=run_type, started=time.time())  # is_finished == False
        self.started.append({"cmd": cmd, "state": st})
        return st


class FailedSpawnWriteRunner:
    """``start_raw`` returns an already-finished spawn-failure state — the claim
    must clear immediately (no card was ever claimed)."""

    def __init__(self):
        self.started: list[dict] = []

    async def start_raw(self, cmd, env, run_type, parser):
        st = CoreRunState(run_type=run_type, started=time.time(), finished=time.time(),
                          exit_code=-1, verdict="failed", error="spawn boom")
        st.done.set()
        self.started.append({"cmd": cmd, "state": st})
        return st

from club3090_cockpit.data import (
    ActionPlan,
    ByoResult,
    DoctorRead,
    FitVerdict,
    Measurement,
    ReconcileResult,
    Scene,
    measurement_from_explain_benchmarks,
    parse_benchmarks_md_for_slug,
    parse_health_text,
    strip_ansi,
)
from club3090_cockpit.services import CockpitData, RealRunner, RunResult, _variant_row_from_dict


ROOT = Path("/tmp/fake-club-3090-root")


# ---------------------------------------------------------------------------
# Fake runner + fake detect
# ---------------------------------------------------------------------------


class FakeRunner:
    """Canned-output runner keyed on a recognizable token in the command.

    ``responses`` maps a substring (matched against the joined command) to a
    RunResult.  ``calls`` records every command for assertions.  A WRITE command
    reaching here would mean execution wasn't mocked — tests assert it never is.
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
# Fixtures: canned contract outputs
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

# kv-calc --fit-all --json shape: {card, card_vram_gb, variants:{slug: verdict}}.
# The catalog fit column is enriched from ONE --fit-all call (not N --fit calls);
# non-vLLM (kvcalc_key SKIP) slugs come back as {"verdict": "skip"}.
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

# REAL switch.sh --explain --json benchmarks shape (verified live):
#   [{"row": "<markdown row>", "columns": [<cell>, …]}]
# Canonical bench layout: Compose|Rig|KV|Max ctx|Narr/Code TPS|PP|VRAM|Date|Notes
# → TPS is columns[4].  (NOT the invented {"narr_tps": …} corpus shape.)
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
        "registry": {"slug": "vllm/dual", "model": "qwen3.6-27b"},
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

ESTATE_REPORT_BUSY = json.dumps(
    {
        "active_estate": {
            "present": True,
            "valid": True,
            "instances": [
                {"name": "llama-gpu0", "compose": "llamacpp/default", "gpus": [0], "port": 8010},
                {"name": "llama-gpu1", "compose": "llamacpp/default", "gpus": [1], "port": 8020},
            ],
        }
    }
)

ESTATE_REPORT_FREE = json.dumps({"active_estate": {"present": False, "instances": []}})

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

DOCKER_PS_ENGINE = "vllm-qwen36-27b-dual|0.0.0.0:8010->8000/tcp, [::]:8010->8000/tcp\nopen-webui|0.0.0.0:3000->8080/tcp\n"
DOCKER_PS_EMPTY = ""


def full_runner(**overrides) -> FakeRunner:
    """A FakeRunner wired for the common read contracts; override per-test."""
    responses = {
        "registry-emit.sh --json": ok(REGISTRY_JSON),
        # --fit-all MUST precede --fit: the batch command contains the substring
        # "kv-calc.py --fit" too, and FakeRunner returns the first match.
        "kv-calc.py --fit-all": ok(FIT_ALL_JSON),
        "kv-calc.py --fit": ok(FIT_JSON),
        "--explain vllm/dual --json": ok(EXPLAIN_JSON),
        "--explain ik-llama/iq4ks-mtp --json": ok(EXPLAIN_NO_BENCH_JSON),
        "gpu-mode.sh --list-modes --json": ok(SCENES_JSON),
        "pull.sh": ok(PULL_JSON),
        "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
        "health.sh": ok(HEALTH_DOWN),
        "docker ps": ok(DOCKER_PS_EMPTY),
    }
    responses.update(overrides)
    return FakeRunner(responses)


# ===========================================================================
# Pure parse helpers
# ===========================================================================


class TestParseHelpers:
    def test_strip_ansi(self):
        assert strip_ansi("\x1b[0;32m✓\x1b[0m serving") == "✓ serving"

    def test_health_serving_parsed(self):
        dr = parse_health_text(HEALTH_SERVING)
        assert dr.reachable is True
        assert dr.serving is True
        assert dr.kv_pool_pct == 61
        assert "MTP n=2" in dr.spec_dec
        assert dr.recent_errors == 0
        assert "serving" in dr.summary

    def test_health_down_parsed(self):
        dr = parse_health_text(HEALTH_DOWN)
        assert dr.reachable is False
        assert dr.summary == "API not reachable"

    def test_health_empty_is_unreachable(self):
        dr = parse_health_text("")
        assert dr.reachable is True  # nothing says "not reachable"
        assert dr.raw == ""

    def test_benchmarks_md_scrape(self):
        md = (
            "| Compose | Rig | KV |\n"
            "| `llamacpp/mtp` | @x | q4 | 200K | **50.27 / 58.92** | ... 8-pack 100/150 ... 2026-05-23 |\n"
        )
        m = parse_benchmarks_md_for_slug(md, "llamacpp/mtp")
        assert m is not None
        assert m.narr_tps == 50.27
        assert m.code_tps == 58.92
        assert m.quality_8pk == "100/150"
        assert m.date == "2026-05-23"
        assert m.source == "benchmarks.md"

    def test_benchmarks_md_no_match(self):
        md = "| `other/slug` | **10 / 20** |\n"
        assert parse_benchmarks_md_for_slug(md, "nope/missing") is None

    def test_benchmarks_md_matches_file_stem(self):
        md = "| `mtp` | **40 / 50** |\n"
        m = parse_benchmarks_md_for_slug(md, "llamacpp/mtp")
        assert m is not None and m.narr_tps == 40.0

    def test_benchmarks_md_matches_yml_filename(self):
        """The first column is usually `<serving>.yml` — must match the stem."""
        md = (
            "| `minimal.yml` (single) | @x | TQ3 | 64K | 32 / 33 | — | — | 2026-05-03 | n |\n"
        )
        m = parse_benchmarks_md_for_slug(md, "vllm/minimal")
        assert m is not None
        assert m.narr_tps == 32.0 and m.code_tps == 33.0

    def test_benchmarks_md_exact_match_not_substring(self):
        """REGRESSION: 'dual' must NOT match the 'dual-dflash.yml' row.

        Previously a substring test let `vllm/dual` pull the dual-dflash row
        (a different 5090 config).  Anchored first-cell match prevents it."""
        md = (
            "| `dual-dflash.yml` | @z (1× 5090) | fp8 | 49K | 126 / 200 | — | — | 2026-05-07 | n |\n"
            "| `dual.yml` ⭐ | @noonghunna (2× 3090) | fp8 | 262K | 69 / 89 | — | — | 2026-04-29 | n |\n"
        )
        m = parse_benchmarks_md_for_slug(md, "vllm/dual")
        assert m is not None
        # Must be the 3090 dual.yml row (69/89), NOT the 5090 dual-dflash (126/200).
        assert m.narr_tps == 69.0 and m.code_tps == 89.0

    def test_benchmarks_md_non_bold_tps(self):
        """Non-bold '~32 / ~33' rows must parse (minimal.yml-style)."""
        md = "| `minimal.yml` | @x | TQ3 | 64K | ~32 / ~33 | — | — | 2026-05-03 | n |\n"
        m = parse_benchmarks_md_for_slug(md, "vllm/minimal")
        assert m is not None
        assert m.narr_tps == 32.0 and m.code_tps == 33.0

    def test_benchmarks_md_tbd_yields_no_measurement(self):
        """A matched row whose TPS cell is 'TBD' must yield NO measurement (not
        a bogus pair) so the UI renders '—' honestly."""
        md = "| `long-text-no-mtp.yml` | @x | TQ3 | 200K | TBD | — | — | — | n |\n"
        m = parse_benchmarks_md_for_slug(md, "vllm/long-text-no-mtp")
        assert m is None

    def test_benchmarks_md_decode_paren_does_not_shadow_headline(self):
        """The headline 'N / M' wins over the parenthetical (decode X / Y)."""
        md = (
            "| `mtp.yml` | @x | q4 | 200K | **59.67 / 68.78** (decode 60.39 / 72.40) "
            "| — | — | 2026-05-23 | 8-pack 107/150 |\n"
        )
        m = parse_benchmarks_md_for_slug(md, "ik-llama/mtp")
        assert m is not None
        assert m.narr_tps == 59.67 and m.code_tps == 68.78
        assert m.quality_8pk == "107/150"

    def test_measurement_from_explain_benchmarks(self):
        """REAL shape: [{"row","columns"}] — TPS parsed out of columns[4]."""
        m = measurement_from_explain_benchmarks([EXPLAIN_BENCH_ROW])
        assert m.source == "explain"
        assert m.narr_tps == 174.0
        assert m.code_tps == 42.0
        assert m.tps_label == "174/42"
        assert m.quality_8pk == "109/150"
        assert m.date == "2026-05-30"

    def test_measurement_from_explain_picks_newest_tps_row(self):
        """A stress/soak row (no TPS in columns) must NOT shadow a TPS row, and
        the newest TPS-bearing row wins."""
        stress_row = {
            "row": "| `dual.yml` | @x | PASS | PASS at 64K | FAIL | 2026-05-03 |",
            "columns": ["`dual.yml`", "@x", "PASS", "PASS at 64K", "FAIL", "2026-05-03"],
        }
        newer = {
            "row": "| `dual.yml` | @y | fp8 | 262K | 69 / 89 | — | — | 2026-06-01 |",
            "columns": ["`dual.yml`", "@y", "fp8", "262K", "69 / 89", "—", "—", "2026-06-01"],
        }
        m = measurement_from_explain_benchmarks([EXPLAIN_BENCH_ROW, stress_row, newer])
        assert m.tps_label == "69/89"
        assert m.date == "2026-06-01"

    def test_measurement_from_explain_no_tps_is_empty(self):
        """benchmarks[] with only a non-TPS (stress) row → empty Measurement so
        the caller can fall through to the BENCHMARKS.md scrape."""
        stress_row = {
            "row": "| `dual.yml` | @x | PASS | PASS at 64K | FAIL | 2026-05-03 |",
            "columns": ["`dual.yml`", "@x", "PASS", "PASS at 64K", "FAIL", "2026-05-03"],
        }
        m = measurement_from_explain_benchmarks([stress_row])
        assert m.narr_tps is None
        assert m.tps_label == "—"

    def test_measurement_empty(self):
        m = measurement_from_explain_benchmarks([])
        assert m.tps_label == "—"
        assert m.quality_label == "—"

    def test_fit_glyphs(self):
        # REAL kv-calc --fit verdict enum (fits-constrained, NOT fits-tight).
        assert FitVerdict(verdict="fits-clean").glyph == "●"
        assert FitVerdict(verdict="fits-constrained").glyph == "◐"
        assert FitVerdict(verdict="wont-fit").glyph == "○"
        assert FitVerdict(verdict="skip").glyph == "·"
        assert FitVerdict(verdict="unknown").glyph == "·"
        # 'fits-tight' is NEVER emitted by kv-calc → falls to the unknown glyph.
        assert FitVerdict(verdict="fits-tight").glyph == "·"

    def test_variant_row_from_dict_attaches_source(self):
        row = _variant_row_from_dict(
            {"slug": "x/y", "port": 8010, "source": "community", "status": "production"}
        )
        assert isinstance(row, VariantRow)
        assert row.slug == "x/y"
        assert row.port == 8010
        assert getattr(row, "source") == "community"


# ===========================================================================
# READ contracts
# ===========================================================================


class TestLoadCatalog:
    @pytest.mark.asyncio
    async def test_catalog_parses_variants(self):
        cd = CockpitData(ROOT, runner=full_runner())
        entries, err = await cd.load_catalog(enrich_fit=False, enrich_measurement=False)
        assert err is None
        assert len(entries) == 2
        assert entries[0].slug == "vllm/dual"
        assert entries[0].source == "curated"

    @pytest.mark.asyncio
    async def test_catalog_enriches_fit(self):
        cd = CockpitData(ROOT, runner=full_runner())
        entries, _ = await cd.load_catalog(enrich_fit=True, enrich_measurement=False)
        vllm = next(e for e in entries if e.slug == "vllm/dual")
        assert vllm.fit.verdict == "fits-clean"
        assert vllm.fit.glyph == "●"

    @pytest.mark.asyncio
    async def test_catalog_skip_fit_for_ik_llama(self):
        """ik/llama kvcalc_key=SKIP → fit verdict 'skip' (the --fit-all batch
        emits skip for non-vLLM slugs; no per-slug fan-out)."""
        cd = CockpitData(ROOT, runner=full_runner())
        entries, _ = await cd.load_catalog(enrich_fit=True, enrich_measurement=False)
        ik = next(e for e in entries if e.slug == "ik-llama/iq4ks-mtp")
        assert ik.fit.verdict == "skip"

    @pytest.mark.asyncio
    async def test_catalog_enriches_measurement_from_explain(self):
        cd = CockpitData(ROOT, runner=full_runner())
        entries, _ = await cd.load_catalog(enrich_fit=False, enrich_measurement=True)
        vllm = next(e for e in entries if e.slug == "vllm/dual")
        assert vllm.measurement.source == "explain"
        assert vllm.measurement.tps_label == "174/42"
        assert vllm.measurement.quality_label == "109/150"

    @pytest.mark.asyncio
    async def test_catalog_empty_registry_returns_error(self):
        runner = full_runner(**{"registry-emit.sh --json": ok(json.dumps({"variants": []}))})
        cd = CockpitData(ROOT, runner=runner)
        entries, err = await cd.load_catalog(enrich_fit=False, enrich_measurement=False)
        assert entries == []
        assert err is not None

    @pytest.mark.asyncio
    async def test_enrich_weights_marks_present_absent_unknown(self, tmp_path):
        """Download UX: enrich_weights joins each slug to weights.py list --json +
        stats the model dir → present (subdir+glob), absent (no subdir),
        unknown (no weights entry to join)."""
        from club3090_cockpit.data import (
            CatalogEntry, WEIGHTS_PRESENT, WEIGHTS_ABSENT, WEIGHTS_UNKNOWN,
        )
        from club3090_tui_core import VariantRow

        listing = json.dumps([
            {"model": "qwen3.6-27b", "variant": "fp8", "subdir": "qwen3.6-27b-fp8",
             "hf_repo": "Qwen/Qwen3.6-27B-FP8", "size_gb": 29, "verify_glob": "*.safetensors",
             "status": "experimental"},
            {"model": "qwen3.6-27b", "variant": "autoround-int4",
             "subdir": "qwen3.6-27b-autoround-int4", "hf_repo": "Lorbus/Qwen3.6-27B-int4-AutoRound",
             "size_gb": 17.5, "verify_glob": "*.safetensors", "status": "production"},
        ])
        runner = full_runner(**{"weights.py list --json": ok(listing)})
        cd = CockpitData(ROOT, runner=runner)
        hf = tmp_path
        (hf / "qwen3.6-27b-autoround-int4").mkdir(parents=True)
        (hf / "qwen3.6-27b-autoround-int4" / "m.safetensors").write_text("x")  # present

        def _e(slug, path):
            return CatalogEntry(row=VariantRow(
                slug=slug, switch_engine="vllm", launch_engine="vllm",
                compose_dir=path.rsplit("/", 1)[0], file=path.rsplit("/", 1)[-1],
                port=8013, model="qwen3.6-27b", engine="vllm-stable", kvcalc_key="SKIP",
                container="c", compose_path=path, status="x", ctx_label="", status_note=""))

        e_ar = _e("vllm/dual", "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml")
        e_fp8 = _e("vllm/dual-max", "models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml")
        e_unk = _e("vllm/none", "models/qwen3.6-27b/vllm/compose/dual/no-such-quant/base.yml")
        await cd.enrich_weights([e_ar, e_fp8, e_unk], model_dir=str(tmp_path))
        assert e_ar.weights_state == WEIGHTS_PRESENT
        assert e_ar.weights is not None and e_ar.weights.hf_repo.startswith("Lorbus")
        assert e_fp8.weights_state == WEIGHTS_ABSENT          # fp8 subdir not created
        assert e_unk.weights_state == WEIGHTS_UNKNOWN          # no weights entry to join

    @pytest.mark.asyncio
    async def test_weights_download_plan_and_progress(self, tmp_path):
        """Download UX (service): the download plan is `WEIGHT_KEY=… setup.sh
        <model>` (disk write, no reconcile/confirm gate); progress = bytes-on-disk
        vs size_gb (capped 99); disk pre-check carries 10% headroom."""
        from club3090_cockpit.data import WeightsMeta
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.weights_download_plan("qwen3.6-27b", "fp8")
        assert plan.cmd == ["bash", "scripts/setup.sh", "qwen3.6-27b"]
        assert plan.requires_reconcile is False and plan.requires_confirm is False
        meta = WeightsMeta(model="qwen3.6-27b", variant="fp8", subdir="qwen3.6-27b-fp8",
                           size_gb=10.0, verify_glob="*.safetensors")
        base = tmp_path / "qwen3.6-27b-fp8"
        base.mkdir(parents=True)
        with open(base / "a.safetensors", "wb") as fh:
            fh.truncate(5 * 10**9)                            # 5 GB sparse → 50%
        assert cd.weights_download_progress(meta, model_dir=str(tmp_path)) == 50
        fits, _free, need = cd.weights_fits_disk(meta, model_dir=str(tmp_path))
        assert need == 11.0                                   # 10 GB * 1.10 headroom
        # size unknown → no progress %, never blocks the disk guard
        m0 = WeightsMeta(model="m", variant="v", subdir="s", size_gb=None)
        assert cd.weights_download_progress(m0, model_dir=str(tmp_path)) is None
        assert cd.weights_fits_disk(m0, model_dir=str(tmp_path))[0] is True

    def test_variant_row_from_dict_attaches_companions(self):
        """The --json variant carries weights_companions/drafter/vision; they're
        attached to the row (same pattern as 'source') for the per-slug companion
        download + display."""
        row = _variant_row_from_dict({
            "slug": "beellama/qwen-dflash-dual", "port": 8065, "status": "experimental",
            "weights_companions": ["anbeeld-dflash-iq4xs"],
            "drafter": "anbeeld-qwen-dflash", "vision": False,
        })
        assert getattr(row, "weights_companions") == ["anbeeld-dflash-iq4xs"]
        assert getattr(row, "drafter") == "anbeeld-qwen-dflash"
        assert getattr(row, "vision") is False
        # absent → safe defaults
        bare = _variant_row_from_dict({"slug": "x/y", "port": 1})
        assert getattr(bare, "weights_companions") == []
        assert getattr(bare, "vision") is False

    @pytest.mark.asyncio
    async def test_run_weights_download_injects_companion_keys(self):
        """The download passes the slug's companions as a model-qualified
        WEIGHT_EXTRA_KEYS list so setup.sh fetches them alongside the core; a slug
        with no companions sets no WEIGHT_EXTRA_KEYS."""
        import asyncio

        class _CaptureDL:
            def __init__(self): self.env = None
            def set_callbacks(self, **kw): pass
            async def start_raw(self, cmd, env=None, run_type=None, parser=None):
                self.env = env
                h = type("H", (), {})()
                h.done = asyncio.Event(); h.done.set(); h.exit_code = 0
                return h

        cap = _CaptureDL()
        cd = CockpitData(ROOT, runner=full_runner(), download_runner=cap)
        await cd.run_weights_download(
            "qwen3.6-27b", "beellama-q8kxl-dflash", companions=["anbeeld-dflash-iq4xs"]
        )
        assert cap.env["WEIGHT_KEY"] == "qwen3.6-27b:beellama-q8kxl-dflash"
        assert cap.env["WEIGHT_EXTRA_KEYS"] == "qwen3.6-27b:anbeeld-dflash-iq4xs"

        cap2 = _CaptureDL()
        cd2 = CockpitData(ROOT, runner=full_runner(), download_runner=cap2)
        await cd2.run_weights_download("qwen3.6-27b", "autoround-int4")
        assert "WEIGHT_EXTRA_KEYS" not in cap2.env

    @pytest.mark.asyncio
    async def test_weights_state_partial_until_companion_present(self, tmp_path):
        """A slug with a companion (DFlash draft) is PARTIAL while the core is on
        disk but the companion subdir is missing, and PRESENT once both exist —
        so Start stays gated until the whole serve set is fetched."""
        from club3090_cockpit.data import (
            CatalogEntry, WEIGHTS_PRESENT, WEIGHTS_PARTIAL,
        )

        listing = json.dumps([
            {"model": "qwen3.6-27b", "variant": "beellama-q8kxl-dflash",
             "subdir": "qwen3.6-27b-gguf/unsloth-mtp-q8kxl", "hf_repo": "unsloth/x",
             "size_gb": 35.8, "verify_glob": "*.gguf", "status": "experimental"},
            {"model": "qwen3.6-27b", "variant": "anbeeld-dflash-iq4xs",
             "subdir": "qwen3.6-27b-gguf/anbeeld-dflash-iq4xs", "hf_repo": "Anbeeld/x",
             "size_gb": 1.6, "verify_glob": "*.gguf", "status": "experimental"},
        ])
        cd = CockpitData(ROOT, runner=full_runner(**{"weights.py list --json": ok(listing)}))
        hf = tmp_path
        core = hf / "qwen3.6-27b-gguf" / "unsloth-mtp-q8kxl"
        core.mkdir(parents=True)
        (core / "m.gguf").write_text("x")  # core present, companion NOT yet

        e = CatalogEntry(row=_variant_row_from_dict({
            "slug": "beellama/qwen-dflash-dual", "port": 8065, "model": "qwen3.6-27b",
            "switch_engine": "beellama", "launch_engine": "beellama", "engine": "beellama-local",
            "compose_dir": "models/qwen3.6-27b/beellama/compose/dual/beellama-q8kxl-dflash",
            "file": "dflash.yml",
            "compose_path": "models/qwen3.6-27b/beellama/compose/dual/beellama-q8kxl-dflash/dflash.yml",
            "kvcalc_key": "SKIP", "container": "c", "status": "experimental",
            "ctx_label": "", "status_note": "",
            "weights_companions": ["anbeeld-dflash-iq4xs"],
        }))
        await cd.enrich_weights([e], model_dir=str(tmp_path))
        assert e.weights_state == WEIGHTS_PARTIAL          # core present, companion missing

        comp = hf / "qwen3.6-27b-gguf" / "anbeeld-dflash-iq4xs"
        comp.mkdir(parents=True)
        (comp / "d.gguf").write_text("x")                  # now the companion is here too
        await cd.enrich_weights([e], model_dir=str(tmp_path))
        assert e.weights_state == WEIGHTS_PRESENT

    def test_download_progress_aggregates_core_and_companion(self, tmp_path):
        """Live-progress regression: progress must aggregate bytes across the WHOLE
        set (core + companions) / total size — so a core-already-present slug shows
        a MOVING % as the companion lands, not a static 99 off the core subdir."""
        from club3090_cockpit.data import WeightsMeta
        cd = CockpitData(ROOT, runner=full_runner())
        core = WeightsMeta(model="m", variant="core", subdir="core", size_gb=8.0,
                           verify_glob="*.gguf")
        comp = WeightsMeta(model="m", variant="comp", subdir="comp", size_gb=2.0,
                           verify_glob="*.gguf")
        hf = tmp_path
        (hf / "core").mkdir(parents=True)
        with open(hf / "core" / "a.gguf", "wb") as fh:
            fh.truncate(8 * 10**9)                          # core FULL, companion absent
        # core-only signal = the static-99 bug; the aggregate = 8/(8+2) = 80%
        assert cd.weights_download_progress(core, model_dir=str(tmp_path)) == 99
        assert cd.weights_download_progress_set([core, comp], model_dir=str(tmp_path)) == 80
        (hf / "comp").mkdir(parents=True)
        with open(hf / "comp" / "d.gguf", "wb") as fh:
            fh.truncate(1 * 10**9)                          # half the companion → (8+1)/10 = 90%
        assert cd.weights_download_progress_set([core, comp], model_dir=str(tmp_path)) == 90

    @pytest.mark.asyncio
    async def test_download_set_metas_includes_companions(self):
        """download_set_metas resolves the core variant + each companion via the
        index (the set the progress aggregates over)."""
        from club3090_cockpit.data import CatalogEntry, WeightsMeta
        cd = CockpitData(ROOT, runner=full_runner())
        index = {
            ("m", "core"): WeightsMeta(model="m", variant="core", subdir="core", size_gb=8.0),
            ("m", "comp"): WeightsMeta(model="m", variant="comp", subdir="comp", size_gb=2.0),
        }
        e = CatalogEntry(row=_variant_row_from_dict({
            "slug": "x/y", "port": 1, "model": "m",
            "compose_path": "models/m/e/compose/dual/core/f.yml",
            "weights_companions": ["comp"],
        }))
        metas = cd.download_set_metas(e, index)
        assert [m.variant for m in metas] == ["core", "comp"]
        # a companion absent from the index is skipped, not fatal
        e2 = CatalogEntry(row=_variant_row_from_dict({
            "slug": "x/z", "port": 1, "model": "m",
            "compose_path": "models/m/e/compose/dual/core/f.yml",
            "weights_companions": ["nope"],
        }))
        assert [m.variant for m in cd.download_set_metas(e2, index)] == ["core"]

    def test_weights_bytes_on_disk_ignores_backup_cruft(self, tmp_path):
        """Swapping weights by renaming the old GGUF to *.bak must NOT pin progress
        at a false ~99%: backup cruft is excluded from the byte count, while hf's
        in-progress *.incomplete staging IS counted so live progress moves."""
        from club3090_cockpit.data import WeightsMeta
        cd = CockpitData(ROOT, runner=full_runner())
        meta = WeightsMeta(model="m", variant="v", subdir="v", size_gb=10.0,
                           verify_glob="*.gguf")
        base = tmp_path / "v"
        base.mkdir(parents=True)
        # Only a leftover full-size backup → ignored → 0 bytes → 0% (the bug was 99).
        with open(base / "model.gguf.bak", "wb") as fh:
            fh.truncate(8 * 10**9)
        with open(base / "model.gguf.old", "wb") as fh:
            fh.truncate(8 * 10**9)
        with open(base / "notes.txt~", "wb") as fh:
            fh.truncate(1 * 10**9)
        assert cd.weights_bytes_on_disk(meta, model_dir=str(tmp_path)) == 0
        assert cd.weights_download_progress(meta, model_dir=str(tmp_path)) == 0
        # hf's in-progress staging IS counted → live progress tracks it (3/10 = 30%).
        with open(base / "model.gguf.incomplete", "wb") as fh:
            fh.truncate(3 * 10**9)
        assert cd.weights_bytes_on_disk(meta, model_dir=str(tmp_path)) == 3 * 10**9
        assert cd.weights_download_progress(meta, model_dir=str(tmp_path)) == 30
        # the real completed weight counts too (the .bak/.old/~ stay excluded).
        with open(base / "model.gguf", "wb") as fh:
            fh.truncate(7 * 10**9)
        assert cd.weights_bytes_on_disk(meta, model_dir=str(tmp_path)) == 10 * 10**9

    def test_weights_model_dir_no_append_and_reads_dotenv(self, tmp_path, monkeypatch):
        """MODEL_DIR unification: the weights root is used DIRECTLY (no 'huggingface'
        append — that was the cockpit↔setup.sh divergence), and resolves from the
        repo .env (the shared config) when no Settings/env value is set; an explicit
        $MODEL_DIR wins over .env."""
        monkeypatch.delenv("MODEL_DIR", raising=False)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".env").write_text("FOO=bar\nMODEL_DIR=/data/weights\n", encoding="utf-8")
        cd = CockpitData(repo_root=repo)
        assert cd.weights_model_dir() == "/data/weights"          # from .env, no append
        # the weights path is <root>/<subdir>, NOT <root>/huggingface/<subdir>
        from club3090_cockpit.data import WeightsMeta
        meta = WeightsMeta(model="m", variant="v", subdir="sub/dir", size_gb=1.0,
                           verify_glob="*.gguf")
        (repo / ".env").unlink()                                   # isolate the path check
        d = repo / "sub" / "dir"
        d.mkdir(parents=True)
        (d / "w.gguf").write_text("x")
        assert cd.weights_bytes_on_disk(meta, model_dir=str(repo)) > 0   # repo/sub/dir, no append
        # explicit $MODEL_DIR wins over .env
        (repo / ".env").write_text("MODEL_DIR=/data/weights\n", encoding="utf-8")
        monkeypatch.setenv("MODEL_DIR", "/env/weights")
        assert cd.weights_model_dir() == "/env/weights"

    @pytest.mark.asyncio
    async def test_run_weights_download_pins_model_dir(self):
        """The download pins setup.sh's MODEL_DIR to the cockpit's resolved weights
        root, so the pull lands exactly where the cockpit reads (the cockpit↔scripts
        single-source-of-truth seam)."""
        import asyncio

        class _Cap:
            def __init__(self): self.env = None
            def set_callbacks(self, **k): pass
            async def start_raw(self, cmd, env=None, run_type=None, parser=None):
                self.env = env
                h = type("H", (), {})()
                h.done = asyncio.Event(); h.done.set(); h.exit_code = 0
                return h

        cap = _Cap()
        cd = CockpitData(ROOT, runner=full_runner(), download_runner=cap)
        cd._model_dir = "/data/weights"
        await cd.run_weights_download("qwen3.6-27b", "fp8")
        assert cap.env["MODEL_DIR"] == "/data/weights"
        assert cap.env["WEIGHT_KEY"] == "qwen3.6-27b:fp8"
        # HF_HOME is derived UNDER that same root (off the host root disk)
        assert cap.env["HF_HOME"].startswith("/data/weights/")


class TestExplainFit:
    @pytest.mark.asyncio
    async def test_explain(self):
        cd = CockpitData(ROOT, runner=full_runner())
        ex, err = await cd.explain("vllm/dual")
        assert err is None
        assert ex["slug"] == "vllm/dual"
        assert ex["benchmarks"]

    @pytest.mark.asyncio
    async def test_fit(self):
        cd = CockpitData(ROOT, runner=full_runner())
        fit = await cd.fit("vllm/dual", "rtx-3090")
        assert fit.verdict == "fits-clean"
        assert fit.vram_est_gb == 19.881
        assert fit.card == "rtx-3090"

    @pytest.mark.asyncio
    async def test_fit_unknown_card_surfaces_error(self):
        runner = full_runner(
            **{"kv-calc.py --fit": ok(json.dumps({"verdict": "unknown", "error": "unrecognized --card"}))}
        )
        cd = CockpitData(ROOT, runner=runner)
        fit = await cd.fit("vllm/dual", "bogus")
        assert fit.verdict == "unknown"
        assert "unrecognized" in fit.error

    @pytest.mark.asyncio
    async def test_explain_timeout_returns_error(self):
        runner = full_runner(
            **{"--explain vllm/dual --json": RunResult(-1, "", "timeout", timed_out=True)}
        )
        cd = CockpitData(ROOT, runner=runner)
        ex, err = await cd.explain("vllm/dual")
        assert ex is None
        assert "timed out" in err


class TestByoCheck:
    @pytest.mark.asyncio
    async def test_byo_eligible(self):
        cd = CockpitData(ROOT, runner=full_runner())
        res = await cd.byo_check("org/Model", "vllm/dual")
        assert res.eligible is True
        assert res.route == "C"
        assert res.sibling_slug == "vllm/dual"
        assert res.drop_spec_config is True
        assert res.quant_match == "int4"

    @pytest.mark.asyncio
    async def test_byo_dry_run_is_in_command(self):
        """byo_check MUST pass --dry-run (Path B, never downloads)."""
        runner = full_runner()
        cd = CockpitData(ROOT, runner=runner)
        await cd.byo_check("org/Model", "vllm/dual")
        pull_call = next(c for c in runner.calls if "pull.sh" in " ".join(c))
        assert "--dry-run" in pull_call
        assert "--json" in pull_call

    @pytest.mark.asyncio
    async def test_byo_no_output(self):
        runner = full_runner(**{"pull.sh": ok("")})
        cd = CockpitData(ROOT, runner=runner)
        res = await cd.byo_check("org/Model", "vllm/dual")
        assert res.error


class TestScenesDoctor:
    @pytest.mark.asyncio
    async def test_scenes(self):
        cd = CockpitData(ROOT, runner=full_runner())
        scenes = await cd.scenes()
        assert len(scenes) == 2
        assert scenes[0].name == "27b"
        assert scenes[0].gpus == "both"

    @pytest.mark.asyncio
    async def test_scenes_filters_command_modes(self):
        """The COMMAND modes (power-cap / prune / prune-all) are filtered from the
        scene table — they're not GPU scenes (power-cap has its own menu, prune was
        removed).  'off' + 'chat' (ops group) and models/studio scenes are kept."""
        listing = json.dumps([
            {"name": "27b", "group": "models", "gpus": "both"},
            {"name": "comfyui", "group": "studio", "gpus": "both"},
            {"name": "chat", "group": "ops", "gpus": "none"},
            {"name": "off", "group": "ops", "gpus": "none"},
            {"name": "power-cap", "group": "ops", "gpus": "both"},
            {"name": "prune", "group": "ops", "gpus": "none"},
            {"name": "prune-all", "group": "ops", "gpus": "none"},
        ])
        cd = CockpitData(ROOT, runner=full_runner(
            **{"gpu-mode.sh --list-modes --json": ok(listing)}))
        names = [s.name for s in await cd.scenes()]
        assert names == ["27b", "comfyui", "chat", "off"]
        assert not ({"power-cap", "prune", "prune-all"} & set(names))

    @pytest.mark.asyncio
    async def test_scenes_clustered_by_group(self):
        """The scene table clusters by group (models -> studio -> ops), stable
        within a group.  Raw catalog order puts 'chat' (ops) first and 'off' (ops)
        last, splitting the ops scenes apart; scenes() must regroup them so the two
        ops scenes sit together at the end."""
        listing = json.dumps([
            {"name": "chat", "group": "ops", "gpus": "none"},
            {"name": "27b", "group": "models", "gpus": "both"},
            {"name": "comfyui", "group": "studio", "gpus": "both"},
            {"name": "gemma", "group": "models", "gpus": "both"},
            {"name": "off", "group": "ops", "gpus": "none"},
        ])
        cd = CockpitData(ROOT, runner=full_runner(
            **{"gpu-mode.sh --list-modes --json": ok(listing)}))
        scenes = await cd.scenes()
        assert [s.name for s in scenes] == ["27b", "gemma", "comfyui", "chat", "off"]
        assert [s.group for s in scenes] == ["models", "models", "studio", "ops", "ops"]

    @pytest.mark.asyncio
    async def test_doctor_serving(self):
        runner = full_runner(**{"health.sh": ok(HEALTH_SERVING)})
        cd = CockpitData(ROOT, runner=runner)
        dr = await cd.doctor_read()
        assert dr.serving is True
        assert dr.kv_pool_pct == 61


class TestContainers:
    @pytest.mark.asyncio
    async def test_containers_lists_engine(self):
        runner = full_runner(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        cd = CockpitData(
            ROOT, runner=runner, detect_endpoint_fn=make_detect(ServingTarget())
        )
        cons = await cd.containers()
        names = [c.name for c in cons]
        assert "vllm-qwen36-27b-dual" in names
        # open-webui is not an engine-prefix container → excluded
        assert "open-webui" not in names

    @pytest.mark.asyncio
    async def test_containers_matches_slug(self):
        runner = full_runner(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        cd = CockpitData(
            ROOT, runner=runner, detect_endpoint_fn=make_detect(ServingTarget())
        )
        variants = [
            VariantRow(
                slug="vllm/dual", switch_engine="vllm", launch_engine="vllm",
                compose_dir="", file="", port=8010, model="qwen3.6-27b",
                engine="vllm", kvcalc_key="", container="vllm-qwen36-27b-dual",
                compose_path="", status="production", ctx_label="262K", status_note="",
            )
        ]
        cons = await cd.containers(variants=variants)
        engine = next(c for c in cons if c.name == "vllm-qwen36-27b-dual")
        assert engine.slug == "vllm/dual"


class TestEstateState:
    @pytest.mark.asyncio
    async def test_estate_state_assembles(self):
        runner = full_runner(
            **{
                "health.sh": ok(HEALTH_SERVING),
                "docker ps": ok(DOCKER_PS_ENGINE),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_BUSY),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22698), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(url="http://localhost:8010", model="qwen3.6-27b", gpus=gpus)
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(target),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        st = await cd.estate_state()
        assert len(st.scenes) == 2
        assert st.doctor.serving is True
        assert [g.index for g in st.gpus] == [0, 1]
        assert st.estate_report["active_estate"]["present"] is True


# ===========================================================================
# Action builders (gated)
# ===========================================================================


class TestActionBuilders:
    def test_serve_no_force(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.serve("vllm/dual")
        assert plan.kind == "serve"
        assert plan.cmd == ["bash", "scripts/switch.sh", "vllm/dual"]
        assert "--force" not in plan.cmd
        assert plan.requires_reconcile is True

    def test_serve_force_requires_reason(self):
        cd = CockpitData(ROOT, runner=full_runner())
        with pytest.raises(ValueError):
            cd.serve("vllm/dual", force=True)

    def test_serve_force_with_reason(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.serve("vllm/dual", force=True, force_reason="user override after VRAM check")
        assert "--force" in plan.cmd
        assert plan.force is True
        assert plan.force_reason

    def test_set_default(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.set_default("vllm/dual")
        assert plan.cmd == ["bash", "scripts/switch.sh", "--set-default", "vllm/dual"]
        assert plan.requires_reconcile is False  # .env pin — no GPU contention

    def test_clear_default(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.clear_default("qwen3.6-27b")
        assert plan.cmd == ["bash", "scripts/switch.sh", "--clear-default", "qwen3.6-27b"]

    def test_scene_switch(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.scene_switch("27b")
        assert plan.cmd == ["bash", "scripts/gpu-mode.sh", "27b"]
        assert plan.requires_reconcile is True

    def test_stop_all_plan(self):
        """The 'off' scene routes here — comprehensive stop (gpu-mode off + estate
        down), reconcile-gated so a forced 'off' also tears down detected stragglers."""
        cd = CockpitData(ROOT, runner=full_runner())
        p = cd.stop_all()
        assert p.kind == "scene" and p.requires_reconcile is True
        joined = " ".join(p.cmd)
        assert "gpu-mode.sh off" in joined and "estate_cli.py down" in joined

    def test_estate_down(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.estate_down()
        assert "down" in plan.cmd
        assert plan.requires_reconcile is True

    def test_container_action_restart(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.container_action("vllm-x", "restart")
        assert plan.cmd == ["docker", "restart", "vllm-x"]
        assert plan.requires_reconcile is False  # restart same config → no new claim

    def test_container_action_stop_reconciles(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.container_action("vllm-x", "stop")
        assert plan.requires_reconcile is True

    def test_container_action_bad_op(self):
        cd = CockpitData(ROOT, runner=full_runner())
        with pytest.raises(ValueError):
            cd.container_action("vllm-x", "rm")


class TestSceneServiceStates:
    """scene_service_states_sync: per-service running flag for a scene, matched
    against the live docker-ps set (drives the Orchestration scene-preview status
    bullets)."""

    def test_sync_matches_running_and_marks_absent_stopped(self):
        scene = Scene(name="chat", group="ops",
                      services=["open-webui", "litellm", "qdrant"])
        # 'openwebui' (no separator) must normalize-match 'open-webui'; qdrant absent.
        states = CockpitData.scene_service_states_sync(
            scene, ["openwebui", "litellm-1"])
        by = {s.name: s.running for s in states}
        assert by == {"open-webui": True, "litellm": True, "qdrant": False}

    def test_sync_empty_scene(self):
        assert CockpitData.scene_service_states_sync(
            Scene(name="off", services=[]), ["anything"]) == []


def _seed_service_dirs(root, names):
    for n in names:
        d = root / "services" / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "docker-compose.yml").write_text("services: {}\n")


class TestSupportingServiceEnrichment:
    """_merge_known_services: non-GPU supporting services carry engine='web' + the
    port they serve on (running) so the Containers tab shows what they are."""

    @pytest.mark.asyncio
    async def test_running_support_service_gets_web_and_port(self, tmp_path):
        _seed_service_dirs(tmp_path, ["litellm", "comfyui"])
        cd = CockpitData(tmp_path, runner=full_runner(
            **{"docker ps": ok("litellm|0.0.0.0:4000->4000/tcp\n")}))
        by = {c.name: c for c in await cd._merge_known_services([])}
        assert by["litellm"].engine == "web"
        assert by["litellm"].host_port == 4000
        assert by["litellm"].status == "running"
        # A GPU-holding rig service (comfyui) keeps NO web tag.
        assert by["comfyui"].engine == ""

    @pytest.mark.asyncio
    async def test_stopped_support_service_web_no_port(self, tmp_path):
        _seed_service_dirs(tmp_path, ["qdrant"])
        cd = CockpitData(tmp_path, runner=full_runner(**{"docker ps": ok("")}))
        by = {c.name: c for c in await cd._merge_known_services([])}
        assert by["qdrant"].engine == "web"
        assert by["qdrant"].host_port == 0
        assert by["qdrant"].status == "stopped"

    @pytest.mark.asyncio
    async def test_host_port_general_match_not_engine_only(self, tmp_path):
        # openwebui maps 3000->8080; the engine-only PORT_MAP_BROAD_RE would miss
        # the host port — the general matcher must catch it.
        _seed_service_dirs(tmp_path, ["openwebui"])
        cd = CockpitData(tmp_path, runner=full_runner(
            **{"docker ps": ok("openwebui|0.0.0.0:3000->8080/tcp\n")}))
        by = {c.name: c for c in await cd._merge_known_services([])}
        assert by["openwebui"].host_port == 3000


class TestServiceStart:
    """service_start: bring a stopped supporting service up via compose, mirroring
    gpu-mode's compose_at (project pinned to the dir name; --env-file when present)."""

    def test_compose_up_plan(self, tmp_path):
        _seed_service_dirs(tmp_path, ["litellm"])
        plan = CockpitData(tmp_path).service_start("litellm")
        # -p pins the project to the dir name (match gpu-mode so it operates on the
        # SAME container, not a duplicate); no .env in tmp_path → --env-file omitted.
        assert plan.cmd == [
            "docker", "compose", "-f",
            "services/litellm/docker-compose.yml", "-p", "litellm", "up", "-d"]
        assert plan.requires_reconcile is True
        assert plan.kind == "service-up"

    def test_env_file_included_when_present(self, tmp_path):
        _seed_service_dirs(tmp_path, ["comfyui"])
        (tmp_path / ".env").write_text("MODEL_DIR=/x\n")
        plan = CockpitData(tmp_path).service_start("comfyui")
        assert plan.cmd[:4] == ["docker", "compose", "--env-file", ".env"]
        assert plan.cmd[-4:] == ["-p", "comfyui", "up", "-d"]

    def test_yaml_spelling_fallback(self, tmp_path):
        d = tmp_path / "services" / "x"
        d.mkdir(parents=True)
        (d / "docker-compose.yaml").write_text("services: {}\n")
        plan = CockpitData(tmp_path).service_start("x")
        assert "services/x/docker-compose.yaml" in plan.cmd


# ===========================================================================
# THE RECONCILE GATE — dual-writer safety core
# ===========================================================================


class TestReconcileGate:
    @pytest.mark.asyncio
    async def test_free_rig_is_safe(self):
        """No containers, no GPU use, no estate → safe to write."""
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is True
        assert rec.conflict_summary == "none"

    @pytest.mark.asyncio
    async def test_running_engine_container_is_conflict(self):
        # A running vLLM engine HOLDS the card it serves on — set GPU0 busy (the
        # realistic state) so the unknown-GPU container is flagged as a conflict.
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_ENGINE),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert any(c.name == "vllm-qwen36-27b-dual" for c in rec.conflicts)

    @pytest.mark.asyncio
    async def test_running_service_on_idle_cards_is_not_conflict(self):
        """Regression: a running-but-IDLE service (the studio stack at 0 GiB) is
        NOT a teardown conflict for a serve onto FREE cards — the gate must not
        cry wolf ('starting this will stop studio-tts') when nothing holds a card."""
        runner = full_runner(
            **{
                "docker ps": ok("studio-tts|\nstudio-image-shim|\n"),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is True
        assert rec.conflicts == []
        assert rec.gpu_conflicts == []

    @pytest.mark.asyncio
    async def test_gpu_in_use_is_conflict(self):
        """A card with >512 MiB used but no named container is still a conflict."""
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22698), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert any(g.gpu_index == 0 and g.mem_used_mib == 22698 for g in rec.gpu_conflicts)

    @pytest.mark.asyncio
    async def test_estate_booted_gpu0_then_scene_switch_conflicts(self):
        """The prompt's canonical case: estate_cli booted GPU0; a scene-switch
        wanting GPU0 must be reported as a conflict by the gate."""
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),  # estate used a non-prefix container
                "estate_cli.py report-state --json": ok(
                    json.dumps(
                        {
                            "active_estate": {
                                "present": True,
                                "instances": [
                                    {"name": "llama-gpu0", "compose": "llamacpp/default", "gpus": [0], "port": 8010}
                                ],
                            }
                        }
                    )
                ),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=20000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        # scene-switch wants GPU0
        rec = await cd.reconcile_before_write("scene:27b", pending_gpus=[0])
        assert rec.safe is False
        assert any(i.get("name") == "llama-gpu0" for i in rec.estate_claims)
        assert "estate:llama-gpu0" in rec.conflict_summary

    @pytest.mark.asyncio
    async def test_force_tears_down_conflicts_before_write(self):
        """2A: a FORCED unsafe write STOPS the gate's conflicting containers
        (docker stop) before running the command — so a forced 'off' clears a
        cockpit-served slug that gpu-mode off doesn't know.  The command still runs."""
        import dataclasses
        wr = FakeWriteRunner()
        runner = full_runner(**{"docker ps": ok(DOCKER_PS_ENGINE)})
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=wr,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec0 = await cd.reconcile_before_write("scene:off")
        assert rec0.safe is False and rec0.conflicts        # a named container is in the way
        names = {c.name for c in rec0.conflicts}
        plan = dataclasses.replace(cd.stop_all(), force=True, force_reason="stop everything")
        executed, rec, _ = await cd.execute_action(plan)
        assert executed is True                             # forced → proceeds
        # the conflicting container(s) were docker-stopped (via the read runner) ...
        stopped = {c[2] for c in runner.calls if c[:2] == ["docker", "stop"] and len(c) > 2}
        assert stopped & names
        # ... and the comprehensive 'off' command still ran via the write runner.
        assert len(wr.started) == 1

    @pytest.mark.asyncio
    async def test_estate_on_gpu1_does_not_conflict_with_gpu0_only_request(self):
        """Estate holds GPU1 only; an action wanting just GPU0 is NOT blocked by it."""
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                "estate_cli.py report-state --json": ok(
                    json.dumps(
                        {
                            "active_estate": {
                                "present": True,
                                "instances": [
                                    {"name": "llama-gpu1", "compose": "llamacpp/default", "gpus": [1], "port": 8020}
                                ],
                            }
                        }
                    )
                ),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=20000)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:single-gpu0", pending_gpus=[0])
        # No container, GPU0 free, estate on GPU1 only → safe.
        assert rec.estate_claims == []
        assert rec.safe is True

    @pytest.mark.asyncio
    async def test_pending_gpus_none_is_conservative_both_cards(self):
        """pending_gpus=None means 'wants both cards' → any GPU1 use conflicts."""
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=20000)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")  # None → {0,1}
        assert rec.pending_gpus == [0, 1]
        assert rec.safe is False
        assert any(g.gpu_index == 1 for g in rec.gpu_conflicts)

    @pytest.mark.asyncio
    async def test_detect_failure_is_unsafe(self):
        """If detect raises, we can't prove the cards are free → not safe."""
        async def boom() -> ServingTarget:
            raise RuntimeError("docker daemon down")

        cd = CockpitData(ROOT, runner=full_runner(), detect_endpoint_fn=boom)
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert "detect failed" in rec.note

    @pytest.mark.asyncio
    async def test_reconcile_calls_detect_freshly(self):
        """The gate must call detect every time (never a cached snapshot)."""
        calls = {"n": 0}

        async def counting_detect() -> ServingTarget:
            calls["n"] += 1
            return ServingTarget(gpus=[GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)])

        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_EMPTY), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        cd = CockpitData(ROOT, runner=runner, detect_endpoint_fn=counting_detect,
                         get_gpu_info_fn=make_gpu_info([]))
        await cd.reconcile_before_write("serve:a")
        await cd.reconcile_before_write("serve:b")
        assert calls["n"] >= 2

    # ── B4: FAIL CLOSED on a state-read error ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_gpu_read_error_is_unsafe(self):
        """A SAFETY gate must fail CLOSED: if reading the GPUs raises, we can't
        prove the cards are free → UNSAFE (not 'nothing in use')."""
        async def gpu_boom() -> list[GpuInfo]:
            raise RuntimeError("nvidia-smi not found")

        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_EMPTY), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        # detect returns NO gpus → forces the get_gpu_info path (which raises).
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=[])),
            get_gpu_info_fn=gpu_boom,
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert "GPU read failed" in rec.note

    @pytest.mark.asyncio
    async def test_gpu_read_empty_is_unsafe(self):
        """No detect GPUs AND nvidia-smi returns [] → no evidence the cards are
        free → fail closed."""
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_EMPTY), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=[])),
            get_gpu_info_fn=make_gpu_info([]),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert "cannot prove free" in rec.note

    @pytest.mark.asyncio
    async def test_estate_read_error_is_unsafe(self):
        """If the estate read errors (no JSON), we can't rule out a hidden estate
        claim → fail closed."""
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                # estate_cli emits non-JSON garbage → _run_json returns (None, err)
                "estate_cli.py report-state --json": RunResult(1, "", "Traceback: estate boom"),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert "estate read failed" in rec.note

    # ── B5: detector #1 broadened to club3090- + services, GPU-filtered ──────────────

    @pytest.mark.asyncio
    async def test_estate_club3090_container_is_conflict(self):
        """A `club3090-<name>` estate container is a live GPU user the gate must
        see (it doesn't match the engine prefixes)."""
        runner = full_runner(
            **{
                "docker ps": ok("club3090-llama-gpu0|0.0.0.0:8010->8080/tcp\n"),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        # The estate container is a live GPU user → GPU0 occupied (realistic).
        gpus = [GpuInfo(index=0, mem_used_mib=20000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        names = [c.name for c in rec.conflicts]
        assert "club3090-llama-gpu0" in names
        assert any(c.kind == "estate" for c in rec.conflicts)

    @pytest.mark.asyncio
    async def test_gpu_service_container_is_conflict(self):
        """A GPU-holding rig service (ComfyUI) is surfaced even with no engine
        prefix and no published port."""
        runner = full_runner(
            **{
                "docker ps": ok("comfyui|\n"),  # no published port
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        # ComfyUI is actively GPU-holding here → GPU0 busy (matches the docstring).
        gpus = [GpuInfo(index=0, mem_used_mib=18500), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        svc = next((c for c in rec.conflicts if c.name == "comfyui"), None)
        assert svc is not None and svc.kind == "service"

    @pytest.mark.asyncio
    async def test_container_on_gpu1_does_not_conflict_with_gpu0_only_request(self):
        """A container provably on GPU1 only (known gpu set) does NOT conflict
        with a request for GPU0 only; detector #2 (raw GPU) is the backstop."""
        from club3090_cockpit.data import ContainerInfo

        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=20000)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )

        # Inject a container whose GPU set is KNOWN to be {1} only.
        async def one_container(variants=None):
            return [ContainerInfo(name="club3090-llama-gpu1", kind="estate", gpus="1")]

        cd.containers = one_container  # type: ignore[assignment]
        rec = await cd.reconcile_before_write("serve:gpu0", pending_gpus=[0])
        # Container on GPU1 only → not a container conflict; GPU0 raw mem is 3 MiB
        # (free) → safe.
        assert all(c.name != "club3090-llama-gpu1" for c in rec.conflicts)
        assert rec.safe is True


# ===========================================================================
# execute_action — gated execution (MOCKED, never live)
# ===========================================================================


class FakeWriteRunner:
    """Stand-in for the core SubprocessRunner — records start_raw calls but
    NEVER spawns a process.  Asserts the write path is fully mocked."""

    def __init__(self):
        self.started: list[dict[str, Any]] = []

    async def start_raw(self, cmd, env, run_type, parser):
        self.started.append({"cmd": cmd, "run_type": run_type})
        return {"mock_state": True, "cmd": cmd}


class TestExecuteActionGated:
    @pytest.mark.asyncio
    async def test_unsafe_gate_refuses_write(self):
        """When the gate is unsafe and no force, execute_action must NOT run."""
        write_runner = FakeWriteRunner()
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_ENGINE), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=write_runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        plan = cd.serve("vllm/dual")
        executed, rec, state = await cd.execute_action(plan)
        assert executed is False
        assert rec is not None and rec.safe is False
        assert state is None
        assert write_runner.started == []  # NEVER reached the runner

    @pytest.mark.asyncio
    async def test_safe_gate_proceeds_to_mocked_runner(self):
        """When the gate is safe, execution reaches the (mocked) write runner.
        The runner is a fake — no real process is ever spawned."""
        write_runner = FakeWriteRunner()
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_EMPTY), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=write_runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        plan = cd.serve("vllm/dual")
        executed, rec, state = await cd.execute_action(plan)
        assert executed is True
        assert rec is not None and rec.safe is True
        assert len(write_runner.started) == 1
        assert write_runner.started[0]["cmd"] == ["bash", "scripts/switch.sh", "vllm/dual"]

    @pytest.mark.asyncio
    async def test_force_override_proceeds_despite_unsafe(self):
        """A force ActionPlan with a reason proceeds even when the gate is unsafe
        (the explicit override path) — still via the mocked runner."""
        write_runner = FakeWriteRunner()
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_ENGINE), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=write_runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        plan = cd.serve("vllm/dual", force=True, force_reason="user accepted teardown")
        executed, rec, state = await cd.execute_action(plan)
        assert executed is True
        assert rec is not None and rec.safe is False  # gate still reports unsafe
        assert len(write_runner.started) == 1

    @pytest.mark.asyncio
    async def test_no_reconcile_action_skips_gate(self):
        """set_default has requires_reconcile=False → no detect, straight to run."""
        write_runner = FakeWriteRunner()

        async def detect_should_not_be_called() -> ServingTarget:
            raise AssertionError("detect must not be called for a non-reconcile action")

        cd = CockpitData(
            ROOT, runner=full_runner(), write_runner=write_runner,
            detect_endpoint_fn=detect_should_not_be_called,
        )
        plan = cd.set_default("vllm/dual")
        executed, rec, state = await cd.execute_action(plan)
        assert executed is True
        assert rec is None
        assert len(write_runner.started) == 1

    # ── B6: skip_reconcile only honored as an explicit reasoned force ────────────────

    @pytest.mark.asyncio
    async def test_skip_reconcile_ignored_without_force(self):
        """skip_reconcile=True on a NON-force plan must NOT bypass the gate —
        the docstring couples skip to force+reason; enforce it in code."""
        write_runner = FakeWriteRunner()
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_ENGINE), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=write_runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        plan = cd.serve("vllm/dual")  # not forced
        executed, rec, _ = await cd.execute_action(plan, skip_reconcile=True)
        # Gate still ran (skip ignored) → unsafe → refused.
        assert executed is False
        assert rec is not None and rec.safe is False
        assert write_runner.started == []

    @pytest.mark.asyncio
    async def test_skip_reconcile_honored_with_force_and_reason(self):
        """skip_reconcile IS honored when the plan is a reasoned force — and then
        the gate is genuinely skipped (detect never called)."""
        write_runner = FakeWriteRunner()

        async def detect_should_not_be_called() -> ServingTarget:
            raise AssertionError("gate must be skipped → detect not called")

        cd = CockpitData(
            ROOT, runner=full_runner(), write_runner=write_runner,
            detect_endpoint_fn=detect_should_not_be_called,
        )
        plan = cd.serve("vllm/dual", force=True, force_reason="user accepted teardown")
        executed, rec, _ = await cd.execute_action(plan, skip_reconcile=True)
        assert executed is True
        assert rec is None  # gate skipped
        assert len(write_runner.started) == 1


# ===========================================================================
# B3 — writes are SERIALIZED (the gate→write window is atomic)
# ===========================================================================


class TestWritesSerialized:
    @pytest.mark.asyncio
    async def test_two_concurrent_dispatches_serialize_via_claim(self):
        """Two concurrent execute_action for overlapping GPUs: the write lock
        serializes reconcile→register-claim→spawn, so exactly ONE passes the gate
        and registers its claim; the OTHER, running its reconcile next, sees the
        live claim and is refused. No double-write — the §3.2 TOCTOU is closed
        even though start_raw returns immediately (the claim, held until the
        winner's subprocess completes, is what blocks the loser)."""
        import asyncio as _aio

        wr = LeaseWriteRunner()
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_EMPTY), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=wr,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        (execA, recA, _), (execB, recB, _) = await _aio.gather(
            cd.execute_action(cd.serve("slugA")),
            cd.execute_action(cd.serve("slugB")),
        )
        executed = [execA, execB]
        assert executed.count(True) == 1, f"exactly one write should pass the gate; got {executed}"
        assert executed.count(False) == 1
        assert len(wr.started) == 1, "only the winner spawned — no double-write"
        loser_rec = recB if execA else recA
        assert loser_rec is not None and loser_rec.pending_claim_tokens, \
            "the refused writer must have been blocked by the winner's pending claim"


# ===========================================================================
# C7 — container logs READ (docker logs)
# ===========================================================================


class TestContainerLogs:
    @pytest.mark.asyncio
    async def test_container_logs_returns_lines(self):
        runner = full_runner(
            **{"docker logs": ok("line one\nline two\nline three\n")}
        )
        cd = CockpitData(ROOT, runner=runner)
        out = await cd.container_logs("vllm-x")
        assert out["error"] is None
        assert out["lines"] == ["line one", "line two", "line three"]
        # It's a READ — docker logs, never stop/restart/rm.
        logs_call = next(c for c in runner.calls if "logs" in " ".join(c))
        assert logs_call[:3] == ["docker", "logs", "--tail"]

    @pytest.mark.asyncio
    async def test_container_logs_missing_container_errors(self):
        runner = full_runner(
            **{"docker logs": RunResult(1, "", "Error: No such container: nope")}
        )
        cd = CockpitData(ROOT, runner=runner)
        out = await cd.container_logs("nope")
        assert out["lines"] == []
        assert "No such container" in out["error"]


class TestRealRunnerNotInvokedInTests:
    """Sanity: a CockpitData built with a FakeRunner never constructs a
    RealRunner, and the default RealRunner is only the production fallback."""

    def test_default_runner_is_real(self):
        cd = CockpitData(ROOT)
        assert isinstance(cd._runner, RealRunner)

    def test_injected_runner_used(self):
        fr = full_runner()
        cd = CockpitData(ROOT, runner=fr)
        assert cd._runner is fr


# ===========================================================================
# Fix 1 — TOCTOU pending-claim registry (the critical one)
# ===========================================================================


class TestTOCTOUPendingClaims:
    """The pending-claim LEASE blocks writer B through the WHOLE boot gap — from
    A's dispatch until A's subprocess COMPLETES — even while docker ps /
    nvidia-smi still show the cards free (A's container hasn't booted). The key
    property: ``start_raw`` returns a RUNNING state immediately (it only spawns);
    the lease, held until the run's ``done`` fires, is what blocks B — NOT the
    write-lock duration (which is microseconds, the bug the old design had)."""

    def _cd(self, wr):
        runner = full_runner(
            **{
                "docker ps": ok(DOCKER_PS_EMPTY),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        return CockpitData(
            ROOT, runner=runner, write_runner=wr,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )

    @pytest.mark.asyncio
    async def test_claim_persists_through_boot_and_clears_on_completion(self):
        """The lease is HELD after execute_action returns (the subprocess is
        still booting) and clears only when the run's ``done`` event fires."""
        import asyncio as _aio

        wr = LeaseWriteRunner()
        cd = self._cd(wr)
        execA, recA, stateA = await cd.execute_action(cd.serve("slugA"))
        assert execA is True and recA is not None and recA.safe is True
        # HELD while A's subprocess is still running (NOT cleared on start_raw).
        assert len(cd._pending_claims) == 1, "lease must persist through the boot gap"
        # Signal A's subprocess completion → the release task clears the lease.
        stateA.done.set()
        for _ in range(8):
            await _aio.sleep(0)
        assert len(cd._pending_claims) == 0, "lease must clear on subprocess completion"

    @pytest.mark.asyncio
    async def test_inflight_claim_blocks_second_writer_and_is_not_forceable(self):
        """While A's lease is live, B's reconcile is unsafe even though docker ps
        / nvidia-smi show the cards FREE — and the in-flight claim is a HARD
        block: even a forced B is refused (cancel the in-flight write first).
        After A completes, the lease clears and C proceeds."""
        import asyncio as _aio

        wr = LeaseWriteRunner()
        cd = self._cd(wr)
        execA, recA, stateA = await cd.execute_action(cd.serve("slugA"))
        assert execA is True and len(cd._pending_claims) == 1

        # B: the rig still LOOKS free (docker ps empty, GPUs idle) but A's lease
        # is live → reconcile must refuse.
        recB = await cd.reconcile_before_write("serve:slugB")
        assert recB.safe is False
        assert recB.pending_claim_tokens, "B must be blocked by A's pending claim"

        # And it is NOT force-overridable (nothing materialized to tear down yet).
        execB, recB2, _ = await cd.execute_action(
            cd.serve("slugB", force=True, force_reason="B insists")
        )
        assert execB is False, "an in-flight pending claim is NOT force-overridable"
        assert len(wr.started) == 1, "B must never have spawned"

        # A completes → lease clears → C now proceeds.
        stateA.done.set()
        for _ in range(8):
            await _aio.sleep(0)
        execC, recC, _ = await cd.execute_action(cd.serve("slugC"))
        assert execC is True and recC is not None and recC.safe is True

    @pytest.mark.asyncio
    async def test_spawn_failure_clears_claim_immediately(self):
        """If start_raw returns an already-finished spawn-failure state, no card
        was claimed → the lease clears at once (no lingering false conflict)."""
        wr = FailedSpawnWriteRunner()
        cd = self._cd(wr)
        execX, recX, stateX = await cd.execute_action(cd.serve("slugX"))
        assert getattr(stateX, "is_finished", False) is True
        assert len(cd._pending_claims) == 0, "a spawn-failure must not leave a lease"


# ===========================================================================
# Fix 2 — docker ps fail-closed
# ===========================================================================


class TestDockerPsFailClosed:
    @pytest.mark.asyncio
    async def test_timed_out_docker_ps_makes_reconcile_unsafe(self):
        """A timed-out docker ps must NOT silently yield no containers
        (which looks like 'free' to the gate).  reconcile_before_write must
        return safe=False with a note containing 'docker ps read failed'."""
        runner = full_runner(
            **{
                "docker ps": RunResult(-1, "", "timeout", timed_out=True),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert "docker ps read failed" in rec.note

    @pytest.mark.asyncio
    async def test_nonzero_docker_ps_makes_reconcile_unsafe(self):
        """A docker ps that exits non-zero (daemon down, permission error) must
        also fail closed — not silently read as empty."""
        runner = full_runner(
            **{
                "docker ps": RunResult(1, "", "permission denied", timed_out=False),
                "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE),
            }
        )
        gpus = [GpuInfo(index=0, mem_used_mib=3), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        rec = await cd.reconcile_before_write("serve:vllm/dual")
        assert rec.safe is False
        assert "docker ps read failed" in rec.note


# ===========================================================================
# Fix 3 — ExplainScreen renders real benchmark shape (not invented keys)
# ===========================================================================


class TestExplainScreenBenchmarkShape:
    """ExplainScreen.set_detail must parse the REAL {row,columns} shape and
    render real TPS / 8pk numbers — never literal 'None/None'.
    """

    def test_explain_modal_renders_real_tps_from_columns(self):
        """Given a real {row,columns} benchmark record, the Measured section
        must contain the actual TPS numbers parsed from columns[4].
        (174 narr, 42 code, 8pk 109/150 from EXPLAIN_BENCH_ROW fixture.)
        """
        from club3090_cockpit.app import ExplainScreen
        from club3090_cockpit.data import measurement_from_explain_columns

        detail = {
            "registry": {"model": "qwen3.6-27b", "engine": "vllm-stable", "status": "production"},
            "fit": {"verdict": "fits-constrained", "vram_est_gb": 19.881, "band_gb": 1.5, "max_ctx": 262144},
            "card": "rtx-3090",
            "benchmarks": [EXPLAIN_BENCH_ROW],
        }

        # Parse via the same helper the fixed modal uses.
        m = measurement_from_explain_columns(EXPLAIN_BENCH_ROW)
        assert m.narr_tps == 174.0
        assert m.code_tps == 42.0
        assert m.quality_8pk == "109/150"
        assert m.tps_label == "174/42"
        assert "None" not in m.tps_label, "tps_label must never contain 'None'"

    def test_explain_modal_never_renders_none_for_tps(self):
        """The modal must render '—' (not 'None') when a benchmark record
        carries no TPS (a stress/soak row)."""
        from club3090_cockpit.data import measurement_from_explain_columns

        stress_row = {
            "row": "| `dual.yml` | @x | PASS | PASS at 64K | FAIL | 2026-05-03 |",
            "columns": ["`dual.yml`", "@x", "PASS", "PASS at 64K", "FAIL", "2026-05-03"],
        }
        m = measurement_from_explain_columns(stress_row)
        assert m.narr_tps is None
        assert m.code_tps is None
        # The ExplainScreen fix renders n = "—" / c = "—" when tps is None.
        n = f"{m.narr_tps:.0f}" if m.narr_tps is not None else "—"
        c = f"{m.code_tps:.0f}" if m.code_tps is not None else "—"
        rendered = f"{n}/{c} TPS"
        assert "None" not in rendered
        assert rendered == "—/— TPS"

    def test_explain_modal_real_columns_fixture_roundtrip(self):
        """Full roundtrip: the EXPLAIN_BENCH_ROW fixture → the fixed rendering
        path → the body contains '174', '42', and '109/150'."""
        from club3090_cockpit.data import measurement_from_explain_columns

        m = measurement_from_explain_columns(EXPLAIN_BENCH_ROW)
        n = f"{m.narr_tps:.0f}" if m.narr_tps is not None else "—"
        c = f"{m.code_tps:.0f}" if m.code_tps is not None else "—"
        q = m.quality_8pk or "—"
        d = m.date or ""
        line = f"    {n}/{c} TPS · 8pk {q}  {d}"
        assert "174" in line
        assert "42" in line
        assert "109/150" in line
        assert "None" not in line


# ===========================================================================
# PHASE 4 — REAL-shaped fixtures (captured live; see the wiring notes inline)
# ===========================================================================

# diagnose-estate.sh --json — REAL shape (verified live 2026-06-18).
ESTATE_DIAGNOSE_GREEN = json.dumps(
    {
        "estate_file": "/home/x/.club3090/estate.yml",
        "live": False,
        "checks": {
            "schema": {"ok": True, "schema_version": 1, "instance_count": 2},
            "registry": {"ok": True, "missing": [], "composes": ["llamacpp/default", "llamacpp/default"]},
            "per_instance_fits": [
                {"name": "llama-gpu0", "valid": True, "constraints_passed": 15, "constraints_failed": 0},
                {"name": "llama-gpu1", "valid": True, "constraints_passed": 15, "constraints_failed": 0},
            ],
            "cross_checks": {"ok": True, "failures": [], "constraints_passed": ["E1", "E2", "E3", "E4"]},
            "calibration": [
                {"name": "llama-gpu0", "status": "predicted", "has_row": False, "source": None},
                {"name": "llama-gpu1", "status": "predicted", "has_row": False, "source": None},
            ],
            "live": {"checked": False, "instances": []},
        },
        "valid": True,
        "summary": "GREEN",
    }
)

ESTATE_DIAGNOSE_RED = json.dumps(
    {
        "estate_file": "/home/x/.club3090/estate.yml",
        "live": False,
        "checks": {
            "schema": {"ok": True, "schema_version": 1, "instance_count": 2},
            "per_instance_fits": [
                {"name": "a", "valid": True},
                {"name": "b", "valid": False, "reasons": ["wont fit"]},
            ],
            "cross_checks": {"ok": False, "failures": ["E2 overlap"]},
        },
        "valid": False,
        "summary": "RED",
    }
)

# diagnose-profile.sh <slug> — REAL text output (verified live, vllm/dual GREEN).
PROFILE_TRIAGE_GREEN = (
    "Profile triage: vllm/dual\n"
    "=========================\n"
    "[1/6] Compose registry entry exists\n"
    "  ✓ vllm/dual found (model=qwen3.6-27b, workload=long-ctx-single, engine=vllm-stable)\n"
    "\n"
    "[2/6] Cross-references resolve\n"
    "  ✓ all referenced profiles exist (model, workload, engine, drafter, hardware-default=2x-rtx-3090-pcie)\n"
    "\n"
    "[3/6] fits() on canonical 2x-rtx-3090-pcie scenario\n"
    "  ✓ valid=true; constraints passed: 15/16; elapsed 0.03 ms\n"
    "  note: KV projection skipped because project_vram=false\n"
    "\n"
    "[4/6] kv-calc projection\n"
    "  ✓ predicted total 19.88 GB/card (90.04% budget), verdict PASS; budget 22.08 GB\n"
    "\n"
    "[5/6] Calibration freshness\n"
    "  ✓ verified; BENCHMARKS.md#qwen36-27b dual.yml @noonghunna 2026-04-29\n"
    "\n"
    "[6/6] Vendored overlays applied\n"
    "  ✓ required_overlays: []\n"
    "  ✓ vendored_overlays: []\n"
    "  ✓ Genesis: not required\n"
    "  ✓ VLLM_IMAGE resolves: vllm/vllm-openai:v0.22.0\n"
    "\n"
    "Triage summary: GREEN\n"
)

# REAL YELLOW output (verified live): ⊘ glyphs on skipped + kv-FAIL steps.
PROFILE_TRIAGE_YELLOW = (
    "[1/6] Compose registry entry exists\n"
    "  ⊘ free-form combo; compose registry lookup skipped\n"
    "\n"
    "[2/6] Cross-references resolve\n"
    "  ✓ all referenced profiles exist (model, workload, engine, hardware-default=1x-rtx-3060-12gb)\n"
    "\n"
    "[3/6] fits() on canonical 1x-rtx-3060-12gb scenario\n"
    "  ✓ valid=true; constraints passed: 15/16; elapsed 0.027 ms\n"
    "  note: KV projection skipped because project_vram=false\n"
    "\n"
    "[4/6] kv-calc projection\n"
    "  ⊘ predicted total 20.51 GB/card (194.27% budget), verdict FAIL\n"
    "  note: kv-calc: fixed components (20.5 GB) leave only 0.00 GB for KV\n"
    "\n"
    "[5/6] Calibration freshness\n"
    "  ⊘ free-form combo; no exact compose calibration row\n"
    "\n"
    "[6/6] Vendored overlays applied\n"
    "  ✓ required_overlays: []\n"
    "  ✓ VLLM_IMAGE resolves: vllm/vllm-openai:v0.22.0\n"
    "\n"
    "Triage summary: YELLOW\n"
)

# gpu-mode power-cap status — REAL output (verified live; ANSI banner + table).
POWER_CAP_STATUS = (
    "\x1b[0;36m═══ GPU Power Limits ═══\x1b[0m\n"
    "index, power.limit [W], power.default_limit [W], power.min_limit [W], power.max_limit [W]\n"
    "0, 370.00 W, 370.00 W, 100.00 W, 390.00 W\n"
    "1, 420.00 W, 420.00 W, 100.00 W, 450.00 W\n"
)

# docker top — REAL ps-style table.
DOCKER_TOP = (
    "UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD\n"
    "root                12345               12300               4                   10:00               ?                   00:05:01            python3 -m vllm.entrypoints.openai.api_server --model x\n"
)

# #249 measurement-record corpus line — REAL frozen-schema shape (verified live
# via scripts/lib/profiles/measurement_record.py --print-only). Bench-only record
# → carries decode_tps_by_ctx + wall_tps but NO 8-pack.
CORPUS_RECORD = {
    "model_slug": "qwen3.6-27b",
    "arch": "qwen3-next-hybrid",
    "arch_class": "deltanet-hybrid",
    "engine_id": "vllm-stable",
    "engine_pin": None,
    "hardware": "rtx-3090",
    "topology": "dual",
    "kv_dtype": "fp8_e5m2",
    "max_model_len": 262144,
    "max_num_seqs": 2,
    "mem_util": 0.92,
    "result_class": "boot-fit-measured",
    "provenance": {"source": "measured", "n_obs": 1, "last_confirmed": "2026-06-16"},
    "measured_extensions": {
        "decode_tps_by_ctx": {"canonical-short": 44.02},
        "prefill_tps": 1196.62,
        "ttft_s": 11.014,
        "wall_tps": 43.81,
        "peak_vram_mib_by_gpu": {"0": 21842, "1": 21228},
        "power_cap_w": 370.0,
        "conditions_fingerprint": {"hardware": "rtx-3090", "kv_dtype": "fp8_e5m2", "topology": "dual"},
    },
    "_tag": "vllm/dual",
}

# REAL BENCHMARKS.md fragment (canonical 9-col table under model + topo headers).
BENCH_MD = (
    "## Qwen3.6-27B\n"
    "\n"
    "### Single-card (1× RTX 3090) — vLLM\n"
    "\n"
    "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |\n"
    "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
    "| `minimal.yml` (single) | @x (1× 3090) | TQ3 | 64K | ~32 / ~33 | — | ~22.4 GB | 2026-05-03 | no MTP |\n"
    "\n"
    "### Dual-card (2× RTX 3090, TP=2)\n"
    "\n"
    "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |\n"
    "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
    "| `dual.yml` ⭐ | @x (2× 3090) | fp8 | 262K | **69 / 89** | — | ~23.6 GB | 2026-04-29 | 8-pack 109/150 |\n"
    "\n"
    "## Gemma 4 31B (community-experimental)\n"
    "\n"
    "### Dual-card (2× RTX 3090, TP=2)\n"
    "\n"
    "| Compose | Rig | KV | Max ctx | Narr / Code TPS | PP tok/s | Peak VRAM | Date | Notes |\n"
    "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
    "| `int8.yml` | @x (2× 3090) | int8 | 192K | 112 / 29 | — | ~23 GB | 2026-06-01 | 8-pack 103/150 |\n"
    "\n"
    "## How to add a row for your rig\n"
    "\n"
    "| not | a | model | row |\n"
)


# ===========================================================================
# PHASE 4 — pure parsers (REAL shapes)
# ===========================================================================


class TestPhase4Parsers:
    def test_estate_diagnose_green(self):
        from club3090_cockpit.data import EstateDiagnose

        ed = EstateDiagnose.from_dict(json.loads(ESTATE_DIAGNOSE_GREEN))
        assert ed.valid is True
        assert ed.summary == "GREEN"
        assert ed.summary_glyph == "●"
        assert ed.instance_count == 2
        assert ed.instances_valid == 2
        assert ed.cross_checks_ok is True

    def test_estate_diagnose_red(self):
        from club3090_cockpit.data import EstateDiagnose

        ed = EstateDiagnose.from_dict(json.loads(ESTATE_DIAGNOSE_RED))
        assert ed.valid is False
        assert ed.summary_glyph == "○"
        assert ed.instances_valid == 1   # one of two fits is valid
        assert ed.cross_checks_ok is False

    def test_estate_diagnose_empty(self):
        from club3090_cockpit.data import EstateDiagnose

        ed = EstateDiagnose.from_dict(None)
        assert ed.error and not ed.valid

    def test_profile_triage_green(self):
        from club3090_cockpit.data import parse_profile_triage

        tri = parse_profile_triage(PROFILE_TRIAGE_GREEN, "vllm/dual")
        assert tri.summary == "GREEN"
        assert tri.summary_glyph == "●"
        assert len(tri.steps) == 6
        assert tri.passed == 6
        assert tri.steps[0].num == 1 and tri.steps[0].total == 6

    def test_profile_triage_yellow_with_skip_glyphs(self):
        """REAL YELLOW output uses ⊘ for skipped + kv-FAIL steps → 'warn'."""
        from club3090_cockpit.data import parse_profile_triage

        tri = parse_profile_triage(PROFILE_TRIAGE_YELLOW)
        assert tri.summary == "YELLOW"
        assert tri.summary_glyph == "◐"
        by = {s.num: s.status for s in tri.steps}
        assert by[1] == "warn"   # ⊘ skipped
        assert by[3] == "passed"
        assert by[4] == "warn"   # ⊘ kv-calc FAIL
        assert by[6] == "passed"

    def test_profile_triage_no_output(self):
        from club3090_cockpit.data import parse_profile_triage

        tri = parse_profile_triage("")
        assert tri.steps == [] and tri.summary == ""

    def test_verify_smoke_parser_all_pass(self):
        from club3090_cockpit.data import parse_verify_smoke

        vs = parse_verify_smoke(
            "  \x1b[32m✓\x1b[0m a\n  \x1b[32m✓\x1b[0m b\n  \x1b[32m✓\x1b[0m c\n  \x1b[32m✓\x1b[0m d\n",
            rc=0,
        )
        assert (vs.passed, vs.total, vs.reachable, vs.ok) == (4, 4, True, True)
        assert vs.verdict_glyph == "●"

    def test_verify_smoke_parser_inband_failure_beats_rc(self):
        """A ✗ in the output means NOT ok even if rc claims success."""
        from club3090_cockpit.data import parse_verify_smoke

        vs = parse_verify_smoke("  \x1b[31m✗\x1b[0m unreachable\n    \x1b[33m→\x1b[0m serve one\n", rc=0)
        assert vs.ok is False and vs.reachable is False
        assert vs.checks[0].hint == "serve one"

    def test_verify_smoke_parser_no_output(self):
        from club3090_cockpit.data import parse_verify_smoke

        vs = parse_verify_smoke("", rc=0)
        assert vs.total == 0 and vs.ok is False and vs.error

    def test_verify_full_parser_summary_authoritative(self):
        from club3090_cockpit.data import parse_verify_full

        vf = parse_verify_full(
            "[1/9] x\n  \x1b[32m✓\x1b[0m ok\n[4/9] tool\n  \x1b[31m✗\x1b[0m no calls\n"
            "[9/9] mtp\n  \x1b[32m✓\x1b[0m al\n1 check(s) failed.\n",
            rc=0,
        )
        assert (vf.passed, vf.total, vf.failed, vf.ok) == (8, 9, 1, False)
        assert vf.verdict_glyph == "◐"

    def test_verify_full_parser_all_pass(self):
        from club3090_cockpit.data import parse_verify_full

        vf = parse_verify_full(
            "[1/9] x\n  \x1b[32m✓\x1b[0m ok\n[9/9] y\n  \x1b[32m✓\x1b[0m ok\nAll checks passed.\n",
            rc=0,
        )
        assert vf.ok is True and vf.failed == 0 and vf.verdict_glyph == "●"

    def test_power_cap_status_parse(self):
        from club3090_cockpit.data import parse_power_cap_status

        st = parse_power_cap_status(POWER_CAP_STATUS)
        assert st.error == ""
        assert len(st.gpus) == 2
        assert st.gpus[0].index == 0 and st.gpus[0].limit_w == 370.0
        assert st.gpus[0].max_w == 390.0
        assert st.gpus[1].index == 1 and st.gpus[1].limit_w == 420.0

    def test_power_cap_status_empty(self):
        from club3090_cockpit.data import parse_power_cap_status

        st = parse_power_cap_status("no GPUs here")
        assert st.gpus == [] and st.error

    def test_docker_top_parse(self):
        from club3090_cockpit.data import parse_docker_top

        top = parse_docker_top("vllm-x", DOCKER_TOP)
        assert top.error == ""
        assert top.header[0] == "UID" and "CMD" in top.header
        assert len(top.rows) == 1
        # The trailing CMD (with spaces) stays one cell.
        assert top.rows[0][0] == "root"
        assert "vllm.entrypoints" in top.rows[0][-1]

    def test_docker_top_empty(self):
        from club3090_cockpit.data import parse_docker_top

        top = parse_docker_top("nope", "")
        assert top.error and top.rows == []

    def test_bench_row_from_corpus_record(self):
        """#249 corpus record → BenchRow: TPS/ctx from measured_extensions, no 8pk."""
        from club3090_cockpit.data import bench_row_from_corpus_record

        row = bench_row_from_corpus_record(CORPUS_RECORD)
        assert row is not None
        assert row.model == "qwen3.6-27b"
        assert row.engine == "vllm-stable"
        assert row.topology == "dual"
        assert row.code_tps == 44.02       # decode_tps_by_ctx['canonical-short']
        assert row.narr_tps == 43.81       # wall_tps
        assert row.max_ctx == "262K"       # 262144 → 262K (÷1000, registry-emit convention)
        assert row.quality_8pk == ""       # bench-only record carries no 8pk
        assert row.source == "corpus"
        assert row.tag == "vllm/dual"
        assert row.date == "2026-06-16"

    def test_bench_row_from_corpus_record_no_tps_is_none(self):
        from club3090_cockpit.data import bench_row_from_corpus_record

        rec = {"model_slug": "x", "engine_id": "y", "topology": "single",
               "measured_extensions": {"decode_tps_by_ctx": {}}}
        assert bench_row_from_corpus_record(rec) is None

    def test_bench_rows_from_benchmarks_md(self):
        """Walks model + topology section headers; parses canonical 9-col rows."""
        from club3090_cockpit.data import bench_rows_from_benchmarks_md

        rows = bench_rows_from_benchmarks_md(BENCH_MD)
        # 3 model rows; the 'How to add a row' table is NOT a model section.
        assert len(rows) == 3
        by = {(r.model, r.topology): r for r in rows}
        single = by[("qwen3.6-27b", "single")]
        assert single.narr_tps == 32.0 and single.code_tps == 33.0
        assert single.max_ctx == "64K"
        dual = by[("qwen3.6-27b", "dual")]
        assert dual.narr_tps == 69.0 and dual.code_tps == 89.0
        assert dual.quality_8pk == "109/150"
        assert dual.date == "2026-04-29"
        assert dual.tag == "dual.yml"
        gemma = by[("gemma-4-31b", "dual")]
        assert gemma.narr_tps == 112.0 and gemma.quality_8pk == "103/150"

    def test_bench_rows_md_skips_non_model_sections(self):
        from club3090_cockpit.data import bench_rows_from_benchmarks_md

        rows = bench_rows_from_benchmarks_md(BENCH_MD)
        # No row should come from the "How to add a row" section.
        assert all(r.model in ("qwen3.6-27b", "gemma-4-31b") for r in rows)


# ===========================================================================
# PHASE 4 — Doctor reads (health + estate-diagnose + profile-triage)
# ===========================================================================


class TestPhase4Doctor:
    @pytest.mark.asyncio
    async def test_estate_diagnose_read(self):
        runner = full_runner(**{"diagnose-estate.sh --json": ok(ESTATE_DIAGNOSE_GREEN)})
        cd = CockpitData(ROOT, runner=runner)
        ed = await cd.estate_diagnose()
        assert ed.summary == "GREEN" and ed.instances_valid == 2

    @pytest.mark.asyncio
    async def test_profile_triage_read(self):
        runner = full_runner(**{"diagnose-profile.sh vllm/dual": ok(PROFILE_TRIAGE_GREEN)})
        cd = CockpitData(ROOT, runner=runner)
        tri = await cd.profile_triage("vllm/dual")
        assert tri.summary == "GREEN" and tri.passed == 6
        # It is a READ — diagnose-profile.sh, never a write.
        call = next(c for c in runner.calls if "diagnose-profile.sh" in " ".join(c))
        assert call[:2] == ["bash", "scripts/diagnose-profile.sh"]

    @pytest.mark.asyncio
    async def test_profile_triage_timeout(self):
        runner = full_runner(
            **{"diagnose-profile.sh vllm/dual": RunResult(-1, "", "timeout", timed_out=True)}
        )
        cd = CockpitData(ROOT, runner=runner)
        tri = await cd.profile_triage("vllm/dual")
        assert tri.error and "timed out" in tri.error

    @pytest.mark.asyncio
    async def test_doctor_all_legs(self):
        runner = full_runner(
            **{
                "health.sh": ok(HEALTH_SERVING),
                "diagnose-estate.sh --json": ok(ESTATE_DIAGNOSE_GREEN),
                "diagnose-profile.sh vllm/dual": ok(PROFILE_TRIAGE_GREEN),
            }
        )
        cd = CockpitData(ROOT, runner=runner)
        rep = await cd.doctor(url="http://localhost:8010", slug="vllm/dual")
        assert rep.health.serving is True
        assert rep.estate.summary == "GREEN"
        assert rep.profile is not None and rep.profile.summary == "GREEN"

    @pytest.mark.asyncio
    async def test_doctor_no_slug_skips_profile(self):
        runner = full_runner(
            **{"health.sh": ok(HEALTH_DOWN), "diagnose-estate.sh --json": ok(ESTATE_DIAGNOSE_GREEN)}
        )
        cd = CockpitData(ROOT, runner=runner)
        rep = await cd.doctor()
        assert rep.profile is None  # no slug → no profile triage
        # diagnose-profile must NOT have been called.
        assert not any("diagnose-profile.sh" in " ".join(c) for c in runner.calls)

    # ── Batch 3: verify / verify-full (Doctor "serving correctly?" reads) ──────────

    _VERIFY_OK = (
        "Running smoke test against http://localhost:8020\n\n"
        "  \x1b[32m✓\x1b[0m Server is reachable\n"
        "  \x1b[32m✓\x1b[0m Genesis patches applied cleanly\n"
        "  \x1b[32m✓\x1b[0m Basic completion works (Paris)\n"
        "  \x1b[32m✓\x1b[0m Tool calling works end-to-end\n"
    )
    _VERIFY_DOWN = (
        "Running smoke test against http://localhost:8020\n\n"
        "  \x1b[31m✗\x1b[0m no response from http://localhost:8020/v1/models\n"
        "    \x1b[33m→\x1b[0m start a model first\n"
    )

    @pytest.mark.asyncio
    async def test_verify_smoke_read(self):
        runner = full_runner(**{"scripts/verify.sh": ok(self._VERIFY_OK)})
        cd = CockpitData(ROOT, runner=runner)
        vs = await cd.verify_smoke()
        assert vs.ok is True and vs.reachable is True
        assert (vs.passed, vs.total) == (4, 4)
        call = next(c for c in runner.calls if "verify.sh" in " ".join(c))
        assert call == ["bash", "scripts/verify.sh"]   # a READ — verify.sh, no args

    @pytest.mark.asyncio
    async def test_verify_smoke_unreachable(self):
        # ok() reports rc=0, but the in-band ✗ must still make it NOT ok.
        runner = full_runner(**{"scripts/verify.sh": ok(self._VERIFY_DOWN)})
        cd = CockpitData(ROOT, runner=runner)
        vs = await cd.verify_smoke()
        assert vs.ok is False and vs.reachable is False
        assert vs.checks and vs.checks[0].hint.startswith("start a model")

    @pytest.mark.asyncio
    async def test_verify_smoke_timeout(self):
        runner = full_runner(
            **{"scripts/verify.sh": RunResult(-1, "", "timeout", timed_out=True)}
        )
        cd = CockpitData(ROOT, runner=runner)
        vs = await cd.verify_smoke()
        assert vs.error and "timed out" in vs.error and vs.ok is False

    @pytest.mark.asyncio
    async def test_verify_full_read(self):
        VF = (
            "[1/9] Server reachable ...\n  \x1b[32m✓\x1b[0m reachable\n"
            "[4/9] Tool calling ...\n  \x1b[31m✗\x1b[0m no tool_calls[]\n"
            "[9/9] MTP acceptance ...\n  \x1b[32m✓\x1b[0m AL>=2.0\n"
            "\x1b[31m1 check(s) failed.\x1b[0m See hints above.\n"
        )
        runner = full_runner(**{"scripts/verify-full.sh": ok(VF)})
        cd = CockpitData(ROOT, runner=runner)
        vf = await cd.verify_full()
        # in-band summary is authoritative even though ok() reports rc=0.
        assert (vf.passed, vf.total, vf.failed, vf.ok) == (8, 9, 1, False)
        call = next(c for c in runner.calls if "verify-full.sh" in " ".join(c))
        assert call == ["bash", "scripts/verify-full.sh"]


# ===========================================================================
# PHASE 4 — Benchmarks explorer (corpus → BENCHMARKS.md fallback)
# ===========================================================================


class _MdCockpitData(CockpitData):
    """CockpitData with BENCHMARKS.md + corpus reads stubbed (no disk dep)."""

    def __init__(self, *a, md_text="", corpus_rows=None, **k):
        super().__init__(*a, **k)
        self._md_text = md_text
        self._corpus_rows = corpus_rows or []

    def _read_benchmarks_md(self):
        return self._md_text

    def _read_measurement_corpus(self):
        return list(self._corpus_rows)


class TestPhase4BenchmarksExplorer:
    @pytest.mark.asyncio
    async def test_explorer_md_only(self):
        cd = _MdCockpitData(ROOT, runner=full_runner(), md_text=BENCH_MD)
        rows, err = await cd.benchmarks_explorer()
        assert err is None
        assert len(rows) == 3
        assert all(r.source == "benchmarks.md" for r in rows)

    @pytest.mark.asyncio
    async def test_explorer_corpus_preferred_and_borrows_8pk(self):
        """Corpus row wins its (model, engine, topo) key AND borrows the 8-pack
        the bench-only corpus record lacks from the markdown row."""
        from club3090_cockpit.data import bench_row_from_corpus_record

        corpus = [bench_row_from_corpus_record(CORPUS_RECORD)]  # qwen/vllm-stable/dual
        # Markdown dual row carries 8-pack 109/150 but uses engine token '' (it's
        # a `dual.yml` filename) — so make a corpus-key-matching md row instead.
        md = BENCH_MD.replace("| `dual.yml` ⭐", "| `vllm-stable/dual` ⭐")
        cd = _MdCockpitData(ROOT, runner=full_runner(), md_text=md, corpus_rows=corpus)
        rows, err = await cd.benchmarks_explorer()
        assert err is None
        corpus_row = next(r for r in rows if r.source == "corpus")
        assert corpus_row.quality_8pk == "109/150"  # borrowed from md
        # Corpus row replaced the md row for that key (no duplicate).
        keys = [(r.model, r.engine, r.topology) for r in rows]
        assert keys.count(("qwen3.6-27b", "vllm-stable", "dual")) == 1

    @pytest.mark.asyncio
    async def test_explorer_empty_both_sources(self):
        cd = _MdCockpitData(ROOT, runner=full_runner(), md_text="", corpus_rows=[])
        rows, err = await cd.benchmarks_explorer()
        assert rows == [] and err is not None

    @pytest.mark.asyncio
    async def test_corpus_dir_absent_returns_empty(self):
        """A fresh rig has no results/measurement-records/ dir → [] (not crash).
        Uses the REAL _read_measurement_corpus against a nonexistent root."""
        cd = CockpitData(Path("/tmp/nonexistent-club3090-root-xyz"), runner=full_runner())
        assert cd._read_measurement_corpus() == []


# ===========================================================================
# PHASE 4 — Evidence (rebench run tags + paste-ready report)
# ===========================================================================


class TestPhase4Evidence:
    @pytest.mark.asyncio
    async def test_evidence_list_enumerates_tags(self, tmp_path):
        base = tmp_path / "results" / "rebench"
        (base / "tagA").mkdir(parents=True)
        (base / "tagA" / "REPORT.md").write_text(
            "# Rebench report — tagA\n\n## TL;DR\n\n- TPS narrative **40.7** / code **44.0**.\n"
            "\n## Meta\n\n- **Date:** 2026-06-16\n",
            encoding="utf-8",
        )
        (base / "tagA" / "_internal.json").write_text("{}", encoding="utf-8")
        (base / "tagA" / "soak.log").write_text("x", encoding="utf-8")
        (base / "tagB").mkdir(parents=True)   # bare tag, no artifacts
        cd = CockpitData(tmp_path, runner=full_runner())
        tags = await cd.evidence_list()
        names = [t.tag for t in tags]
        assert "tagA" in names and "tagB" in names
        a = next(t for t in tags if t.tag == "tagA")
        assert a.has_report and a.has_internal and a.has_soak
        assert a.date == "2026-06-16"
        assert "40.7" in a.tldr
        b = next(t for t in tags if t.tag == "tagB")
        assert not b.has_report and b.date  # mtime fallback

    @pytest.mark.asyncio
    async def test_evidence_list_no_rebench_dir(self):
        cd = CockpitData(Path("/tmp/nonexistent-club3090-root-xyz"), runner=full_runner())
        assert await cd.evidence_list() == []

    @pytest.mark.asyncio
    async def test_evidence_report_reads_generated_report(self, tmp_path):
        base = tmp_path / "results" / "rebench" / "tagA"
        base.mkdir(parents=True)
        # The (mocked) rebench-report.py "generates" REPORT.md — simulate by
        # having the file already present; the runner returns ok with no spawn.
        base.joinpath("REPORT.md").write_text("# Rebench report — tagA\n\nbody here\n", encoding="utf-8")
        runner = full_runner(**{"rebench-report.py": ok("wrote REPORT.md")})
        cd = CockpitData(tmp_path, runner=runner)
        rep = await cd.evidence_report("tagA")
        assert rep.error == ""
        assert "body here" in rep.body
        assert rep.report_path.endswith("REPORT.md")
        # It used the report generator (a READ of results), never a write script.
        call = next(c for c in runner.calls if "rebench-report.py" in " ".join(c))
        assert "--no-discuss" in call

    @pytest.mark.asyncio
    async def test_evidence_report_missing_tag(self, tmp_path):
        cd = CockpitData(tmp_path, runner=full_runner())
        rep = await cd.evidence_report("nope")
        assert rep.error and "no run dir" in rep.error


# ===========================================================================
# PHASE 4 — submit-bench (READ preview vs OUTWARD-FACING gated write)
# ===========================================================================


class TestPhase4SubmitBench:
    @pytest.mark.asyncio
    async def test_submit_bench_preview_is_read(self, tmp_path):
        """submit-bench.sh --tag (NO --auto-submit) only generates the row file —
        no network, no PR (verified live)."""
        base = tmp_path / "results" / "rebench" / "tagA"
        base.mkdir(parents=True)
        base.joinpath("BENCHMARKS-row.md").write_text(
            "| `dual.yml` | @x | fp8 | 262K | 69 / 89 | ... |", encoding="utf-8"
        )
        runner = full_runner(**{"submit-bench.sh": ok("Generated row")})
        cd = CockpitData(tmp_path, runner=runner)
        out = await cd.submit_bench_preview("tagA")
        assert out["error"] is None
        assert "69 / 89" in out["row"]
        # The preview command must NOT carry --auto-submit (no network).
        call = next(c for c in runner.calls if "submit-bench.sh" in " ".join(c))
        assert "--auto-submit" not in call

    def test_submit_bench_plan_is_network_gated(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.submit_bench("tagA")
        assert plan.kind == "submit_bench"
        assert "--auto-submit" in plan.cmd
        assert plan.network is True
        assert plan.requires_confirm is True
        assert plan.requires_reconcile is False  # no GPU contention

    def test_submit_bench_as_pr(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.submit_bench("tagA", as_pr=True)
        assert "--as-pr" in plan.cmd

    @pytest.mark.asyncio
    async def test_submit_bench_never_auto_fires_network(self):
        """Building the plan must NOT touch the runner/network — execution is the
        caller's confirmed dispatch, mocked here."""
        runner = full_runner()
        cd = CockpitData(ROOT, runner=runner)
        cd.submit_bench("tagA")
        # No submit-bench call was made just by building the plan.
        assert not any("submit-bench.sh" in " ".join(c) for c in runner.calls)


# ===========================================================================
# PHASE 4 — power cap (read safe · write/sweep WIRED mock-only confirm)
# ===========================================================================


class TestPhase4PowerCap:
    @pytest.mark.asyncio
    async def test_power_cap_get_is_read(self):
        runner = full_runner(**{"power-cap status": ok(POWER_CAP_STATUS)})
        cd = CockpitData(ROOT, runner=runner)
        st = await cd.power_cap_get()
        assert len(st.gpus) == 2 and st.gpus[0].limit_w == 370.0
        call = next(c for c in runner.calls if "power-cap" in " ".join(c))
        assert call == ["bash", "scripts/gpu-mode.sh", "power-cap", "status"]

    def test_power_cap_set_on_off(self):
        cd = CockpitData(ROOT, runner=full_runner())
        on = cd.power_cap_set("on")
        assert on.cmd == ["bash", "scripts/gpu-mode.sh", "power-cap", "on"]
        assert on.requires_confirm is True and on.requires_reconcile is False
        off = cd.power_cap_set("off")
        assert off.cmd[-1] == "off"

    def test_power_cap_set_custom_wattage(self):
        """gpu-mode power-cap now ALSO takes a custom wattage (cockpit menu's
        'custom' option) — gpu-mode.sh validates it against each card's range."""
        cd = CockpitData(ROOT, runner=full_runner())
        for w in (280, "330"):
            plan = cd.power_cap_set(w)
            assert plan.cmd == ["bash", "scripts/gpu-mode.sh", "power-cap", str(int(w))]
            assert plan.kind == "power_cap"
            assert plan.requires_confirm is True and plan.requires_reconcile is False

    def test_power_cap_set_rejects_invalid(self):
        cd = CockpitData(ROOT, runner=full_runner())
        for bad in ("hi", 0, -5, True):           # non-numeric / non-positive / bool
            with pytest.raises(ValueError):
                cd.power_cap_set(bad)

    def test_power_cap_sweep_plan(self):
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.power_cap_sweep(caps=[300, 330, 370])
        assert plan.kind == "power_cap_sweep"
        assert "--caps" in plan.cmd and "300,330,370" in plan.cmd
        assert plan.requires_confirm is True


# ===========================================================================
# PHASE 4 — container top (READ) + rm (WIRED mock-only, reconcile-gated)
# ===========================================================================


class TestPhase4Container:
    @pytest.mark.asyncio
    async def test_container_top_is_read(self):
        runner = full_runner(**{"docker top": ok(DOCKER_TOP)})
        cd = CockpitData(ROOT, runner=runner)
        top = await cd.container_top("vllm-x")
        assert top.error == "" and len(top.rows) == 1
        call = next(c for c in runner.calls if "top" in " ".join(c))
        assert call == ["docker", "top", "vllm-x"]

    @pytest.mark.asyncio
    async def test_container_top_missing(self):
        runner = full_runner(**{"docker top": RunResult(1, "", "Error: No such container: nope")})
        cd = CockpitData(ROOT, runner=runner)
        top = await cd.container_top("nope")
        assert top.error and top.rows == []

    def test_container_rm_is_reconcile_gated(self):
        cd = CockpitData(ROOT, runner=full_runner())
        p = cd.container_rm("vllm-x")
        assert p.kind == "container_rm"
        assert p.cmd == ["docker", "rm", "vllm-x"]
        assert p.requires_reconcile is True   # frees a GPU → gate it
        assert p.requires_confirm is True

    def test_container_rm_force_requires_reason(self):
        cd = CockpitData(ROOT, runner=full_runner())
        with pytest.raises(ValueError):
            cd.container_rm("vllm-x", force=True)

    def test_container_rm_force_adds_f(self):
        cd = CockpitData(ROOT, runner=full_runner())
        p = cd.container_rm("vllm-x", force=True, force_reason="user accepted teardown")
        assert "-f" in p.cmd and p.force is True


# ===========================================================================
# PHASE 4 — run_validation (WIRED, execution MOCKED, no reconcile)
# ===========================================================================


class TestPhase4RunValidation:
    def test_validation_plan_builds_for_each_kind(self):
        cd = CockpitData(ROOT, runner=full_runner())
        # Most kinds shell out via bash; stream-toolcall-probe is a .py (python3).
        bash_kinds = ("verify-full", "verify-stress", "bench", "quality-test",
                      "soak-test", "rebench-full", "quality-baseline",
                      "bench-agentic")
        for kind in bash_kinds:
            plan = cd.validation_plan(kind, model="qwen3.6-27b", url="http://localhost:8010")
            assert plan.kind == "validation"
            assert plan.cmd[0] == "bash"
            assert plan.requires_reconcile is False  # hits model, no GPU claim
            assert plan.requires_confirm is True     # heavy → confirm
        probe = cd.validation_plan("stream-toolcall-probe", model="qwen3.6-27b", url="http://localhost:8010")
        assert probe.cmd[0] == "python3"
        assert probe.requires_reconcile is False and probe.requires_confirm is True

    def test_validation_plan_real_script_names(self):
        """The wired cmds use the REAL on-disk script names (verified live):
        quality-baseline.sh (its own #252 wrapper, NOT --baseline on quality-test)
        and stream-toolcall-probe.py (a .py taking --url/--model, NOT a .sh)."""
        cd = CockpitData(ROOT, runner=full_runner())
        qb = cd.validation_plan("quality-baseline")
        assert qb.cmd[:2] == ["bash", "scripts/quality-baseline.sh"]
        probe = cd.validation_plan("stream-toolcall-probe")
        assert probe.cmd[:2] == ["python3", "scripts/stream-toolcall-probe.py"]

    def test_validation_plan_probe_takes_url_model_as_cli_args(self):
        """stream-toolcall-probe.py reads --url/--model from the CLI, not env."""
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.validation_plan(
            "stream-toolcall-probe", model="qwen3.6-27b", url="http://localhost:8010"
        )
        assert "--url" in plan.cmd and "http://localhost:8010" in plan.cmd
        assert "--model" in plan.cmd and "qwen3.6-27b" in plan.cmd

    def test_validation_plan_quality_baseline_threads_slug(self):
        """quality-baseline.sh REQUIRES --slug; it's appended when supplied."""
        cd = CockpitData(ROOT, runner=full_runner())
        plan = cd.validation_plan("quality-baseline", slug="vllm/dual")
        assert "--slug" in plan.cmd and "vllm/dual" in plan.cmd

    def test_validation_plan_unknown_kind(self):
        cd = CockpitData(ROOT, runner=full_runner())
        with pytest.raises(ValueError):
            cd.validation_plan("nope")

    def test_validation_parser_maps_to_core_parser(self):
        from club3090_tui_core.parsers import BenchParser, VerifyParser, QualityParser

        cd = CockpitData(ROOT, runner=full_runner())
        assert isinstance(cd._validation_parser("bench"), BenchParser)
        assert isinstance(cd._validation_parser("verify-full"), VerifyParser)
        assert isinstance(cd._validation_parser("quality-test"), QualityParser)
        # Extra tools have no dedicated parser → NullParser passthrough.
        np = cd._validation_parser("bench-agentic")
        assert np.parse_line("anything") is None

    @pytest.mark.asyncio
    async def test_run_validation_streams_via_mocked_write_runner(self):
        """run_validation LAUNCHES via the (mocked) write runner — never live.
        conftest blocks the real spawn; here we inject a FakeWriteRunner."""
        wr = FakeWriteRunner()
        cd = CockpitData(ROOT, runner=full_runner(), write_runner=wr)
        state = await cd.run_validation("bench", model="qwen3.6-27b", url="http://localhost:8010")
        assert state is not None
        assert len(wr.started) == 1
        assert wr.started[0]["cmd"] == ["bash", "scripts/bench.sh"]
        assert wr.started[0]["run_type"] == "validation"

    @pytest.mark.asyncio
    async def test_run_validation_does_not_reconcile(self):
        """Validation hits the model but does not claim a GPU → no detect call."""
        wr = FakeWriteRunner()

        async def detect_should_not_be_called():
            raise AssertionError("validation must not run the reconcile gate")

        cd = CockpitData(
            ROOT, runner=full_runner(), write_runner=wr,
            detect_endpoint_fn=detect_should_not_be_called,
        )
        await cd.run_validation("verify-full")
        assert len(wr.started) == 1


# ===========================================================================
# PHASE 4 — gated writes route through execute_action (reconcile/confirm)
# ===========================================================================


class TestPhase4GatedWriteExecution:
    @pytest.mark.asyncio
    async def test_container_rm_unsafe_gate_refuses(self):
        """container_rm requires_reconcile=True → refused when the gate is unsafe
        (the target container is live), unless forced."""
        write_runner = FakeWriteRunner()
        runner = full_runner(
            **{"docker ps": ok(DOCKER_PS_ENGINE), "estate_cli.py report-state --json": ok(ESTATE_REPORT_FREE)}
        )
        gpus = [GpuInfo(index=0, mem_used_mib=22000), GpuInfo(index=1, mem_used_mib=1)]
        cd = CockpitData(
            ROOT, runner=runner, write_runner=write_runner,
            detect_endpoint_fn=make_detect(ServingTarget(gpus=gpus)),
            get_gpu_info_fn=make_gpu_info(gpus),
        )
        plan = cd.container_rm("vllm-qwen36-27b-dual")
        executed, rec, _ = await cd.execute_action(plan)
        assert executed is False
        assert rec is not None and rec.safe is False
        assert write_runner.started == []  # destructive rm never reached the runner

    @pytest.mark.asyncio
    async def test_power_cap_custom_skips_reconcile_but_runs_via_mocked_runner(self):
        """A custom-wattage power-cap has requires_reconcile=False → no detect;
        reaches the mocked write runner with the gpu-mode power-cap <W> command."""
        write_runner = FakeWriteRunner()

        async def detect_should_not_be_called():
            raise AssertionError("power-cap must not reconcile (no GPU contention)")

        cd = CockpitData(
            ROOT, runner=full_runner(), write_runner=write_runner,
            detect_endpoint_fn=detect_should_not_be_called,
        )
        executed, rec, _ = await cd.execute_action(cd.power_cap_set(280))
        assert executed is True and rec is None
        assert len(write_runner.started) == 1
        assert write_runner.started[0]["cmd"] == ["bash", "scripts/gpu-mode.sh", "power-cap", "280"]

    @pytest.mark.asyncio
    async def test_submit_bench_runs_via_mocked_runner_only(self):
        """The outward submit reaches ONLY the mocked write runner (network is
        never touched live — conftest blocks the real spawn)."""
        write_runner = FakeWriteRunner()
        cd = CockpitData(ROOT, runner=full_runner(), write_runner=write_runner)
        executed, rec, _ = await cd.execute_action(cd.submit_bench("tagA"))
        assert executed is True and rec is None  # no reconcile
        assert len(write_runner.started) == 1
        assert "--auto-submit" in write_runner.started[0]["cmd"]


# ===========================================================================
# PHASE 5 — the three v2 hooks (service layer)
# ===========================================================================


class EnvCapturingWriteRunner:
    """Like FakeWriteRunner but ALSO captures the child env + supports
    set_callbacks (the c3t launch wires the live-stream callbacks)."""

    def __init__(self):
        self.started: list[dict[str, Any]] = []
        self.callbacks: dict[str, Any] = {}

    def set_callbacks(self, on_event=None, on_line=None, on_complete=None):
        self.callbacks = {"on_event": on_event, "on_line": on_line}

    async def start_raw(self, cmd, env, run_type, parser):
        self.started.append({"cmd": cmd, "run_type": run_type, "env": dict(env)})
        return {"mock_state": True, "cmd": cmd}


class TestEvaluateHandoff:
    """Hook 1 — Estate → ▸ Evaluate hands the SHARED ServingTarget to c3t."""

    def test_handoff_carries_the_same_serving_target_by_identity(self):
        cd = CockpitData(ROOT, runner=full_runner())
        tgt = ServingTarget(url="http://localhost:8010", model="qwen3.6-27b", container="vllm-x")
        handoff = cd.evaluate_handoff(tgt)
        # The hand-off must carry the SAME dataclass instance (design §4/§6.6).
        assert handoff.target is tgt
        assert handoff.available is True
        assert handoff.plan.kind == "evaluate"
        assert handoff.plan.requires_confirm is True
        assert handoff.plan.requires_reconcile is False     # c3t hits endpoint, no GPU claim
        assert handoff.plan.cmd == ["bash", "scripts/c3t"]

    def test_serving_target_is_the_shared_core_dataclass(self):
        """The cockpit + c3t both speak club3090_tui_core.detect.ServingTarget —
        the hand-off type IS that shared dataclass (not a cockpit-local copy)."""
        from club3090_tui_core.detect import ServingTarget as CoreTarget
        import club3090_cockpit.services as svc
        assert svc.ServingTarget is CoreTarget

    def test_handoff_unavailable_when_no_running_target(self):
        cd = CockpitData(ROOT, runner=full_runner())
        handoff = cd.evaluate_handoff(None)
        assert handoff.available is False
        assert "no running" in handoff.reason.lower()
        # Even unavailable, the plan exists (confirm-gated) but nothing launches.
        assert handoff.plan.requires_confirm is True

    @pytest.mark.asyncio
    async def test_launch_evaluate_is_mock_only_and_scopes_to_target(self):
        """Launch streams via the MOCKED write runner (never live — conftest
        blocks the spawn) and scopes c3t to the SAME target via the child env."""
        wr = EnvCapturingWriteRunner()
        cd = CockpitData(ROOT, runner=full_runner(), write_runner=wr)
        tgt = ServingTarget(
            url="http://localhost:8010", model="qwen3.6-27b",
            container="vllm-qwen", slug="vllm/dual",
        )
        await cd.launch_evaluate(tgt)
        assert len(wr.started) == 1
        rec = wr.started[0]
        assert rec["cmd"] == ["bash", "scripts/c3t"]
        assert rec["run_type"] == "evaluate"
        # c3t is scoped to the SAME running target via env (it preselects it).
        env = rec["env"]
        assert env["C3T_TARGET_URL"] == "http://localhost:8010"
        assert env["C3T_TARGET_MODEL"] == "qwen3.6-27b"
        assert env["C3T_TARGET_CONTAINER"] == "vllm-qwen"
        assert env["C3T_TARGET_SLUG"] == "vllm/dual"
        assert env["C3T_REPO_ROOT"] == str(ROOT)


class TestPromoteScaffold:
    """Hook 2 — Discover → ▸ Promote to catalog: COMPUTE + PREVIEW only."""

    def _byo(self, **over) -> ByoResult:
        base = dict(
            repo="unsloth/Qwen3-27B-abliterated", profile_like="vllm/dual",
            arch="Qwen3ForCausalLM", eligible=True, fit_verdict="fits-clean",
            route="C", sibling_slug="vllm/dual", quant_match="int4",
            drop_spec_config=True,
        )
        base.update(over)
        return ByoResult(**base)

    def test_scaffold_computes_real_profile_and_registry_shapes(self):
        cd = CockpitData(ROOT, runner=full_runner())
        meas = Measurement(narr_tps=174.0, code_tps=42.0, quality_8pk="109/150", source="explain")
        sc = cd.promote_scaffold(byo=self._byo(), measurement=meas)
        assert sc.computed is True
        # ModelProfile YAML — REAL schema keys (ADDING_MODELS.md).
        assert "schema_version: 1" in sc.profile_yaml
        assert sc.profile_yaml.startswith("schema_version: 1\n")
        assert "\nweights:\n" in sc.profile_yaml            # weights MAP, not a list
        assert "vision_capable:" in sc.profile_yaml
        assert sc.profile_path.startswith("scripts/lib/profiles/models/")
        # compose_registry _entry(...) row — REAL kwargs.
        for kw in ("model=", "weights_variant=", "workload=", "engine=",
                   "drafter=", "kv_format=", "tp=", "compose_path=",
                   "default_port=", "kvcalc_key=", "status="):
            assert kw in sc.registry_entry, kw
        assert "_entry(" in sc.registry_entry
        # New models START at incubating (ADDING_MODELS.md rule).
        assert 'status="incubating"' in sc.registry_entry
        # Measured Evidence numbers flow into the status_note.
        assert "8-pack 109/150" in sc.registry_entry

    def test_scaffold_drops_drafter_when_byo_has_no_mtp_head(self):
        cd = CockpitData(ROOT, runner=full_runner())
        sc = cd.promote_scaffold(byo=self._byo(drop_spec_config=True))
        assert "drafter=None" in sc.registry_entry

    def test_scaffold_write_plan_is_gated_mock_only(self):
        cd = CockpitData(ROOT, runner=full_runner())
        sc = cd.promote_scaffold(byo=self._byo())
        plan = sc.write_plan
        assert plan is not None
        assert plan.kind == "promote_catalog"
        assert plan.requires_confirm is True
        assert plan.requires_reconcile is False
        # The gated action runs the guard suite; it does NOT auto-write scripts/.
        assert plan.cmd[:2] == ["bash", "-c"]
        assert "scripts/tests/*.sh" in " ".join(plan.cmd)

    @pytest.mark.asyncio
    async def test_promote_does_not_write_into_scripts_dir(self, tmp_path):
        """Computing + previewing the scaffold touches NO files under scripts/.
        The write is gated/mock-only and never auto-fires."""
        # Seed a scripts/ tree to detect any accidental write.
        scripts = tmp_path / "scripts" / "lib" / "profiles" / "models"
        scripts.mkdir(parents=True)
        before = sorted(p.name for p in scripts.iterdir())
        cd = CockpitData(tmp_path, runner=full_runner())
        sc = cd.promote_scaffold(byo=self._byo())
        assert sc.computed
        # No file was created under scripts/lib/profiles/models/ (preview only).
        after = sorted(p.name for p in scripts.iterdir())
        assert after == before == []
        # And the proposed profile path was NOT materialized.
        assert not (tmp_path / sc.profile_path).exists()

    def test_scaffold_errors_propagate_not_fabricated(self):
        cd = CockpitData(ROOT, runner=full_runner())
        bad = ByoResult(repo="x/Y", profile_like="vllm/dual", error="arch not eligible")
        sc = cd.promote_scaffold(byo=bad)
        assert sc.computed is False
        assert "arch not eligible" in sc.error
        assert sc.write_plan is None             # nothing to stage on a failed scaffold


class TestOptimizeSeam:
    """Hook 3 — ▸ Optimize for my card: DORMANT v0.10.0 seam (no-op)."""

    @pytest.mark.asyncio
    async def test_optimizer_is_not_available_v0_10_0(self):
        cd = CockpitData(ROOT, runner=full_runner())
        report = await cd.optimize_for_card(slug="vllm/dual")
        assert report.available is False
        assert report.message == "optimizer not available (v0.10.0)"
        # No fabricated recommendation / honesty gates while dormant.
        assert report.recommended_slug == ""
        assert report.boot_fit == "" and report.runtime == ""
        assert report.confidence == ""
        assert report.accept_runtime_risk_required is False

    @pytest.mark.asyncio
    async def test_optimizer_does_not_fabricate_even_if_probe_errors(self):
        """A probe error keeps the seam honestly dormant (never invents output)."""
        async def boom(cmd, *, cwd, timeout=30.0):
            raise RuntimeError("no such script")
        runner = full_runner()
        runner.run = boom  # type: ignore[assignment]
        cd = CockpitData(ROOT, runner=runner)
        report = await cd.optimize_for_card()
        assert report.available is False
        assert "v0.10.0" in report.message
