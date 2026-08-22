"""P4 boot-log KV back-solve — parser units, service seam, app render.

Three layers, mirroring the ADDING_MODELS.md Step-5 automation:

  1. PARSER — ``scripts/lib/profiles/bootlog_solve.py`` against captured-style
     boot-log FIXTURES (tests/fixtures/bootlogs/*.log): match / half /
     mismatch / insufficient-log, plus classification band edges and the
     log-vs-registry field-precedence rules.
  2. SERVICE — ``CockpitData.bootlog_solve``: docker logs via the read-runner
     seam + the ``bootlog_solve.py --json`` subprocess contract, both faked —
     no subprocess ever spawns (conftest enforces it).
  3. APP — the producer ③ Gate [K] binding renders the verdict card in
     ``BootlogSolveScreen`` with a FakeRunner; honest failures render as such.

Predicted constants in the fixtures' assertions are the kv-calc formula values
(KV_MATH): qwen3.6-27b dual fp8 = 16 attn × 4 kv-heads × 256 head_dim × 2 × 1.0
= 32,768 B/tok TP1; gemma-4-31b int8-PTH tied = 10 × 16 × 512 × 1 × 1.01 =
82,739.2 B/tok TP1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bootlogs"

# The parser module lives OUTSIDE the cockpit package (scripts/lib/profiles) —
# load it by file location.  Its top level is stdlib-only, so importing it
# never pulls PyYAML / kv-calc (those are lazy, inside slug_facts()).
_spec = importlib.util.spec_from_file_location(
    "bootlog_solve", REPO_ROOT / "scripts" / "lib" / "profiles" / "bootlog_solve.py"
)
bootlog_solve = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("bootlog_solve", bootlog_solve)
_spec.loader.exec_module(bootlog_solve)

parse_bootlog = bootlog_solve.parse_bootlog
back_solve = bootlog_solve.back_solve
classify = bootlog_solve.classify

from club3090_cockpit.services import CockpitData, RunResult

from tests.test_app_headless import (
    DOCKER_PS_ENGINE,
    ServingTarget,
    _settle,
    fake_responses,
    make_app,
    ok,
)
from tests.test_services import FakeRunner

# GpuInfo is needed for the app-level serving target (mirrors test_app_headless
# usage); imported lazily-shaped to keep this header readable.
from club3090_tui_core.detect import GpuInfo


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# Predicted per-token bytes (TP1-equivalent) — kv-calc formula values.
QWEN27B_DUAL_FP8_TP1 = 32768.0
GEMMA31B_INT8_TP1 = 82739.2


# ===========================================================================
# 1. Parser — captured fixtures
# ===========================================================================


class TestParseBootlog:
    def test_match_fixture_fields(self):
        p = parse_bootlog(_fixture("match-qwen27b-dual.log"))
        assert p["available_kv_gib"] == 8.00
        assert p["kv_pool_tokens"] == 524288
        assert p["max_ctx"] == 262144
        assert p["max_num_seqs"] == 2
        assert p["tp"] == 2
        assert "Available KV cache / card = 8.00 GiB" in p["evidence"]["available_kv_line"]
        assert "GPU KV cache size: 524,288 tokens" in p["evidence"]["pool_tokens_line"]
        assert p["evidence"]["gpu_memory_lines"]  # weights line kept as evidence

    def test_half_fixture_fields(self):
        # The real matched-config rebench anchor: 10.82 GiB @262K seqs=2 TP=2.
        # The registry compose says seqs=4 — the LOG value must win downstream.
        p = parse_bootlog(_fixture("half-gemma31b-int8.log"))
        assert p["available_kv_gib"] == 10.82
        assert p["max_num_seqs"] == 2
        assert p["max_ctx"] == 262144
        assert p["kv_pool_tokens"] is None  # no tokens line → back-solve path

    def test_insufficient_fixture_has_no_kv_fields(self):
        p = parse_bootlog(_fixture("insufficient-crash.log"))
        # Config lines parse fine — what's MISSING is the KV-pool report
        # (the boot OOM'd before allocation), so the solve can't happen.
        assert p["available_kv_gib"] is None
        assert p["kv_pool_tokens"] is None
        assert p["max_ctx"] == 262144
        assert p["max_num_seqs"] == 2
        assert p["tp"] == 2


class TestClassifyBands:
    def test_exact_match(self):
        verdict, _ = classify(100.0, 100.0)
        assert verdict == "match"

    def test_band_edges(self):
        # ±10% inclusive → match; just outside → not match.
        assert classify(110.0, 100.0)[0] == "match"
        assert classify(90.0, 100.0)[0] == "match"
        assert classify(110.01, 100.0)[0] != "match"

    def test_half_both_directions(self):
        # 2× either way → half (K=V tying suspect).
        assert classify(200.0, 100.0)[0] == "half"
        assert classify(50.0, 100.0)[0] == "half"
        # Band edges 1.75–2.25 inclusive.
        assert classify(180.0, 100.0)[0] == "half"
        assert classify(225.0, 100.0)[0] == "half"
        assert classify(174.0, 100.0)[0] == "mismatch"
        assert classify(226.0, 100.0)[0] == "mismatch"

    def test_half_causes_name_tying(self):
        assert "k_v_tensors=1" in classify(50.0, 100.0)[1]
        assert "NOT active" in classify(200.0, 100.0)[1]

    def test_far_off_is_mismatch_naming_growing_layers(self):
        verdict, cause = classify(320.0, 100.0)
        assert verdict == "mismatch"
        assert "growing-layer" in cause


class TestFixtureVerdicts:
    def test_match_fixture(self):
        r = back_solve(
            parse_bootlog(_fixture("match-qwen27b-dual.log")),
            predicted_tp1=QWEN27B_DUAL_FP8_TP1,
            registry_cfg={"max_ctx": 262144, "max_num_seqs": 2, "tp": 2},
        )
        assert r["verdict"] == "match"
        assert r["solve_path"] == "direct-bytes-over-tokens"
        assert r["ratio"] == 1.0
        assert r["measured_per_token_bytes_tp1"] == QWEN27B_DUAL_FP8_TP1

    def test_half_fixture(self):
        r = back_solve(
            parse_bootlog(_fixture("half-gemma31b-int8.log")),
            predicted_tp1=GEMMA31B_INT8_TP1,
            registry_cfg={"max_ctx": 262144, "max_num_seqs": 4, "tp": 2},
        )
        assert r["verdict"] == "half"
        assert r["solve_path"] == "back-solve-ctx-seqs-tp"
        # 10.82 GiB / (262144 * 2 / 2) = 44,319.23 B/tok TP1 → ratio 0.5356.
        assert r["measured_per_token_bytes_tp1"] == pytest.approx(44319.23, abs=1.0)
        assert r["ratio"] == pytest.approx(0.5356, abs=0.001)
        assert "tying" in r["suspected_cause"]
        # The LOG's seqs=2 won over the registry's 4.
        assert r["field_sources"]["max_num_seqs"] == "log"

    def test_mismatch_fixture(self):
        r = back_solve(
            parse_bootlog(_fixture("mismatch-qwen27b-dual.log")),
            predicted_tp1=QWEN27B_DUAL_FP8_TP1,
            registry_cfg={"max_ctx": 262144, "max_num_seqs": 2, "tp": 2},
        )
        assert r["verdict"] == "mismatch"
        # 2.50 GiB / 262144 = 10,240 B/tok → ratio 0.3125 (3.2× off).
        assert r["ratio"] == pytest.approx(0.3125, abs=0.001)
        assert "growing-layer" in r["suspected_cause"]

    def test_insufficient_fixture(self):
        r = back_solve(
            parse_bootlog(_fixture("insufficient-crash.log")),
            predicted_tp1=QWEN27B_DUAL_FP8_TP1,
            registry_cfg={"max_ctx": 262144, "max_num_seqs": 2, "tp": 2},
        )
        assert r["verdict"] == "insufficient-log"
        assert r["measured_per_token_bytes_tp1"] is None
        assert any("available-kv-gib" in m for m in r["missing_fields"])
        assert r["suspected_cause"]  # honest reason, never a guess


class TestFieldPrecedence:
    def test_registry_fallback_when_log_silent(self):
        parsed = parse_bootlog("INFO Available KV cache / card = 4.00 GiB")
        r = back_solve(
            parsed,
            predicted_tp1=QWEN27B_DUAL_FP8_TP1,
            registry_cfg={"max_ctx": 262144, "max_num_seqs": 2, "tp": 2},
        )
        # 4 GiB / (262144*2/2) = 16,384 B/tok TP1 → ratio 0.5 → HALF.
        assert r["verdict"] == "half"
        assert r["field_sources"] == {
            "max_ctx": "registry",
            "max_num_seqs": "registry",
            "tp": "registry",
        }

    def test_missing_registry_fallback_lists_all(self):
        parsed = parse_bootlog("INFO Available KV cache / card = 4.00 GiB")
        r = back_solve(parsed, predicted_tp1=QWEN27B_DUAL_FP8_TP1, registry_cfg=None)
        assert r["verdict"] == "insufficient-log"
        assert len(r["missing_fields"]) == 3
        assert r["measured_per_token_bytes_tp1"] is None

    def test_direct_path_still_needs_tp_for_comparison(self):
        parsed = parse_bootlog(
            "INFO GPU KV cache size: 100,000 tokens\n"
            "INFO Available KV cache / card = 1.00 GiB"
        )
        r = back_solve(parsed, predicted_tp1=QWEN27B_DUAL_FP8_TP1, registry_cfg=None)
        assert r["verdict"] == "insufficient-log"
        assert any(m.startswith("tp") for m in r["missing_fields"])


# ===========================================================================
# 2. Service seam — CockpitData.bootlog_solve (FakeRunner, no subprocess)
# ===========================================================================

_SOLVER_JSON = json.dumps(
    {
        "verdict": "match",
        "suspected_cause": "measured per-token KV matches the kv-calc prediction within ±10%",
        "measured_per_token_bytes": 16384.0,
        "measured_per_token_bytes_tp1": 32768.0,
        "predicted_per_token_bytes": 16384.0,
        "predicted_per_token_bytes_tp1": 32768.0,
        "ratio": 1.0,
        "solve_path": "direct-bytes-over-tokens",
        "available_kv_gib": 8.0,
        "kv_pool_tokens": 524288,
        "max_ctx": 262144,
        "max_num_seqs": 2,
        "tp": 2,
        "field_sources": {},
        "missing_fields": [],
        "evidence": {},
        "slug": "vllm/dual",
        "model": "qwen3.6-27b",
        "kv_format": "fp8_e4m3",
    }
)


def _bootlog_service(**overrides) -> tuple[CockpitData, FakeRunner]:
    responses = {
        "docker logs": ok(_fixture("match-qwen27b-dual.log")),
        "bootlog_solve.py": ok(_SOLVER_JSON),
    }
    responses.update(overrides)
    runner = FakeRunner(responses)
    return CockpitData(REPO_ROOT, runner=runner), runner


class TestBootlogSolveService:
    @pytest.mark.asyncio
    async def test_happy_path_pulls_logs_then_solver(self):
        cd, runner = _bootlog_service()
        res = await cd.bootlog_solve(slug="vllm/dual", container="vllm-qwen36-27b-dual")
        assert res["ok"] is True
        assert res["report"]["verdict"] == "match"
        assert res["report"]["container"] == "vllm-qwen36-27b-dual"
        # Both legs went through the injected read runner — docker logs FIRST,
        # then the solver with the slug + a temp log file.
        assert runner.calls[0][:2] == ["docker", "logs"]
        assert "vllm-qwen36-27b-dual" in runner.calls[0]
        solve_cmd = runner.calls[1]
        assert any(c.endswith("bootlog_solve.py") for c in solve_cmd)
        assert "--slug" in solve_cmd and "vllm/dual" in solve_cmd
        assert "--json" in solve_cmd

    @pytest.mark.asyncio
    async def test_docker_failure_is_honest(self):

        cd, _ = _bootlog_service(
            **{
                "docker logs": RunResult(
                    returncode=1, stdout="", stderr="Error: No such container: gone"
                )
            }
        )
        res = await cd.bootlog_solve(slug="vllm/dual", container="gone")
        assert res["ok"] is False
        assert "docker logs unavailable" in res["message"]
        assert res["report"] is None

    @pytest.mark.asyncio
    async def test_no_container_is_honest(self):
        cd, runner = _bootlog_service()
        res = await cd.bootlog_solve(slug="vllm/dual", container=None)
        assert res["ok"] is False
        assert res["report"] is None
        assert runner.calls == []  # nothing ran

    @pytest.mark.asyncio
    async def test_no_slug_is_honest(self):
        cd, runner = _bootlog_service()
        res = await cd.bootlog_solve(slug=None, container="c")
        assert res["ok"] is False
        assert "catalog slug" in res["message"]
        assert runner.calls == []

    @pytest.mark.asyncio
    async def test_solver_garbage_is_honest(self):
        cd, _ = _bootlog_service(**{"bootlog_solve.py": ok("Traceback (most recent call last)")})
        res = await cd.bootlog_solve(slug="vllm/dual", container="c")
        assert res["ok"] is False
        assert "back-solve failed" in res["message"]
        assert res["report"] is None


# ===========================================================================
# 3. App render — the [K] binding + BootlogSolveScreen with a FakeRunner
# ===========================================================================


def _serving_responses() -> dict:
    """Default fakes + a serving engine container + canned solver verdict."""
    return fake_responses(
        **{
            "docker ps": ok(DOCKER_PS_ENGINE),
            "docker logs": ok(_fixture("half-gemma31b-int8.log")),
            "bootlog_solve.py": ok(
                json.dumps(
                    {
                        "verdict": "half",
                        "suspected_cause": "measured is ~1.87x BELOW predicted — K==V tying "
                        "likely ACTIVE in the engine but not modelled (set k_v_tensors=1)",
                        "measured_per_token_bytes": 22159.61,
                        "measured_per_token_bytes_tp1": 44319.23,
                        "predicted_per_token_bytes": 41369.6,
                        "predicted_per_token_bytes_tp1": 82739.2,
                        "ratio": 0.5356,
                        "solve_path": "back-solve-ctx-seqs-tp",
                        "available_kv_gib": 10.82,
                        "kv_pool_tokens": None,
                        "max_ctx": 262144,
                        "max_num_seqs": 2,
                        "tp": 2,
                        "field_sources": {"max_ctx": "log", "max_num_seqs": "log", "tp": "log"},
                        "missing_fields": [],
                        "evidence": {
                            "available_kv_line": "Available KV cache / card = 10.82 GiB",
                        },
                        "slug": "vllm/dual",
                        "model": "qwen3.6-27b",
                        "kv_format": "fp8_e4m3",
                    }
                )
            ),
        }
    )


class TestBootlogSolveApp:
    @pytest.mark.asyncio
    async def test_k_binding_renders_verdict_card(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        app, runner, _ = make_app(responses=_serving_responses(), gpus=gpus, target=target)
        from textual.widgets import TabbedContent

        from club3090_cockpit.app import BootlogSolveScreen

        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")  # the producer Bring & Validate lane
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"  # ③ Gate
            await _settle(pilot)
            await pilot.press("K")
            await _settle(pilot)
            assert isinstance(app.screen, BootlogSolveScreen)
            body = str(app.screen.query_one("#bootlog-body").render())
            assert "HALF" in body
            assert "44,319.23" in body and "82,739.20" in body
            assert "tying" in body
            assert "10.82 GiB" in body
            # Both read legs hit the FakeRunner (docker logs → solver), never a
            # real subprocess (conftest would raise).
            assert any("docker logs" in " ".join(c) for c in runner.calls)
            assert any("bootlog_solve.py" in " ".join(c) for c in runner.calls)

    @pytest.mark.asyncio
    async def test_no_serving_model_notifies_without_modal(self):
        app, _, _ = make_app()  # default fakes: nothing serving
        from textual.widgets import TabbedContent

        from club3090_cockpit.app import BootlogSolveScreen

        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await _settle(pilot)
            await pilot.press("K")
            await _settle(pilot)
            assert not isinstance(app.screen, BootlogSolveScreen)

    @pytest.mark.asyncio
    async def test_honest_failure_renders_message_not_numbers(self):
        gpus = [GpuInfo(index=0, mem_used_mib=1), GpuInfo(index=1, mem_used_mib=1)]
        target = ServingTarget(container="vllm_qwen36_27b", host_port=8010, gpus=gpus)
        responses = fake_responses(
            **{
                "docker ps": ok(DOCKER_PS_ENGINE),
                "docker logs": RunResult(
                    returncode=1, stdout="", stderr="Error: No such container: x"
                ),
            }
        )
        app, _, _ = make_app(responses=responses, gpus=gpus, target=target)
        from textual.widgets import TabbedContent

        from club3090_cockpit.app import BootlogSolveScreen

        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot)
            await pilot.press("2")
            app.query_one("#validate-tabs", TabbedContent).active = "tab-run"
            await _settle(pilot)
            await pilot.press("K")
            await _settle(pilot)
            assert isinstance(app.screen, BootlogSolveScreen)
            body = str(app.screen.query_one("#bootlog-body").render())
            assert "unavailable" in body
            assert "docker logs unavailable" in body

    def test_verdict_card_formatter_insufficient_lists_missing(self):
        from club3090_cockpit.app import _bootlog_verdict_lines

        lines, plain = _bootlog_verdict_lines(
            {
                "ok": True,
                "message": "",
                "report": {
                    "verdict": "insufficient-log",
                    "suspected_cause": "fields missing: available-kv-gib",
                    "measured_per_token_bytes": None,
                    "measured_per_token_bytes_tp1": None,
                    "predicted_per_token_bytes": None,
                    "predicted_per_token_bytes_tp1": None,
                    "ratio": None,
                    "available_kv_gib": None,
                    "kv_pool_tokens": None,
                    "max_ctx": None,
                    "max_num_seqs": None,
                    "tp": None,
                    "field_sources": {},
                    "missing_fields": ["available-kv-gib ('Available KV cache / card = X GiB')"],
                    "evidence": {},
                },
            }
        )
        blob = "\n".join(lines)
        assert "INSUFFICIENT-LOG" in blob
        assert "available-kv-gib" in blob
        assert "—" in blob  # em-dash placeholders, never fabricated numbers
        assert plain and "INSUFFICIENT-LOG" in plain
