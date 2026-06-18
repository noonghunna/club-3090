"""Tests for the parsers module — exercises all output formats from spec Section 10."""

from __future__ import annotations

import pytest

from club3090_test_console.parsers import (
    BenchParser,
    VerifyParser,
    StressParser,
    QualityParser,
    SoakParser,
    RebenchParser,
    Status,
    strip_ansi,
)


# ============================================================================
# Test strip_ansi
# ============================================================================

def test_strip_ansi():
    assert strip_ansi("\033[32m✓\033[0m test") == "✓ test"
    assert strip_ansi("no codes here") == "no codes here"
    assert strip_ansi("\033[1;31m✗\033[0m fail") == "✗ fail"


# ============================================================================
# Test BenchParser
# ============================================================================

class TestBenchParser:
    def test_section_header(self):
        p = BenchParser()
        event = p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        assert event is not None
        assert event.event_type == "bench_section"
        assert event.data["section"] == "narrative"

    def test_warmup_line(self):
        p = BenchParser()
        p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        event = p.parse_line("  warm-1     wall= 12.05s  ttft=   120ms  toks=1000  wall_TPS= 83.03  decode_TPS= 83.86")
        assert event is not None
        assert event.event_type == "bench_run"
        assert event.data["type"] == "warm"
        assert event.data["run"] == 1
        assert event.data["wall_s"] == 12.05
        assert event.data["ttft_ms"] == 120
        assert event.data["tokens"] == 1000
        assert event.data["wall_tps"] == 83.03
        assert event.data["decode_tps"] == 83.86

    def test_measured_run_line(self):
        p = BenchParser()
        p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        event = p.parse_line("  run-1      wall= 11.53s  ttft=   118ms  toks=1000  wall_TPS= 86.68  decode_TPS= 87.58")
        assert event is not None
        assert event.event_type == "bench_run"
        assert event.data["type"] == "run"
        assert event.data["wall_tps"] == 86.68
        assert event.data["decode_tps"] == 87.58
        assert len(p.runs["narrative"]) == 1

    def test_summary_metric(self):
        p = BenchParser()
        p.current_section = "narrative"
        event = p.parse_line("  wall_TPS       mean=  84.14   std=  2.41   CV= 2.9%   min=81.89   max=86.68")
        assert event is not None
        assert event.event_type == "summary_metric"
        assert event.data["metric"] == "wall_TPS"
        assert event.data["mean"] == 84.14
        assert event.data["cv"] == 2.9

    def test_decode_tps_summary(self):
        p = BenchParser()
        p.current_section = "narrative"
        event = p.parse_line("  decode_TPS     mean=  84.99   std=  2.45   CV= 2.9%   min=82.70   max=87.58")
        assert event is not None
        assert event.data["metric"] == "decode_TPS"
        assert event.data["mean"] == 84.99

    def test_ttft_summary(self):
        p = BenchParser()
        p.current_section = "narrative"
        event = p.parse_line("  TTFT          mean=   119ms  std=    1ms  min=118ms  max=120ms")
        assert event is not None
        assert event.event_type == "summary_ttft"
        assert event.data["ttft_mean_ms"] == 119

    def test_multiple_runs_accumulate(self):
        p = BenchParser()
        p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        p.parse_line("  run-1      wall= 11.53s  ttft=   118ms  toks=1000  wall_TPS= 86.68  decode_TPS= 87.58")
        p.parse_line("  run-2      wall= 11.93s  ttft=   119ms  toks=1000  wall_TPS= 83.81  decode_TPS= 84.65")
        assert len(p.runs["narrative"]) == 2

    def test_unmatched_line_returns_none(self):
        p = BenchParser()
        assert p.parse_line("some random log line") is None


# ============================================================================
# Test VerifyParser
# ============================================================================

class TestVerifyParser:
    def test_step_header(self):
        p = VerifyParser()
        event = p.parse_line("[3/9] Basic completion — capital of France ...")
        assert event is not None
        assert event.event_type == "verify_step"
        assert event.data["step"] == 3
        assert event.data["total"] == 9
        assert "capital of France" in event.data["name"]

    def test_pass_check(self):
        p = VerifyParser()
        event = p.parse_line("  ✓ reply contains 'Paris'")
        assert event is not None
        assert event.event_type == "verify_check"
        assert event.data["status"] == "passed"
        assert event.data["glyph"] == "✓"

    def test_fail_check(self):
        p = VerifyParser()
        event = p.parse_line("  ✗ tool-call request failed")
        assert event is not None
        assert event.data["status"] == "failed"
        assert event.data["glyph"] == "✗"

    def test_skip_check(self):
        p = VerifyParser()
        event = p.parse_line("  ⊘ Genesis patches applied (skipped)")
        assert event is not None
        assert event.data["status"] == "skipped"
        assert event.data["glyph"] == "⊘"

    def test_hint(self):
        p = VerifyParser()
        p.current_step = {"num": 1, "checks": [{"message": "test"}]}
        event = p.parse_line("    → Check docker logs vllm-qwen36-27b")
        assert event is not None
        assert event.event_type == "verify_hint"
        assert "docker logs" in event.data["hint"]

    def test_all_passed_verdict(self):
        p = VerifyParser()
        event = p.parse_line("All checks passed. Stack is ready for full-functionality use.")
        assert event is not None
        assert event.event_type == "verdict"
        assert event.data["status"] == Status.PASSED

    def test_checks_failed_verdict(self):
        p = VerifyParser()
        event = p.parse_line("3 check(s) failed. See hints above.")
        assert event is not None
        assert event.event_type == "verdict"
        assert event.data["status"] == Status.FAILED
        assert event.data["failed"] == 3

    def test_ansi_stripped(self):
        p = VerifyParser()
        event = p.parse_line("  \033[32m✓\033[0m reply contains 'Paris'")
        assert event is not None
        assert event.data["status"] == "passed"

    def test_counters(self):
        p = VerifyParser()
        p.parse_line("  ✓ test 1")
        p.parse_line("  ✗ test 2")
        p.parse_line("  ⊘ test 3")
        assert p.passed == 1
        assert p.failed == 1
        assert p.skipped == 1


# ============================================================================
# Test StressParser
# ============================================================================

class TestStressParser:
    def test_probe_header(self):
        p = StressParser()
        event = p.parse_line("[1/8] Long-context needle small rungs (10K / 30K) ...")
        assert event is not None
        assert event.event_type == "stress_probe"
        assert event.data["probe"] == 1

    def test_token_pass(self):
        p = StressParser()
        event = p.parse_line("    ✓  10000 tokens: recalled '…' (got: …)")
        assert event is not None
        assert event.event_type == "niah_token"
        assert event.data["tokens"] == 10000
        assert event.data["status"] == "passed"

    def test_token_partial(self):
        p = StressParser()
        event = p.parse_line("    △  30000 tokens: recall MISS (…) — system OK, quality ceiling reached")
        assert event is not None
        assert event.data["status"] == "partial"
        assert event.data["glyph"] == "△"

    def test_rung_pass(self):
        p = StressParser()
        event = p.parse_line("    ✓ rung 1/6: target=95K  actual=95K tok (36%)  recalled '…'  prefill=… t/s (…s)  VRAM_free=…MB")
        assert event is not None
        assert event.event_type == "niah_rung"
        assert event.data["rung"] == 1
        assert event.data["total"] == 6
        assert event.data["target_k"] == 95
        assert event.data["status"] == "passed"

    def test_rung_partial(self):
        p = StressParser()
        event = p.parse_line("    △ rung 2/6: target=125K  actual=125K tok (47%)  recall MISS (…) — quality ceiling reached")
        assert event is not None
        assert event.data["status"] == "partial"

    def test_rung_failed(self):
        p = StressParser()
        event = p.parse_line("    ✗ rung 3/6: target=155K  HTTP 500 (OOM at ~59% of n_ctx=262000)")
        assert event is not None
        assert event.data["status"] == "failed"

    def test_rung_skipped(self):
        p = StressParser()
        event = p.parse_line("    ⊘ rung 4/6: target=185K  HTTP 400 (exceeds engine limit — clean rejection)")
        assert event is not None
        assert event.data["status"] == "skipped"

    def test_ladder_info(self):
        p = StressParser()
        event = p.parse_line("    n_ctx=262000  ladder: 95000 → 125000 → 155000 → 185000 → 215000 → 241000 (6 rungs)")
        assert event is not None
        assert event.event_type == "niah_ladder"
        assert event.data["n_ctx"] == 262000

    def test_vram_free(self):
        p = StressParser()
        event = p.parse_line("    VRAM free (ladder start): 12345 MB")
        assert event is not None
        assert event.data["vram_free_mb"] == 12345

    def test_all_passed(self):
        p = StressParser()
        event = p.parse_line("All stress / boundary checks passed. KV-cache and prefill paths are sound for the deployed config.")
        assert event is not None
        assert event.data["status"] == Status.PASSED

    def test_some_failed(self):
        p = StressParser()
        event = p.parse_line("3 stress check(s) failed. See hints above.")
        assert event is not None
        assert event.data["status"] == Status.FAILED
        assert event.data["failed"] == 3


# ============================================================================
# Test QualityParser
# ============================================================================

class TestQualityParser:
    def test_scenario_pass(self):
        p = QualityParser()
        event = p.parse_line("  [1/15] TC-01 ✓ passed (2.3s)")
        assert event is not None
        assert event.event_type == "quality_scenario"
        assert event.data["num"] == 1
        assert event.data["total"] == 15
        assert event.data["scenario_id"] == "TC-01"
        assert event.data["passed"] is True
        assert event.data["elapsed_s"] == 2.3
        assert event.data["pack_id"] == "toolcall-15"

    def test_scenario_fail(self):
        p = QualityParser()
        event = p.parse_line("  [7/15] TC-07 ✗ verifier_fail (3.1s)")
        assert event is not None
        assert event.data["passed"] is False
        assert event.data["failure_mode"] == "verifier_fail"
        assert event.data["elapsed_s"] == 3.1

    def test_pack_id_mapping(self):
        p = QualityParser()
        # IF = instructfollow
        event = p.parse_line("  [1/15] IF-01 ✓ passed (1.6s)")
        assert event.data["pack_id"] == "instructfollow-15"

        # SO = structoutput
        event = p.parse_line("  [1/15] SO-01 ✓ passed (1.2s)")
        assert event.data["pack_id"] == "structoutput-15"

        # RM = reasonmath
        event = p.parse_line("  [1/15] RM-01 ✓ passed (2.0s)")
        assert event.data["pack_id"] == "reasonmath-15"

    def test_totals_accumulate(self):
        p = QualityParser()
        p.parse_line("  [1/15] TC-01 ✓ passed (2.3s)")
        p.parse_line("  [2/15] TC-02 ✓ passed (1.5s)")
        p.parse_line("  [3/15] TC-03 ✗ verifier_fail (3.1s)")
        assert p.total_passed == 2
        assert p.total_count == 3

    def test_pack_tracking(self):
        p = QualityParser()
        p.parse_line("  [1/15] TC-01 ✓ passed (2.3s)")
        p.parse_line("  [2/15] TC-02 ✓ passed (1.5s)")
        assert p.packs["toolcall-15"]["passed"] == 2
        assert p.packs["toolcall-15"]["total"] == 2

    def test_total_line(self):
        p = QualityParser()
        event = p.parse_line("TOTAL 120/150")
        assert event is not None
        assert event.event_type == "quality_total"
        assert event.data["passed"] == 120
        assert event.data["total"] == 150


# ============================================================================
# Test SoakParser
# ============================================================================

class TestSoakParser:
    def test_config_line(self):
        p = SoakParser()
        event = p.parse_line("[soak] mode=fresh sessions=20 turns=5 max_growth=200MiB timeout=1800s")
        assert event is not None
        assert event.event_type == "soak_config"
        assert event.data["mode"] == "fresh"
        assert event.data["sessions"] == 20
        assert event.data["turns"] == 5

    def test_session(self):
        p = SoakParser()
        event = p.parse_line("[soak] session 1/20")
        assert event is not None
        assert event.event_type == "soak_session"
        assert event.data["session"] == 1
        assert event.data["total"] == 20

    def test_turn(self):
        p = SoakParser()
        p.current_session = 1
        event = p.parse_line("[soak]   turn 1/5: status=200 wall=5159ms ttft=481ms decode_tps=42.113 vram=43104MiB")
        assert event is not None
        assert event.event_type == "soak_turn"
        assert event.data["turn"] == 1
        assert event.data["status"] == 200
        assert event.data["wall_ms"] == 5159
        assert event.data["ttft_ms"] == 481
        assert event.data["decode_tps"] == 42.113
        assert event.data["vram_mib"] == 43104

    def test_baseline(self):
        p = SoakParser()
        event = p.parse_line("[soak] warm baseline after session 1: 43104 MiB")
        assert event is not None
        assert event.event_type == "soak_baseline"
        assert event.data["vram_mib"] == 43104

    def test_verdict_pass(self):
        p = SoakParser()
        event = p.parse_line("[soak]   verdict              PASS")
        assert event is not None
        assert event.event_type == "verdict"
        assert event.data["status"] == Status.PASSED
        assert event.data["verdict"] == "PASS"

    def test_verdict_fail(self):
        p = SoakParser()
        event = p.parse_line("[soak]   verdict              FAIL")
        assert event.data["status"] == Status.FAILED

    def test_metrics(self):
        p = SoakParser()
        event = p.parse_line("[soak]   silent_empty         0 / 100 (0.0%)")
        assert event is not None
        assert event.data["key"] == "silent_empty"
        assert "0 / 100" in event.data["value"]

        event = p.parse_line("[soak]   tps_retention        100.0%")
        assert event.data["key"] == "tps_retention"

        event = p.parse_line("[soak]   p50_decode_tps       42.23")
        assert event.data["key"] == "p50_decode_tps"


# ============================================================================
# Test RebenchParser
# ============================================================================

class TestRebenchParser:
    def test_step_running(self):
        p = RebenchParser()
        event = p.parse_line("[verify-full] running…")
        assert event is not None
        assert event.event_type == "rebench_step_start"
        assert event.data["step"] == "verify-full"
        assert p.current_step == "verify-full"

    def test_step_passed(self):
        p = RebenchParser()
        event = p.parse_line("[verify-full] ✓ 96s — log: results/rebench/test-tag/verify-full.log")
        assert event is not None
        assert event.event_type == "rebench_step_done"
        assert event.data["step"] == "verify-full"
        assert event.data["status"] == Status.PASSED
        assert event.data["elapsed_s"] == 96

    def test_step_failed(self):
        p = RebenchParser()
        event = p.parse_line("[bench] ✗ 14s — failed (rc=1) — log: …")
        assert event is not None
        assert event.data["status"] == Status.FAILED
        assert event.data["rc"] == 1

    def test_step_skipped(self):
        p = RebenchParser()
        event = p.parse_line("[quality-full] skipped — 8-pack is opt-in (pass --with-8pack-thinking=off|both)")
        assert event is not None
        assert event.data["status"] == Status.SKIPPED

    def test_report_path(self):
        p = RebenchParser()
        event = p.parse_line("  report:      results/rebench/test-tag/REPORT.md")
        assert event is not None
        assert event.data["path"] == "results/rebench/test-tag/REPORT.md"

    def test_artifacts_dir(self):
        p = RebenchParser()
        event = p.parse_line("  artifacts:   results/rebench/test-tag")
        assert event is not None
        assert event.data["dir"] == "results/rebench/test-tag"

    def test_complete(self):
        p = RebenchParser()
        event = p.parse_line(" rebench complete")
        assert event is not None
        assert event.event_type == "rebench_complete"
        assert p.complete is True

    def test_full_sequence(self):
        """Test a complete rebench sequence."""
        p = RebenchParser()
        p.parse_line("[verify-full] running…")
        assert p.steps["verify-full"]["status"] == Status.RUNNING

        p.parse_line("[verify-full] ✓ 96s — log: results/rebench/tag/verify-full.log")
        assert p.steps["verify-full"]["status"] == Status.PASSED

        p.parse_line("[bench] running…")
        p.parse_line("[bench] ✓ 312s — log: results/rebench/tag/bench.log")
        assert p.steps["bench"]["status"] == Status.PASSED

        p.parse_line("[quality-full] skipped — 8-pack is opt-in")
        assert p.steps["quality-full"]["status"] == Status.SKIPPED

        p.parse_line(" rebench complete")
        assert p.complete is True


# ============================================================================
# Test parser factory
# ============================================================================

class TestParserFactory:
    def test_get_parser(self):
        from club3090_test_console.parsers import get_parser, TestType
        assert isinstance(get_parser(TestType.BENCH), BenchParser)
        assert isinstance(get_parser(TestType.VERIFY), VerifyParser)
        assert isinstance(get_parser(TestType.VERIFY_FULL), VerifyParser)
        assert isinstance(get_parser(TestType.VERIFY_STRESS), StressParser)
        assert isinstance(get_parser(TestType.QUALITY), QualityParser)
        assert isinstance(get_parser(TestType.SOAK), SoakParser)
        assert isinstance(get_parser(TestType.REBENCH), RebenchParser)
