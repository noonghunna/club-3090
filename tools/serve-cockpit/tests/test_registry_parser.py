"""Unit tests for the registry catalog tab-row parser.

These tests use a fixture string — the real script is never called.
They are designed to run headless with no TTY, no GPU, and no Docker.
"""

from __future__ import annotations

import pytest

from club3090_cockpit.registry import parse_variant_rows, VariantRow


# ---------------------------------------------------------------------------
# Fixture data — representative tab-delimited output from registry_variant_rows
# ---------------------------------------------------------------------------

FIXTURE_OUTPUT = """\
VARIANT\tvllm/dual\tvllm\tvllm\tmodels/qwen3.6-27b/vllm/compose/dual/autoround-int4\tfp8-mtp.yml\t8010\tqwen3.6-27b\tvllm\tqwen3.6-27b:fp8-mtp\tvllm_qwen36_27b\tmodels/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml\tproduction\t295K\t
VARIANT\tbeellama/dflash\tbeellama\tbeellama\tmodels/qwen3.6-27b/beellama/compose/dual/autoround-int4\tdflash.yml\t8065\tqwen3.6-27b\tbeellama\tSKIP\tbeellama_qwen_dflash\tmodels/qwen3.6-27b/beellama/compose/dual/autoround-int4/dflash.yml\tcaveats\t102K\tDFlash prose regression
VARIANT\tik-llama/iq4ks-single\tik-llama\tik-llama\tmodels/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks\tturbo.yml\t8063\tqwen3.6-27b\tik-llama\tSKIP\tik_llama_qwen_single\tmodels/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/turbo.yml\texperimental\t200K\t
DEFAULT\tqwen3.6-27b\tvllm\tdual\tvllm/dual\t
VARIANT\tgemma-int8/dual\tvllm\tvllm\tmodels/gemma-4-31b/vllm/compose/dual/autoround-int4\tint8.yml\t8066\tgemma-4-31b\tvllm\tgemma-4-31b:int8\tvllm_gemma4_31b\tmodels/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml\tproduction\t192K\t
"""

FIXTURE_EMPTY = ""
FIXTURE_NO_VARIANTS = "DEFAULT\tqwen3.6-27b\tvllm\tdual\tvllm/dual\n"
FIXTURE_MALFORMED_LINE = "VARIANT\tonly-two-fields\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseVariantRows:
    """Tests for parse_variant_rows()."""

    def test_parses_all_variant_lines(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert len(rows) == 4

    def test_ignores_default_lines(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        slugs = [r.slug for r in rows]
        assert "qwen3.6-27b" not in slugs  # DEFAULT line slug must not appear

    def test_slug_field(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[0].slug == "vllm/dual"

    def test_engine_field(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[0].engine == "vllm"
        assert rows[1].engine == "beellama"

    def test_status_field(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[0].status == "production"
        assert rows[1].status == "caveats"
        assert rows[2].status == "experimental"

    def test_ctx_label_present(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[0].ctx_label == "295K"
        assert rows[2].ctx_label == "200K"

    def test_ctx_label_absent_becomes_empty(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        # rows[2] has empty status_note but non-empty ctx_label
        assert rows[2].ctx_label == "200K"

    def test_status_note_populated(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[1].status_note == "DFlash prose regression"

    def test_status_note_empty_when_absent(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[0].status_note == ""

    def test_port_parsed_as_int(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[0].port == 8010
        assert rows[1].port == 8065

    def test_model_field(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        assert rows[3].model == "gemma-4-31b"

    def test_stub_columns_present(self):
        """Stub columns must be present with their placeholder glyphs."""
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        for r in rows:
            assert r.fit == "·"
            assert r.tps == "—"
            assert r.quality_8pk == "—"
            assert r.source == "·"

    def test_empty_output_returns_empty_list(self):
        assert parse_variant_rows(FIXTURE_EMPTY) == []

    def test_no_variant_lines_returns_empty_list(self):
        assert parse_variant_rows(FIXTURE_NO_VARIANTS) == []

    def test_malformed_line_skipped(self):
        """A line with fewer than 13 fields is silently skipped."""
        rows = parse_variant_rows(FIXTURE_MALFORMED_LINE)
        assert rows == []

    def test_returns_variant_row_objects(self):
        rows = parse_variant_rows(FIXTURE_OUTPUT)
        for r in rows:
            assert isinstance(r, VariantRow)
