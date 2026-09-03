"""Compose viewer (Stage 1, READ-ONLY) — the data layer and the [c] surface.

The gap this closes: Catalog rows, container drills and lane stages all name a
config the user has only ever seen summarised. The serve confirm shows the
CATALOG'S CLAIMS (ctx, measured TPS, fit) rather than the file, and a Route-K or
generated serve shows only the docker command — so a user commits a compose they
have not read and cannot reopen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static, TabbedContent

from club3090_cockpit.app import ComposeViewScreen
from club3090_cockpit.data import (
    COMPOSE_STATUS_EMOJI,
    classify_compose_provenance,
    derive_compose_facts,
    parse_profile_header,
)

from .test_app_headless import _settle, make_app

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo(tmp_path):
    """A THROWAWAY repo root holding one real curated compose.

    App-level tests must never be handed the real checkout: `make_app` seeds two
    profile stubs under `<root>/scripts/lib/profiles/`, so pointing it at the
    real tree truncated two tracked files. The data-layer tests below still read
    the real composes directly — that is a pure read and is the point of them."""
    src = (
        REPO / "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml"
    )
    if not src.is_file():
        pytest.skip("needs the repo tree")
    dst = tmp_path / "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml"
    dst.parent.mkdir(parents=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


# ── Header parser ────────────────────────────────────────────────────────────


class TestProfileHeaderParser:
    def test_block_scoped_ignores_prose_status_further_down(self):
        """A `# Status:` outside the schema block must NOT be read as the header.

        This is the whole reason for a block-scoped parser rather than
        ComposeFacts.status_header's any-line regex: the gate
        (compose_registry.compose_header_status) stops at the `# ---` separator,
        and the viewer must agree with the gate or it shows a status the drift
        check disagrees with."""
        text = (
            "# ===========================\n"
            "# Profile (at-a-glance):\n"
            "#   Model:     Test\n"
            "#   Status:    🐣 Incubating\n"
            "#   Caveats:   not validated\n"
            "# ---------------------------\n"
            "#   Status:    ✅ Production   <- prose, must be ignored\n"
            "services:\n"
            "  x:\n"
            "    image: busybox\n"
        )
        h = parse_profile_header(text)
        assert h.present
        assert h.status_word == "incubating"
        assert h.fields["Model"] == "Test"
        assert not h.caveats_missing

    def test_continuation_lines_append_to_the_previous_field(self):
        """84 of 112 live composes carry a multi-line Caveats; dropping the
        continuations would render a truncated caveat, which is worse than none."""
        text = (
            "# Profile (at-a-glance):\n"
            "#   Status:    ⚠️ Production w/ caveats\n"
            "#   Caveats:   first clause;\n"
            "#              second clause;\n"
            "#              third clause\n"
            "# ---\n"
        )
        h = parse_profile_header(text)
        assert h.status_word == "caveats"
        assert "first clause" in h.fields["Caveats"]
        assert "second clause" in h.fields["Caveats"]
        assert "third clause" in h.fields["Caveats"]

    def test_missing_required_caveats_is_flagged(self):
        """A status in the caveats-required set with no Caveats line reds
        test-compose-status-drift on the user's own checkout — worth seeing in
        the viewer rather than at commit time."""
        text = "# Profile (at-a-glance):\n#   Status:    🐣 Incubating\n# ---\n"
        h = parse_profile_header(text)
        assert h.caveats_required and h.caveats_missing

    def test_no_header_is_not_an_error(self):
        h = parse_profile_header("services:\n  x:\n    image: busybox\n")
        assert not h.present and h.status_word == ""

    def test_status_emoji_map_parity_with_the_gate(self):
        """data.py DUPLICATES the map (it is stdlib-only and must not import the
        profiles package). Parity is asserted rather than assumed."""
        import sys

        sys.path.insert(0, str(REPO / "scripts" / "lib" / "profiles"))
        from compose_registry import COMPOSE_STATUS_EMOJI as GATE

        assert COMPOSE_STATUS_EMOJI == GATE

    @pytest.mark.skipif(not (REPO / "models").is_dir(), reason="needs the repo tree")
    def test_agrees_with_the_gate_on_every_shipped_compose(self):
        import sys

        sys.path.insert(0, str(REPO / "scripts" / "lib" / "profiles"))
        from compose_registry import compose_header_status

        files = sorted((REPO / "models").glob("*/*/compose/**/*.yml"))
        assert files, "no composes found"
        for f in files:
            text = f.read_text(encoding="utf-8")
            assert parse_profile_header(text).status_word == (
                compose_header_status(text) or ""
            ), f


# ── Provenance ───────────────────────────────────────────────────────────────


class TestComposeProvenance:
    def test_curated_is_not_editable(self, tmp_path):
        """The asymmetry the whole design turns on: a git-tracked curated compose
        edited in place silently diverges the checkout from upstream."""
        p = tmp_path / "models" / "m" / "vllm" / "compose" / "dual" / "q" / "base.yml"
        p.parent.mkdir(parents=True)
        p.write_text("image: x\n", encoding="utf-8")
        prov = classify_compose_provenance(str(p), tmp_path)
        assert prov.kind == "curated" and prov.editable is False
        assert "read-only" in prov.reason

    def test_local_layer_is_editable(self, tmp_path):
        p = tmp_path / "scripts" / "lib" / "profiles-local" / "composes" / "x" / "base.yml"
        p.parent.mkdir(parents=True)
        p.write_text("image: x\n", encoding="utf-8")
        prov = classify_compose_provenance(str(p), tmp_path)
        assert prov.kind == "local" and prov.editable is True

    def test_generated_is_flagged_ephemeral(self, tmp_path):
        p = tmp_path / "c3-genc-abc.yml"
        p.write_text("image: x\n", encoding="utf-8")
        prov = classify_compose_provenance(str(p), tmp_path)
        assert prov.kind == "generated" and "ephemeral" in prov.reason

    def test_external_file_outside_the_repo(self, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        p = outside / "mine.yml"
        p.write_text("image: x\n", encoding="utf-8")
        prov = classify_compose_provenance(str(p), tmp_path / "repo")
        assert prov.kind == "external" and prov.editable is True

    def test_missing_path(self, tmp_path):
        prov = classify_compose_provenance(str(tmp_path / "nope.yml"), tmp_path)
        assert prov.kind == "missing" and prov.editable is False


# ── The flag resolver regression ─────────────────────────────────────────────


class TestFlagResolverShellCollision:
    def test_bash_dash_c_does_not_hijack_max_ctx(self):
        """`entrypoint: [bash, -c, |...]` is the standard vLLM compose shape.

        flag() matched the FIRST token equal to ANY alias, so bash's -c beat
        --max-model-len and max_ctx came back as the YAML block scalar "|".
        125 of 133 shipped composes reported the wrong context this way, and
        derive_compose_facts is what ① Bring's Route-K ingestion reads."""
        text = (
            "services:\n"
            "  vllm:\n"
            "    image: vllm/vllm-openai:v0.27.1\n"
            "    entrypoint:\n"
            "      - bash\n"
            "      - -c\n"
            "      - |\n"
            "        exec vllm serve \\\n"
            "    command:\n"
            "      - --max-model-len\n"
            '      - "${MAX_MODEL_LEN:-262144}"\n'
        )
        facts = derive_compose_facts(text)
        assert facts.max_ctx == "${MAX_MODEL_LEN:-262144}"

    def test_llama_cpp_short_ctx_flag_still_read(self):
        """The -c alias must keep working when it is genuinely the engine's."""
        text = (
            "services:\n"
            "  llama:\n"
            "    image: llama.cpp:latest\n"
            "    command:\n"
            "      - -c\n"
            "      - 65536\n"
        )
        assert derive_compose_facts(text).max_ctx == "65536"

    @pytest.mark.skipif(not (REPO / "models").is_dir(), reason="needs the repo tree")
    def test_no_shipped_compose_reports_a_block_scalar_ctx(self):
        for f in sorted((REPO / "models").glob("*/*/compose/**/*.yml")):
            ctx = derive_compose_facts(f.read_text(encoding="utf-8")).max_ctx
            assert ctx not in ("|", ">", "|-", ">-", "|+", ">+"), f


# ── The [c] surface ──────────────────────────────────────────────────────────


class TestViewComposeKey:
    @pytest.mark.asyncio
    async def test_c_on_catalog_opens_the_slugs_compose(self, repo):
        """The anchor placement: [c] on a Catalog row opens the file that slug
        will actually run."""
        app, _, _ = make_app(repo_root=repo, surface="producer")
        async with app.run_test(size=(120, 44)) as pilot:
            await _settle(pilot)
            path, title = app._compose_target_for_focus()
            assert path.endswith(".yml") and title
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, ComposeViewScreen)
            prov = str(app.screen.query_one("#compose-provenance", Static).render())
            assert "CURATED" in prov and "read-only" in prov

    @pytest.mark.asyncio
    async def test_viewer_shows_facts_and_yaml_and_copies_raw(self, repo):
        app, _, _ = make_app(repo_root=repo, surface="producer")
        async with app.run_test(size=(120, 44)) as pilot:
            await _settle(pilot)
            path, _ = app._compose_target_for_focus()
            await pilot.press("c")
            await _settle(pilot)
            facts = str(app.screen.query_one("#compose-facts", Static).render())
            assert "Status" in facts
            # ${VAR:-default} is unwrapped for display, not shown raw.
            assert "${MAX_MODEL_LEN" not in facts
            # [Y] copies the RAW file, not the rendering.
            raw = (repo / path).read_text(encoding="utf-8")
            assert app.screen.copyable_text() == raw

    @pytest.mark.asyncio
    async def test_c_is_inert_on_orchestration_where_it_is_power_cap(self, repo):
        """The two `c` bindings must have DISJOINT gates — Orchestration keeps
        power-cap, so the viewer must not fire there."""
        app, _, _ = make_app(repo_root=repo, surface="producer")
        async with app.run_test(size=(120, 44)) as pilot:
            await _settle(pilot)
            app.query_one("#operate-tabs", TabbedContent).active = "tab-orchestration"
            await _settle(pilot)
            assert app.check_action("view_compose", ()) is False
            assert app.check_action("power_cap", ()) is not False

    @pytest.mark.asyncio
    async def test_esc_and_c_both_close(self, repo):
        app, _, _ = make_app(repo_root=repo, surface="producer")
        async with app.run_test(size=(120, 44)) as pilot:
            await _settle(pilot)
            await pilot.press("c")
            await _settle(pilot)
            assert isinstance(app.screen, ComposeViewScreen)
            await pilot.press("c")
            await _settle(pilot)
            assert not isinstance(app.screen, ComposeViewScreen)
