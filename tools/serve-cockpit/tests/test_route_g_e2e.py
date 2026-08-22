"""Route-G ALL-LEGS integration test — one test walks the full GGUF journey.

Completeness-audit follow-up: the individual legs each had unit coverage, but
nothing proved the CHAIN hands off — each leg consumes the PREVIOUS leg's
actual output. This module is that one walk, hermetic (no network, no GPU):

  Leg 1  INSPECT   — deriver.inspect_repo() against a recorded-fixture fetcher
                     (a fake HF ?blobs=true API whose sibling size IS the
                     synthetic fixture's real byte length) → inventory shows
                     the Q4_K_M gguf variant.
  Leg 2  FIT       — gguf_header_facts() reads the ACTUAL fixture bytes from
                     the CONTRACT-2 pull dir (HF_HOME pointed at tmp), then
                     byo_check_gguf() judges the picked variant against a stub
                     sibling slug → verdict fits-clean, route "G", header dims
                     threaded onto ByoResult.facts.
  Leg 3  SCAFFOLD  — compute_promote_scaffold() consumes that ByoResult →
                     arch placeholders AUTO-FILLED from the leg-2 facts.
  Leg 4  PROMOTE   — the REAL promote.py CLI (--layer local, throwaway --root
                     built like scripts/tests/test_promote.py) consumes the
                     edited leg-3 spec → profiles-local artifacts written +
                     get_registry(root) sees the incubating entry.

Real vs stubbed (honest map):
  REAL:   deriver inventory mapping, local GGUF header parse + spec-fact
          mapping, byo_check_gguf verdict math + facts threading,
          compute_promote_scaffold, the promote.py executor subprocess, the
          compose_registry local-layer loader/get_registry merge.
  STUBBED: the HF API transport (recorded-fetcher fake — the deriver's own
          documented test seam), the sibling catalog slug (a token-stub
          ``llamacpp/dual/…`` outside the registry; topology falls back to the
          slug's own ``dual`` token), and the compose body handed to the
          scaffold (a minimal literal — the route-G emit path has its own
          phase-4 unit coverage and is not one of the audited legs).

    tools/serve-cockpit/.venv/bin/python -m pytest tools/serve-cockpit/tests/test_route_g_e2e.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from club3090_cockpit.data import compute_promote_scaffold  # noqa: E402
from club3090_cockpit.services import CockpitData  # noqa: E402
from scripts.lib.profiles import deriver as D  # noqa: E402

REPO = "org/Synth-7B-GGUF"
GGUF_NAME = "Synth-7B-Q4_K_M.gguf"
# Stub sibling: NOT a registry slug — topology_cards_for_profile falls back to
# scanning the string itself, so "dual" ⇒ 2 cards ⇒ 48 GiB budget.
SIBLING = "llamacpp/dual/synth-7b-gguf"
MID = "route-g-synth-7b"


# ---------------------------------------------------------------------------
# Synthetic GGUF builder — reused verbatim from scripts/tests/
# test-gguf-header.sh's in-test builder (stdlib struct; the scalar format table
# comes from the deriver itself so the two builders cannot drift).
# ---------------------------------------------------------------------------
def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def build_gguf(kv: dict, version: int = 3, tensor_count: int = 291) -> bytes:
    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kv))
    for key, (vt, val) in kv.items():
        out += _s(key) + struct.pack("<I", vt)
        if vt == 8:
            out += _s(val)
        elif vt == 9:  # array of strings (the tokenizer shape)
            out += struct.pack("<I", 8) + struct.pack("<Q", len(val))
            for item in val:
                out += _s(item)
        else:
            out += struct.pack(D._GGUF_SCALAR_FMTS[vt], val)
    return out


ARCH_KV = {
    "general.architecture": (8, "llama"),
    "general.name": (8, "Synth-7B"),
    "general.file_type": (4, 15),              # LLAMA_FTYPE_MOSTLY_Q4_K_M
    "general.quantization_version": (10, 2),
    "llama.block_count": (4, 32),
    "llama.embedding_length": (10, 4096),
    "llama.attention.head_count": (10, 32),
    "llama.attention.head_count_kv": (10, 8),
    "llama.attention.key_length": (10, 128),
    "llama.context_length": (10, 131072),
}
TOKENIZER_KV = {"tokenizer.ggml.tokens": (9, ["token-" + "x" * 16] * 4096)}
GGUF_BLOB = build_gguf({**ARCH_KV, **TOKENIZER_KV})


class _RecordedApiFetcher:
    """The deriver's injectable-fetcher seam, fed a recorded HF API payload.

    No socket is ever opened; the sibling ``size`` is the FIXTURE's true byte
    length so the leg-1 inventory numbers are real, not invented."""

    def __init__(self, api: dict):
        self._api = api
        self.urls: list[str] = []

    def get(self, url, headers=None, range_=None):
        self.urls.append(url)
        return D.FetchResponse(status=200, body=json.dumps(self._api).encode("utf-8"))


def test_route_g_all_legs_end_to_end(tmp_path: Path, monkeypatch):
    hf_home = tmp_path / "hf-home"

    # ── Leg 1: INSPECT — synthetic fixture + recorded API → gguf inventory ──
    api = {
        "id": REPO,
        "siblings": [{"rfilename": GGUF_NAME, "size": len(GGUF_BLOB)}],
    }
    inv = D.inspect_repo(REPO, fetcher=_RecordedApiFetcher(api))

    assert inv.get("error") is None, inv
    assert inv["repo"] == REPO
    assert inv["formats"] == ["gguf"], inv["formats"]
    [variant] = inv["gguf_variants"]
    # Single-variant repo: the display label IS the file stem (the common
    # prefix only strips when >1 variant shares it) — the real contract.
    assert variant["quant"] == "Synth-7B-Q4_K_M"
    assert variant["parts"] == 1
    assert variant["files"] == [GGUF_NAME]
    assert variant["size_gb"] == round(len(GGUF_BLOB) / 1024**3, 4)

    # HANDOFF 1→2: the fixture lands in the CONTRACT-2 pull dir the cockpit
    # computes (same sanitize/pull-dir code pull.sh uses), so leg 2 reads the
    # very bytes leg 1 inventoried.
    monkeypatch.setenv("HF_HOME", str(hf_home))
    data = CockpitData(REPO_ROOT)
    pull_dir = data.bring_pull_dir(REPO)
    pull_dir.mkdir(parents=True, exist_ok=True)
    (pull_dir / GGUF_NAME).write_bytes(GGUF_BLOB)
    assert data.bring_weights_present(REPO) is True

    # ── Leg 2: FIT — header facts off the real bytes, then byo_check_gguf ──
    # ModelSpec M3: gguf_header_facts returns the TYPED, provenance-labeled
    # spec (every Fact sourced "gguf-header:<arch>.<kv>", provenance
    # "derived-estimate").
    mspec = asyncio.run(
        data.gguf_header_facts(REPO, [GGUF_NAME], size_gb=variant["size_gb"])
    )
    assert mspec, "header facts must come from the on-disk fixture"
    assert mspec.confidence == "estimated-lower-bound"
    assert mspec.num_hidden_layers.value == 32
    assert mspec.hidden_size.value == 4096
    assert mspec.num_attn_heads.value == 32
    assert mspec.num_kv_heads.value == 8
    assert mspec.head_dim_attn.value == 128
    assert mspec.max_ctx_supported.value == 131072
    # weights_total_gb is the LEG-1 inventory number, threaded through.
    assert mspec.weights_total_gb.value == variant["size_gb"]
    for dim in ("hidden_size", "num_hidden_layers", "num_attn_heads",
                "num_kv_heads", "head_dim_attn", "max_ctx_supported"):
        f = getattr(mspec, dim)
        assert f.provenance == "derived-estimate"
        assert f.source.startswith("gguf-header:llama."), dim

    byo = data.byo_check_gguf(
        REPO,
        SIBLING,
        quant=variant["quant"],
        size_gb=variant["size_gb"],
        spec=mspec,
    )
    assert byo.error == ""
    assert byo.arch == "gguf"
    assert byo.eligible is True
    assert byo.route == "G"
    assert byo.sibling_slug == SIBLING
    assert byo.quant_match == variant["quant"]
    assert byo.fit_verdict == "fits-clean"
    assert "48 GiB (dual)" in byo.note  # budget from the stub sibling's topology
    # HANDOFF 2→3: the header-derived dims ride ON the result.
    assert byo.facts.num_hidden_layers.value == 32
    assert byo.facts.head_dim_attn.value == 128

    # ── Leg 3: SCAFFOLD — compute_promote_scaffold fills the arch dims ──
    compose_text = (
        "services:\n"
        "  llm:\n"
        "    image: ghcr.io/ggerganov/llama.cpp:server\n"
        "    command: >-\n"
        "      -m /models/model.gguf\n"
        "    ports:\n"
        "      - '8080:8080'\n"
    )
    scaffold = compute_promote_scaffold(
        byo=byo,
        measurement=None,
        model_id=MID,
        compose_text=compose_text,
        layer="local",
    )
    assert not scaffold.error, scaffold.error
    assert scaffold.computed
    assert scaffold.layer == "local"
    spec = scaffold.spec
    assert spec["model_id"] == MID

    arch = spec["arch"]
    assert arch["num_hidden_layers"] == 32
    assert arch["hidden_size"] == 4096
    assert arch["num_attn_heads"] == 32
    assert arch["num_kv_heads"] == 8
    assert arch["head_dim_attn"] == 128
    assert arch["max_ctx_supported"] == 131072
    # AUTO-FILLED, not placeholder: no `<int>` survives anywhere in arch.
    assert "<" not in json.dumps(arch)

    [weight_meta] = spec["weights"].values()
    assert weight_meta["size_gb"] == round(variant["size_gb"], 2)
    assert weight_meta["hf_repo"] == REPO

    slug = spec["registry_entry"]["slug"]
    assert slug.startswith("local/")
    assert spec["registry_entry"]["kwargs"]["model"] == MID
    assert spec["registry_entry"]["kwargs"]["max_ctx"] == 131072
    assert spec["registry_entry"]["kwargs"]["status"] == "incubating"
    assert spec["compose"]["content"] == compose_text

    # The screen's REQUIRED inline edits before staging (display_name/family
    # stay placeholders by design) — applied here as the 3→4 handoff.
    spec["display_name"] = "Synth 7B (GGUF)"
    spec["family"] = "generic-dense"

    # ── Leg 4: PROMOTE — the real CLI against a throwaway repo root ──
    root = tmp_path / "throwaway-repo"
    # ignore_patterns must cover boot residue: real rigs accumulate
    # models/*/vllm/cache/torch_compile/... (root-owned container writes)
    # that both explode copytree on permission and are irrelevant to the
    # registry under test.
    _copy_ignore = shutil.ignore_patterns("__pycache__", "cache", "*.log")
    for rel in ("scripts/lib", "tools/tui-core", "models"):
        shutil.copytree(
            REPO_ROOT / rel,
            root / rel,
            ignore=_copy_ignore,
        )
    # Deterministic local layer: drop any dev-time registry.local.json so the
    # post-write get_registry assertion sees ONLY this promotion.
    local_reg = root / "scripts/lib/profiles-local/registry.local.json"
    if local_reg.exists():
        local_reg.unlink()

    spec_file = tmp_path / "promote-spec.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ)
    env.pop("C3_ALLOW_CORE_PROMOTE", None)  # local path must never need it
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/lib/profiles/promote.py"),
            "--spec-file",
            str(spec_file),
            "--layer",
            "local",
            "--root",
            str(root),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"promote.py failed rc={proc.returncode}\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert f"PROMOTE_OK {slug}" in proc.stdout

    # The three profiles-local artifacts exist…
    profile_yml = root / "scripts/lib/profiles-local/models.d" / f"{MID}.yml"
    compose_yml = root / spec["compose"]["path"]
    assert profile_yml.is_file()
    assert compose_yml.is_file()
    raw_local = json.loads(local_reg.read_text(encoding="utf-8"))
    assert slug in raw_local
    assert raw_local[slug]["status"] == "incubating"

    # …and the SINGLE merged view every runtime consumer reads sees the entry.
    from scripts.lib.profiles.compose_registry import get_registry

    merged = get_registry(root=str(root))
    entry = merged[slug]
    assert entry["model"] == MID
    assert entry["status"] == "incubating"
    assert entry["max_ctx"] == 131072
    assert entry["compose_path"] == spec["compose"]["path"]
