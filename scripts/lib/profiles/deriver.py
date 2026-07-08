"""v0.8.0 Pull-Gate — `[A]` transformers deriver.

Given an HF repo slug, derive a ModelProfile-shaped spec from the repo's own
`config.json` + HF Hub model API (`?blobs=true` for LFS-resolved sizes) +
a bounded, pre-download safetensors-header probe. This is the `[A]` slice
that produces the spec P1's generic-dense `[B]` branch consumes.

Public API (stable for P3/P4):

    from scripts.lib.profiles import deriver

    res = deriver.derive(slug, *, hf_token=None, hf_home=None, fetcher=None,
                          profiles=None)
    #   res: DeriveResult
    #     .error           -> DeriverError | None  (stratum-1 structured error)
    #     .tier1           -> Tier1Match | None     (curated lookup hit)
    #     .confidence      -> Confidence enum
    #     .generic_dense_eligible -> bool | None
    #     .spec            -> dict | None   (kv-calc generic-dense spec shape)
    #     .profile         -> dict | None   (derived ModelProfile-shaped dict)

P2 ONLY classifies. The stratum-5 `no-fit-model` abort, `[C0]`/`[C2a]`/`[C1]`
and the orchestrator are P3/P4 — this module never raises a traceback for a
stratum-1 condition; it returns a structured `DeriverError`.

Network: all HTTP goes through an injectable `fetcher` (see `HttpFetcher`).
Tests pass a recorded-fixture fetcher so there is NO live network and NO
weight file is ever downloaded — the header probe is range-bounded.

How P1's kv-calc is imported (per the in-file import contract at
`tools/kv-calc.py` ~line 645): via importlib, registering the module in
`sys.modules["kv_calc"]` BEFORE `exec_module` so `@dataclass` resolves
`cls.__module__`. See `_load_kv_calc()`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
_HF_API = "https://huggingface.co/api/models"
_HF_RESOLVE = "https://huggingface.co"
_NET_TIMEOUT = 30  # seconds (per brief)
_MAX_HEADER_BYTES = 16 * 1024 * 1024  # 16 MiB safetensors-header ceiling

# Name patterns that mark a *.safetensors blob as an adapter/LoRA, not a
# complete weight set. Matched case-insensitively on the basename.
_ADAPTER_PATTERNS = (
    "adapter_model",
    "adapter-model",
    "lora",
    "/adapter",
)


# ---------------------------------------------------------------------------
# Structured stratum-1 errors (NEVER raised as raw tracebacks)
# ---------------------------------------------------------------------------
class DeriverErrorKind(str, Enum):
    REPO_NOT_FOUND = "repo-not-found"
    GATED_NO_TOKEN = "gated-no-token"
    UNSUPPORTED_FORMAT = "unsupported-format"
    AMBIGUOUS_WEIGHT_SET = "ambiguous-weight-set"
    QUANT_DTYPE_UNKNOWN = "quant-dtype-unknown"


@dataclass(frozen=True)
class DeriverError:
    """A stratum-1 structured error. Returned, never raised."""

    kind: DeriverErrorKind
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind.value}: {self.detail}" if self.detail else self.kind.value


class Confidence(str, Enum):
    """§4 confidence tiers. `DERIVED` is RESERVED for the future
    override-registry phase and unused in v0.8.0."""

    EXACT = "exact"
    ESTIMATED_LOWER_BOUND = "estimated-lower-bound"
    DERIVED = "derived"  # RESERVED — not assigned this phase
    NOT_ELIGIBLE = "not-generic-dense-eligible"


@dataclass(frozen=True)
class Tier1Match:
    """A curated lookup hit: slug ∈ a curated model variant's `hf_repos`."""

    model_id: str
    weights_variant: str
    slug: str


@dataclass
class DeriveResult:
    slug: str
    error: Optional[DeriverError] = None
    tier1: Optional[Tier1Match] = None
    confidence: Optional[Confidence] = None
    generic_dense_eligible: Optional[bool] = None
    spec: Optional[dict[str, Any]] = None
    profile: Optional[dict[str, Any]] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP fetcher abstraction (injectable for fixture-driven tests)
# ---------------------------------------------------------------------------
@dataclass
class FetchResponse:
    status: int
    body: bytes


class HttpFetcher:
    """Real-network fetcher. Tests inject a recorded-fixture replacement.

    `get(url, headers=None, range_=None)` returns a FetchResponse. HTTP errors
    surface as the response's status (not exceptions) where the caller maps
    them to structured stratum-1 errors; only true network failures raise
    `NetworkError`.
    """

    timeout = _NET_TIMEOUT

    def get(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        range_: Optional[tuple[int, int]] = None,
    ) -> FetchResponse:
        req_headers = dict(headers or {})
        if range_ is not None:
            lo, hi = range_
            req_headers["Range"] = f"bytes={lo}-{hi}"
        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return FetchResponse(status=resp.status, body=resp.read())
        except urllib.error.HTTPError as exc:  # 4xx/5xx — surface status
            try:
                body = exc.read()
            except Exception:  # pragma: no cover - defensive
                body = b""
            return FetchResponse(status=exc.code, body=body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise NetworkError(str(exc)) from exc


class NetworkError(RuntimeError):
    """True transport failure (timeout / DNS / connection). Distinct from an
    HTTP status code, which the fetcher returns as a FetchResponse."""


Fetcher = HttpFetcher  # type alias for callers


# ---------------------------------------------------------------------------
# kv-calc import (per the documented sys.modules contract)
# ---------------------------------------------------------------------------
_KV_CALC = None


def _load_kv_calc():
    """Load tools/kv-calc.py via importlib, registering it in sys.modules
    BEFORE exec_module (required: kv-calc.py uses @dataclass, which resolves
    cls.__module__ via sys.modules during class creation)."""
    global _KV_CALC
    if _KV_CALC is not None:
        return _KV_CALC
    if "kv_calc" in sys.modules:
        _KV_CALC = sys.modules["kv_calc"]
        return _KV_CALC
    kv_path = REPO_ROOT / "tools" / "kv-calc.py"
    spec = importlib.util.spec_from_file_location("kv_calc", kv_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kv_calc"] = mod  # MUST precede exec_module
    spec.loader.exec_module(mod)
    _KV_CALC = mod
    return _KV_CALC


# ---------------------------------------------------------------------------
# HF_HOME resolution
#   --hf-home > $HF_HOME > $MODEL_DIR/.cache/huggingface > $XDG_CACHE_HOME/hf > ~
# ---------------------------------------------------------------------------
def _model_dir_from_env_or_dotenv() -> Optional[str]:
    """MODEL_DIR from the environment, else parsed from the repo `.env` (the
    SAME value switch.sh / launch.sh / c3 resolve). `None` if set in neither.
    Read with `encoding="utf-8"` (non-UTF-8-locale rigs, #599)."""
    env = os.environ.get("MODEL_DIR")
    if env:
        return env
    try:
        for raw in (REPO_ROOT / ".env").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            s = raw.strip().rstrip("\r")
            if not s or s.startswith("#") or "=" not in s:
                continue
            if s.startswith("export "):
                s = s[len("export "):]
            key, _, val = s.partition("=")
            if key.strip() == "MODEL_DIR":
                val = val.strip().strip('"').strip("'")
                return val or None
    except OSError:
        pass
    return None


def resolve_hf_home(hf_home: Optional[str] = None) -> Path:
    """HF_HOME precedence: `--hf-home > $HF_HOME > $MODEL_DIR/.cache/huggingface
    > $XDG_CACHE_HOME/huggingface > ~/.cache/huggingface`.

    The MODEL_DIR step keeps a bare `pull.sh <repo>` — run with only `.env`'s
    MODEL_DIR set and no explicit HF_HOME — on the MODEL DISK, instead of
    silently falling to `~/.cache` on root (the footgun that misplaced a brought
    model's weights, #617-followup). c3 is unaffected: it sets HF_HOME
    explicitly, which still wins here."""
    if hf_home:
        return Path(hf_home).expanduser()
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env).expanduser()
    model_dir = _model_dir_from_env_or_dotenv()
    if model_dir:
        return Path(model_dir).expanduser() / ".cache" / "huggingface"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


# v0.8.0 [E] E2 (additive): the canonical fetcher E1's deferred dtype
# header-probe step (2) uses. E1's `_resolve_compute_dtype()` calls the
# deriver's existing `probe_safetensors_dtype()` ONLY when a usable fetcher
# is present at `einput.diagnostics["fetcher"]`. E2 standardizes the SOURCE
# of that fetcher here (a real range-bounded `HttpFetcher`) so E4 can wire
# `diagnostics["fetcher"]` deterministically and tests can inject a fixture.
# This does NOT change E1's resolution ORDER/semantics — it only makes the
# probe path live by supplying the fetcher E1 already looks for.
def default_probe_fetcher() -> "HttpFetcher":
    """The canonical real bounded-header-probe fetcher (range-GET only;
    never downloads a full weight). Tests inject a recorded fixture
    instead."""
    return HttpFetcher()


# ---------------------------------------------------------------------------
# Tier-1 curated lookup
# ---------------------------------------------------------------------------
def _tier1_lookup(slug: str, profiles) -> Optional[Tier1Match]:
    """slug ∈ a curated model variant's `hf_repos` → that (model, variant).

    Matched case-insensitively (per brief: `hf_repos` entries are full HF
    slugs, matched case-insensitively)."""
    needle = slug.strip().lower()
    for model in profiles.models.values():
        for variant, meta in model.weights.items():
            for repo in meta.get("hf_repos", []) or []:
                if str(repo).strip().lower() == needle:
                    return Tier1Match(
                        model_id=model.id, weights_variant=variant, slug=str(repo)
                    )
    return None


# ---------------------------------------------------------------------------
# HF fetch helpers
# ---------------------------------------------------------------------------
def _auth_headers(hf_token: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {hf_token}"} if hf_token else {}


def _fetch_model_api(
    slug: str, fetcher: HttpFetcher, hf_token: Optional[str]
) -> tuple[Optional[dict], Optional[DeriverError]]:
    url = f"{_HF_API}/{slug}?blobs=true"
    resp = fetcher.get(url, headers=_auth_headers(hf_token))
    if resp.status == 404:
        return None, DeriverError(DeriverErrorKind.REPO_NOT_FOUND, slug)
    if resp.status in (401, 403):
        if not hf_token:
            return None, DeriverError(DeriverErrorKind.GATED_NO_TOKEN, slug)
        # token present but still denied → treat as not-found (no access)
        return None, DeriverError(DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (auth denied)")
    if resp.status != 200:
        return None, DeriverError(
            DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (HF API status {resp.status})"
        )
    try:
        return json.loads(resp.body.decode("utf-8")), None
    except (ValueError, UnicodeDecodeError) as exc:
        return None, DeriverError(
            DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (malformed HF API: {exc})"
        )


def _fetch_config_json(
    slug: str, fetcher: HttpFetcher, hf_token: Optional[str]
) -> tuple[Optional[dict], Optional[DeriverError]]:
    url = f"{_HF_RESOLVE}/{slug}/resolve/main/config.json"
    resp = fetcher.get(url, headers=_auth_headers(hf_token))
    if resp.status == 404:
        return None, DeriverError(
            DeriverErrorKind.UNSUPPORTED_FORMAT, f"{slug} (no config.json)"
        )
    if resp.status in (401, 403):
        if not hf_token:
            return None, DeriverError(DeriverErrorKind.GATED_NO_TOKEN, slug)
        return None, DeriverError(DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (auth denied)")
    if resp.status != 200:
        return None, DeriverError(
            DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (config.json status {resp.status})"
        )
    try:
        return json.loads(resp.body.decode("utf-8")), None
    except (ValueError, UnicodeDecodeError) as exc:
        return None, DeriverError(
            DeriverErrorKind.QUANT_DTYPE_UNKNOWN, f"{slug} (malformed config.json: {exc})"
        )


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def _is_adapter(name: str) -> bool:
    low = name.lower()
    return any(p in low for p in _ADAPTER_PATTERNS)


def _siblings(api: dict) -> list[dict]:
    out = []
    for s in api.get("siblings", []) or []:
        if isinstance(s, dict) and s.get("rfilename"):
            out.append(s)
    return out


def _declares_mtp(config: dict) -> bool:
    """True when config.json declares a multi-token-prediction head — Qwen3-Next
    uses `mtp_num_hidden_layers`; other families use `num_nextn_predict_layers`.
    Checks the top level AND a nested `text_config` (VLMs nest the LM config)."""
    for cfg in (config or {}, (config or {}).get("text_config") or {}):
        if not isinstance(cfg, dict):
            continue
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
            v = cfg.get(key)
            if isinstance(v, int) and v > 0:
                return True
    return False


def _has_mtp_weight_file(api: dict) -> bool:
    """True when the repo ships a dedicated MTP-head weights file — the layout
    fine-tune re-quants use (e.g. `model_mtp_bf16.safetensors`)."""
    for s in _siblings(api):
        name = (s.get("rfilename") or "").lower()
        if name.endswith(".safetensors") and ("mtp" in name or "nextn" in name):
            return True
    return False


def detect_mtp_head(config: dict, api: dict) -> bool:
    """Whether a brought checkpoint actually carries an MTP draft head, so the
    Route-C weight-swap keeps `--speculative-config` instead of blanket-dropping
    it. The blanket drop was a bug: fine-tunes that PRESERVE the head (e.g.
    ThinkingCap) were served MTP-off. Signal (no extra fetch — config + siblings
    are already in hand): config DECLARES the MTP layers AND a dedicated mtp
    weights file is present. Ground-truth for the separate-file layout every
    fine-tune uses. An embedded-head repo (head baked into the shards with no
    named file) still falls back to drop — the named-file layout is the norm and
    the alternative is reading each shard's index weight_map."""
    return _declares_mtp(config) and _has_mtp_weight_file(api)


def select_weight_files(
    api: dict,
) -> tuple[Optional[list[str]], Optional[DeriverError]]:
    """Per brief file selection:

      - `*.safetensors.index.json` present → the shard set in its weight_map.
        (The index itself must be fetched separately to read weight_map; here
        we only need the shard filenames the index points to, which equal the
        set of top-level shard *.safetensors. We resolve the shard set from
        the siblings list filtered to non-adapter *.safetensors.)
      - Else → after excluding adapter/LoRA patterns, accept EXACTLY ONE
        top-level *.safetensors regardless of basename; multiple plausible
        complete sets → `ambiguous-weight-set`.
      - No *.safetensors → `unsupported-format`.
    """
    sibs = _siblings(api)
    names = [s["rfilename"] for s in sibs]

    safet = [
        n
        for n in names
        if n.endswith(".safetensors") and "/" not in n and not _is_adapter(n)
    ]
    if not safet:
        return None, DeriverError(
            DeriverErrorKind.UNSUPPORTED_FORMAT,
            "no top-level *.safetensors (GGUF/.bin not supported — this path is vLLM + safetensors only)",
        )

    index_files = [
        n
        for n in names
        if n.endswith(".safetensors.index.json") and "/" not in n
    ]
    if index_files:
        # Sharded set: the shards are the non-adapter top-level *.safetensors.
        # A single complete sharded set is unambiguous; >1 distinct index
        # implies >1 plausible complete set.
        if len(index_files) > 1:
            return None, DeriverError(
                DeriverErrorKind.AMBIGUOUS_WEIGHT_SET,
                f"multiple safetensors index files: {sorted(index_files)}",
            )
        shards = sorted(
            n for n in safet if "-of-" in n or n.startswith("model-")
        )
        if not shards:
            # Index present but no obvious shard naming — fall back to all
            # non-adapter top-level safetensors as the set.
            shards = sorted(safet)
        else:
            # A dedicated MTP/nextn head (e.g. `mtp_grafted.safetensors`) is a
            # real weight the model needs with MTP enabled, but it's neither a
            # `model-*` nor `-of-` shard, so the filter above drops it —
            # which silently omitted Tess-4-27B-FP8's MTP head and would break
            # MTP serving (club-3090 #617). `detect_mtp_head` already sees such a
            # file; union it into the download set so it's actually fetched.
            mtp_head = [
                n for n in safet
                if n not in shards
                and ("mtp" in n.lower() or "nextn" in n.lower())
            ]
            if mtp_head:
                shards = sorted(set(shards) | set(mtp_head))
        return shards, None

    # No index: must be exactly one complete set.
    if len(safet) == 1:
        return safet, None
    # Multiple top-level safetensors with no index → ambiguous.
    return None, DeriverError(
        DeriverErrorKind.AMBIGUOUS_WEIGHT_SET,
        f"{len(safet)} top-level *.safetensors and no index.json: {sorted(safet)}",
    )


def _sum_blob_gb(api: dict, selected: list[str]) -> float:
    by_name = {}
    for s in _siblings(api):
        size = s.get("size")
        if size is None and isinstance(s.get("lfs"), dict):
            size = s["lfs"].get("size")
        if size is not None:
            by_name[s["rfilename"]] = size
    total = 0
    wanted = set(selected) | {"config.json"}
    for name, size in by_name.items():
        base = name
        if name in wanted or (
            name.endswith("config.json") and "/" not in name
        ) or any(name == w for w in selected):
            total += int(size)
            continue
        # tokenizer files (required) count toward footprint
        if base in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            total += int(size)
    # weights authority is the summed selected blobs (config/tokenizer are
    # negligible but included for footprint completeness).
    return round(total / (1024 ** 3), 4)


def _selected_weight_gb(api: dict, selected: list[str]) -> float:
    by_name = {}
    for s in _siblings(api):
        size = s.get("size")
        if size is None and isinstance(s.get("lfs"), dict):
            size = s["lfs"].get("size")
        if size is not None:
            by_name[s["rfilename"]] = int(size)
    total = sum(by_name.get(n, 0) for n in selected)
    return round(total / (1024 ** 3), 4)


# ---------------------------------------------------------------------------
# v0.8.0 [E] CONTRACT-3 — the SINGLE shared download allowlist.
#
# `select_weight_files()` returns only `*.safetensors`; vLLM also needs the
# config/tokenizer/template assets. CONTRACT-3 reconciles v2's `*.jinja`
# addition with the legacy `[C2a]` footprint's `vocab.json`/`merges.txt` into
# ONE union, used identically by `[C2a]` sizing (gates.c2a_disk), E2 download
# (downloader.download_model -> snapshot_download allow_patterns), and E3
# smoke. There is exactly ONE function — no parallel lists that can drift.
# ---------------------------------------------------------------------------
# Exact non-glob metadata basenames (the brief's REQUIRED_METADATA, minus the
# `*.jinja` glob which is matched separately). "those that exist in siblings".
REQUIRED_METADATA = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def download_set(api: dict) -> list[str]:
    """CONTRACT-3 reconciled union — the ONE allowlist:

        select_weight_files(api)                  # *.safetensors (no adapters)
      + the *.safetensors.index.json if present
      + REQUIRED_METADATA siblings that exist     # config/tokenizer/...
      + every top-level *.jinja sibling           # chat templates

    Deterministic ordering: weights (as `select_weight_files` returns them),
    then the index, then metadata in REQUIRED_METADATA order, then sorted
    `*.jinja`. Only siblings that ACTUALLY EXIST are included ("those that
    exist in siblings"). On an unselectable weight set this returns `[]`
    (the caller already surfaced the structured `select_weight_files` error;
    `download_set` never raises — it is a pure projection of `api`).

    This is the literal list E2 passes as `snapshot_download(...,
    allow_patterns=...)` and the exact set `[C2a]` sizes — a test asserts
    fetched-set == sized-set.
    """
    selected, err = select_weight_files(api or {})
    if err is not None or not selected:
        return []
    names = {s["rfilename"] for s in _siblings(api or {})}
    out: list[str] = list(selected)
    # the *.safetensors.index.json (top-level) if present
    for n in sorted(names):
        if n.endswith(".safetensors.index.json") and "/" not in n:
            out.append(n)
    # REQUIRED_METADATA — exact basenames that exist (top-level)
    for meta in REQUIRED_METADATA:
        if meta in names:
            out.append(meta)
    # every top-level *.jinja (chat templates)
    for n in sorted(names):
        if n.endswith(".jinja") and "/" not in n and n not in out:
            out.append(n)
    # de-dup while preserving first-seen order (a weight is never metadata,
    # but be defensive against an index/metadata name collision).
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def sized_download_gb(api: dict) -> float:
    """Σ size of EXACTLY `download_set(api)` (LFS-resolved), GiB.

    This is the CONTRACT-3 footprint: `[C2a]` sizes precisely the set E2
    fetches, so `[C2a]`/E2/E3 cannot drift. Replaces the legacy
    `_sum_blob_gb` heuristic (which approximated the same union but with a
    hand-listed metadata set that omitted `*.jinja`); behaviour-equivalent
    to within KB on any real model (metadata is negligible vs multi-GB
    weights) and now provably == the fetched set."""
    by_name = {}
    for s in _siblings(api or {}):
        size = s.get("size")
        if size is None and isinstance(s.get("lfs"), dict):
            size = s["lfs"].get("size")
        if size is not None:
            by_name[s["rfilename"]] = int(size)
    total = sum(by_name.get(n, 0) for n in download_set(api or {}))
    return round(total / (1024 ** 3), 4)


# ---------------------------------------------------------------------------
# Bring-funnel stage-1 INSPECT — artifact inventory (design §2 / §2b)
# ---------------------------------------------------------------------------
# GGUF quant token in a basename (Q4_K_M / IQ4_XS / UD-Q5_K_XL / Q8_0 / TQ1_0 /
# BF16 / F16), tolerant of multi-part suffixes which are stripped first.
_GGUF_PART_RE = re.compile(r"-(\d{5})-of-(\d{5})$", re.I)
_GGUF_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:I?Q\d|TQ\d|BF16|F16|F32)[A-Z0-9_]*)$", re.I
)


def _blob_sizes(api: dict) -> dict[str, int]:
    by_name: dict[str, int] = {}
    for s in _siblings(api or {}):
        size = s.get("size")
        if size is None and isinstance(s.get("lfs"), dict):
            size = s["lfs"].get("size")
        if size is not None:
            by_name[s["rfilename"]] = int(size)
    return by_name


def artifact_inventory(api: dict) -> dict:
    """What servable artifacts does this repo carry? — WITHOUT gating on
    format.  A GGUF-only repo is a first-class bring here (design §2b-1/2:
    the staged Bring UI reveals nothing template-side until this says the
    repo is supported, and presents ALL discovered GGUF variants for the
    user to pick BEFORE any engine/slug appears).  `select_weight_files`
    stays the vLLM/safetensors gate — this never replaces it.

    Returns (all sizes GiB, from the ?blobs=true siblings):
      formats          ["safetensors", "gguf"] — whichever are present
      safetensors      {weight_files, size_gb} | None (top-level, non-adapter)
      gguf_variants    [{quant, size_gb, parts, files}] sorted by size —
                       multi-part files grouped under one quant token; a file
                       with no parseable token keys by its stem (never dropped)
      gguf_mmproj      vision-projector *.gguf names (NOT variants)
      lineage_base_model  cardData.base_model when the API carries it
                          (friction #11 — ⑤'s taxonomy default + credits)"""
    sizes = _blob_sizes(api)
    names = [s["rfilename"] for s in _siblings(api or {})]

    safet = [
        n for n in names
        if n.endswith(".safetensors") and "/" not in n and not _is_adapter(n)
    ]
    st = None
    if safet:
        st = {
            "weight_files": sorted(safet),
            "size_gb": round(sum(sizes.get(n, 0) for n in safet) / (1024 ** 3), 4),
        }

    # GGUF: any depth (quant subdirs are common), mmproj split out.
    # Grouping key = the STEM (basename minus the -NNNNN-of-NNNNN part
    # suffix), NOT the quant token — a repo can ship DISTINCT artifacts
    # sharing a token (live dogfood 2026-07-05: Qwythos ships
    # `…-Q4_K_M.gguf` AND `…-MTP-Q4_K_M.gguf` per quant; token-keying
    # merged them into one "2-part variant" with a summed, wrong size).
    # True multi-part shards share a stem, so `parts` still counts them.
    variants: dict[str, dict] = {}
    mmproj: list[str] = []
    for n in names:
        if not n.lower().endswith(".gguf"):
            continue
        base = n.rsplit("/", 1)[-1][: -len(".gguf")]
        if base.lower().startswith("mmproj"):
            mmproj.append(n)
            continue
        stem = _GGUF_PART_RE.sub("", base)
        v = variants.setdefault(stem, {"size_gb": 0.0, "parts": 0, "files": []})
        v["size_gb"] += sizes.get(n, 0) / (1024 ** 3)
        v["parts"] += 1
        v["files"].append(n)
    # Display label: the stem minus the repo-wide COMMON prefix — for
    # standard repos that IS the quant token ("Q4_K_M"); for multi-artifact
    # repos it keeps the distinguishing part ("MTP-Q4_K_M").  Falls back to
    # the parsed token, then the full stem (labels stay unique: stems are).
    common = os.path.commonprefix(list(variants)) if len(variants) > 1 else ""
    out_variants = []
    for stem, v in variants.items():
        label = stem[len(common):].strip("-_. ")
        if not label:
            m = _GGUF_QUANT_RE.search(stem)
            label = m.group(1).upper() if m else stem
        out_variants.append(
            {"quant": label, "size_gb": round(v["size_gb"], 4),
             "parts": v["parts"], "files": sorted(v["files"])}
        )
    gguf_variants = sorted(out_variants, key=lambda v: (v["size_gb"], v["quant"]))

    formats = []
    if st:
        formats.append("safetensors")
    if gguf_variants or mmproj:
        formats.append("gguf")

    card = api.get("cardData") if isinstance(api, dict) else None
    base_model = (card or {}).get("base_model") if isinstance(card, dict) else None

    return {
        "formats": formats,
        "safetensors": st,
        "gguf_variants": gguf_variants,
        "gguf_mmproj": sorted(mmproj),
        "lineage_base_model": base_model,
    }


def inspect_repo(
    slug: str,
    *,
    hf_token: Optional[str] = None,
    fetcher: Optional[HttpFetcher] = None,
) -> dict:
    """Stage-1 INSPECT entry: fetch the model API + return the inventory.
    Structured errors, never a traceback (same discipline as derive())."""
    if fetcher is None:
        fetcher = HttpFetcher()
    hf_token = hf_token or os.environ.get("HF_TOKEN") or None
    try:
        api, err = _fetch_model_api(slug, fetcher, hf_token)
    except NetworkError as exc:
        return {"repo": slug, "error": f"network error: {exc}"}
    if err is not None:
        return {"repo": slug, "error": str(err)}
    inv = artifact_inventory(api or {})
    inv["repo"] = slug
    if not inv["formats"]:
        inv["error"] = "no servable artifacts (no safetensors weight set, no *.gguf)"
    return inv


# ---------------------------------------------------------------------------
# Bounded safetensors-header probe (pre-download, range-bounded)
# ---------------------------------------------------------------------------
def probe_safetensors_dtype(
    slug: str,
    weight_file: str,
    fetcher: HttpFetcher,
    hf_token: Optional[str],
) -> Optional[str]:
    """Range-bounded header probe. NEVER downloads a full weight file.

      1. Range-GET bytes=0-7 → little-endian u64 = header length N.
      2. if N > 16 MiB → None (caller maps to quant-dtype-unknown).
      3. Range-GET bytes=8-(8+N-1)  [HTTP ranges inclusive; read exactly N].
      4. parse JSON; read __metadata__ / first tensor dtype.

    Any failure/malformed → None.
    """
    url = f"{_HF_RESOLVE}/{slug}/resolve/main/{weight_file}"
    try:
        r1 = fetcher.get(url, headers=_auth_headers(hf_token), range_=(0, 7))
    except NetworkError:
        return None
    if r1.status not in (200, 206) or len(r1.body) < 8:
        return None
    n = struct.unpack("<Q", r1.body[:8])[0]
    if n <= 0 or n > _MAX_HEADER_BYTES:
        return None
    # bytes 8 .. 8+N-1 inclusive == exactly N bytes from offset 8.
    try:
        r2 = fetcher.get(
            url, headers=_auth_headers(hf_token), range_=(8, 8 + n - 1)
        )
    except NetworkError:
        return None
    if r2.status not in (200, 206):
        return None
    blob = r2.body[:n]
    if len(blob) < n:
        return None
    try:
        hdr = json.loads(blob.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(hdr, dict):
        return None
    meta = hdr.get("__metadata__")
    if isinstance(meta, dict):
        for k in ("dtype", "torch_dtype", "format"):
            if isinstance(meta.get(k), str):
                return meta[k]
    for key, tinfo in hdr.items():
        if key == "__metadata__":
            continue
        if isinstance(tinfo, dict) and isinstance(tinfo.get("dtype"), str):
            return tinfo["dtype"]
    return None


# ---------------------------------------------------------------------------
# Quant / dtype chain + effective bits-per-weight
# ---------------------------------------------------------------------------
_DTYPE_BPW = {
    "BF16": 16.0,
    "F16": 16.0,
    "FP16": 16.0,
    "FLOAT16": 16.0,
    "BFLOAT16": 16.0,
    "F32": 32.0,
    "FP32": 32.0,
    "FLOAT32": 32.0,
    "F8_E5M2": 8.0,
    "F8_E4M3": 8.0,
    "FLOAT8_E5M2": 8.0,
    "FLOAT8_E4M3FN": 8.0,
    "I8": 8.0,
    "INT8": 8.0,
    "U8": 8.0,
    "I4": 4.0,
    "INT4": 4.0,
}


def _quant_bpw(quant_cfg: dict) -> Optional[float]:
    bits = quant_cfg.get("bits") or quant_cfg.get("w_bit") or quant_cfg.get(
        "weight_bits"
    )
    if isinstance(bits, (int, float)) and bits > 0:
        return float(bits)
    # compressed-tensors (llm-compressor) checkpoints nest the bit-width per
    # config-group instead of top-level: config_groups.<g>.weights =
    # {num_bits: 8, type: "float"|"int", ...}. Take the widest weights
    # num_bits across groups (mixed-precision groups exist; the widest
    # dominates the VRAM footprint the fit-check cares about). Explicit
    # structure beats the method-name heuristics below — "compressed-tensors"
    # as a method name matches none of them (the Agents-A1-FP8-dynamic
    # producer-zero dogfood finding, 2026-07-02).
    groups = quant_cfg.get("config_groups")
    if isinstance(groups, dict):
        bits_seen = []
        for g in groups.values():
            w = g.get("weights") if isinstance(g, dict) else None
            nb = w.get("num_bits") if isinstance(w, dict) else None
            if isinstance(nb, (int, float)) and nb > 0:
                bits_seen.append(float(nb))
        if bits_seen:
            return max(bits_seen)
    method = str(
        quant_cfg.get("quant_method")
        or quant_cfg.get("method")
        or quant_cfg.get("quant_algo")
        or ""
    ).lower()
    if any(k in method for k in ("awq", "gptq", "autoround", "int4", "4bit")):
        return 4.0
    if "fp8" in method or "8bit" in method or "int8" in method:
        return 8.0
    return None


def resolve_quant_dtype(
    slug: str,
    config: dict,
    selected: list[str],
    fetcher: HttpFetcher,
    hf_token: Optional[str],
) -> tuple[Optional[str], Optional[float], Optional[DeriverError]]:
    """`quantization_config` → `torch_dtype` → bounded header probe →
    else `quant-dtype-unknown`. Returns (weight_format, bpw, error)."""
    qcfg = config.get("quantization_config")
    if isinstance(qcfg, dict) and qcfg:
        method = str(
            qcfg.get("quant_method") or qcfg.get("method") or "quantized"
        ).lower()
        bpw = _quant_bpw(qcfg)
        if bpw is not None:
            return method, bpw, None
        # known method, undeterminable bits
        return None, None, DeriverError(
            DeriverErrorKind.QUANT_DTYPE_UNKNOWN,
            f"{slug} (quantization_config method={method!r} bits undeterminable)",
        )

    td = config.get("torch_dtype") or config.get("dtype")
    if isinstance(td, str) and td.strip():
        bpw = _DTYPE_BPW.get(td.strip().upper())
        if bpw is not None:
            return td.strip(), bpw, None

    # bounded header probe (sharded → first shard only)
    if selected:
        dtype = probe_safetensors_dtype(
            slug, sorted(selected)[0], fetcher, hf_token
        )
        if dtype:
            bpw = _DTYPE_BPW.get(dtype.strip().upper())
            if bpw is not None:
                return dtype.strip(), bpw, None

    return None, None, DeriverError(
        DeriverErrorKind.QUANT_DTYPE_UNKNOWN,
        f"{slug} (no quantization_config / torch_dtype / probeable header dtype)",
    )


# ---------------------------------------------------------------------------
# Spec / profile assembly
# ---------------------------------------------------------------------------
def _int(config: dict, key: str) -> Optional[int]:
    v = config.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _build_generic_dense_spec(
    slug: str, config: dict, weight_gb: float
) -> dict[str, Any]:
    hidden = _int(config, "hidden_size")
    n_layers = _int(config, "num_hidden_layers")
    n_heads = _int(config, "num_attention_heads")
    n_kv = _int(config, "num_key_value_heads")
    head_dim = _int(config, "head_dim")
    if head_dim is None and hidden and n_heads and hidden % n_heads == 0:
        head_dim = hidden // n_heads
    arch_list = config.get("architectures") or []
    arch = str(arch_list[0]) if arch_list else None
    return {
        "model_id": slug,
        "model_family": "generic-dense",
        "arch": arch,
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attn_heads": n_heads,
        "num_kv_heads": n_kv,
        "head_dim_attn": head_dim,
        "weights_total_gb": weight_gb,
        "valid_tp": [1, 2],
        "max_ctx_supported": _int(config, "max_position_embeddings") or 131072,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def derive(
    slug: str,
    *,
    hf_token: Optional[str] = None,
    hf_home: Optional[str] = None,
    fetcher: Optional[HttpFetcher] = None,
    profiles=None,
) -> DeriveResult:
    """Derive a ModelProfile-shaped result for an HF repo slug.

    Resolution order (§4):
      1. Tier-1 curated lookup (slug ∈ a curated variant's hf_repos) →
         confidence EXACT, no network for the spec (curated profile is
         authoritative). Stratum-1 file/quant checks are NOT run for a
         curated hit (the curated profile already encodes them); P3/P4 do
         the Path-A weights_variant compat check.
      2. else fetch config.json + HF siblings; run stratum-1 deriver checks;
         if is_generic_dense_eligible → confidence ESTIMATED_LOWER_BOUND.
      3. else → NOT_ELIGIBLE (P4 wires the stratum-5 abort; P2 only marks).
    """
    res = DeriveResult(slug=slug)
    if fetcher is None:
        fetcher = HttpFetcher()
    if profiles is None:
        from .compat import load_profiles

        profiles = load_profiles()

    hf_token = hf_token or os.environ.get("HF_TOKEN") or None

    # --- §4 Tier-1 curated lookup ------------------------------------------
    t1 = _tier1_lookup(slug, profiles)
    if t1 is not None:
        model = profiles.models[t1.model_id]
        variant_meta = model.weights.get(t1.weights_variant, {})
        # Schema invariant: hf_repos must only attach to safetensors-compatible
        # variants. If a curated slug somehow resolves to a gguf / non-
        # safetensors variant, surface stratum-1 unsupported-format honestly
        # rather than a silent mismatch (Codex-r5 Med-3).
        fmt = str(variant_meta.get("format", "")).lower()
        if fmt in ("gguf",):
            res.error = DeriverError(
                DeriverErrorKind.UNSUPPORTED_FORMAT,
                f"{slug} resolves to {t1.model_id}.{t1.weights_variant} "
                f"(format={fmt!r}); GGUF not supported — vLLM + safetensors only",
            )
            return res
        res.tier1 = t1
        res.confidence = Confidence.EXACT
        res.profile = {
            "model_id": model.id,
            "weights_variant": t1.weights_variant,
            "arch": None,
            "family": model.family,
            "hidden_size": model.hidden_size,
            "num_hidden_layers": model.num_hidden_layers,
            "num_attn_heads": model.num_attn_heads,
            "num_kv_heads": model.num_kv_heads,
            "weight_format": variant_meta.get("format"),
            "weights_variant_size_gb": variant_meta.get("size_gb"),
        }
        res.diagnostics["resolution"] = "tier1-curated"
        return res

    # --- stratum-1: HF model API first (repo existence / gating) -----------
    # Order matters: the HF model API is the authority for repo-not-found
    # (404) and gated-no-token (401/403 w/o token). config.json 404 on an
    # existing repo means "no transformers config" -> unsupported-format.
    try:
        api, err = _fetch_model_api(slug, fetcher, hf_token)
    except NetworkError as exc:
        res.error = DeriverError(
            DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (network error: {exc})"
        )
        return res
    if err is not None:
        res.error = err
        return res

    try:
        config, err = _fetch_config_json(slug, fetcher, hf_token)
    except NetworkError as exc:
        res.error = DeriverError(
            DeriverErrorKind.REPO_NOT_FOUND, f"{slug} (network error: {exc})"
        )
        return res
    if err is not None:
        res.error = err
        return res

    selected, err = select_weight_files(api or {})
    if err is not None:
        res.error = err
        return res

    weight_format, bpw, err = resolve_quant_dtype(
        slug, config or {}, selected or [], fetcher, hf_token
    )
    if err is not None:
        res.error = err
        return res

    weight_gb = _selected_weight_gb(api or {}, selected or [])
    # v0.8.0 [E] CONTRACT-3: footprint sizes EXACTLY the shared download_set
    # (the same union E2 fetches + [C2a] gates on) — single source, no drift.
    # (`_sum_blob_gb` kept above as append-only history; superseded here.)
    footprint_gb = sized_download_gb(api or {})

    # --- §4 generic-dense eligibility (reuse P1's predicate) ---------------
    kv = _load_kv_calc()
    eligible = bool(kv.is_generic_dense_eligible(config or {}))
    res.generic_dense_eligible = eligible

    arch_list = (config or {}).get("architectures") or []
    arch = str(arch_list[0]) if arch_list else None
    has_auto_map = bool((config or {}).get("auto_map"))

    # v0.8.0 [E] CONTRACT-2 — scoped ADDITIVE surface: expose config.json's
    # raw `torch_dtype` (or its `dtype` alias) so [E]'s derived-vllm template
    # can resolve `--dtype` for quantized rows (resolve_quant_dtype()
    # short-circuits at quantization_config and never records it). Additive
    # field ONLY — every existing field is byte-unchanged; no behaviour
    # depends on it inside the deriver. None when config.json omits it.
    _raw_td = (config or {}).get("torch_dtype") or (config or {}).get("dtype")
    config_torch_dtype = _raw_td.strip() if isinstance(_raw_td, str) and _raw_td.strip() else None

    res.profile = {
        "model_id": slug,
        "weights_variant": None,
        "arch": arch,
        "family": "generic-dense" if eligible else None,
        "auto_map": has_auto_map,
        "weight_format": weight_format,
        "torch_dtype": config_torch_dtype,
        "effective_bpw": bpw,
        "weights_total_gb": weight_gb,
        "footprint_gb": footprint_gb,
        "selected_weight_files": selected,
        # v0.8.0 [E] CONTRACT-3 (additive): the raw HF siblings API so
        # gates.c2a_disk can size the SHARED download_set() directly (single
        # function, [C2a]/E2/E3 cannot drift). Additive field ONLY — no
        # existing field/behaviour changes; absent for a tier-1 curated hit
        # (curated footprint comes from the variant size_gb, not the API).
        "_hf_api": api or {},
        "download_set": download_set(api or {}),
        "config_hidden_size": _int(config or {}, "hidden_size"),
        "config_num_hidden_layers": _int(config or {}, "num_hidden_layers"),
        "config_num_attention_heads": _int(config or {}, "num_attention_heads"),
        "config_num_key_value_heads": _int(config or {}, "num_key_value_heads"),
        # Additive: does the brought checkpoint carry an MTP draft head? The
        # Route-C weight-swap (pull.sh _swap_path) reads this to keep vs drop
        # --speculative-config, instead of the old blanket "fine-tune → no MTP"
        # drop that silently served head-preserving fine-tunes MTP-off.
        "has_mtp_head": detect_mtp_head(config or {}, api or {}),
    }
    res.diagnostics["resolution"] = "derived"

    if eligible:
        res.confidence = Confidence.ESTIMATED_LOWER_BOUND
        res.spec = _build_generic_dense_spec(slug, config or {}, weight_gb)
    else:
        res.confidence = Confidence.NOT_ELIGIBLE  # P4 wires stratum-5 abort

    return res


# ---------------------------------------------------------------------------
# CLI — stage-1 INSPECT for the Bring funnel (c3 subprocess + standalone use)
#   python3 scripts/lib/profiles/deriver.py --inventory <org/Model> [--json]
# ---------------------------------------------------------------------------
def _cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="deriver", description="Bring-funnel stage-1 INSPECT (artifact inventory)"
    )
    ap.add_argument("repo", help="HF repo slug, e.g. org/Model")
    ap.add_argument("--inventory", action="store_true", required=True,
                    help="emit the artifact inventory (the only CLI mode)")
    ap.add_argument("--json", action="store_true", help="JSON output (default: pretty)")
    ns = ap.parse_args(argv)
    inv = inspect_repo(ns.repo)
    if ns.json:
        print(json.dumps(inv))
    else:
        if inv.get("error"):
            print(f"error: {inv['error']}")
        else:
            print(f"repo: {inv['repo']}  formats: {', '.join(inv['formats'])}")
            if inv.get("safetensors"):
                st = inv["safetensors"]
                print(f"  safetensors: {len(st['weight_files'])} file(s), {st['size_gb']:.1f} GiB")
            for v in inv.get("gguf_variants") or []:
                print(f"  gguf {v['quant']}: {v['size_gb']:.1f} GiB ({v['parts']} file(s))")
            for m in inv.get("gguf_mmproj") or []:
                print(f"  mmproj: {m}")
            if inv.get("lineage_base_model"):
                print(f"  base_model: {inv['lineage_base_model']}")
    return 1 if inv.get("error") else 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(_cli(sys.argv[1:]))
