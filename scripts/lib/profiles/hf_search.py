#!/usr/bin/env python3
"""Hugging Face model search — stdlib-only CLI (c3 Bring lane, search before Inspect).

The cockpit app NEVER does network I/O itself: every external fact arrives as
subprocess JSON through the Runner seam (the ``bring_inspect`` → ``deriver.py``
pattern).  This script is that subprocess for repo DISCOVERY:

    python3 scripts/lib/profiles/hf_search.py <query> [--limit N] [--json]

It GETs

    https://huggingface.co/api/models?search=<q>&limit=<N>&sort=downloads&direction=-1&full=false

and emits one row per hit (``--json``):

    {"id": "org/Model", "downloads": 123456, "likes": 42,
     "last_modified": "2026-08-01T...", "gguf": true, "safetensors": false,
     "pipeline_tag": "text-generation"}

The gguf/safetensors booleans derive from the API's ``tags`` list so the Bring
lane can pre-warn route-G (GGUF-only repos) before any Inspect call.

Authorization: when ``HF_TOKEN`` is set, an ``Authorization: Bearer <token>``
header is attached — gated/private repos only surface in search results with
a valid token; anonymous queries work for public repos.

Exit codes: 0 on success — an EMPTY result list is valid output, not an error.
Non-zero + a message on stderr for any failure (network error, HTTP error,
bad payload).  Without ``--json`` the rows render as aligned text for humans;
stdout stays empty on failure so callers never parse partial output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

_HF_API = "https://huggingface.co/api/models"
_NET_TIMEOUT = 20          # seconds — the cockpit's Runner seam allows 20
_MAX_LIMIT = 100           # the hub API's own page-size ceiling


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def build_url(query: str, limit: int) -> str:
    """The exact search endpoint the cockpit contract expects: downloads-sorted,
    descending (most-downloaded first), lean payloads (full=false)."""
    qs = urllib.parse.urlencode(
        {
            "search": query,
            "limit": str(limit),
            "sort": "downloads",
            "direction": "-1",
            "full": "false",
        }
    )
    return f"{_HF_API}?{qs}"


def _request_headers(token: Optional[str]) -> dict[str, str]:
    """JSON accept + optional Bearer auth.  An explicitly-passed token wins over
    $HF_TOKEN (tests inject "" to force the anonymous path deterministically)."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "club3090-hf-search/1.0",
    }
    tok = token if token is not None else (os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def row_from_api(model: dict[str, Any]) -> dict[str, Any]:
    """One hub /api/models entry → the flat row the cockpit renders."""
    tags = model.get("tags") or []
    return {
        "id": model.get("id") or model.get("modelId") or "",
        "downloads": int(model.get("downloads") or 0),
        "likes": int(model.get("likes") or 0),
        "last_modified": model.get("lastModified") or "",
        "gguf": "gguf" in tags,
        "safetensors": "safetensors" in tags,
        "pipeline_tag": model.get("pipeline_tag") or "",
    }


def search(
    query: str,
    *,
    limit: int = 20,
    token: Optional[str] = None,
    timeout: int = _NET_TIMEOUT,
    urlopen: Optional[Callable[..., Any]] = None,
) -> list[dict[str, Any]]:
    """Run the search and return normalized rows.  Raises on any failure —
    the caller decides how to surface it.  ``urlopen`` is injectable so tests
    can monkeypatch the transport (NO live network in the test suite)."""
    open_url = urlopen if urlopen is not None else urllib.request.urlopen
    req = urllib.request.Request(
        build_url(query, limit), headers=_request_headers(token)
    )
    with open_url(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("unexpected payload: expected a JSON array of models")
    return [row_from_api(m) for m in payload if isinstance(m, dict)]


# ---------------------------------------------------------------------------
# rendering + CLI
# ---------------------------------------------------------------------------
def _human_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def format_row(row: dict[str, Any]) -> str:
    fmts = "/".join(
        f for f in ("safetensors" if row["safetensors"] else "", "GGUF" if row["gguf"] else "") if f
    )
    updated = (row["last_modified"] or "")[:10] or "?"
    pipeline = f" · {row['pipeline_tag']}" if row["pipeline_tag"] else ""
    return (
        f"{row['id']:<48} {_human_count(row['downloads']):>7} ↓ "
        f"{_human_count(row['likes']):>5} ♥  {updated}  {fmts or '?'}{pipeline}"
    )


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hf_search.py",
        description="Search Hugging Face models (stdlib urllib; HF_TOKEN honored).",
    )
    parser.add_argument("query", help="search terms (org/Model or keywords)")
    parser.add_argument("--limit", type=int, default=20, help="max results (default 20)")
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="emit the JSON row contract instead of human-readable text",
    )
    args = parser.parse_args(argv)

    if args.limit < 1 or args.limit > _MAX_LIMIT:
        print(f"hf_search: --limit must be 1..{_MAX_LIMIT}", file=sys.stderr)
        return 2

    try:
        rows = search(args.query, limit=args.limit)
    except Exception as exc:  # network, HTTP, JSON — one stderr line, rc≠0
        print(f"hf_search: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(rows))
    else:
        for row in rows:
            print(format_row(row))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    sys.exit(_main(sys.argv[1:]))
