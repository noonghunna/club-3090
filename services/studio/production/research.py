"""Web research for documentaries — ground the plan in real facts via SearXNG.

The director plans from a 4B's frozen (~2024) knowledge; for a factual/documentary brief that
risks vague or invented history. When the operator asks for research (or approves the director's
offer), we query the rig's SearXNG (:8088 JSON, the same instance OWUI uses) and hand the top
result snippets to the planner as RESEARCH NOTES, so the treatment + shots use real names, dates,
events, and places instead of guesses.

FAILS OPEN: SearXNG down / no results / timeout / bad JSON → empty notes, and the planner proceeds
from its own knowledge. Research is an enhancement, never a hard dependency — a render must never
fail because the web was unreachable.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from . import config

SEARCH_SUFFIX = " key facts history timeline dates events"   # nudge the query toward factual sources
MAX_SNIPPET = 240


def web_search(query: str, *, n: int = 6, timeout: int = 20) -> list[dict]:
    """Top SearXNG results as [{title, content, url}], best-effort. [] on any failure."""
    url = config.SEARXNG_URL.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
    except Exception:
        return []
    out: list[dict] = []
    seen = set()
    for item in (data.get("results") or []):
        content = (item.get("content") or "").strip()
        title = (item.get("title") or "").strip()
        key = content[:80].lower()
        if not content or key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "content": content, "url": item.get("url", "")})
        if len(out) >= n:
            break
    return out


def format_notes(results: list[dict]) -> str:
    """A RESEARCH NOTES block for the planner prompts ('' if no results)."""
    if not results:
        return ""
    lines = ["RESEARCH NOTES (real facts from the web — ground every shot and narration in THESE; "
             "use the real names, dates, events, and places below, not invented ones):"]
    for i, r in enumerate(results, 1):
        snippet = r["content"][:MAX_SNIPPET].strip()
        tail = f"  [{r['title']}]" if r.get("title") else ""
        lines.append(f"{i}. {snippet}{tail}")
    return "\n".join(lines)


def research_notes(brief: str, *, n: int = 6, timeout: int = 20) -> str:
    """Search the web for a brief and return a formatted notes block ('' if nothing useful)."""
    return format_notes(web_search((brief or "").strip() + SEARCH_SUFFIX, n=n, timeout=timeout))
