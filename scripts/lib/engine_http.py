#!/usr/bin/env python3
"""engine_http — the one entry point for our own HTTP requests to a serving engine.

Why one entry point: Strata 0.2+ answers nothing but GET /health without its key, and every
script that hand-rolled its own urllib request re-derived the key rule — or forgot it. The
2026-10-30 rebench died exactly this way (a container with a custom key 401'd the whole
chain), and compaction/deep-context probes never had it at all. The rules stay where they
are: club-containers.sh's club_container_for_url answers WHICH container publishes the URL
(#1584), engine-auth.sh's club_engine_api_key answers WHICH key (#1795). This module CALLS
those helpers; it never copies their rules.

Lazy, like the gateway probe (#1795): the 401 itself is the trigger. Keyless engines are
probed exactly as before, byte-identical; a keyed engine pays one extra round trip per URL,
and the key is cached per URL for the rest of the process. A 401 means the server did not
process the request, so the retry is safe even for POSTs.

Precedence (same as engine-auth.sh, which is asked last): ENGINE_API_KEY (the caller's
override, e.g. a remote URL with no container here) -> CLUB_ENGINE_API_KEY (a parent shell
already resolved it) -> the URL's container, through the bash helpers.

Usage:
    import engine_http
    with engine_http.urlopen(url + "/v1/chat/completions", data=body, timeout=600) as r: ...
    # or for hand-built Request objects:
    req = urllib.request.Request(url, data=body, headers=engine_http.headers(url))
    with engine_http.urlopen(req, timeout=600) as r: ...   # engine_http attaches any known key
"""
import os
import subprocess
import urllib.error
import urllib.request

_LIB = os.path.dirname(os.path.abspath(__file__))
_KEYS = {}  # url -> key ("" = already resolved, none applies)

# Ask the repo's own helpers — never a copy of their rules.
_RESOLVE = ('source "$1/club-containers.sh"; source "$1/engine-auth.sh"; '
            'club_engine_api_key "$(club_container_for_url "$2")"')


def _env_key():
    return os.environ.get("ENGINE_API_KEY") or os.environ.get("CLUB_ENGINE_API_KEY") or ""


def key(url):
    """The key that applies to URL, or "" when none does. Cached per URL."""
    if url not in _KEYS:
        k = _env_key()
        if not k:
            try:
                p = subprocess.run(["bash", "-c", _RESOLVE, "engine-http", _LIB, url],
                                   capture_output=True, text=True, timeout=15)
                k = p.stdout.strip()
            except Exception:
                k = ""
        _KEYS[url] = k
    return _KEYS[url]


def _known(url):
    """The key for URL if already known (env or cache) — never triggers a resolution."""
    return _KEYS.get(url) or _env_key()


def headers(url, base=None):
    """Base headers plus the engine's key when one is already known (lazy: no resolution)."""
    h = dict(base if base is not None else {"Content-Type": "application/json"})
    k = _known(url)
    if k:
        h["Authorization"] = "Bearer " + k
    return h


def urlopen(req_or_url, data=None, timeout=None, headers=None, method=None):
    """urllib.request.urlopen with the engine's key — resolved only after a 401 asks for it."""
    if isinstance(req_or_url, urllib.request.Request):
        req, url = req_or_url, req_or_url.full_url
    else:
        url = req_or_url
        base = dict(headers) if headers is not None else (
            {"Content-Type": "application/json"} if data is not None else {})
        req = urllib.request.Request(url, data=data, headers=base, method=method)
    k = _known(url)
    if k and not req.has_header("Authorization"):
        req.add_header("Authorization", "Bearer " + k)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        # Only a 401 we have not already answered for this URL retries: the key rule may
        # resolve nothing (keyless engine, no container here) — then the 401 stands.
        if e.code != 401 or _KEYS.get(url):
            raise
        fresh = key(url)
        if not fresh or fresh == k:
            raise
        retry = urllib.request.Request(
            url, data=req.data,
            headers={h: v for h, v in req.header_items() if h.lower() != "authorization"}
                    | {"Authorization": "Bearer " + fresh},
            method=req.get_method())
        return urllib.request.urlopen(retry, timeout=timeout)
