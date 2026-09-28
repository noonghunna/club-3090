#!/usr/bin/env python3
"""Render the LiteLLM config the gateway ACTUALLY serves, from what is running.

See scripts/lib/litellm-sync.sh for the full rationale. In short:

  config.yaml          tracked CATALOG view — one canonical slug per model,
                       port-pinned, registry-derived. Right for a file in git.
  config.runtime.yaml  gitignored RUNTIME view — what the container mounts,
                       rendered from endpoints that actually answered.

The catalog view cannot be what a gateway serves: its route for a model names
ONE slug's default_port, so running a sibling slug for that model advertises a
model on a port with nothing behind it, and 13 of 22 models have no route at all.
Both failures are silent — the model list looks populated either way.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

try:                                    # run as a file from scripts/lib (litellm-sync.sh)
    import litellm_local
except ImportError:                     # imported as scripts.lib.litellm_sync
    from scripts.lib import litellm_local

BEGIN = "  # === BEGIN GENERATED LOCAL BLOCK"
END = "  # === END GENERATED LOCAL BLOCK ==="
LOCAL_BEGIN = "  # === BEGIN THIS RIG'S OWN ROUTES — {src} (not tracked) ==="
LOCAL_END = "  # === END THIS RIG'S OWN ROUTES ==="


def registry_variants(root: str) -> list[dict]:
    try:
        out = subprocess.run(
            ["bash", os.path.join(root, "scripts/lib/registry-emit.sh"), "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        ).stdout
        return json.loads(out).get("variants", [])
    except Exception:
        return []


def registry_ports(variants: list[dict]) -> list[int]:
    """Every port the catalog owns. These are the ONLY ports this sync probes,
    and the only ones it may prune — see PRUNE SCOPE in the wrapper."""
    return sorted({v["port"] for v in variants if v.get("port")})


# A live route: (port, served id, context window or None, whether Claude Code's
# /v1/messages goes to the engine's own Anthropic endpoint — see serves_messages).
Live = tuple[int, str, "int | None", bool]


def probe(port: int) -> list[Live]:
    """Ask the server what it serves. Names come from /v1/models, NOT from the
    registry's `served_name`: 72 of 138 variants do not declare one, so a
    registry-derived name would have nothing to emit for half the catalog.
    vLLM and SGLang also report the booted `max_model_len` there — the real
    context window of THIS boot."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            data = json.load(r).get("data") or []
        owned_by = next((str(m["owned_by"]) for m in data if m.get("owned_by")), "")
        first_id = next((str(m["id"]) for m in data if m.get("id")), "")
        native = serves_messages(port, owned_by, first_id) if first_id else False
        out: list[Live] = []
        for m in data:
            if not m.get("id"):
                continue
            ml = m.get("max_model_len")
            ml = int(ml) if isinstance(ml, int) and ml > 0 else None
            if ml is None:
                ml = llamacpp_ctx(port)   # llama.cpp reports its window on /props instead
            out.append((port, m["id"], ml, native))
        return out
    except Exception:
        return []


def engine_kind_from_owned_by(owned_by: str) -> str:
    """The engine family, decided by scripts/lib/engine-kind.sh (#1282) — never here."""
    lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine-kind.sh")
    try:
        out = subprocess.run(
            ["bash", "-c", 'source "$1" && engine_kind_from_owned_by "$2"', "_", lib, owned_by],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def count_tokens(port: int, model: str, body: dict) -> "int | None":
    """Anthropic /v1/messages/count_tokens: renders the prompt, generates nothing."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages/count_tokens",
        data=json.dumps({"model": model, **body}).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            n = json.load(r).get("input_tokens")
    except Exception:
        return None
    return n if isinstance(n, int) else None


def serves_messages(port: int, owned_by: str, model: str) -> bool:
    """Whether the gateway should hand Claude Code's Anthropic /v1/messages to the
    engine's own endpoint instead of translating it. Both must hold:
      - the engine is SGLang or vLLM, the two checked with the Claude Code CLI:
        thinking blocks out, replayed thinking and a mid-array system message in,
        tool calls, prefix reuse on follow-ups.
      - the endpoint keeps a mid-conversation `system` message where it is.
        Claude Code sends one every turn. An endpoint that moves it into the
        leading system block changes the front of the prompt each turn, and the
        whole conversation after the system prompt is re-prefilled (vLLM v0.30.0
        without `--chat-template`: new tokens per turn grew 7.8K → 9.9K → 12.8K
        over three turns, against 2.9K). count_tokens tells the two apart without
        generating anything: moved, the message renders exactly like the same text
        appended to the top-level system prompt; kept in place, it adds its own
        turn markers. A server without the endpoint (an older build) fails the
        request and stays on LiteLLM's translation, which still works.
    vLLM v0.30.0 keeps the message in place only when started with
    `--chat-template` — it tests the flag, not the template it loaded; our Qwen3.8
    composes pass it. vllm#58754 fixes the check (docs/UPSTREAM.md)."""
    if engine_kind_from_owned_by(owned_by) not in ("sglang", "vllm"):
        return False
    turns = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    inline = count_tokens(port, model, {"system": "Be brief.", "messages": [
        turns[0], {"role": "system", "content": "Env: linux"}, turns[1]]})
    merged = count_tokens(port, model, {"system": [
        {"type": "text", "text": "Be brief."}, {"type": "text", "text": "Env: linux"}], "messages": turns})
    return inline is not None and merged is not None and inline > merged + 1


def llamacpp_ctx(port: int) -> "int | None":
    """llama.cpp's runtime context window (/props n_ctx); None anywhere else."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=2) as r:
            props = json.load(r)
    except Exception:
        return None
    gen = props.get("default_generation_settings") or {}
    for v in (props.get("n_ctx"), gen.get("n_ctx")):
        if isinstance(v, int) and v > 0:
            return v
    return None


def live_routes(ports: list[int]) -> list[Live]:
    # TEST SEAM. Probing real sockets makes a gate depend on whatever the rig
    # happens to be serving, which is neither deterministic nor reproducible on
    # a contributor's machine. `C3_LITELLM_FAKE_LIVE="8182=a,8091=b"` substitutes
    # the probe result wholesale; `8113=qwen@262144` adds the context window,
    # and a trailing `+messages` marks a server that serves /v1/messages itself.
    # Deliberately env-only and undocumented in --help: it is for tests, not
    # operators.
    fake = os.environ.get("C3_LITELLM_FAKE_LIVE")
    if fake is not None:
        out: list[Live] = []
        for item in fake.split(","):
            if "=" in item:
                p, rest = item.split("=", 1)
                rest, _, flag = rest.partition("+")
                mid, _, ml = rest.partition("@")
                if p.strip().isdigit():
                    out.append((int(p.strip()), mid.strip(),
                                int(ml) if ml.strip().isdigit() else None,
                                flag.strip() == "messages"))
        return out
    found: list[Live] = []
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        for res in ex.map(probe, ports):
            found += res
    return found


def port_facts(variants: list[dict]) -> dict[int, dict]:
    """Per port, the registry facts every variant on that port agrees on. A port
    can host several slugs (:8020 is multi-slug); a fact they disagree on is
    left out rather than guessed."""
    by_port: dict[int, list[dict]] = {}
    for v in variants:
        if v.get("port"):
            by_port.setdefault(int(v["port"]), []).append(v)
    facts: dict[int, dict] = {}
    for port, rows in by_port.items():
        f: dict = {}
        for key in ("configured_ctx", "vision"):
            vals = {json.dumps(r.get(key), sort_keys=True) for r in rows}
            if len(vals) == 1 and rows[0].get(key) not in (None, "", []):
                f[key] = rows[0].get(key)
        # Reasoning: only from slugs that DECLARE sampler profiles (a "thinking"
        # profile means a thinking model). No profiles is "unknown", not "no" —
        # most slugs declare none, and a false `supports_reasoning` would make a
        # client turn thinking off on a thinking model.
        declared = [r["sampler_profiles"] for r in rows if isinstance(r.get("sampler_profiles"), dict) and r["sampler_profiles"]]
        thinking = {any("think" in str(k).lower() for k in p) for p in declared}
        if len(thinking) == 1:
            f["thinking"] = thinking.pop()
        facts[port] = f
    return facts


# Every local route: the openai provider, plus reasoning_effort let through.
#   - openai, NOT hosted_vllm: LiteLLM's hosted_vllm transform drops
#     `reasoning_content` from past assistant turns (it keeps `reasoning`) — the
#     field omp replays and SGLang answers with. Qwen3.8's template re-renders
#     past reasoning, so the model would see an empty think block where the
#     reasoning its client chose to keep should be. openai forwards the messages
#     untouched. (Prefix reuse is unaffected either way: measured 2026-09-27.)
#   - allowed_openai_params: the openai provider otherwise answers a top-level
#     `reasoning_effort` with HTTP 400 (UnsupportedParamsError) before the request
#     reaches the engine. vLLM and SGLang render it into the chat template;
#     llama.cpp ignores it.
# chat_template_kwargs, thinking_token_budget and top_k pass through as-is.
# ⚠️ omp's `discovery: litellm` talks the Responses API to a route whose provider
# is `openai` (chat completions otherwise); /v1/responses is served natively by
# vLLM, SGLang and llama.cpp — see docs/CODING_AGENTS.md.
# Claude Code talks Anthropic /v1/messages. On an `openai` route LiteLLM translates
# that to the engine's Responses API, and that bridge builds thinking blocks only
# from a reasoning SUMMARY, which vLLM and SGLang never produce: Claude Code got the
# answer but none of the model's reasoning, so it had nothing to replay. Its
# chat-completions bridge maps reasoning both ways, but it re-routes a thinking
# request to Responses unless the route says `supports_reasoning: false`, which
# /model_group/info publishes (pi-setup.sh would turn thinking off). So a route on
# an engine whose own /v1/messages holds up (SGLang, and vLLM started with
# --chat-template — see serves_messages) lists it in `supported_endpoints`, and
# LiteLLM forwards the Anthropic request to the engine untranslated.
MESSAGES_ENDPOINTS = '      supported_endpoints: ["/v1/chat/completions", "/v1/responses", "/v1/messages"]'
ROUTE_PARAMS = ["      allowed_openai_params: [reasoning_effort]"]


def model_info(ctx: "int | None", facts: dict, native_messages: bool = False) -> list[str]:
    """What omp's `discovery: litellm` reads from /model_group/info. Without it
    an agent falls back to a default context window and output cap: compaction
    fires far too early and busts the prefix cache, and long file writes get
    cut off at the output cap."""
    ctx = ctx or (facts.get("configured_ctx") if isinstance(facts.get("configured_ctx"), int) else None)
    lines = ["    model_info:", "      mode: chat", "      supports_function_calling: true"]
    if ctx:
        lines += [f"      max_input_tokens: {ctx}", f"      max_output_tokens: {min(32768, ctx // 2)}"]
    if "thinking" in facts:
        lines.append(f"      supports_reasoning: {'true' if facts['thinking'] else 'false'}")
    if facts.get("vision") not in (None, "", "no", False):
        lines.append("      supports_vision: true")
    if native_messages:
        lines.append(MESSAGES_ENDPOINTS)
    return lines


def render_block(live: list[Live], facts: "dict[int, dict] | None" = None) -> tuple[str, int]:
    lines = [
        BEGIN.rstrip() + " — RUNTIME VIEW, rendered by scripts/lib/litellm-sync.sh ===",
        "  # Routes for endpoints that answered /v1/models at render time. Names come",
        "  # from each server's own model list, so a slug with no registry served_name",
        "  # still gets a route. Regenerated on every switch.sh launch and teardown.",
        "  # DO NOT EDIT — edit services/litellm/config.yaml (the catalog view) instead.",
    ]
    seen: set[str] = set()
    facts = facts or {}
    for port, mid, ctx, native in sorted(live, key=lambda t: (t[1], t[0])):
        if mid in seen:
            lines.append(f"  # ⚠️ '{mid}' is also served on :{port}; keeping the first route only")
            continue
        seen.add(mid)
        lines += [
            f"  - model_name: {mid}",
            "    litellm_params:",
            f"      model: openai/{mid}",
            f"      api_base: http://host.docker.internal:{port}/v1",
            "      api_key: EMPTY",
            *ROUTE_PARAMS,
        ]
        lines += model_info(ctx, facts.get(port, {}), native) + [""]
    if not live:
        lines.append("  # (nothing serving right now — no local routes)")
    return "\n".join(lines).rstrip() + "\n" + END + "\n", len(seen)


def prune(text: str, reg_ports: set[str], live_ports: set[str]) -> str:
    """Drop OUR dead routes. A route is ours when its api_base is
    host.docker.internal on a registry-owned port. Everything else — the cloud
    block, anything on a port we do not own — passes through untouched: not ours
    to garbage-collect, and a cloud endpoint is not dead because a GPU is idle."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    for ln in text.split("\n"):
        # A route chunk ends at the next route OR at any top-level key: a block
        # like `litellm_settings:` after the last route must never ride along
        # with (and be pruned together with) that route.
        top_level = bool(ln) and not ln[0].isspace() and not ln.startswith("#")
        if (ln.startswith("  - model_name:") or top_level) and cur:
            chunks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append(cur)
    kept: list[str] = []
    for c in chunks:
        body = "\n".join(c)
        m = re.search(r"api_base:\s*http://host\.docker\.internal:(\d+)/", body)
        if m and m.group(1) in reg_ports and m.group(1) not in live_ports:
            continue
        kept.append(body)
    return "\n".join(kept)


def local_routes(root: str) -> str:
    """This rig's own routes — cloud endpoints, private services — so they never go
    into the tracked catalog: litellm/config.local.yaml in the club-3090 config dir,
    else (an older install) the checkout's gitignored services/litellm/
    config.local.yaml — see litellm_local.py. Its `model_list:` entries are copied
    VERBATIM (comments kept; stdlib only, like the rest of the switch path) and
    re-indented to match the catalog. They are not on registry ports, so the prune
    never touches them. Returns '' when there is no file or it lists no routes."""
    found, _origin = litellm_local.active_routes(root)
    if found is None or not found.is_file():
        return ""
    path = str(found)
    body, inside = [], False
    for ln in io.open(path, encoding="utf-8").read().splitlines():
        top = bool(ln) and not ln[0].isspace() and not ln.startswith("#") and not ln.startswith("-")
        if top:
            inside = ln.split("#", 1)[0].strip() == "model_list:"
            continue
        if inside:
            body.append(ln)
    items = [ln for ln in body if ln.lstrip().startswith("- ")]
    if not items:
        print(f"[litellm-sync] WARN: {path} has no model_list entries — nothing added", file=sys.stderr)
        return ""
    shift = 2 - min(len(ln) - len(ln.lstrip()) for ln in items)   # catalog items sit at 2 spaces
    fixed = []
    for ln in body:
        if not ln.strip():
            fixed.append("")
        elif shift >= 0:
            fixed.append(" " * shift + ln)
        else:
            fixed.append(ln[min(-shift, len(ln) - len(ln.lstrip())):])
    return (LOCAL_BEGIN.format(src=litellm_local.display_path(found, root)) + "\n"
            + "\n".join(fixed).strip("\n") + "\n" + LOCAL_END + "\n")


def mounted_runtime_path() -> "str | None":
    """Host path of the file the running `litellm` container mounts as its config."""
    try:
        out = subprocess.run(
            ["docker", "inspect", "litellm", "--format",
             '{{range .Mounts}}{{if eq .Destination "/app/config.yaml"}}{{.Source}}{{end}}{{end}}'],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return path if out.returncode == 0 and path else None


def gateway_env_names(container: str = "litellm") -> "set[str] | None":
    """The variable NAMES in the gateway container's environment. docker's own
    template drops the values, so none leaves docker. None when there is no such
    container or docker can't be asked."""
    try:
        out = subprocess.run(
            ["docker", "inspect", container, "--format",
             '{{range .Config.Env}}{{index (split . "=") 0}}{{"\\n"}}{{end}}'],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    def log(msg: str) -> None:
        if not a.quiet:
            print(f"[litellm-sync] {msg}")

    # Routes or keys still read from the checkout: say so (every run unless
    # --quiet; quiet — switch.sh, gpu-mode — only once). Never on a --check.
    if not a.check:
        litellm_local.notice(a.root, quiet=a.quiet)

    template = os.path.join(a.root, "services/litellm/config.yaml")
    runtime = os.path.join(a.root, "services/litellm/config.runtime.yaml")
    # The gateway serves the file its container MOUNTS. A switch.sh run from a
    # worktree used to re-render the worktree's copy while the gateway kept the
    # main checkout's stale routes. Real renders follow the mount; the test seam
    # never does (a test must not rewrite a live gateway's config).
    if os.environ.get("C3_LITELLM_FAKE_LIVE") is None:
        mounted = mounted_runtime_path()
        if mounted and os.path.abspath(mounted) != os.path.abspath(runtime):
            log(f"gateway mounts {mounted} — rendering that file")
            runtime = mounted
    if not os.path.exists(template):
        print(f"[litellm-sync] no template at {template}", file=sys.stderr)
        return 1

    src = io.open(template, encoding="utf-8").read()
    b, e = src.find(BEGIN), src.find(END)
    if b < 0 or e < 0:
        print("[litellm-sync] template has no BEGIN/END markers", file=sys.stderr)
        return 1

    variants = registry_variants(a.root)
    ports = registry_ports(variants)
    live = live_routes(ports)
    reg_ports = {str(p) for p in ports}
    live_ports = {str(r[0]) for r in live}

    block, n_routes = render_block(live, port_facts(variants))
    out = (prune(src[:b], reg_ports, live_ports)
           + block
           + local_routes(a.root)
           + prune(src[e + len(END):], reg_ports, live_ports))

    cur = io.open(runtime, encoding="utf-8").read() if os.path.exists(runtime) else None
    if a.check:
        if cur == out:
            log("runtime config is up to date")
            return 0
        log("runtime config is STALE — run: bash scripts/lib/litellm-sync.sh")
        return 1

    # A route whose saved key the running gateway was created without (a new route,
    # or a key saved since): the restart below can't add it, only a recreate can.
    # Skipped under --no-restart (the caller is about to start the gateway itself,
    # with its keys) and under the test seam.
    if not a.no_restart and os.environ.get("C3_LITELLM_FAKE_LIVE") is None:
        names = gateway_env_names()
        missing = litellm_local.keys_missing_from_gateway(a.root, names) if names is not None else []
        if missing:
            print(f"[litellm-sync] WARN: the running gateway has no {', '.join(missing)}, which your routes use. "
                  "A restart can't add it; recreate the gateway: bash scripts/gpu-mode.sh gateway", file=sys.stderr)

    if cur == out:
        log(f"no change ({n_routes} live route(s))")
        return 0

    tmp = runtime + ".tmp"                       # temp + replace: a failed encode
    io.open(tmp, "w", encoding="utf-8").write(out)   # never truncates the original
    os.replace(tmp, runtime)
    log(f"rendered {n_routes} live route(s) -> services/litellm/config.runtime.yaml")

    # LiteLLM has NO config-reload endpoint on the pinned image (POST
    # /config/reload -> 404), so a changed file needs a restart. Only on a real
    # change, or every launch would bounce the gateway for nothing.
    if not a.no_restart:
        try:
            names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                   capture_output=True, text=True, timeout=15).stdout.split()
            if "litellm" in names:
                subprocess.run(["docker", "restart", "litellm"],
                               capture_output=True, timeout=90, check=True)
                log("restarted litellm to pick it up")
        except Exception:
            log("WARN: could not restart litellm — routes apply on its next start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
