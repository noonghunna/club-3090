#!/usr/bin/env bash
#
# Agentic prefill stress benchmark.
#
# Simulates a multi-turn coding-agent session and measures TTFT + decode TPS
# as context accumulates over N turns of tool calls. This is the relevant
# workload for Cline / Cursor / Claude Code — NOT single large prompts.
#
# Why this matters:
#   bench.sh uses single-prompt, single-turn requests. It measures decode
#   throughput but does NOT stress incremental prefill — the cost that
#   dominates every round >5 in a real coding session.
#
#   On standard-attention engines, TTFT may stay roughly flat after warmup
#   when cached context can be reused. On DeltaNet/SSM hybrid models served
#   through vLLM, the recurrent SSM state is not prefix-cacheable, so TTFT can
#   grow O(n) with accumulated context even when attention KV caching works.
#   Treat this as a per-(engine, arch_class, config) curve-shape producer, not
#   a universal cache verdict.
#
# Fixture:
#   scripts/fixtures/agentic-bench-fixture.json — 15 turns of real tool
#   call results extracted from an actual Claude Code session against this
#   repo (filesystem paths redacted). The payload is opaque context used only
#   to grow prompt depth; this script does not parse fixture paths or commands
#   as live config. Sizes range from 300 chars (ls output) to 35K chars (large
#   file reads), reaching ~53K accumulated prompt tokens by turn 15.
#
# Cliff 2 context (vLLM DeltaNet observation — llama.cpp is not affected):
#   Qwen3.6-27B is a DeltaNet/Mamba hybrid. The DeltaNet SSM recurrent
#   state CANNOT be prefix-cached — it must be recomputed from the full
#   sequence on every turn. On one measured vLLM single-card 24 GB
#   Qwen3-Next cell, TTFT degraded noticeably above ~35K accumulated tokens
#   and requests timed out around ~74K. Treat those as informational
#   per-arch_class observations, not universal thresholds.
#
# Output:
#   Per-turn table (turn, prompt_tokens, ttft_ms, decode_tps)
#   TTFT growth analysis: flat = low incremental prefill; linear = O(n)
#
# Usage:
#   bash scripts/bench-agentic.sh
#   SESSIONS=3 bash scripts/bench-agentic.sh       # 3 sessions for stats
#   TURNS=10 bash scripts/bench-agentic.sh         # stop at turn 10
#   QUIET=1 bash scripts/bench-agentic.sh          # suppress per-req lines
#
# Env vars:
#   URL          Endpoint. Default: auto-detect running service, else bench.sh fallback
#   MODEL        Served model name. Default: auto-detected from /v1/models
#   CONTAINER    For GPU + spec-decode log scrape. Default: auto-detect
#   SESSIONS     Sessions to run (for per-turn TTFT statistics). Default: 2
#   TURNS        Turns per session (1-15). Default: 12
#   QUIET        Set to 1 to suppress per-request status lines. Default: 0

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint || true
fi
URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-}"
CONTAINER="${CONTAINER:-}"
SESSIONS="${SESSIONS:-2}"
TURNS="${TURNS:-12}"
QUIET="${QUIET:-0}"

FIXTURE="${ROOT_DIR}/scripts/fixtures/agentic-bench-fixture.json"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }; }
need curl
need python3

if [[ ! -f "$FIXTURE" ]]; then
  echo "ERROR: fixture not found: $FIXTURE" >&2
  exit 1
fi

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: bash scripts/launch.sh (or bash scripts/switch.sh <variant>)" >&2
  exit 1
fi

# Auto-detect model name from API if not explicitly provided
if [[ -z "$MODEL" ]]; then
  MODEL=$(curl -sf "${URL}/v1/models" | python3 -c \
    "import json,sys; d=json.load(sys.stdin).get('data',[]); print(d[0]['id'] if d else '')" 2>/dev/null || true)
fi
if [[ -z "$MODEL" ]]; then
  echo "ERROR: could not detect model name from ${URL}/v1/models — set MODEL=<name>" >&2
  exit 1
fi

python3 - "$URL" "$MODEL" "$SESSIONS" "$TURNS" "$QUIET" "$FIXTURE" << 'PYEOF'
import json, sys, time, urllib.request, statistics as s, pathlib
sys.stdout.reconfigure(line_buffering=True)  # flush after every \n

URL, MODEL, SESSIONS, TURNS, QUIET, FIXTURE_PATH = sys.argv[1:7]
SESSIONS = int(SESSIONS); TURNS = int(TURNS); QUIET = int(QUIET) == 1

# Load real-session fixtures (tool results from an actual Claude Code session)
FIXTURE = json.loads(pathlib.Path(FIXTURE_PATH).read_text())
# Cap to requested TURNS
FIXTURE = FIXTURE[:TURNS]

# ---------------------------------------------------------------------------
# System prompt + tool schemas (fixed across all turns/sessions so prefix
# caching can warm up after the first turn of the first session).
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are an autonomous coding assistant working inside a Python repository. "
    "The user is investigating a performance regression. When file contents, "
    "search results, or command output would materially change your answer, "
    "call the appropriate tool — don't speculate. After each tool call, "
    "briefly state what you learned and what your next planned step is. "
    "Keep responses concise (under 100 words); defer to tools for raw data.\n\n"
    "Repository layout:\n"
    "  scripts/         — bench, verify, soak, launch helper scripts\n"
    "  models/          — per-model compose configs + patches\n"
    "  docs/            — architecture and cliff notes\n"
    "  BENCHMARKS.md    — measured performance numbers\n"
    "  CHANGELOG.md     — version history\n"
)

TOOLS = [
    {"type": "function", "function": {
        "name": n, "description": d,
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "command": {"type": "string"},
            "pattern": {"type": "string"},
            "recursive": {"type": "boolean"},
        }, "required": []}}}
    for n, d in [
        ("Read",          "Read a UTF-8 file from the repository."),
        ("Bash",          "Execute a shell command and return stdout+stderr."),
        ("Edit",          "Apply a string replacement edit to a file."),
        ("Write",         "Write or overwrite a file."),
        ("Grep",          "Search for a regex pattern across the codebase."),
        ("LS",            "List files in a directory."),
        ("TodoRead",      "Read the current task/todo list."),
        ("TodoWrite",     "Create or update a task/todo list."),
        ("WebSearch",     "Search the web for information."),
        ("WebFetch",      "Fetch a URL and return the HTML/text."),
    ]
]


def run_turn(messages, fixture_turn, session_id, turn_idx):
    user_msg = fixture_turn["user_msg"]
    tool_result_content = fixture_turn["tool_result"]

    messages.append({"role": "user", "content": user_msg})

    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "required",  # guarantee a tool call every turn
        "max_tokens": 150,
        "temperature": 0.3,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    content_parts = []
    tool_calls_acc = {}

    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if ttft is None and (delta.get("content") or delta.get("tool_calls")):
                    ttft = time.time() - t_send
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"): slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"): slot["name"] = fn["name"]
                    if fn.get("arguments"): slot["args"] += fn["arguments"]
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall

    # Reconstruct assistant message from real tool calls.
    # tool_choice=required guarantees at least one; treat empty as a server bug.
    tool_calls_response = [
        {"id": s["id"] or f"call_t{turn_idx}_s{session_id}_{i}",
         "type": "function",
         "function": {"name": s["name"], "arguments": s["args"] or "{}"}}
        for i, s in sorted(tool_calls_acc.items()) if s["name"]
    ]
    if not tool_calls_response:
        raise RuntimeError(
            f"server returned no tool calls despite tool_choice=required "
            f"(turn {turn_idx+1}). Check that the endpoint supports tool_choice=required."
        )
    # Sanitize: strip lone surrogates that json.dumps would emit as
    # invalid \uD800-\uDFFF sequences, causing server-side 400s.
    def _clean(s):
        return s.encode("utf-8", errors="replace").decode("utf-8")
    content = _clean("".join(content_parts)) or None
    assistant_msg = {"role": "assistant", "tool_calls": tool_calls_response}
    if content:
        assistant_msg["content"] = content
    messages.append(assistant_msg)

    # Inject the fixture tool result onto the model's real tool call IDs.
    # The model may have called a different tool than the original session;
    # that's intentional — fixed results make TTFT measurements reproducible
    # across runs and engines. Only the first call gets the full result; any
    # additional calls (rare with max_tokens=150) get a placeholder so the
    # context size matches the single-tool-call case.
    messages.append({
        "role": "tool",
        "tool_call_id": tool_calls_response[0]["id"],
        "content": tool_result_content,
    })
    for tc in tool_calls_response[1:]:
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "(done)"})

    decode_s = max(wall - ttft, 1e-6)
    decode_tps = completion_tokens / decode_s if completion_tokens > 0 else 0

    return {
        "ttft_ms": ttft * 1000,
        "wall_ms": wall * 1000,
        "decode_tps": decode_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "tool_calls": len(tool_calls_response),
        "result_chars": len(tool_result_content),
    }


# ---------------------------------------------------------------------------
# Run sessions and collect per-turn metrics
# ---------------------------------------------------------------------------
per_turn_metrics = [[] for _ in range(TURNS)]

for session in range(1, SESSIONS + 1):
    print(f"\n{'='*72}")
    print(f"SESSION {session}/{SESSIONS} — {TURNS} turns, context grows to ~{sum(f['chars'] for f in FIXTURE)//4:,} tokens")
    print(f"{'='*72}")
    print(f"  {'Turn':<5} {'Prompt tok':>10} {'TTFT ms':>9} {'Decode TPS':>11} {'Result chars':>13}")
    print(f"  {'-'*5} {'-'*10} {'-'*9} {'-'*11} {'-'*13}")

    messages = [{"role": "system", "content": SYSTEM}]

    for turn_idx in range(TURNS):
        fixture_turn = FIXTURE[turn_idx]
        try:
            m = run_turn(messages, fixture_turn, session, turn_idx)
            per_turn_metrics[turn_idx].append(m)
            if not QUIET:
                print(f"  {turn_idx+1:<5} {m['prompt_tokens']:>10,} {m['ttft_ms']:>9.0f} "
                      f"{m['decode_tps']:>11.1f} {m['result_chars']:>13,}", flush=True)
        except Exception as e:
            print(f"  turn {turn_idx+1}: FAIL — {e}", flush=True)
            break


# ---------------------------------------------------------------------------
# Summary table: per-turn means across sessions
# ---------------------------------------------------------------------------
print(f"\n\n{'='*72}")
print(f"SUMMARY — multi-turn prefill stress ({SESSIONS} session(s) × {TURNS} turns)")
print(f"{'='*72}")
print(f"  {'Turn':<5} {'Prompt tok':>10} {'TTFT ms':>9} {'σ ms':>6} {'Decode TPS':>11}  Notes")
print(f"  {'-'*5} {'-'*10} {'-'*9} {'-'*6} {'-'*11}  {'─'*35}")

turn1_ttft = None
active_turns = 0
for turn_idx in range(TURNS):
    ms_list = per_turn_metrics[turn_idx]
    if not ms_list:
        break
    active_turns += 1
    ttfts = [m["ttft_ms"] for m in ms_list]
    tpss  = [m["decode_tps"] for m in ms_list if m["decode_tps"] > 0]
    ptoks = [m["prompt_tokens"] for m in ms_list]
    mean_ttft = s.mean(ttfts)
    std_ttft  = s.stdev(ttfts) if len(ttfts) > 1 else 0
    mean_tps  = s.mean(tpss) if tpss else 0
    mean_ptok = s.mean(ptoks)

    if turn_idx == 0:
        turn1_ttft = mean_ttft

    note = ""
    if turn1_ttft and turn_idx > 0 and mean_ttft > 0:
        ratio = mean_ttft / turn1_ttft
        if ratio > 4.0:
            note = f"⚠  TTFT {ratio:.1f}× turn-1 (O(n)-like growth for this arch_class)"
        elif ratio > 2.0:
            note = f"↑  TTFT {ratio:.1f}× turn-1"
        elif ratio > 1.4:
            note = f"~  TTFT {ratio:.1f}× turn-1"

    print(f"  {turn_idx+1:<5} {mean_ptok:>10,.0f} {mean_ttft:>9.0f} {std_ttft:>6.0f} {mean_tps:>11.1f}  {note}")

# TTFT growth analysis
filled = [per_turn_metrics[i] for i in range(active_turns) if per_turn_metrics[i]]
if len(filled) >= 2:
    first_ttft = s.mean([m["ttft_ms"] for m in filled[0]])
    last_ttft  = s.mean([m["ttft_ms"] for m in filled[-1]])
    first_ptok = s.mean([m["prompt_tokens"] for m in filled[0]])
    last_ptok  = s.mean([m["prompt_tokens"] for m in filled[-1]])

    ttft_growth  = last_ttft / first_ttft if first_ttft > 0 else 0
    token_growth = last_ptok / first_ptok if first_ptok > 0 else 0

    print(f"\n{'─'*72}")
    print(f"  TTFT growth by accumulated context ({active_turns} turns, {SESSIONS} sessions):")
    print(f"    Turn 1:  {first_ttft:.0f} ms TTFT @ {first_ptok:,.0f} prompt tokens")
    print(f"    Turn {active_turns}: {last_ttft:.0f} ms TTFT @ {last_ptok:,.0f} prompt tokens")
    print(f"    Context grew {token_growth:.1f}×,  TTFT grew {ttft_growth:.1f}×")
    if ttft_growth <= 1.5:
        print("    ✓  TTFT stable for this engine/arch/config cell.")
    elif ttft_growth <= 2.5:
        print(f"    ~  TTFT grew modestly ({ttft_growth:.1f}× vs {token_growth:.1f}× context).")
    elif ttft_growth <= token_growth * 0.5:
        print(f"    ~  TTFT sub-linear for this cell ({ttft_growth:.1f}× vs {token_growth:.1f}× context).")
    else:
        print(f"    ⚠  TTFT grew near-linearly — O(n)-like accumulated-context cost for this cell.")
    print(f"    (Full-context O(n) growth would approach {token_growth:.1f}× with context)")
    print(f"")
    print(f"  Note — DeltaNet/SSM state is NOT prefix-cacheable on vLLM Qwen3-Next cells.")
    print(f"  Attention KV caching can still work, but recurrent-state recomputation scales")
    print(f"  O(n) with sequence length. Prior single-card 24 GB vLLM Qwen3-Next observations")
    print(f"  saw degradation above ~35K tokens and timeouts around ~74K; treat those as")
    print(f"  informational per-arch_class guideposts. llama.cpp is not affected.")

PYEOF

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi

# MTP / spec-decode stats
if command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo ""
  echo "=== Last 3 SpecDecoding metrics ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi
