#!/usr/bin/env python3
import csv
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to read."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files under a directory for a text pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "dir": {"type": "string"}},
                "required": ["pattern", "dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a non-destructive shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]


def base_req(model, messages, max_tokens, temp=0.4, thinking=False, tools=False):
    req = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # `chat_template_kwargs.enable_thinking` is a Qwen3-family feature.
    # Other model families (Gemma 4, etc.) reject it with 400. Skip via:
    #   SOAK_NO_CHAT_TEMPLATE_KWARGS=1
    #
    # WHICH key controls reasoning is model-specific, and a family that uses a
    # different one IGNORES this silently — no 400, no warning — so the model
    # reasons at full effort for the whole soak while the run believes thinking
    # is off. soak-test.sh resolves the real key via preflight.sh
    # ::preflight_detect_thinking_control and exports it as final JSON.
    if os.environ.get("SOAK_NO_CHAT_TEMPLATE_KWARGS") != "1":
        raw = os.environ.get("THINK_ON_KW" if thinking else "THINK_OFF_KW")
        try:
            req["chat_template_kwargs"] = (
                json.loads(raw) if raw else {"enable_thinking": thinking})
        except Exception:
            req["chat_template_kwargs"] = {"enable_thinking": thinking}
        # The OpenAI-standard top-level parameter, for engines that honour it.
        effort = os.environ.get("THINK_ON_EFFORT" if thinking else "THINK_OFF_EFFORT")
        if effort:
            req["reasoning_effort"] = effort
    if tools:
        req["tools"] = TOOLS
        req["tool_choice"] = "auto"
    return req


# ---- Continuous-mode fixtures (SOAK_MODE=continuous) -----------------------
# Each session is a single multi-turn agentic-coding conversation with growing
# context — mirrors the hermes/openhands workload pattern that bit GuiPerPT
# in club-3090#41 (fresh-mode reset-each-turn fixtures don't surface that
# accretion class). By turn 5, accumulated context ≈ 22-25K tokens.

CONTINUOUS_SYSTEM = (
    "You are an autonomous coding assistant working inside a small Python "
    "service repository. The user is debugging a production issue. When file "
    "contents, search results, or command output would materially change "
    "your answer, call the appropriate tool — don't speculate. Keep "
    "responses concise; defer to the tools for raw data.\n\n"
    "Repository layout you can assume:\n"
    "  src/handlers.py  — webhook handler entry points\n"
    "  src/payloads.py  — payload validation + parsing\n"
    "  src/db.py        — database access layer\n"
    "  tests/           — pytest suite mirrors src/\n"
    "  logs/app.log     — recent service logs\n"
)


def _filler_python_code(target_chars):
    """Generate plausible-looking Python code text of approximately target_chars."""
    block = (
        "def handle_webhook(payload, db_conn=None):\n"
        "    validated = validate_payload(payload)\n"
        "    if validated is None:\n"
        "        raise InvalidPayloadError('payload missing required fields')\n"
        "    txn = validated.get('transaction_id')\n"
        "    cust = validated.get('customer_id')\n"
        "    amount = float(validated.get('amount', 0))\n"
        "    record = persist_record(db_conn, txn, cust, amount)\n"
        "    notify_downstream(record)\n"
        "    return {'status': 'ok', 'record_id': record.id}\n"
        "\n"
        "def validate_payload(payload):\n"
        "    if not isinstance(payload, dict):\n"
        "        return None\n"
        "    required = ('transaction_id', 'customer_id', 'amount')\n"
        "    if not all(k in payload for k in required):\n"
        "        return None\n"
        "    return payload\n"
        "\n"
        "def persist_record(conn, txn, cust, amount):\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(\n"
        "        'INSERT INTO transactions (txn_id, cust_id, amount, ts) '\n"
        "        'VALUES (%s, %s, %s, NOW()) RETURNING id',\n"
        "        (txn, cust, amount),\n"
        "    )\n"
        "    return cursor.fetchone()\n"
        "\n"
    )
    repeats = (target_chars // len(block)) + 1
    return (block * repeats)[:target_chars]


def _filler_grep_output(target_chars):
    """Generate plausible-looking grep -rn output of approximately target_chars."""
    block = (
        "src/handlers.py:14:    txn = validated.get('transaction_id')\n"
        "src/handlers.py:42:    log.info('processed transaction_id=%s', txn)\n"
        "src/payloads.py:28:    REQUIRED_KEYS = ('transaction_id', 'customer_id', 'amount')\n"
        "src/payloads.py:55:        log.error('missing transaction_id in payload %r', raw)\n"
        "src/db.py:88:    SELECT * FROM transactions WHERE transaction_id = %s\n"
        "tests/test_handlers.py:21:    payload = {'transaction_id': 'txn_001', ...}\n"
        "tests/test_handlers.py:43:    assert result['transaction_id'] == 'txn_001'\n"
        "tests/test_payloads.py:11:    bad = {'customer_id': 'c1', 'amount': 12.5}\n"
        "tests/test_payloads.py:12:    # missing transaction_id intentionally\n"
        "tests/test_payloads.py:18:    assert validate_payload(bad) is None\n"
        "logs/app.log:142:KeyError: 'transaction_id' at handlers.py:14\n"
        "logs/app.log:148:KeyError: 'transaction_id' at handlers.py:14\n"
    )
    repeats = (target_chars // len(block)) + 1
    return (block * repeats)[:target_chars]


def _filler_command_output(target_chars):
    """Generate plausible-looking pytest output of approximately target_chars."""
    block = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.5.0\n"
        "rootdir: /workspace, configfile: pyproject.toml\n"
        "collected 14 items\n"
        "\n"
        "tests/test_handlers.py::test_happy_path PASSED                             [  7%]\n"
        "tests/test_handlers.py::test_missing_amount FAILED                         [ 14%]\n"
        "tests/test_handlers.py::test_invalid_customer FAILED                       [ 21%]\n"
        "tests/test_payloads.py::test_validate_full PASSED                          [ 28%]\n"
        "tests/test_payloads.py::test_validate_missing_txn FAILED                   [ 35%]\n"
        "\n"
        "=================================== FAILURES ===================================\n"
        "____________________ test_missing_amount ____________________\n"
        "    def test_missing_amount():\n"
        "        payload = {'transaction_id': 'txn_002', 'customer_id': 'c2'}\n"
        ">       result = handle_webhook(payload)\n"
        "E       KeyError: 'transaction_id'\n"
        "src/handlers.py:14: KeyError\n"
        "----------------------------- captured log call --------------------------------\n"
        "ERROR    src.handlers:handlers.py:14 KeyError on payload {'transaction_id': 'txn_002'...}\n"
        "\n"
    )
    repeats = (target_chars // len(block)) + 1
    return (block * repeats)[:target_chars]


# (turn, role, content_or_call_spec). Driven entirely by data so the request
# generator + ingestion are simple table lookups.
CONTINUOUS_TURNS = [
    # turn 1: opening user message — agent is expected to call read_file.
    {
        "turn": 1,
        "user": (
            "We're seeing a KeyError 'transaction_id' in production every few "
            "minutes when handle_webhook runs. Can you investigate the handler "
            "and figure out where this is coming from? Start with src/handlers.py."
        ),
        "tool_synth": None,
        "max_tokens": 350,
        "temp": 0.3,
        "thinking": False,
    },
    # turn 2: ingest synthetic file contents, ask follow-up.
    {
        "turn": 2,
        "user": (
            "OK now show me the test file at tests/test_handlers.py — I want to "
            "see whether this case has a regression test."
        ),
        "tool_synth": ("read_file", "python_code", 20000),  # ~5K toks of Python
        "max_tokens": 350,
        "temp": 0.25,
        "thinking": False,
    },
    # turn 3: ingest more synthetic content, request a grep.
    {
        "turn": 3,
        "user": (
            "Now grep across the whole codebase for 'transaction_id' so we can "
            "see every place this key is used or referenced. Make sure to "
            "include test files and log lines."
        ),
        "tool_synth": ("read_file", "python_code", 24000),  # ~6K toks of Python
        "max_tokens": 400,
        "temp": 0.25,
        "thinking": False,
    },
    # turn 4: ingest grep output, ask for command run.
    {
        "turn": 4,
        "user": (
            "Run the test suite and show me the full failing-test output for "
            "any test that exercises the handler path."
        ),
        "tool_synth": ("grep", "grep_output", 24000),  # ~6K toks of grep results
        "max_tokens": 500,
        "temp": 0.3,
        "thinking": False,
    },
    # turn 5: ingest command output, final summary + fix request — heaviest turn,
    # operating at ~22-25K accumulated context which is GuiPerPT's #41 territory.
    {
        "turn": 5,
        "user": (
            "Based on everything we've looked at, write a fix for the KeyError "
            "and explain in 4-6 bullets what was wrong, what change closes it, "
            "and what regression test should be added. Write the fix as a "
            "code block at the top, then the explanation."
        ),
        "tool_synth": ("run_command", "command_output", 32000),  # ~8K toks of pytest output
        "max_tokens": 1500,
        "temp": 0.35,
        "thinking": False,
    },
]


def _continuous_synth_filler(kind, target_chars):
    if kind == "python_code":
        return _filler_python_code(target_chars)
    if kind == "grep_output":
        return _filler_grep_output(target_chars)
    if kind == "command_output":
        return _filler_command_output(target_chars)
    raise ValueError(f"unknown filler kind: {kind}")


def continuous_initial_state(session):
    """Initial state for a continuous session — system prompt + tools, no user yet."""
    return {
        "session_id": int(session),
        "messages": [
            {"role": "system", "content": CONTINUOUS_SYSTEM},
        ],
        "tool_calls_seen": 0,
        "fallback_tool_calls_synthesized": 0,
    }


def fixture(model, session, turn):
    if turn == 1:
        return base_req(
            model,
            [
                {"role": "system", "content": "You are a concise coding assistant."},
                {"role": "user", "content": f"Session {session}: give a short checklist for reviewing a small Python patch."},
            ],
            220,
            temp=0.3,
        )
    if turn == 2:
        return base_req(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "You are working inside a repository. Prefer tools when file "
                        "contents or command output would materially change the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Session {session}: inspect scripts/verify-full.sh and tell me "
                        "whether there is a server reachability check. Use the tools."
                    ),
                },
            ],
            320,
            temp=0.2,
            tools=True,
        )
    if turn == 3:
        block = (
            "src/example.py: def handle_request(payload):\n"
            "    validate(payload)\n"
            "    result = service.call(payload)\n"
            "    return {'ok': True, 'result': result}\n"
            "tests/test_example.py: assert handle_request({'x': 1})['ok'] is True\n"
            "logs/app.log: INFO request completed in 42ms\n"
        )
        payload = (block * 120)[:12000]
        call_id = f"call_soak_{session}"
        return base_req(
            model,
            [
                {"role": "system", "content": "You are a concise coding assistant."},
                {"role": "user", "content": "Read the relevant files and summarize the failure mode."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": json.dumps({"pattern": "handle_request", "dir": "."}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": payload},
                {"role": "user", "content": "Identify the most likely missing test and keep it under 8 bullets."},
            ],
            700,
            temp=0.35,
            tools=True,
        )
    if turn == 4:
        return base_req(
            model,
            [
                {"role": "system", "content": "You write direct, production-quality code."},
                {
                    "role": "user",
                    "content": (
                        "Implement a Python function parse_size(s) that accepts values like "
                        "'128MiB', '2 GiB', and '4096', returns bytes, and raises ValueError "
                        "for invalid input. Include compact tests."
                    ),
                },
            ],
            900,
            temp=0.45,
        )
    return base_req(
        model,
        [
            {"role": "system", "content": "Solve carefully and show the final answer clearly."},
            {
                "role": "user",
                "content": (
                    f"Session {session}: Cache A grows by 6 MiB per request until it resets every "
                    "11 requests. Cache B grows by 14 MiB on prime-numbered requests and never "
                    "resets during the run. Starting from 21000 MiB used on a 24576 MiB card, "
                    "after 25 requests what is peak used memory, and which request first exceeds 23000 MiB?"
                ),
            },
        ],
        2000,
        temp=0.2,
        thinking=True,
    )


def cmd_model(path):
    with open(path) as f:
        data = json.load(f)
    models = data.get("data") or []
    print(models[0].get("id", "qwen3.6-27b") if models else "qwen3.6-27b")


def cmd_baseline(out_dir, container, endpoint, model, sessions, turns, growth):
    out = pathlib.Path(out_dir)
    models = {}
    try:
        models = json.loads((out / "models.json").read_text())
    except Exception:
        pass
    doc = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "container": container,
        "endpoint": endpoint,
        "model": model,
        "soak_sessions": int(sessions),
        "soak_turns": int(turns),
        "soak_max_growth_mib": int(growth),
        "models": models,
    }
    (out / "baseline.json").write_text(json.dumps(doc, indent=2) + "\n")


def cmd_request(model, session, turn, path):
    req = fixture(model, int(session), int(turn))
    pathlib.Path(path).write_text(json.dumps(req) + "\n")


def cmd_init_session(state_path, session):
    """Continuous mode — write the initial session state file (system + tools, no user yet)."""
    state = continuous_initial_state(session)
    pathlib.Path(state_path).write_text(json.dumps(state, indent=2) + "\n")


def cmd_request_continuous(model, state_path, turn, req_path):
    """Continuous mode — generate next turn's request from accumulated state.

    Side-effect: appends the new user message to the state file BEFORE the
    request is issued, so cmd_ingest later only has to append the assistant
    response and (if applicable) a synthetic tool result.
    """
    turn = int(turn)
    state = json.loads(pathlib.Path(state_path).read_text())
    spec = next(t for t in CONTINUOUS_TURNS if t["turn"] == turn)

    # Append the new user message into the running history.
    state["messages"].append({"role": "user", "content": spec["user"]})

    req = base_req(
        model,
        state["messages"],
        max_tokens=spec["max_tokens"],
        temp=spec["temp"],
        thinking=spec["thinking"],
        tools=True,  # tools available across the whole session
    )
    pathlib.Path(req_path).write_text(json.dumps(req) + "\n")
    pathlib.Path(state_path).write_text(json.dumps(state, indent=2) + "\n")


def cmd_ingest(state_path, metrics_path, turn):
    """Continuous mode — append assistant response + synthetic tool result(s) to state.

    For each tool_call the model emitted, we synthesize a tool message of the
    size specified by the next turn's spec (so the NEXT request includes the
    accumulated tool result). If the model didn't emit a tool_call but the
    next turn's spec expects one, we synthesize a fallback assistant tool_call
    + tool result so the conversation keeps growing context as designed.
    """
    turn = int(turn)
    state = json.loads(pathlib.Path(state_path).read_text())
    metrics = json.loads(pathlib.Path(metrics_path).read_text())

    # Append the assistant's response. tool_calls take precedence over content
    # in the OpenAI message schema; if both present, both fields populate.
    assistant_msg = {"role": "assistant"}
    if metrics.get("tool_calls"):
        assistant_msg["tool_calls"] = metrics["tool_calls"]
        assistant_msg["content"] = metrics.get("content") or None
        state["tool_calls_seen"] = state.get("tool_calls_seen", 0) + len(metrics["tool_calls"])
    else:
        assistant_msg["content"] = metrics.get("content") or "(empty response)"
    state["messages"].append(assistant_msg)

    # Look at the NEXT turn's spec to decide whether to synthesize a tool
    # result. The synthetic tool message is what makes context accumulate
    # to the 22-25K target by turn 5 — without it, sessions don't reach
    # GuiPerPT's #41 territory regardless of the model's tool-use behavior.
    next_turn = turn + 1
    next_spec = next((t for t in CONTINUOUS_TURNS if t["turn"] == next_turn), None)
    if next_spec is None or next_spec["tool_synth"] is None:
        pathlib.Path(state_path).write_text(json.dumps(state, indent=2) + "\n")
        return

    expected_tool_name, kind, target_chars = next_spec["tool_synth"]
    filler = _continuous_synth_filler(kind, target_chars)

    if metrics.get("tool_calls"):
        # Use the model's actual tool_call IDs so the tool messages link
        # correctly. Tool name need not match spec — model may pick a
        # different tool, that's fine for soak purposes (we just need
        # the tool message to flow back).
        for tc in metrics["tool_calls"]:
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": filler,
            })
    else:
        # Fallback — model didn't emit a tool_call when the conversation
        # design expected one. Synthesize an assistant tool_call retroactively
        # (insert BEFORE we append the tool message) so the schema is valid.
        synth_id = f"call_fallback_t{turn}_s{state['session_id']}"
        # Replace the just-appended assistant message with one that has a
        # synthetic tool_call. This is a soak-test-only patch-up — real
        # production agents would handle this differently.
        if state["messages"][-1].get("role") == "assistant":
            state["messages"][-1] = {
                "role": "assistant",
                "content": metrics.get("content") or None,
                "tool_calls": [{
                    "id": synth_id,
                    "type": "function",
                    "function": {
                        "name": expected_tool_name,
                        "arguments": json.dumps({"_synthetic": True}),
                    },
                }],
            }
        state["messages"].append({
            "role": "tool",
            "tool_call_id": synth_id,
            "content": filler,
        })
        state["fallback_tool_calls_synthesized"] = state.get("fallback_tool_calls_synthesized", 0) + 1

    pathlib.Path(state_path).write_text(json.dumps(state, indent=2) + "\n")


# --- engine-side decode counters (#1268) -------------------------------------
# A decode rate inferred from SSE arrival times is only honest when the decode
# window (wall - ttft) is wide enough to time. Below DECODE_FLOOR_MS the window
# is dominated by scheduling and transport noise, so the harness refuses to
# quote a rate. That floor is right for the inference and wrong as a
# measurement policy: on fast hardware a legitimate turn finishes decoding
# INSIDE it (#849 measured 82-83 ms windows on dual 5090s; #1261 measured
# 49 / 65 / 97 ms on 3 of 5 turns), so a healthy run can end up with no decode
# figure on most of its turns. Soak's per-turn decode series is how a Cliff-2b
# decay would show up, and a series full of holes can hide one.
#
# So ask the ENGINE for its own decode rate first: it is computed over real
# decode steps and does not care how narrow the wall-clock window is. Client
# timing stays as the FALLBACK, and the floor stays with it, where it is doing
# its job. Sources, in the order they are tried:
#
#   prom-tpot     <engine>:time_per_output_token_seconds on <endpoint>/metrics
#                 (vLLM always; SGLang with --enable-metrics). The DELTA of
#                 (_sum, _count) across one turn is that turn's mean seconds per
#                 output token, i.e. decode steps only — prefill is excluded by
#                 construction. Attributing the delta to one turn is valid
#                 because soak is single-stream by design.
#   sglang-log    "gen throughput (token/s): N" on SGLang's decode-batch lines.
#   llamacpp-log  llama.cpp's per-request "eval time = ... (N tokens per
#                 second)" summary. NOT "prompt eval time", which is the prefill
#                 rate — the lookbehind below is all that keeps them apart.
#
# Every turn records WHICH source produced its number (decode_source), because a
# figure whose provenance a reader has to guess is the defect #1267 is about.
# SOAK_ENGINE_COUNTER=off restores pure client-side timing for the whole run.
#
# NOTE (club-3090#1282): none of this classifies the engine — it probes for a
# counter and matches signatures. When the canonical engine resolver lands
# (scripts/lib/engine-kind.sh) the probe can key off it instead of trying each
# signature in turn. A private engine classifier here would be its own defect,
# so there deliberately is not one.
DECODE_FLOOR_MS = 100.0

# Prometheus histogram lines: `<name>_sum{labels} <value>` / `<name>_count ...`.
# A `_bucket` line cannot match: the suffix group accepts only sum|count.
_TPOT_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_:]*?time_per_output_token_seconds)"
    r"_(?P<field>sum|count)(?:\{[^}]*\})?\s+(?P<value>\S+)\s*$"
)
_SGLANG_LOG_RE = re.compile(r"gen throughput \(token/s\):\s*([0-9]+(?:\.[0-9]+)?)")
_LLAMACPP_LOG_RE = re.compile(
    r"(?<!prompt )eval time\s*=[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second"
)
# A counter reading this far out is a parse error, not a measurement.
ENGINE_TPS_SANE_MAX = 100000.0
ENGINE_SOURCE_LABELS = {
    "prom-tpot": "engine-reported: /metrics time_per_output_token_seconds",
    "sglang-log": "engine-reported: SGLang 'gen throughput (token/s)' log line",
    "llamacpp-log": "engine-reported: llama.cpp 'eval time' log line",
}


def _metrics_url(endpoint):
    return endpoint.rstrip("/") + "/metrics"


def _read_metrics_text(url, timeout=3.0):
    """GET a Prometheus page. Any failure means 'no counter', never a run failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return ""
            return resp.read(4 * 1024 * 1024).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _tpot_series(text):
    """{metric_name: [sum_seconds, count]}, summed across every label series."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _TPOT_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue  # NaN / Inf: the engine has no sample yet
        slot = out.setdefault(m.group("name"), [0.0, 0.0])
        slot[0 if m.group("field") == "sum" else 1] += value
    return out


def engine_rate_from_metrics(before_text, after_text, metric):
    """Decode tok/s for the turn that ran between the two /metrics snapshots."""
    before = _tpot_series(before_text).get(metric)
    after = _tpot_series(after_text).get(metric)
    if not before or not after:
        return None
    d_sum = after[0] - before[0]
    d_count = after[1] - before[1]
    if d_sum <= 0 or d_count <= 0:
        return None
    rate = d_count / d_sum
    if not 0 < rate < ENGINE_TPS_SANE_MAX:
        return None
    return round(rate, 3)


def _rfc3339_nano(ts):
    """`docker logs --since` wants RFC3339. Nanoseconds keep a fast PREVIOUS
    turn's decode line from falling inside this turn's window."""
    return "%s.%09dZ" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)), int((ts % 1) * 1e9))


def _docker_logs_since(container, since_ts, timeout=15.0):
    """Engine log tail for one turn. stderr is MERGED into stdout, not
    discarded: every engine in this repo logs to stderr, so a `2>/dev/null`
    here would read as 'the counter never fired'."""
    try:
        proc = subprocess.run(
            ["docker", "logs", "--since", _rfc3339_nano(since_ts), "--tail", "2000", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""
    return proc.stdout or ""


def engine_rate_from_logs(text):
    """(tok/s, source) from the LAST engine decode-rate line in `text`."""
    for regex, source in ((_SGLANG_LOG_RE, "sglang-log"), (_LLAMACPP_LOG_RE, "llamacpp-log")):
        hits = regex.findall(text)
        if not hits:
            continue
        try:
            value = float(hits[-1])
        except ValueError:
            continue
        if 0 < value < ENGINE_TPS_SANE_MAX:
            return round(value, 3), source
    return None, ""


def decode_label(basis, source, window_ms):
    """Provenance for one turn's decode figure (#1267).

    The contract: an UNMEASURABLE turn and a genuine SILENT-EMPTY turn must
    never render the same way. Pre-#1267 both printed a bare `decode_tps=0.0`,
    so a harmless turn and the failure soak exists to catch were byte-identical
    in the log and the reader had to redo `wall - ttft` by hand to tell them
    apart.
    """
    if basis == "engine":
        return ENGINE_SOURCE_LABELS.get(source, "engine-reported: %s" % (source or "engine counter"))
    if basis == "decode":
        return "client-timed: %d ms decode window" % window_ms
    if basis == "wall":
        return "wall-derived, canvas"
    if basis == "empty":
        return "SILENT-EMPTY: 0 completion tokens"
    if basis == "unmeasurable":
        return ("no decode figure: %d ms decode window is under the %d ms client-timing "
                "floor and no engine counter was available" % (window_ms, int(DECODE_FLOOR_MS)))
    return basis


def cmd_engine_counter_probe(endpoint, container=""):
    """Resolve ONCE per run where decode rates come from; print '<kind> <detail>'.

    prom <metric>    — scrape /metrics before + after each turn
    log <container>  — scrape `docker logs` for a decode-rate signature per turn
    none <reason>    — client-side SSE timing, with the DECODE_FLOOR_MS floor
    """
    mode = (os.environ.get("SOAK_ENGINE_COUNTER") or "auto").strip().lower()
    if mode == "off":
        print("none SOAK_ENGINE_COUNTER=off")
        return
    names = sorted(_tpot_series(_read_metrics_text(_metrics_url(endpoint))))
    if names:
        print("prom " + names[0])
        return
    if container and container != "none" and shutil.which("docker"):
        # Matched per turn by signature: a container whose logs carry no
        # decode-rate line just falls back to client timing, labelled as such.
        print("log " + container)
        return
    print("none no decode counter on %s and no container log to read" % _metrics_url(endpoint))


def cmd_run(endpoint, req_path, timeout_s, metrics_path):
    body = pathlib.Path(req_path).read_bytes()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Engine-side decode counter (#1268). The source is resolved ONCE per run by
    # cmd_engine_counter_probe and handed down in the environment; the /metrics
    # snapshot has to be taken BEFORE the request so its delta belongs to this
    # turn and this turn only (valid because soak is single-stream).
    counter_kind = (os.environ.get("SOAK_ENGINE_COUNTER_KIND") or "none").strip().lower()
    if (os.environ.get("SOAK_ENGINE_COUNTER") or "auto").strip().lower() == "off":
        counter_kind = "none"
    counter_metric = (os.environ.get("SOAK_ENGINE_COUNTER_METRIC") or "").strip()
    counter_container = (os.environ.get("SOAK_ENGINE_LOG_CONTAINER") or "").strip()
    metrics_before = _read_metrics_text(_metrics_url(endpoint)) if counter_kind == "prom" else ""
    t0 = time.time()
    ttft = None
    completion_tokens = 0
    status = 0
    error = ""
    # Continuous-mode response capture — accumulate streamed deltas so the
    # next turn can extend the conversation. Fresh mode ignores these fields.
    content_parts = []
    reasoning_parts = []
    tool_calls_acc = {}  # idx → {id, type, name, args}
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            status = getattr(resp, "status", 200)
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk and not error:
                    error = str(chunk["error"])[:240]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    # vLLM emits reasoning under either `delta.reasoning_content`
                    # (older qwen3 reasoner path) or `delta.reasoning` (current
                    # nightly as of vllm-0.20.2rc1+; legacy field name). Watch
                    # both so the soak harness doesn't go silent when the
                    # underlying field name shifts under us.
                    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                    if ttft is None and (delta.get("content") or reasoning_delta or delta.get("tool_calls")):
                        ttft = time.time() - t0
                    # Accumulate streamed parts. vLLM splits content/reasoning
                    # across many small deltas; tool_calls stream as indexed
                    # objects whose fields (name, arguments) arrive in pieces.
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                    for tc in (delta.get("tool_calls") or []):
                        idx = tc.get("index", 0)
                        slot = tool_calls_acc.setdefault(idx, {"id": "", "type": "function", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            error = e.read(500).decode("utf-8", errors="replace")
        except Exception:
            error = str(e)
    except Exception as e:
        status = 0
        error = f"{type(e).__name__}: {e}"

    wall = time.time() - t0
    # Ask the engine for its own decode rate (#1268). A miss is not a failure:
    # the client-side window below is the labelled fallback. Only asked when the
    # turn produced tokens — a zero-token turn is silent-empty, full stop.
    engine_tps = None
    engine_source = ""
    if completion_tokens > 0 and counter_kind == "prom" and metrics_before and counter_metric:
        engine_tps = engine_rate_from_metrics(
            metrics_before, _read_metrics_text(_metrics_url(endpoint)), counter_metric)
        if engine_tps is not None:
            engine_source = "prom-tpot"
    elif completion_tokens > 0 and counter_kind == "log" and counter_container:
        engine_tps, engine_source = engine_rate_from_logs(_docker_logs_since(counter_container, t0))
    # --- decode-rate basis (#809) --------------------------------------------
    # decode TPS = completion_tokens / (wall - ttft) assumes token-by-token
    # autoregressive streaming. Canvas-granularity (block-diffusion) models do
    # not stream that way: they denoise a whole N-token canvas in parallel and
    # the endpoint emits roughly ONE chunk per completed canvas. A response
    # shorter than one canvas therefore arrives as a single chunk, ttft == wall,
    # and the decode window is zero-width. Emitting a hard 0.0 there converts a
    # wrong number into a misleading one — 20 of 25 turns unmeasurable in the
    # #809 soak, and the summary silently described only the multi-canvas
    # subset. So: when the window is unmeasurable but the model DID produce
    # tokens, derive from wall time and label the basis, rather than zeroing.
    #
    # decode_basis, carried into the metrics JSON and turn-log.csv:
    #   engine        the ENGINE's own decode counter answered (#1268). Beats
    #                 client-side timing whenever it is available, because it is
    #                 computed over real decode steps and is therefore immune to
    #                 a decode window too narrow to time. decode_source says
    #                 which counter.
    #   decode        real decode window observed; decode_tps IS a decode rate
    #   wall          window unmeasurable, derived as completion_tokens / wall.
    #                 Equals wall TPS by construction: it INCLUDES prefill and
    #                 must never be presented as a bare decode rate.
    #   empty         completion_tokens == 0 — genuine silent-empty. Keeps the
    #                 0.0 that the silent-empty discriminator depends on; this
    #                 case must not regress (club-3090 #43, #47).
    #   unmeasurable  window unmeasurable on a run NOT classified as canvas and
    #                 no engine counter available. decode_tps stays 0.0 in the
    #                 CSV for column compatibility, but it is NOT a measurement
    #                 and must never be RENDERED as one (#1267) — see
    #                 decode_label(), which is what soak-test.sh prints.
    #
    # The canvas SIGNATURE is ttft ≈ wall, i.e. a zero-width window — NOT merely
    # "narrow". That distinction is load-bearing: a fast autoregressive rig
    # produces genuinely narrow windows (#849 measured 82-83 ms on a dual-NVFP4
    # 5090 pair) which are sub-threshold but nowhere near zero, and those turns
    # must keep behaving exactly as they did. SOAK_CANVAS_WINDOW_MS sets the
    # zero-width bound (default 5 ms); SOAK_DECODE_GRANULARITY forces the
    # classification (canvas | autoregressive | auto, default auto).
    single_chunk = ttft is None
    if single_chunk:
        # No content/reasoning/tool_calls delta was observed before the final
        # chunk. Either a canvas arriving whole, or a thinking-mode response
        # where the engine bundled everything into the terminal chunk.
        ttft = wall
    decode_s = wall - ttft
    try:
        canvas_window_ms = float(os.environ.get("SOAK_CANVAS_WINDOW_MS", "5"))
    except ValueError:
        canvas_window_ms = 5.0
    granularity = (os.environ.get("SOAK_DECODE_GRANULARITY") or "auto").strip().lower()
    if granularity not in ("auto", "canvas", "autoregressive"):
        granularity = "auto"
    canvas_signature = completion_tokens > 0 and (
        single_chunk or decode_s * 1000.0 <= canvas_window_ms
    )
    # Branch order is load-bearing and unchanged from #809 for the client-side
    # cases: a measurable window still beats the canvas branch (a multi-canvas
    # turn IS a real decode measurement), and the canvas branch still claims a
    # zero-width window before "unmeasurable" does. #1268 adds exactly one
    # thing: where the client would have timed the stream, the engine's own
    # counter is preferred. Canvas turns are deliberately NOT handed to the
    # engine counter — an engine's per-output-token accounting is not meaningful
    # for a model that denoises a whole canvas per step.
    canvas_turn = granularity == "canvas" or (granularity == "auto" and canvas_signature)
    measurable_window = decode_s >= DECODE_FLOOR_MS / 1000.0
    if completion_tokens <= 0:
        decode_tps = 0.0
        decode_basis = "empty"
        decode_source = "none"
    elif measurable_window or not canvas_turn:
        if engine_tps is not None:
            decode_tps = engine_tps
            decode_basis = "engine"
            decode_source = engine_source
        elif measurable_window:
            decode_tps = round(completion_tokens / decode_s, 3)
            decode_basis = "decode"
            decode_source = "stream"
        else:
            # Window under the floor, no engine counter: streaming closed before
            # decode steps were observable separately from prefill. NOT a zero.
            decode_tps = 0.0
            decode_basis = "unmeasurable"
            decode_source = "none"
    else:
        decode_tps = round(completion_tokens / wall, 3) if wall > 0 else 0.0
        decode_basis = "wall"
        decode_source = "wall"
    # Reassemble the captured response for continuous-mode ingestion.
    # `tool_calls_response` is in OpenAI tool_calls format, ready to drop
    # into the next turn's assistant message.
    tool_calls_response = []
    for idx in sorted(tool_calls_acc.keys()):
        slot = tool_calls_acc[idx]
        if not slot["name"]:
            continue
        tool_calls_response.append({
            "id": slot["id"] or f"call_synth_{idx}",
            "type": slot["type"],
            "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
        })
    data = {
        "status": int(status),
        "error": error.replace("\n", " ")[:300],
        "t_ms": round(wall * 1000),
        "ttft_ms": round(ttft * 1000),
        "decode_tps": decode_tps,
        "decode_basis": decode_basis,
        "decode_source": decode_source,
        "decode_label": decode_label(decode_basis, decode_source, round(decode_s * 1000)),
        "decode_window_ms": round(decode_s * 1000),
        "completion_tokens": completion_tokens,
        # Continuous-mode capture (ignored in fresh mode):
        "content": "".join(content_parts)[:4000],
        "reasoning_content": "".join(reasoning_parts)[:4000],
        "tool_calls": tool_calls_response,
    }
    pathlib.Path(metrics_path).write_text(json.dumps(data) + "\n")


def cmd_append_log(log_path, session, turn, vram, metrics_path):
    metrics = json.loads(pathlib.Path(metrics_path).read_text())
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                session,
                turn,
                metrics.get("t_ms", 0),
                vram,
                metrics.get("ttft_ms", 0),
                metrics.get("decode_tps", 0),
                metrics.get("completion_tokens", 0),
                metrics.get("status", 0),
                metrics.get("error", ""),
                # decode_basis, then decode_source, appended last so a consumer
                # reading the pre-#809 column order positionally still lines up.
                metrics.get("decode_basis", "decode"),
                metrics.get("decode_source", ""),
            ]
        )


def cmd_metric(metrics_path):
    m = json.loads(pathlib.Path(metrics_path).read_text())
    # Field 5 (err_flag) lets soak-test.sh decide in bash whether a session ran
    # CLEAN — without re-parsing the metrics JSON. The condition mirrors
    # cmd_summary's errors[] exactly: non-200 status OR a stream error payload.
    # Used to pick the session the warm VRAM baseline anchors on (#829).
    err_flag = 1 if (int(m.get("status", 0)) != 200 or m.get("error")) else 0
    # Field 6 (decode_basis) lets soak-test.sh label a wall-derived figure on the
    # per-turn line and make the canvas classification sticky for the run (#809).
    # Fields 7-9 carry the provenance the per-turn line must show (#1267/#1268):
    # the decode window in ms, which source produced the figure, and the label
    # itself. The label is LAST and contains spaces — soak-test.sh reads it with
    # a trailing `read` variable, so the wording lives here, in one place.
    basis = m.get("decode_basis", "decode")
    source = m.get("decode_source", "") or "none"
    window_ms = int(m.get("decode_window_ms", 0) or 0)
    label = m.get("decode_label") or decode_label(basis, source, window_ms)
    print(m.get("status", 0), m.get("t_ms", 0), m.get("ttft_ms", 0), m.get("decode_tps", 0),
          err_flag, basis, window_ms, source, label)


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def med(xs):
    return statistics.median(xs) if xs else 0.0


def cmd_summary(turn_log, summary_path, boot_vram, growth_limit, timed_out,
                expected_sessions, baseline_session="1"):
    boot_vram = int(boot_vram)
    growth_limit = int(growth_limit)
    timed_out = int(timed_out) == 1
    expected_sessions = int(expected_sessions)
    # Which session the warm VRAM baseline was anchored at the END of, or 0 when
    # NO session completed error-free and therefore no warm baseline exists
    # (#829). Defaulted to 1 so older callers / replayed CSVs keep the historical
    # "baseline == end of session 1, compare against every row" behaviour.
    baseline_session = int(baseline_session)
    vram_unmeasurable = baseline_session <= 0
    rows = []
    with open(turn_log) as f:
        reader = csv.DictReader(f)
        # completion_tokens column added 2026-05-04. Its presence selects the
        # silent-empty discriminator below: genuine completion_tokens==0 when
        # available, else the legacy decode_tps==0 proxy for older CSVs.
        has_completion_tokens = "completion_tokens" in (reader.fieldnames or [])
        # decode_basis column added 2026-08-01 (#809). Absent on older CSVs, in
        # which case every row is treated as a real decode measurement — exactly
        # the pre-#809 reading of that data.
        has_decode_basis = "decode_basis" in (reader.fieldnames or [])
        # decode_source column added 2026-09-13 (#1268). Absent on older CSVs.
        has_decode_source = "decode_source" in (reader.fieldnames or [])
        for row in reader:
            for key in ("session_id", "turn_id", "t_ms", "vram_mib", "ttft_ms", "status"):
                row[key] = int(float(row[key] or 0))
            row["decode_tps"] = float(row["decode_tps"] or 0)
            # completion_tokens is new (added 2026-05-04) — back-compat for old CSVs
            row["completion_tokens"] = int(float(row.get("completion_tokens", 0) or 0))
            row["decode_basis"] = (row.get("decode_basis") or "decode") if has_decode_basis else "decode"
            row["decode_source"] = (row.get("decode_source") or "") if has_decode_source else ""
            rows.append(row)

    sessions = sorted({r["session_id"] for r in rows})
    first = sessions[:5]
    last = sessions[-5:]
    # Filter unrealistic TPS values (>500 t/s) — these come from streaming
    # responses where ttft ≈ wall (no separate decode time observable),
    # yielding a divide-by-tiny artifact. The cmd_run path now guards this
    # for fresh runs but we filter defensively in case of future regressions
    # or data from older runs that pre-date the fix.
    def realistic(t):
        return 0 < t <= 500
    # Wall-derived (canvas) turns are kept OUT of every decode statistic (#809):
    # a wall-derived figure includes prefill, so averaging it with real decode
    # rates would silently redefine what p50/p95/retention mean. They get their
    # own labelled series below. On an autoregressive run `derived` is empty and
    # every pool here is identical to the pre-#809 pools.
    derived = [r for r in rows if r["decode_basis"] == "wall"]
    measured_rows = [r for r in rows if r["decode_basis"] != "wall"]
    # Basis pools for the denominator the summary owes the reader (#1267). An
    # `engine` row IS a decode rate, so it belongs in the decode series with the
    # client-timed ones; an `unmeasurable` row carries no rate at all and is
    # dropped by realistic() below — which is numerically right and was, until
    # now, invisible: p50 described an unstated subset with no denominator shown.
    engine_rows = [r for r in rows if r["decode_basis"] == "engine"]
    stream_rows = [r for r in rows if r["decode_basis"] == "decode"]
    unmeasurable_rows = [r for r in rows if r["decode_basis"] == "unmeasurable"]
    empty_rows = [r for r in rows if r["decode_basis"] == "empty"]
    engine_sources = sorted({r["decode_source"] for r in engine_rows if r["decode_source"]})
    tps = [r["decode_tps"] for r in measured_rows if realistic(r["decode_tps"])]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] > 0]
    first_tps = [r["decode_tps"] for r in measured_rows if r["session_id"] in first and realistic(r["decode_tps"])]
    last_tps = [r["decode_tps"] for r in measured_rows if r["session_id"] in last and realistic(r["decode_tps"])]
    dtps = [r["decode_tps"] for r in derived if realistic(r["decode_tps"])]
    first_dtps = [r["decode_tps"] for r in derived if r["session_id"] in first and realistic(r["decode_tps"])]
    last_dtps = [r["decode_tps"] for r in derived if r["session_id"] in last and realistic(r["decode_tps"])]
    first_ttft = [r["ttft_ms"] for r in rows if r["session_id"] in first and r["ttft_ms"] > 0]
    last_ttft = [r["ttft_ms"] for r in rows if r["session_id"] in last and r["ttft_ms"] > 0]
    # VRAM accretion is only meaningful FROM the warm baseline forward, and the
    # baseline is only meaningful if the session it was taken at the end of ran
    # clean (#829). Two consequences, both of which the pre-#829 code got wrong
    # when session 1 died:
    #   1. Rows recorded BEFORE the baseline was captured must not feed the peak.
    #      They pre-date the comparison point, so a pre-baseline reading can only
    #      manufacture growth that was never accretion.
    #   2. If NO session ran clean there is no warm baseline at all — sampling
    #      nvidia-smi anyway reads a dying/dead engine and every later comparison
    #      lands multi-GB high. #827: a 31823 MiB corpse baseline against the
    #      healthy 63876 MiB peak printed "VRAM grew 32053 MiB" + "oscillation
    #      63866 MiB" — the loudest FAIL in the report, pointing at a memory leak
    #      that did not exist, on a run whose actual fault was an engine crash.
    vram_rows = [] if vram_unmeasurable else [r for r in rows if r["session_id"] >= baseline_session]
    vram_sessions = sorted({r["session_id"] for r in vram_rows})
    if vram_unmeasurable:
        max_vram = 0
        growth = 0
    else:
        max_vram = max([r["vram_mib"] for r in vram_rows] + [boot_vram])
        growth = max_vram - boot_vram
    errors = [r for r in rows if r["status"] != 200 or r["error"]]
    # Silent-empty turns: HTTP 200 + no transport error + the model produced
    # NO observable output despite t_ms ≥ 1s. These slip past errors[] — the
    # engine ACK'd the request and the stream closed cleanly, but nothing came
    # back. The discriminator is completion_tokens == 0 (genuine empty) when
    # that column is present — NOT decode_tps == 0. cmd_run zeroes decode_tps
    # when decode_s < 0.1s OR completion_tokens <= 0, so the old decode_tps==0
    # proxy ALSO fired on turns that DID produce output but decoded it in a
    # sub-100ms burst: a tool-call turn (small tool_calls payload, empty
    # `content`) or a block-diffusion canvas emitted all at once. That
    # false-flagged real output as silent-empty (DiffusionGemma re-soak
    # 2026-06-11: 4/25 tool-call turns mislabelled). Genuine causes:
    # xgrammar mask rejecting every candidate (club-3090 #43, #47),
    # client-side max_tokens exhausted by the <think> block, or spec-decode
    # returning an empty draft batch. Pre-2026-05-04 CSVs lack the
    # completion_tokens column → fall back to the decode_tps==0 heuristic.
    def _is_silent_empty(r):
        if r["status"] != 200 or r["error"] or r["t_ms"] < 1000:
            return False
        if has_completion_tokens:
            return r["completion_tokens"] == 0
        return r["decode_tps"] == 0
    silent_empty = [r for r in rows if _is_silent_empty(r)]
    silent_empty_pct = (100.0 * len(silent_empty) / len(rows)) if rows else 0.0
    first_med = med(first_tps)
    last_med = med(last_tps)
    tps_retention = last_med / first_med if first_med > 0 else 0.0
    d_first_med = med(first_dtps)
    d_last_med = med(last_dtps)
    d_retention = d_last_med / d_first_med if d_first_med > 0 else 0.0
    ttft_ratio = med(last_ttft) / med(first_ttft) if med(first_ttft) > 0 else 0.0
    session_max = [max(r["vram_mib"] for r in vram_rows if r["session_id"] == s) for s in vram_sessions]
    oscillation = max([abs(b - a) for a, b in zip(session_max, session_max[1:])] or [0])
    slow_turns = [r for r in rows if r["t_ms"] > 30000]

    warnings = []
    failures = []
    if errors:
        failures.append(f"{len(errors)} request(s) returned non-200 status or stream error.")
    if vram_unmeasurable:
        # Loud, and deliberately NOT a number (#829). A numeric growth figure
        # here would be an artifact of baselining a dead engine, and it is the
        # single most misdirecting line a soak report can carry.
        warnings.append(
            f"VRAM growth + oscillation: UNMEASURABLE — no session completed without errors "
            f"({len(errors)} errored turn(s)), so no warm baseline could be anchored. "
            f"These metrics are NOT reported rather than reported wrong; fix the errors "
            f"above and re-run before reading anything into VRAM."
        )
    elif growth > growth_limit:
        failures.append(f"VRAM grew {growth} MiB > {growth_limit} MiB threshold.")
    if first_med > 0 and tps_retention < 0.80:
        failures.append(f"Decode TPS retention was {tps_retention * 100:.1f}% < 80%.")
    elif first_med == 0 and d_first_med > 0:
        # Every measurable turn was canvas-derived (#809). Evaluate retention on
        # the wall-derived series rather than reporting "no samples" — but say
        # which series it is, because wall TPS includes prefill.
        if d_retention < 0.80:
            failures.append(
                f"Wall-derived (canvas) TPS retention was {d_retention * 100:.1f}% < 80%. "
                f"No turn had a measurable decode window, so this is a wall-time series."
            )
    elif first_med == 0 and rows:
        warnings.append("No positive decode TPS samples; retention could not be evaluated.")
    if ttft_ratio > 1.5:
        warnings.append(f"TTFT grew {ttft_ratio:.2f}x from first sessions to last sessions.")
    if slow_turns:
        warnings.append(f"{len(slow_turns)} turn(s) exceeded 30s.")
    if not vram_unmeasurable and oscillation > 500:
        warnings.append(f"VRAM session-to-session oscillation reached {oscillation} MiB.")
    if sessions and sessions[-1] < expected_sessions:
        warnings.append(f"Only {sessions[-1]} of {expected_sessions} sessions completed.")
    # Silent-empty handling — separate from errors[] because HTTP 200 was
    # returned. ≥50% silent-empty = workload broken (FAIL); 1-49% = WARN.
    if silent_empty:
        msg = (
            f"{len(silent_empty)} of {len(rows)} turn(s) "
            f"({silent_empty_pct:.1f}%) returned HTTP 200 with empty completion "
            f"(model thought ≥1s, then emitted zero tokens). Common causes: "
            f"xgrammar mask rejection (club-3090 #43, #47), client max_tokens "
            f"exhausted by <think>, or spec-decode empty-draft return."
        )
        if silent_empty_pct >= 50.0:
            failures.append(msg)
        else:
            warnings.append(msg)

    # A run with no warm baseline can never be a PASS: the VRAM-accretion class
    # this test exists to detect went unmeasured (#829).
    verdict = "INCONCLUSIVE" if timed_out else (
        "FAIL" if failures else ("INCONCLUSIVE" if vram_unmeasurable else "PASS"))
    exit_code = 2 if timed_out else (1 if failures else (2 if vram_unmeasurable else 0))
    if vram_unmeasurable:
        vram_lines = [
            "- Boot VRAM baseline: **UNMEASURABLE** — no error-free session to anchor on",
            "- Max VRAM observed: n/a (no warm baseline to compare against)",
            "- Max growth observed: **INCONCLUSIVE** — see Warnings",
        ]
    else:
        vram_lines = [
            f"- Boot VRAM baseline: {boot_vram} MiB",
            f"- Max VRAM observed: {max_vram} MiB",
            f"- Max growth observed: {growth} MiB",
        ]
        if baseline_session > 1:
            vram_lines.append(
                f"- Warm baseline anchored at END of session {baseline_session} "
                f"(sessions 1-{baseline_session - 1} had errored turns; their rows are "
                f"excluded from growth + oscillation)"
            )
    # Decode-window basis block. Pre-#1267 this was gated on `derived`, i.e. it
    # appeared ONLY when a canvas turn existed — so an autoregressive run with
    # unmeasurable turns reported its p50 over an unstated subset and said
    # nothing about the turns it had dropped. It now fires whenever the run is
    # not uniformly client-timed decode turns, which keeps a plain healthy run's
    # summary byte-identical while giving every other run its denominator.
    basis_seg = []
    if engine_rows:
        basis_seg.append(
            f"{len(engine_rows)} engine-reported"
            f" ({', '.join(engine_sources) if engine_sources else 'engine counter'})")
    if stream_rows:
        basis_seg.append(f"{len(stream_rows)} client-timed")
    if unmeasurable_rows:
        basis_seg.append(
            f"{len(unmeasurable_rows)} unmeasurable (decode window under the "
            f"{int(DECODE_FLOOR_MS)} ms client-timing floor, no engine counter)")
    if derived:
        basis_seg.append(f"{len(derived)} wall-derived (canvas)")
    if empty_rows:
        basis_seg.append(f"{len(empty_rows)} silent-empty")
    basis_lines = []
    basis_rows = []
    if derived or engine_rows or unmeasurable_rows:
        basis_text = (
            f"- Decode-window basis: {' / '.join(basis_seg)} of {len(rows)} turn(s). "
            f"The decode percentiles and retention below are computed over the {len(tps)} "
            f"turn(s) carrying a decode figure — that is the denominator; turns without one "
            f"are excluded rather than counted as zero."
        )
        if derived:
            basis_text += (
                " A wall-derived turn arrived in a single chunk (canvas granularity), so its "
                "decode window is zero-width and the figure is completion_tokens / wall — it "
                "INCLUDES prefill and is not a decode rate."
            )
        basis_lines = [basis_text]
    if derived:
        basis_rows = [
            f"| p50 wall-derived TPS (canvas) | {percentile(dtps, 0.50):.2f} |",
            f"| p95 wall-derived TPS (canvas) | {percentile(dtps, 0.95):.2f} |",
            f"| wall-derived (canvas) turns | {len(derived)} / {len(rows)} |",
        ]
        if d_first_med > 0:
            basis_rows.append(f"| wall-derived TPS retention (canvas) | {d_retention * 100:.1f}% |")
    lines = [
        "# Soak test summary",
        "",
        f"- Verdict: **{verdict}**",
        *vram_lines,
        f"- Sessions completed: {len(sessions)}",
        f"- Request errors: {len(errors)}",
        f"- Silent-empty turns (HTTP 200 + 0 completion tokens): {len(silent_empty)} / {len(rows)} ({silent_empty_pct:.1f}%)",
        *basis_lines,
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| p50 decode TPS | {percentile(tps, 0.50):.2f} |",
        f"| p95 decode TPS | {percentile(tps, 0.95):.2f} |",
        f"| first-5 median TPS | {first_med:.2f} |",
        f"| last-5 median TPS | {last_med:.2f} |",
        f"| TPS retention | {tps_retention * 100:.1f}% |",
        f"| p50 TTFT | {percentile(ttft, 0.50):.0f} ms |",
        f"| p95 TTFT | {percentile(ttft, 0.95):.0f} ms |",
        f"| TTFT first/last ratio | {ttft_ratio:.2f}x |",
        (f"| VRAM oscillation | INCONCLUSIVE |" if vram_unmeasurable
         else f"| VRAM oscillation | {oscillation} MiB |"),
        *basis_rows,
        "",
    ]
    if failures:
        lines += ["## Failures", "", *[f"- {x}" for x in failures], ""]
    if warnings:
        lines += ["## Warnings", "", *[f"- {x}" for x in warnings], ""]
    if errors[:10]:
        lines += ["## First request errors", ""]
        lines += [f"- session {r['session_id']} turn {r['turn_id']}: status={r['status']} error={r['error'][:160]}" for r in errors[:10]]
        lines += [""]
    rec = "Runtime VRAM growth and throughput retention stayed within v1 soak thresholds."
    if verdict == "FAIL":
        rec = "Inspect docker logs and compare turn-log.csv against GPU snapshots to identify the accreting path."
    elif verdict == "INCONCLUSIVE":
        rec = "Re-run with a larger SOAK_TIMEOUT_S or fewer/lighter sessions before treating this config as soak-clean."
    if vram_unmeasurable:
        rec = ("Every session errored, so VRAM accretion was never measured. Diagnose the "
               "request errors first (`docker logs <container> 2>&1 | tail -50`) — a mid-run "
               "engine death is the common cause — then re-run the soak.")
    lines += ["## Recommendation", "", f"- {rec}"]
    pathlib.Path(summary_path).write_text("\n".join(lines) + "\n")

    print("")
    print("[soak] summary")
    print(f"[soak]   verdict              {verdict}")
    if vram_unmeasurable:
        print(f"[soak]   boot_vram_mib        UNMEASURABLE (no error-free session)")
        print(f"[soak]   max_vram_mib         n/a")
        print(f"[soak]   max_growth_mib       INCONCLUSIVE / {growth_limit}")
    else:
        print(f"[soak]   boot_vram_mib        {boot_vram}")
        print(f"[soak]   max_vram_mib         {max_vram}")
        print(f"[soak]   max_growth_mib       {growth} / {growth_limit}")
        if baseline_session > 1:
            print(f"[soak]   baseline_session     {baseline_session} (sessions 1-{baseline_session - 1} errored; excluded)")
    print(f"[soak]   errors               {len(errors)}")
    print(f"[soak]   silent_empty         {len(silent_empty)} / {len(rows)} ({silent_empty_pct:.1f}%)")
    print(f"[soak]   p50_decode_tps       {percentile(tps, 0.50):.2f}")
    if basis_lines:
        print(f"[soak]   decode_basis         {' / '.join(basis_seg)} of {len(rows)}")
    if derived:
        print(f"[soak]   p50_wall_tps_canvas  {percentile(dtps, 0.50):.2f}  (includes prefill — NOT a decode rate)")
    print(f"[soak]   p95_ttft_ms          {percentile(ttft, 0.95):.0f}")
    print(f"[soak]   tps_retention        {tps_retention * 100:.1f}%")
    if derived and d_first_med > 0:
        print(f"[soak]   wall_tps_retention   {d_retention * 100:.1f}%  (canvas series)")
    for label, items in (("failures", failures), ("warnings", warnings)):
        if items:
            print(f"[soak] {label}:")
            for item in items:
                print(f"[soak]   - {item}")
    if verdict == "PASS":
        print("[soak]   note                 PASS = no failure signal on this sample;")
        print("[soak]                        not patch validation (topology alone can")
        print("[soak]                        sidestep what overlays target). See")
        print("[soak]                        scripts/soak-test.sh --help and docs/CLIFFS.md.")
    sys.exit(exit_code)


def main():
    cmd = sys.argv[1]
    args = sys.argv[2:]
    {
        "model": cmd_model,
        "baseline": cmd_baseline,
        "request": cmd_request,
        "init-session": cmd_init_session,
        "request-continuous": cmd_request_continuous,
        "ingest": cmd_ingest,
        "run": cmd_run,
        "append-log": cmd_append_log,
        "metric": cmd_metric,
        "engine-counter-probe": cmd_engine_counter_probe,
        "summary": cmd_summary,
    }[cmd](*args)


if __name__ == "__main__":
    main()
