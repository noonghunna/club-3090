#!/usr/bin/env python3
import csv
import json
import pathlib
import statistics
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
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
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
    print(models[0].get("id", "qwen3.6-27b-autoround") if models else "qwen3.6-27b-autoround")


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


def cmd_run(endpoint, req_path, timeout_s, metrics_path):
    body = pathlib.Path(req_path).read_bytes()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
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
                    if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                        ttft = time.time() - t0
                    # Accumulate streamed parts. vLLM splits content/reasoning
                    # across many small deltas; tool_calls stream as indexed
                    # objects whose fields (name, arguments) arrive in pieces.
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
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
    if ttft is None:
        # No streaming content/reasoning/tool_calls delta was observed before
        # the final chunk arrived. Common with thinking-mode responses where
        # vLLM occasionally bundles the reasoning into the terminal chunk.
        # We can't compute a meaningful TTFT or decode rate in that case —
        # report TTFT as wall and decode_tps as 0 (signals "couldn't measure"
        # without producing the spurious 2-billion-tps artifact).
        ttft = wall
        decode_tps = 0.0
    else:
        decode_s = wall - ttft
        if decode_s < 0.1 or completion_tokens <= 0:
            # decode_s < 100ms means streaming closed before any decode steps
            # were observable separately from prefill — same not-measurable
            # case as ttft=None above.
            decode_tps = 0.0
        else:
            decode_tps = round(completion_tokens / decode_s, 3)
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
                metrics.get("status", 0),
                metrics.get("error", ""),
            ]
        )


def cmd_metric(metrics_path):
    m = json.loads(pathlib.Path(metrics_path).read_text())
    print(m.get("status", 0), m.get("t_ms", 0), m.get("ttft_ms", 0), m.get("decode_tps", 0))


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


def cmd_summary(turn_log, summary_path, boot_vram, growth_limit, timed_out, expected_sessions):
    boot_vram = int(boot_vram)
    growth_limit = int(growth_limit)
    timed_out = int(timed_out) == 1
    expected_sessions = int(expected_sessions)
    rows = []
    with open(turn_log) as f:
        for row in csv.DictReader(f):
            for key in ("session_id", "turn_id", "t_ms", "vram_mib", "ttft_ms", "status"):
                row[key] = int(float(row[key] or 0))
            row["decode_tps"] = float(row["decode_tps"] or 0)
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
    tps = [r["decode_tps"] for r in rows if realistic(r["decode_tps"])]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] > 0]
    first_tps = [r["decode_tps"] for r in rows if r["session_id"] in first and realistic(r["decode_tps"])]
    last_tps = [r["decode_tps"] for r in rows if r["session_id"] in last and realistic(r["decode_tps"])]
    first_ttft = [r["ttft_ms"] for r in rows if r["session_id"] in first and r["ttft_ms"] > 0]
    last_ttft = [r["ttft_ms"] for r in rows if r["session_id"] in last and r["ttft_ms"] > 0]
    max_vram = max([r["vram_mib"] for r in rows] + [boot_vram])
    growth = max_vram - boot_vram
    errors = [r for r in rows if r["status"] != 200 or r["error"]]
    first_med = med(first_tps)
    last_med = med(last_tps)
    tps_retention = last_med / first_med if first_med > 0 else 0.0
    ttft_ratio = med(last_ttft) / med(first_ttft) if med(first_ttft) > 0 else 0.0
    session_max = [max(r["vram_mib"] for r in rows if r["session_id"] == s) for s in sessions]
    oscillation = max([abs(b - a) for a, b in zip(session_max, session_max[1:])] or [0])
    slow_turns = [r for r in rows if r["t_ms"] > 30000]

    warnings = []
    failures = []
    if errors:
        failures.append(f"{len(errors)} request(s) returned non-200 status or stream error.")
    if growth > growth_limit:
        failures.append(f"VRAM grew {growth} MiB > {growth_limit} MiB threshold.")
    if first_med > 0 and tps_retention < 0.80:
        failures.append(f"Decode TPS retention was {tps_retention * 100:.1f}% < 80%.")
    elif first_med == 0 and rows:
        warnings.append("No positive decode TPS samples; retention could not be evaluated.")
    if ttft_ratio > 1.5:
        warnings.append(f"TTFT grew {ttft_ratio:.2f}x from first sessions to last sessions.")
    if slow_turns:
        warnings.append(f"{len(slow_turns)} turn(s) exceeded 30s.")
    if oscillation > 500:
        warnings.append(f"VRAM session-to-session oscillation reached {oscillation} MiB.")
    if sessions and sessions[-1] < expected_sessions:
        warnings.append(f"Only {sessions[-1]} of {expected_sessions} sessions completed.")

    verdict = "INCONCLUSIVE" if timed_out else ("FAIL" if failures else "PASS")
    exit_code = 2 if timed_out else (1 if failures else 0)
    lines = [
        "# Soak test summary",
        "",
        f"- Verdict: **{verdict}**",
        f"- Boot VRAM baseline: {boot_vram} MiB",
        f"- Max VRAM observed: {max_vram} MiB",
        f"- Max growth observed: {growth} MiB",
        f"- Sessions completed: {len(sessions)}",
        f"- Request errors: {len(errors)}",
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
        f"| VRAM oscillation | {oscillation} MiB |",
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
    lines += ["## Recommendation", "", f"- {rec}"]
    pathlib.Path(summary_path).write_text("\n".join(lines) + "\n")

    print("")
    print("[soak] summary")
    print(f"[soak]   verdict              {verdict}")
    print(f"[soak]   boot_vram_mib        {boot_vram}")
    print(f"[soak]   max_vram_mib         {max_vram}")
    print(f"[soak]   max_growth_mib       {growth} / {growth_limit}")
    print(f"[soak]   errors               {len(errors)}")
    print(f"[soak]   p50_decode_tps       {percentile(tps, 0.50):.2f}")
    print(f"[soak]   p95_ttft_ms          {percentile(ttft, 0.95):.0f}")
    print(f"[soak]   tps_retention        {tps_retention * 100:.1f}%")
    for label, items in (("failures", failures), ("warnings", warnings)):
        if items:
            print(f"[soak] {label}:")
            for item in items:
                print(f"[soak]   - {item}")
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
        "summary": cmd_summary,
    }[cmd](*args)


if __name__ == "__main__":
    main()
