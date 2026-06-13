#!/usr/bin/env python3
"""Streaming tool-call probe — the streaming validation benchlocal-cli can't do.

benchlocal-cli runs every pack NON-streaming (see benchlocal-cli#68), so it is
blind to streaming-only tool-call regressions — the vLLM Qwen3 family
(vllm#39056 / club-3090#145): the tool-call XML is emitted/parsed fine
non-streaming, but the STREAMING extractor drops it at the </think>-><tool_call>
boundary (finish_reason:stop, XML leaks into delta.content, no delta.tool_calls).
That bug is invisible to the 8-pack and only shows up over `stream:true`.

This probe sends tool-requiring prompts with `stream:true` (thinking-on by
default — the #145 trigger), reassembles the SSE deltas, and classifies each
response:

  PASS   tool-call streamed correctly: delta.tool_calls accumulated a name,
         finish_reason==tool_calls, valid JSON args, no <tool_call> leaked into
         delta.content.
  DROP   the #145 signature: no streamed tool-call AND (finish_reason==stop OR
         <tool_call> markup leaked into content). This is the regression.
  OTHER  anything else (e.g. model declined, or tool-call present but
         finish_reason!=tool_calls) — surfaced but not counted as a hard fail.

Exit non-zero if any DROP is seen (CI-friendly). Run it against the candidate
AND the control engine and compare DROP rates — that's the streaming A/B.

Usage:
  python3 scripts/stream-toolcall-probe.py --url http://localhost:8010 \
      --model qwen3.6-27b-autoround --thinking on --tool-choice required --repeat 3

  # A/B: run twice (control vs candidate) and diff the DROP counts.
"""
import argparse
import json
import sys
import urllib.request

# Tools the prompts can call. Standard JSON-args tools (equivalent across
# qwen3_coder / qwen3_xml per club-3090#145).
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

# Single-turn prompts that should each trigger exactly one tool call. With
# thinking on, the model reasons in <think>…</think> first, then emits the
# call — exercising the </think>-><tool_call> boundary (#145).
SINGLE_TURN = [
    "Read the file /etc/hosts and summarize it.",
    "What's in /var/log/syslog? Read it.",
    "Run `df -h` and tell me the root filesystem usage.",
    "Check the current directory contents by running `ls -la`.",
    "Search the web for the latest Qwen3 release notes.",
    "Look up who won the 2022 FIFA World Cup.",
    "Compute 17 * 23 + 100 using the calculator.",
    "What is 2 to the power of 16? Use the calculator tool.",
]

# Multi-turn: turn 1 gets a tool result, turn 2 must call again — exercises the
# boundary mid-conversation (where #145's intermittent turn-N failures appeared).
MULTI_TURN = [
    [
        {"role": "user", "content": "Read /etc/os-release."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/etc/os-release"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "NAME=\"Ubuntu\"\nVERSION=\"24.04\""},
        {"role": "user", "content": "Now read /etc/hostname too."},
    ],
    [
        {"role": "user", "content": "Run `uname -r`."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_shell", "arguments": '{"command": "uname -r"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "6.8.0-124-generic"},
        {"role": "user", "content": "Good. Now run `nproc` to count CPUs."},
    ],
]


def stream_request(url, model, messages, tool_choice, thinking, temperature, max_tokens, timeout):
    body = {
        "model": model, "messages": messages, "tools": TOOLS,
        "tool_choice": tool_choice, "stream": True,
        "temperature": temperature, "top_p": 0.95, "top_k": 20,
        "max_tokens": max_tokens,
    }
    if thinking:
        # vLLM Qwen3 thinking gate.
        body["chat_template_kwargs"] = {"enable_thinking": True}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    content = ""
    tool_calls = {}   # index -> {"name": str, "arguments": str}
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return content, tool_calls, finish


def classify(content, tool_calls, finish):
    named = [tc for tc in tool_calls.values() if tc["name"]]
    got_call = bool(named)
    leaked = ("<tool_call>" in content) or ("</tool_call>" in content)
    args_ok = True
    for tc in named:
        try:
            json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError:
            args_ok = False
    if got_call and finish == "tool_calls" and not leaked and args_ok:
        return "PASS", ""
    # #145 signature: tool-call dropped on the streaming path.
    if not got_call and (leaked or finish == "stop"):
        why = "no delta.tool_calls; " + ("XML leaked into content" if leaked else f"finish={finish}")
        return "DROP", why
    bits = []
    if not got_call:
        bits.append("no tool-call")
    if got_call and finish != "tool_calls":
        bits.append(f"finish={finish}")
    if leaked:
        bits.append("XML leaked")
    if not args_ok:
        bits.append("bad JSON args")
    return "OTHER", "; ".join(bits) or f"finish={finish}"


def main():
    ap = argparse.ArgumentParser(description="Streaming tool-call probe (#145 / vllm#39056 class).")
    ap.add_argument("--url", default="http://localhost:8010", help="OpenAI-compatible base URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="enable_thinking (default on — the #145 trigger)")
    ap.add_argument("--tool-choice", choices=["required", "auto", "both"], default="required")
    ap.add_argument("--repeat", type=int, default=3, help="repeats per prompt (default 3; #145 is intermittent)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--label", default="", help="optional label for the report header")
    args = ap.parse_args()

    choices = ["required", "auto"] if args.tool_choice == "both" else [args.tool_choice]
    thinking = args.thinking == "on"

    cases = []  # (name, messages)
    for i, p in enumerate(SINGLE_TURN):
        cases.append((f"S{i+1}", [{"role": "user", "content": p}]))
    for i, conv in enumerate(MULTI_TURN):
        cases.append((f"M{i+1}", conv))

    hdr = f"streaming tool-call probe :: {args.label or args.url} :: model={args.model} thinking={args.thinking}"
    print(hdr)
    print(f"  {len(cases)} prompts x {len(choices)} tool-choice x {args.repeat} repeats "
          f"= {len(cases)*len(choices)*args.repeat} streamed requests\n")

    counts = {"PASS": 0, "DROP": 0, "OTHER": 0, "ERROR": 0}
    drops = []
    for tc_mode in choices:
        for name, messages in cases:
            for r in range(args.repeat):
                tag = f"{name}/{tc_mode}#{r+1}"
                try:
                    content, calls, finish = stream_request(
                        args.url, args.model, messages, tc_mode, thinking,
                        args.temperature, args.max_tokens, args.timeout)
                    verdict, why = classify(content, calls, finish)
                except Exception as exc:  # noqa: BLE001 — surface any transport error
                    verdict, why = "ERROR", f"{type(exc).__name__}: {exc}"
                counts[verdict] = counts.get(verdict, 0) + 1
                mark = {"PASS": "✓", "DROP": "✗ DROP", "OTHER": "·", "ERROR": "‼ ERR"}[verdict]
                if verdict in ("DROP", "ERROR"):
                    drops.append(f"{tag}: {why}")
                if verdict != "PASS":
                    print(f"  {mark} {tag}  {why}")

    total = sum(counts.values())
    print(f"\nsummary [{args.label or args.url}]: "
          f"PASS {counts['PASS']}/{total} · DROP {counts['DROP']} · OTHER {counts['OTHER']} · ERROR {counts['ERROR']}")
    if counts["DROP"]:
        print("  ✗ STREAMING TOOL-CALL DROPS (the #145 signature):")
        for d in drops:
            print(f"    - {d}")
    # Hard fail only on DROP (the regression we gate on). OTHER/ERROR are surfaced.
    sys.exit(1 if counts["DROP"] else 0)


if __name__ == "__main__":
    main()
