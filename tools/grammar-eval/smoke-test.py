#!/usr/bin/env python3
"""Smoke-test Holiday tagline grammar on the bounded-thinking vLLM stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GRAMMAR = REPO_ROOT / "tools/grammar-eval/holiday-tagline.gbnf"
STRUCTURED_COT_DIR = pathlib.Path("/home/wasif/structured-cot")

HOLIDAY_SYSTEM = (
    "You are an expert Python programmer. Think only inside the required "
    "tagline fields. Q is intent, M is method, K and R are comma-separated "
    "keywords with no spaces, V is status. After </think>, write the final "
    "answer or code directly."
)
FREE_SYSTEM = (
    "You are an expert Python programmer. Think carefully in your <think> "
    "block, then write the final answer or code."
)

BODY_RE = re.compile(
    r"^Q=(solve|prove|route|debug|patch|code|calc|compare|explain)\n"
    r"M=(case|enum|check|derive|edit|test|trace|rank)\n"
    r"K=[A-Za-z][A-Za-z0-9_.!-]{0,18}(,[A-Za-z][A-Za-z0-9_.!-]{0,18}){0,4}\n"
    r"R=[A-Za-z][A-Za-z0-9_.!-]{0,18}(,[A-Za-z][A-Za-z0-9_.!-]{0,18}){0,4}\n"
    r"V=(ok|fail|done|blocked|candidate|verify)$"
)


def api_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read(1000).decode("utf-8", errors="replace")
        return e.code, body
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def get_json(url: str, timeout: float) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read(1000).decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def wait_models(base_url: str, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    models_url = api_url(base_url, "models")
    last = ""
    while time.time() < deadline:
        status, body = get_json(models_url, 10)
        if status == 200 and isinstance(body, dict) and body.get("data"):
            return body["data"][0].get("id", "qwen3.6-27b-autoround")
        last = str(body)[:200]
        time.sleep(5)
    raise SystemExit(f"[grammar-smoke] endpoint did not become ready: {last}")


def load_humaneval_prompts(task_nums: list[int]) -> list[tuple[str, str]]:
    script = STRUCTURED_COT_DIR / "fsm_vs_free_eval.py"
    if not script.exists():
        return []
    try:
        spec = importlib.util.spec_from_file_location("fsm_vs_free_eval", script)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rows = mod.load_benchmark("humaneval", 164)
        by_id = {r["task_id"]: r for r in rows}
        prompts = []
        for n in task_nums:
            task_id = f"HumanEval/{n}"
            if task_id in by_id:
                prompts.append((f"HE/{n}", mod.build_user_prompt(by_id[task_id], "humaneval")))
        return prompts
    except Exception as e:
        print(f"[grammar-smoke] warning: could not load HE+ prompts: {e}", file=sys.stderr)
        return []


def prompts() -> list[tuple[str, str]]:
    out = [
        ("simple-math", "What is 17 * 19? Answer briefly."),
        ("simple-code", "Write a Python function square(x) that returns x * x."),
        ("simple-debug", "Explain the bug in: for i in range(len(xs)): xs.pop(i)"),
    ]
    out.extend(load_humaneval_prompts([97, 137]))
    if len(out) < 5:
        out.extend(
            [
                ("fallback-hard-1", "Write a function that returns the longest palindromic substring."),
                ("fallback-hard-2", "Patch a binary search implementation that loops forever on two elements."),
            ][: 5 - len(out)]
        )
    return out[:5]


def message_parts(body: dict) -> tuple[str, str]:
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not reasoning and "<think>" in content:
        match = re.search(r"<think>\n?(.*?)\n?</think>", content, re.S)
        if match:
            reasoning = match.group(1)
    reasoning = re.sub(r"^<think>\n?", "", str(reasoning))
    reasoning = re.sub(r"\n?</think>.*$", "", reasoning, flags=re.S).strip()
    return reasoning, content


def run_one(base_url: str, model: str, grammar: str, label: str, prompt: str, use_grammar: bool, max_tokens: int, timeout: float) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": HOLIDAY_SYSTEM if use_grammar else FREE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if use_grammar:
        payload["structured_outputs"] = {"grammar": grammar}
    status, body = post_json(api_url(base_url, "chat/completions"), payload, timeout)
    if status != 200 or not isinstance(body, dict):
        return False, f"{label}: HTTP {status}: {str(body)[:220]}"
    reasoning, content = message_parts(body)
    if not reasoning:
        return False, f"{label}: missing reasoning_content"
    if use_grammar and not BODY_RE.match(reasoning):
        return False, f"{label}: grammar shape mismatch: {reasoning[:180]!r}"
    if not content:
        return False, f"{label}: empty answer content"
    return True, f"{label}: ok"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8020/v1")
    p.add_argument("--model", default="", help="Default: first /v1/models id.")
    p.add_argument("--grammar-file", default=str(DEFAULT_GRAMMAR))
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--request-timeout", type=float, default=300)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--boot", action="store_true", help="Run scripts/switch.sh vllm/bounded-thinking-andthattoo first.")
    args = p.parse_args()

    if args.boot:
        print("[grammar-smoke] booting vllm/bounded-thinking-andthattoo")
        subprocess.run(["bash", "scripts/switch.sh", "vllm/bounded-thinking-andthattoo"], cwd=REPO_ROOT, check=True)

    model = args.model or wait_models(args.base_url, args.ready_timeout)
    grammar = pathlib.Path(args.grammar_file).read_text()
    failures = []
    print(f"[grammar-smoke] endpoint={args.base_url} model={model}")

    for label, prompt in prompts():
        for mode, use_grammar in (("tagline", True), ("free", False)):
            ok, msg = run_one(args.base_url, model, grammar, f"{label}/{mode}", prompt, use_grammar, args.max_tokens, args.request_timeout)
            print(f"[grammar-smoke] {'PASS' if ok else 'FAIL'} {msg}")
            if not ok:
                failures.append(msg)

    if failures:
        print("[grammar-smoke] failures:")
        for failure in failures:
            print(f"[grammar-smoke]   - {failure}")
        return 1
    print("[grammar-smoke] all requests returned 200 and tagline reasoning matched shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
