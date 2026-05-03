#!/usr/bin/env python3
"""30-prompt HE+ grammar A/B bench: FREE vs current FSM vs Holiday tagline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import random
import re
import statistics
import sys
import time
from types import SimpleNamespace


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURED_COT_DIR = pathlib.Path("/home/wasif/structured-cot")
DEFAULT_PRIOR = DEFAULT_STRUCTURED_COT_DIR / "runs/full-humaneval-2026-04-30/results.jsonl"
DEFAULT_CURRENT_GRAMMAR = DEFAULT_STRUCTURED_COT_DIR / "grammars/fsm_grammar_no_open.gbnf"
DEFAULT_TAGLINE_GRAMMAR = REPO_ROOT / "tools/grammar-eval/holiday-tagline.gbnf"

TARGET_REGRESSIONS = [97, 101, 108, 129, 137, 151]
CURRENT_RE = re.compile(r"^GOAL: [^\n]+\nAPPROACH: [^\n]+\nEDGE: [^\n]+$")
TAGLINE_RE = re.compile(
    r"^Q=(solve|prove|route|debug|patch|code|calc|compare|explain)\n"
    r"M=(case|enum|check|derive|edit|test|trace|rank)\n"
    r"K=[A-Za-z][A-Za-z0-9_.!-]{0,18}(,[A-Za-z][A-Za-z0-9_.!-]{0,18}){0,4}\n"
    r"R=[A-Za-z][A-Za-z0-9_.!-]{0,18}(,[A-Za-z][A-Za-z0-9_.!-]{0,18}){0,4}\n"
    r"V=(ok|fail|done|blocked|candidate|verify)$",
    re.S,
)

HOLIDAY_SYSTEM = (
    "You are an expert Python programmer. Think only inside the required "
    "tagline fields. Q is intent, M is method, K and R are comma-separated "
    "keywords with no spaces, V is status. After </think>, write correct, "
    "efficient, well-tested code in a ```python ... ``` fenced block."
)

CONDITIONS = ("free", "current", "holiday", "prompt_terse")
LABELS = {
    "free": "FREE",
    "current": "GOAL/APPROACH/EDGE",
    "holiday": "Holiday tagline",
    "prompt_terse": "PROMPT_TERSE",
}


def load_harness(path: pathlib.Path):
    script = path / "fsm_vs_free_eval.py"
    if not script.exists():
        raise SystemExit(f"missing structured-cot harness: {script}")
    spec = importlib.util.spec_from_file_location("fsm_vs_free_eval", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_prior(path: pathlib.Path) -> list[dict]:
    rows = []
    if path.exists():
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def task_num(task_id: str) -> int:
    return int(str(task_id).split("/")[-1])


def choose_subset(prior_rows: list[dict], seed: int) -> tuple[list[int], dict[int, str]]:
    selected = list(TARGET_REGRESSIONS)
    buckets = {n: "target_regress" for n in selected}

    wins = [
        task_num(r["task_id"])
        for r in prior_rows
        if r.get("free", {}).get("pass") is False and r.get("fsm", {}).get("pass") is True
    ]
    if not wins:
        wins = [10, 22, 36, 39]
    for n in wins[:4]:
        if n not in buckets:
            selected.append(n)
            buckets[n] = "current_win"

    rng = random.Random(seed)
    candidates = [n for n in range(164) if n not in buckets]
    for n in rng.sample(candidates, 30 - len(selected)):
        selected.append(n)
        buckets[n] = "random"
    return selected, buckets


def grammar_violation(condition: str, think: str) -> str:
    think = re.sub(r"^<think>\n?", "", think or "")
    think = re.sub(r"\n?</think>.*$", "", think, flags=re.S).strip()
    if condition == "current" and not CURRENT_RE.match(think):
        return "current_shape_mismatch"
    if condition == "holiday" and not TAGLINE_RE.match(think):
        return "holiday_shape_mismatch"
    return ""


def generate(mod, client, args, condition: str, user_prompt: str, grammars: dict[str, str]) -> tuple[str, int]:
    if condition == "free":
        return mod.generate_free(client, args.model, user_prompt, args.max_tokens)
    if condition == "prompt_terse":
        return mod.generate_prompt_terse(client, args.model, user_prompt, args.max_tokens)
    if condition == "current":
        grammar = grammars["current"]
        prompt = mod.fsm_user_prompt_for_grammar(user_prompt, grammar)
        return mod.generate_fsm(
            client,
            args.model,
            prompt,
            grammar,
            args.max_tokens,
            mod.fsm_system_prompt_for_grammar(grammar),
        )
    if condition == "holiday":
        grammar = grammars["holiday"]
        prompt = mod.fsm_user_prompt_for_grammar(user_prompt, grammar)
        return mod.generate_fsm(client, args.model, prompt, grammar, args.max_tokens, HOLIDAY_SYSTEM)
    raise ValueError(condition)


def run_condition(mod, client, args, prob: dict, condition: str, grammars: dict[str, str]) -> dict:
    entry_point = prob.get("entry_point") or "candidate"
    user_prompt = mod.build_user_prompt(prob, "humaneval")
    t0 = time.time()
    try:
        text, total_tokens = generate(mod, client, args, condition, user_prompt, grammars)
        gen_ms = round((time.time() - t0) * 1000)
        think = mod.extract_think(text)
        code, extraction = mod.extract_code_with_info(text)
        think_tokens = mod.count_tokens(think, args.tokenizer)
        output_tokens = max(int(total_tokens) - think_tokens, 0)
        violation = grammar_violation(condition, think)
        entry_found = mod._entry_point_found(code, "humaneval", entry_point, prob)
        if extraction["extraction_issue"] == "empty_code":
            passed, err = False, "empty_code"
        else:
            passed, err = mod.run_tests(code, prob.get("test", ""), entry_point, args.timeout)
        wall_ms = round((time.time() - t0) * 1000)
        return {
            "verdict": "pass" if passed else "fail",
            "pass": bool(passed),
            "err": (err or "")[:300],
            "think_token_count": think_tokens,
            "output_token_count": output_tokens,
            "total_tokens": int(total_tokens),
            "grammar_violations": violation,
            "wall_time_ms": wall_ms,
            "generation_wall_time_ms": gen_ms,
            "extracted_think": think[:500],
            "extracted_code": code[:500],
            "raw_response": text if args.save_raw else "",
            "entry_point_found": bool(entry_found),
            **extraction,
        }
    except Exception as e:
        return {
            "verdict": "fail",
            "pass": False,
            "err": f"{type(e).__name__}: {e}"[:300],
            "think_token_count": 0,
            "output_token_count": 0,
            "total_tokens": 0,
            "grammar_violations": f"generation_error:{type(e).__name__}",
            "wall_time_ms": round((time.time() - t0) * 1000),
            "generation_wall_time_ms": round((time.time() - t0) * 1000),
            "entry_point_found": False,
            "extraction_method": "not_run",
            "extraction_issue": "generation_error",
        }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def condition_summary(results: list[dict], condition: str) -> dict:
    rows = [r["conditions"][condition] for r in results]
    thinks = [r.get("think_token_count", 0) for r in rows]
    return {
        "pass_count": sum(1 for r in rows if r.get("pass")),
        "n": len(rows),
        "pass_rate": mean([1.0 if r.get("pass") else 0.0 for r in rows]),
        "mean_think_tokens": mean(thinks),
        "median_think_tokens": statistics.median(thinks) if thinks else 0,
    }


def ids_where(results: list[dict], pred) -> list[str]:
    return [r["task_id"] for r in results if pred(r)]


def write_summary(out_dir: pathlib.Path, results: list[dict], args, selected: list[int]) -> None:
    summaries = {c: condition_summary(results, c) for c in CONDITIONS}
    target_ids = {f"HumanEval/{n}" for n in TARGET_REGRESSIONS}
    target_rows = [r for r in results if r["task_id"] in target_ids]
    holiday_target_pass = [r["task_id"] for r in target_rows if r["conditions"]["holiday"].get("pass")]
    holiday_rescues_current = ids_where(
        results,
        lambda r: not r["conditions"]["current"].get("pass") and r["conditions"]["holiday"].get("pass"),
    )
    holiday_new_failures = ids_where(
        results,
        lambda r: r["conditions"]["current"].get("pass") and not r["conditions"]["holiday"].get("pass"),
    )

    lines = [
        "# Grammar A/B subset bench",
        "",
        f"- Endpoint: `{args.base_url}`",
        f"- Model: `{args.model}`",
        f"- Subset seed: `{args.seed}`",
        f"- Target prior HE+ regressions: {', '.join(f'HE/{n}' for n in TARGET_REGRESSIONS)}",
        f"- Holiday target prior-regressions passed: **{len(holiday_target_pass)}/6**",
        f"- Holiday rescues vs current in this run: **{len(holiday_rescues_current)}**",
        f"- Holiday new failures vs current in this run: **{len(holiday_new_failures)}**",
        "",
        "| Grammar | Pass@1 (30) | Mean think tokens | Median think tokens | Failures rescued vs current | New failures introduced |",
        "|---|---:|---:|---:|---|---|",
    ]
    for condition in CONDITIONS:
        s = summaries[condition]
        if condition == "current":
            rescues = "baseline"
            new = "baseline"
        else:
            rescues_ids = ids_where(
                results,
                lambda r, c=condition: not r["conditions"]["current"].get("pass")
                and r["conditions"][c].get("pass"),
            )
            new_ids = ids_where(
                results,
                lambda r, c=condition: r["conditions"]["current"].get("pass")
                and not r["conditions"][c].get("pass"),
            )
            rescues = ", ".join(rescues_ids) if rescues_ids else "-"
            new = ", ".join(new_ids) if new_ids else "-"
        lines.append(
            f"| {LABELS[condition]} | {s['pass_count']}/{s['n']} ({s['pass_rate']*100:.1f}%) "
            f"| {s['mean_think_tokens']:.0f} | {s['median_think_tokens']:.0f} | {rescues} | {new} |"
        )

    if len(holiday_target_pass) >= 3:
        verdict = "Run phase 3: Holiday rescued enough of the prior regression cluster to justify a full HE+164 + LCB v6 bench."
    elif len(holiday_target_pass) >= 1:
        verdict = "Marginal signal: document the result, inspect rescued cases manually, and defer phase 3 unless quality looks clearly better."
    else:
        verdict = "Stop: Holiday did not rescue the targeted HE+ regression cluster."
    lines += [
        "",
        "## Target details",
        "",
        f"- Holiday target passes: {', '.join(holiday_target_pass) if holiday_target_pass else '-'}",
        f"- Holiday rescues vs current: {', '.join(holiday_rescues_current) if holiday_rescues_current else '-'}",
        f"- Holiday new failures vs current: {', '.join(holiday_new_failures) if holiday_new_failures else '-'}",
        "",
        "## Next step",
        "",
        verdict,
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))
    args_json = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": args_json,
                "selected_problem_nums": selected,
                "condition_summaries": summaries,
                "holiday_target_prior_regression_passes": holiday_target_pass,
                "holiday_rescues_vs_current": holiday_rescues_current,
                "holiday_new_failures_vs_current": holiday_new_failures,
                "next_step": verdict,
            },
            indent=2,
        )
    )
    print("\n".join(lines[:16]))
    print(f"\nSaved -> {out_dir / 'results.jsonl'}")
    print(f"Saved -> {out_dir / 'summary.md'}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8020/v1")
    p.add_argument("--model", default="qwen3.6-27b-autoround")
    p.add_argument("--tokenizer", default="/opt/ai/github/qwen36-dual-3090/models/qwen3.6-27b-autoround-int4")
    p.add_argument("--structured-cot-dir", type=pathlib.Path, default=DEFAULT_STRUCTURED_COT_DIR)
    p.add_argument("--prior-results", type=pathlib.Path, default=DEFAULT_PRIOR)
    p.add_argument("--current-grammar", type=pathlib.Path, default=DEFAULT_CURRENT_GRAMMAR)
    p.add_argument("--holiday-grammar", type=pathlib.Path, default=DEFAULT_TAGLINE_GRAMMAR)
    p.add_argument("--out-dir", type=pathlib.Path, default=REPO_ROOT / f"results/grammar-ab-{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--timeout", type=int, default=30, help="Per-test execution timeout.")
    p.add_argument("--save-raw", action="store_true", help="Store full raw responses in results.jsonl.")
    p.add_argument("--list-only", action="store_true", help="Print the selected 30-problem subset and exit.")
    args = p.parse_args()

    prior_rows = read_prior(args.prior_results)
    selected, buckets = choose_subset(prior_rows, args.seed)
    if args.list_only:
        for n in selected:
            print(f"HumanEval/{n}\t{buckets[n]}")
        return 0

    mod = load_harness(args.structured_cot_dir)
    grammars = {
        "current": args.current_grammar.read_text(),
        "holiday": args.holiday_grammar.read_text(),
    }
    client_args = SimpleNamespace(
        base_url=args.base_url,
        api_key_env="DUMMY_KEY",
        request_timeout=args.request_timeout,
    )
    client = mod.make_client(client_args)
    problems = mod.load_benchmark("humaneval", 164)
    by_num = {task_num(p["task_id"]): p for p in problems}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, n in enumerate(selected, start=1):
        prob = by_num[n]
        row = {"task_id": prob["task_id"], "subset_bucket": buckets[n], "conditions": {}}
        print(f"[grammar-ab] [{i:02d}/{len(selected)}] {prob['task_id']} ({buckets[n]})")
        for condition in CONDITIONS:
            t0 = time.time()
            result = run_condition(mod, client, args, prob, condition, grammars)
            row["conditions"][condition] = result
            print(
                f"[grammar-ab]   {LABELS[condition]:<20s} {result['verdict']:<4s} "
                f"think={result['think_token_count']:<4d} total={result['total_tokens']:<4d} "
                f"wall={round(time.time() - t0):>4d}s violation={result['grammar_violations'] or '-'}"
            )
        results.append(row)
        with (args.out_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    write_summary(args.out_dir, results, args, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
