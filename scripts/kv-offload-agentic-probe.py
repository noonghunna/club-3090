#!/usr/bin/env python3
"""kv-offload-agentic-probe — real agent workload for the KV offload tier (#1419).

The word-soup probe (kv-offload-probe.py) proves the tier with one prompt. This one runs the
workload the discussion asked for: N long agentic conversations (15 turns of a real Claude Code
session fixture, shared coding-agent system prompt + tools, one tool call per turn), INTERLEAVED —
the conversations are switched between turn by turn — then all evicted at once, then each one is
revisited and proven to come back warm from the tier:

  RAM  (default): build all N conversations -> fillers larger than the GPU pool push every
                  conversation off the GPU -> revisit each: it must come back from host RAM.
  DISK (--disk):  build all N conversations -> `docker restart <container>` (the GPU and RAM tiers
                  die with the process) -> revisit each: it must come back from the disk tier.

Each conversation carries a needle ("the access code is ...") planted in its unique first user
message, so a hit that returns the wrong state also shows up. PASS = every conversation's return
takes less than half of a fresh cold-reference prompt of the same length AND the engine's own
offload counters moved during the revisits. The probe runs thinking-off by default (a fast,
deterministic build); --thinking switches the whole run (build and revisit) to
enable_thinking: true and widens the build-turn budget (thinking turns run ~20 reasoning tokens
plus the preamble). The needle check is unaffected: this family emits the answer in `content`
even with thinking on (the reasoning block is a separate `reasoning` field the probe does not
read). Pass --seed to vary the needles.

  python3 scripts/kv-offload-agentic-probe.py --url http://localhost:8142
  python3 scripts/kv-offload-agentic-probe.py --url http://localhost:8142 --disk --container sglang-qwen38-27b-mtp-dual
  python3 scripts/kv-offload-agentic-probe.py --conversations 4 --turns 15 --out results/probe-1419-ram.md
  python3 scripts/kv-offload-agentic-probe.py --thinking --out results/probe-1419-vllm-ram-think.md

Needs the slug booted with KV_OFFLOAD_GB (>= 64 recommended, so the tier holds everything the probe
writes; too small a tier evicts the conversations too and reads as a false FAIL). RAM mode takes
~15 min on a 2x 3090 dual-fast slug in RAM mode (build + fillers must exceed its ~550K-token GPU pool).
Stdlib only.
"""
import argparse, json, os, random, re, subprocess, sys, time, urllib.error, urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone amber cedar maple "
         "copper silver harbor meadow canyon glacier prairie lantern compass anchor beacon falcon heron").split()
# Offload evidence, per engine. Substring match on the metric name, summed over every label set.
EVIDENCE = {
    "vllm": ["external_prefix_cache_hits", "kv_offload_load_bytes"],
    "sglang": ['cached_tokens_total{.*cache_source="host"', 'cached_tokens_total{.*cache_source="storage"',
               "load_back_tokens_total"],
}

# System prompt + tool schemas, copied verbatim from scripts/bench-agentic.sh so this probe
# exercises the same realistic coding-agent prefix as the TTFT bench (fixed across all
# turns/sessions so prefix caching can warm up after the first turn of the first session).
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


def http(url, body=None, timeout=3600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        raw = f.read().decode()
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def detect(base):
    """(engine, model, max_model_len, gpu_pool_tokens or None)"""
    m = http(base + "/v1/models")["data"][0]
    model, max_len = m["id"], int(m.get("max_model_len") or 131072)
    try:
        info = http(base + "/server_info", timeout=30)
        if isinstance(info, dict) and ("max_total_num_tokens" in info or "internal_states" in info):
            pool = info.get("max_total_num_tokens") or (info.get("internal_states") or [{}])[0].get("max_total_num_tokens")
            return "sglang", model, max_len, int(pool) if pool else None
    except (urllib.error.URLError, ValueError, KeyError, IndexError, TypeError):
        pass
    metrics = http(base + "/metrics", timeout=30)
    # vLLM publishes the pool it logs as "GPU KV cache size: N tokens" as a cache_config_info label.
    # ⚠️ Don't multiply num_gpu_blocks x block_size: on hybrid (GDN/mamba) models the block is a padded
    # page, and the product read 11,840 against a real 591,422 on Qwen3.8 dual-fast.
    size = re.search(r'cache_config_info\{[^}]*\bkv_cache_size_tokens="(\d+)"', metrics)
    return "vllm", model, max_len, int(size.group(1)) if size else None


def evidence(base, engine):
    out = {}
    for line in http(base + "/metrics", timeout=30).splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        for pat in EVIDENCE[engine]:
            if re.search(pat, name) and not name.split("{")[0].endswith(("_created", "_bucket", "_count", "_sum")):
                key = re.sub(r"\{.*", "", name) + (f' [{m.group(1)}]' if (m := re.search(r'cache_source="(\w+)"', name)) else "")
                try:
                    out[key] = out.get(key, 0.0) + float(value)
                except ValueError:
                    pass
    return out


def seed_user_msg(rng, orig):
    """(code, first user message) — plant the needle in a copy of the fixture's first turn."""
    code = f"{rng.choice(WORDS).upper()}-{rng.randrange(1000, 9999)}"
    return code, f"Also, for this conversation, remember: the access code is {code}.\n{orig}"


def needle_ok(code, answer):
    return code in (answer or "")


def default_url():
    """http://localhost:<port> — the port of the slug currently running, resolved from the registry
    the same way bench-agentic.sh does (running container -> registry row -> port); 8020 if the
    registry can't be consulted (bench-agentic.sh's last-resort literal)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=15).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return "http://localhost:8020"
    try:
        out = subprocess.run(["bash", os.path.join(root, "scripts/lib/registry-emit.sh"), "--json", root],
                             capture_output=True, text=True, timeout=120).stdout
        cat = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return "http://localhost:8020"
    by_container = {v.get("container"): v.get("port") for v in cat.get("variants", [])
                    if v.get("container") and v.get("port")}
    for n in names:
        if n in by_container:
            return f"http://localhost:{by_container[n]}"
    return "http://localhost:8020"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="the slug's endpoint, e.g. http://localhost:8142 (default: the running slug's port from the registry, else :8020)")
    ap.add_argument("--conversations", type=int, default=3, help="number of distinct long conversations (default 3)")
    ap.add_argument("--turns", type=int, default=15, help="fixture turns per conversation (default 15, capped to the fixture length)")
    ap.add_argument("--fill-tokens", type=int, help="total filler tokens (RAM mode). Default: GPU pool x 1.1, auto-detected")
    ap.add_argument("--disk", action="store_true", help="disk mode: restart the container instead of filling the GPU")
    ap.add_argument("--container", help="container to restart in --disk mode (docker ps shows it)")
    # Time-derived by default: a re-run with the same seed on the same still-running slug
    # re-generates byte-identical fillers/soups and re-hits the previous run's KV, turning the
    # "cold reference" into a cache hit (2026-09-27: 100K-token fillers ran 1.2 s vs 20 s cold).
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for the needles (default: time-derived so re-runs stay cold; pass 1096 to match bench-agentic.sh)")
    ap.add_argument("--thinking", action="store_true", help="run the whole probe (build + revisit) with enable_thinking: true instead of off")
    ap.add_argument("--out", help="also write the final report block as markdown to this path")
    a = ap.parse_args()
    base = (a.url or default_url()).rstrip("/")
    if a.seed is None:
        a.seed = int(time.time() * 1000) % (10 ** 9)
        print(f"[probe] seed {a.seed} (time-derived; pass --seed to fix the needles)", flush=True)
    if a.disk and not a.container:
        ap.error("--disk needs --container (the serving container's name from `docker ps`)")

    fixture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "agentic-bench-fixture.json")
    if not os.path.isfile(fixture_path):
        sys.exit(f"[probe] fixture not found: {fixture_path}")
    with open(fixture_path) as f:
        fixture = json.load(f)
    turns = min(a.turns, len(fixture))

    engine, model, max_len, pool = detect(base)
    rng = random.Random(a.seed)
    mode = "DISK" if a.disk else "RAM"
    print(f"[probe] {engine} · model {model} · max_model_len {max_len:,} · GPU pool "
          f"{f'{pool:,} tokens' if pool else 'not detected'} · mode {mode} · "
          f"thinking {'ON' if a.thinking else 'off'} · "
          f"{a.conversations} conversations x {turns} turns", flush=True)

    # Each conversation starts with the shared system prompt; its turn 0 is the seeded needle
    planted = [seed_user_msg(rng, fixture[0]["user_msg"]) for _ in range(a.conversations)]
    codes, first_user = [p[0] for p in planted], [p[1] for p in planted]
    convs = [[{"role": "system", "content": SYSTEM}] for _ in range(a.conversations)]

    def chat(messages, max_tokens=2000 if a.thinking else 600):
        t0 = time.time()
        r = http(base + "/v1/chat/completions", {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "required",
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": a.seed,
            "chat_template_kwargs": {"enable_thinking": a.thinking},
        })
        return time.time() - t0, r

    # Phase 1 — interleaved build: the conversations are switched between turn by turn.
    # #665 guard: a reasoning model can burn max_tokens mid-object -> truncated, unterminated
    # `arguments` JSON; replaying that assistant turn poisons every subsequent turn (HTTP 500 when
    # the server re-parses the client-supplied history). Substitute "{}" for any args that don't
    # parse. #255 guard: a turn that emits no parseable tool call still grows the context via a
    # synthetic call, so the ramp keeps running.
    def safe_args(s):
        s = s or "{}"
        try:
            json.loads(s)
            return s
        except (json.JSONDecodeError, ValueError):
            return "{}"

    prompt_tokens = {}
    for p in range(turns):
        for c in range(a.conversations):
            user = first_user[c] if p == 0 else fixture[p]["user_msg"]
            msgs = convs[c]
            msgs.append({"role": "user", "content": user})
            tt, r = chat(msgs)
            msg = r["choices"][0]["message"]
            calls = []
            for i, s in enumerate(msg.get("tool_calls") or []):
                fn = s.get("function") or {}
                if not fn.get("name"):
                    continue
                calls.append({"id": s.get("id") or f"call_c{c}_t{p}_{i}",
                              "type": "function",
                              "function": {"name": fn["name"], "arguments": safe_args(fn.get("arguments"))}})
            if not calls:  # #255
                calls = [{"id": f"call_c{c}_t{p}_synthetic", "type": "function",
                          "function": {"name": TOOLS[0]["function"]["name"], "arguments": "{}"}}]
            am = {"role": "assistant", "tool_calls": calls}
            if msg.get("content"):
                am["content"] = msg["content"]
            msgs.append(am)
            msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": fixture[p]["tool_result"]})
            for tc in calls[1:]:
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "(done)"})
            prompt_tokens[c] = (r.get("usage") or {}).get("prompt_tokens") or prompt_tokens.get(c, 0)
            print(f"[probe] build t{p} conv{c} {tt:7.2f} s  ({prompt_tokens[c]:,} tokens)", flush=True)
    L = max(prompt_tokens.values())

    # Phase 2 — evict: fillers larger than the GPU pool (RAM), or a container restart (DISK).
    cal = " ".join(rng.choice(WORDS) for _ in range(3000))
    t0 = time.time()
    r = http(base + "/v1/completions", {"model": model, "prompt": cal, "max_tokens": 1, "temperature": 0})
    per_word = ((r.get("usage") or {}).get("prompt_tokens") or 3000) / 3000

    def complete(prompt, max_tokens=24):
        t0 = time.time()
        r = http(base + "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
        return time.time() - t0, (r.get("usage") or {}).get("prompt_tokens") or 0, r["choices"][0]["text"]

    def text(tag, n):
        return f"[{tag} {rng.getrandbits(64):016x}] " + " ".join(rng.choice(WORDS) for _ in range(int(n / per_word)))

    if a.disk:
        print(f"[probe] waiting 20 s for the disk tier to flush, then: docker restart {a.container}", flush=True)
        time.sleep(20)
        subprocess.run(["docker", "restart", a.container], check=True, capture_output=True)
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                urllib.request.urlopen(base + "/health", timeout=5)
                break
            except (urllib.error.URLError, OSError, ConnectionError):
                time.sleep(5)
        else:
            sys.exit("[probe] the endpoint did not come back within 30 min of the restart")
        complete("warm up " * 8, 1)
        print("[probe] container restarted and serving (the GPU and RAM tiers are empty now)", flush=True)
    else:
        fill = a.fill_tokens or (int(pool * 1.1) if pool else None)
        if not fill:
            sys.exit("[probe] could not detect the GPU pool; pass --fill-tokens (e.g. 600000 for a 2x 3090 dual-fast slug)")
        chunk = min(100000, max_len - 2000)
        n = -(-fill // chunk)
        print(f"[probe] {n} fillers x ~{chunk:,} tokens to push every conversation off the GPU (this is the slow part)", flush=True)
        for i in range(n):
            dt, pt, _ = complete(text(f"F{i}", chunk), 1)
            print(f"[probe]   filler {i + 1}/{n}       {dt:7.2f} s  ({pt:,} tokens)", flush=True)

    # Phase 3 — revisit each conversation (round-robin = switching back between them) and prove the hit.
    before = evidence(base, engine)
    results = []
    for c in range(a.conversations):
        msgs = convs[c]
        msgs.append({"role": "user", "content": "Question: What is the access code for this conversation? Answer with the code only."})
        t0 = time.time()
        # ⚠️ `tools` MUST match the build turns: the Qwen3.8 template renders the tool block at the
        # top of the system turn, so a revisit without it diverges from the stored conversation and
        # no tier (GPU, host or disk) can match it.
        # Thinking-on revisits spend ~20 reasoning tokens before the answer, so 24 truncates
        # the answer text itself (finish_reason length, content None) — widen to 128 there.
        r = http(base + "/v1/chat/completions", {
            "model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 128 if a.thinking else 24, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": a.thinking},
        })
        t_back = time.time() - t0
        # With `tools` on the revisit the model may answer through a tool call, so a correct
        # answer in the tool-call arguments is still a hit — check them as well as content.
        m = r["choices"][0]["message"]
        answer = (m.get("content") or "") + " ".join(
            (tc.get("function") or {}).get("arguments") or "" for tc in (m.get("tool_calls") or []))
        time.sleep(2)
        after = evidence(base, engine)
        moved = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) - before.get(k, 0) > 0}
        before = after
        ok_needle = needle_ok(codes[c], answer)
        results.append({"conv": c, "back": t_back, "needle": ok_needle, "moved": moved})
        print(f"[probe] conv{c} back {t_back:7.2f} s  needle {'OK' if ok_needle else 'MISSING'}  "
              f"counters {', '.join(sorted(moved)) or 'none'}", flush=True)
    t_fresh, _, _ = complete(text("B", L) + "\nAnswer:")
    print(f"[probe] fresh cold ref {t_fresh:7.2f} s ({L:,} tokens)", flush=True)
    for res in results:
        for k, v in sorted(res["moved"].items()):
            print(f"[probe]   conv{res['conv']} counter moved: {k} +{v:,.0f}", flush=True)

    max_back = max(res["back"] for res in results)
    any_moved = any(res["moved"] for res in results)
    ok = max_back < 0.5 * t_fresh and any_moved
    tier = "disk" if a.disk else "host RAM"
    print("[probe] RESULT " + (f"PASS — all {a.conversations} conversations came back from {tier}: {max_back:.2f} s vs {t_fresh:.2f} s cold"
                               if ok else
                               f"FAIL — no {tier} hit: {max_back:.2f} s vs {t_fresh:.2f} s cold, counters moved: {any_moved}"),
          flush=True)
    missing = [res["conv"] for res in results if not res["needle"]]
    if ok and missing:
        print(f"[probe] ⚠️ the hit returned, but needle answers are missing in conv{missing} — post this output", flush=True)
    if a.out:
        lines = [
            f"# KV offload agentic probe — {engine} · {model} · mode {mode}",
            "",
            f"- {a.conversations} conversations x {turns} turns, switched between; cold ref {L:,} tokens",
            "",
            "| conv | back (s) | cold ref (s) | needle | counters moved |",
            "|---|---:|---:|---|---|",
        ]
        for res in results:
            lines.append(f"| {res['conv']} | {res['back']:.2f} | {t_fresh:.2f} | "
                         f"{'OK' if res['needle'] else 'MISSING'} | {', '.join(sorted(res['moved'])) or 'none'} |")
        lines += ["", f"RESULT: {'PASS' if ok else 'FAIL'} — {a.conversations} conversations, {max_back:.2f} s back vs {t_fresh:.2f} s cold, counters moved: {any_moved}", ""]
        with open(a.out, "w") as f:
            f.write("\n".join(lines))
        print(f"[probe] wrote {a.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
