#!/usr/bin/env python3
"""kv-offload-probe — prove the KV offload tier (KV_OFFLOAD_GB / KV_OFFLOAD_DISK) serves prefix hits.

A clean boot with offload enabled proves nothing: a tier can accept every write and never serve a
read back (vLLM v0.29.0 did exactly that, and so do the vLLM DFlash tiers today). This probe checks
the thing that matters, on vLLM or SGLang:

  RAM  (default): prompt A cold -> A again (GPU hit) -> fillers larger than the GPU pool, which push A
                  off the GPU -> A again. With a working tier, A comes back from host RAM.
  DISK (--disk):  prompt A cold -> A again -> `docker restart <container>` (the GPU and RAM tiers die
                  with the process) -> A again. With a working disk tier, A comes back from disk.

Both end with a fresh prompt B of the same length as the cold reference. PASS = A's return takes less
than half of B's time AND the engine's own offload counters moved. A carries a needle ("the access code
is ...") so a hit that returns the wrong state also shows up.

  python3 scripts/kv-offload-probe.py --url http://localhost:8142
  python3 scripts/kv-offload-probe.py --url http://localhost:8142 --disk --container sglang-qwen38-27b-mtp-dual

Needs the slug booted with KV_OFFLOAD_GB (>= 64 recommended, so the tier holds everything the probe writes;
too small a tier evicts A from RAM too and reads as a false FAIL). RAM mode takes ~10 min on a 2x 3090
dual-fast slug, because the fillers must exceed its ~550K-token GPU pool. Stdlib only.
"""
import argparse, json, random, re, subprocess, sys, time, urllib.error, urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone amber cedar maple "
         "copper silver harbor meadow canyon glacier prairie lantern compass anchor beacon falcon heron").split()
# Offload evidence, per engine. Substring match on the metric name, summed over every label set.
EVIDENCE = {
    "vllm": ["external_prefix_cache_hits", "kv_offload_load_bytes"],
    "sglang": ['cached_tokens_total{.*cache_source="host"', 'cached_tokens_total{.*cache_source="storage"',
               "load_back_tokens_total"],
}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="the slug's endpoint, e.g. http://localhost:8142")
    ap.add_argument("--prompt-tokens", type=int, default=40000, help="size of the probed prompt A (default 40000)")
    ap.add_argument("--fill-tokens", type=int, help="total filler tokens (RAM mode). Default: GPU pool + 10%%, auto-detected")
    ap.add_argument("--disk", action="store_true", help="disk mode: restart the container instead of filling the GPU")
    ap.add_argument("--container", help="container to restart in --disk mode (docker ps shows it)")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    if a.disk and not a.container:
        ap.error("--disk needs --container (the serving container's name from `docker ps`)")

    engine, model, max_len, pool = detect(base)
    rng = random.Random()
    print(f"[probe] {engine} · model {model} · max_model_len {max_len:,} · GPU pool "
          f"{f'{pool:,} tokens' if pool else 'not detected'} · mode {'DISK' if a.disk else 'RAM'}", flush=True)

    def complete(prompt, max_tokens=24):
        t0 = time.time()
        r = http(base + "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
        return time.time() - t0, (r.get("usage") or {}).get("prompt_tokens") or 0, r["choices"][0]["text"]

    cal = " ".join(rng.choice(WORDS) for _ in range(3000))
    per_word = complete(cal, 1)[1] / 3000

    def text(tag, n):
        return f"[{tag} {rng.getrandbits(64):016x}] " + " ".join(rng.choice(WORDS) for _ in range(int(n / per_word)))

    code = f"{rng.choice(WORDS).upper()}-{rng.randrange(1000, 9999)}"
    A = (f"[A {rng.getrandbits(64):016x}] Remember this: the access code is {code}.\n"
         + text("A-body", a.prompt_tokens) + "\nQuestion: What is the access code? Answer with the code only.\nAnswer: The access code is")

    t_cold, n_a, _ = complete(A)
    print(f"[probe] A cold              {t_cold:7.2f} s  ({n_a:,} tokens)", flush=True)
    t_dev, _, _ = complete(A)
    print(f"[probe] A again (GPU)       {t_dev:7.2f} s", flush=True)

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
        print(f"[probe] {n} fillers x ~{chunk:,} tokens to push A off the GPU (this is the slow part)", flush=True)
        for i in range(n):
            dt, pt, _ = complete(text(f"F{i}", chunk), 1)
            print(f"[probe]   filler {i + 1}/{n}       {dt:7.2f} s  ({pt:,} tokens)", flush=True)

    before = evidence(base, engine)
    t_back, _, answer = complete(A)
    time.sleep(2)
    after = evidence(base, engine)
    moved = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) - before.get(k, 0) > 0}
    t_fresh, _, _ = complete(text("B", n_a) + "\nAnswer:")
    needle = code in answer
    print(f"[probe] A after {'restart' if a.disk else 'eviction'}  {t_back:7.2f} s  needle {'OK' if needle else 'MISSING'}", flush=True)
    print(f"[probe] B fresh (cold ref)  {t_fresh:7.2f} s", flush=True)
    for k, v in sorted(moved.items()):
        print(f"[probe]   counter moved: {k} +{v:,.0f}", flush=True)

    ok = t_back < 0.5 * t_fresh and bool(moved)
    tier = "disk" if a.disk else "host RAM"
    print("[probe] RESULT " + (f"PASS — A came back from {tier}: {t_back:.2f} s vs {t_fresh:.2f} s cold"
                               if ok else
                               f"FAIL — no {tier} hit: {t_back:.2f} s vs {t_fresh:.2f} s cold, counters moved: {bool(moved)}"),
          flush=True)
    if ok and not needle:
        print("[probe] ⚠️ the hit returned, but the needle answer is missing — post this output", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
