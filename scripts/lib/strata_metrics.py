#!/usr/bin/env python3
"""Read a Strata server's GET /metrics JSON: where our tools get Strata's runtime numbers.

Strata writes its per-request lines (throughput, drafts accepted, expert-cache hits) to a log
file inside the container, not to `docker logs`, so the log parsers the other engines use find
nothing there and would report a dead drafter or an empty run. /metrics carries the same numbers:
`totals` (cumulative since the server started), `requests` (the newest 12, newest first) and
`engine` (version, KV, expert slots). One reader, so every consumer reads them the same way.

  strata_metrics.py snapshot <base-url> [--container C | --log-file F]
                                                       totals + engine as JSON on stdout
  strata_metrics.py delta <before.json> <after.json>   what happened between two snapshots
  strata_metrics.py recent <base-url> [N] [--container C | --log-file F]
                                                       the newest N finished requests (default 12)
  strata_metrics.py host-ram <base-url> [--container C | --log-file F]
                                                       evidence that experts sit in host RAM

`host-ram` prints one line of evidence and exits 0, or exits 1 when there is none: the expert
slots in RAM that /metrics reports (`tiers.now.ram_slots`, from the engine's STAT lines, so only
once a request has run), else, for Project Maya, the "RAM tier X GB pinned" lines of its engine
log (the latest one per card), or for its DeepSeek engine (maya-ds4) the latest "N routed experts on
the CPU lane (X GB)" line, or for its Hy4-Flash engine (maya-hy4) the latest "CPU lane: N of M routed experts in
VRAM" line (the other M - N on the CPU, from the model file); for its MiniMax engine (maya-mm3), which logs no RAM line, the RAM tier its
/metrics carries (engine.ram_slots, tiers.now.ram_gb). bench.sh uses it for Maya, whose docker log doesn't say.

`delta` and `recent` print key=value lines:
  requests  output_tokens  decode_tok_s  prefill_tok_s  drafts_offered  drafts_accepted  accept_rate
`recent` adds hit_rate and the engine's version, kv, max_context and expert_slots. A value is
empty when there is nothing to divide by (no drafts offered: accept_rate is empty, not 0).

Project Maya (the strata-maya engine, a Strata fork) serves the same document with its version in
`engine.maya_version` and no draft counts in `totals`: its per-request "glm spec: N of M drafts
accepted" lines (DeepSeek's maya-ds4: "maya ds4 serve: drafts N of M kept"; MiniMax's maya-mm3: "maya mm3
serve: DSpark kept N of M drafts") go only to the engine log: the `log` of the run config the server's
command line names (--config), for every engine, else `<storage.path>/maya.log` (the GLM recipe's
layout; from v1.1.31 DeepSeek's and MiniMax's servers send a storage.path too, which is their folder,
not their log's). Given
the container (--container, read with docker exec) or the log itself (--log-file), `snapshot` sums
those lines into totals.drafts_offered / drafts_accepted, so `delta` reads Maya's drafts like
Strata's, and `recent` sums the newest N of them. Without either, a Maya document reads with no
draft counts (drafts_offered=0), as before.
Exit 0 with the lines; 2 on bad usage; 3 when /metrics is unreachable, is not JSON, or is not
Strata's (no `engine.version` / `engine.maya_version` or `totals`), or when a Maya log named by a
given container can't be read. A consumer must report exit 3 as "could not read",
never as "no drafts" or "nothing ran". The engine's API key, when set (CLUB_ENGINE_API_KEY from
scripts/lib/engine-auth.sh, else STRATA_API_KEY), is sent as a bearer token: Strata gates /metrics
behind its key when it has one, and Strata 0.2+ always has one. Standard library only.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

TIMEOUT_S = 10


def fail(msg, code=3):
    print(f"[strata-metrics] {msg}", file=sys.stderr)
    sys.exit(code)


def fetch(base):
    url = base.rstrip("/") + "/metrics"
    # Accept: application/json, because Strata answers text/plain with Prometheus text instead
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    # CLUB_ENGINE_API_KEY: what scripts/lib/engine-auth.sh found (the container's own key, first)
    key = os.environ.get("CLUB_ENGINE_API_KEY", "") or os.environ.get("STRATA_API_KEY", "")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        fail(f"{url}: {e}")
    return parse(body, url)


def parse(body, where):
    try:
        data = json.loads(body)
    except ValueError:
        fail(f"{where}: not JSON")
    if not isinstance(data, dict) or not isinstance(data.get("totals"), dict) \
            or not isinstance(data.get("engine"), dict) \
            or not (data["engine"].get("version") or data["engine"].get("maya_version")):
        fail(f"{where}: not a Strata /metrics document (no engine.version or totals)")
    return data


# GLM: "glm spec: 15 of 16 drafts accepted (93.8%)"; DeepSeek (maya-ds4): "maya ds4 serve: drafts 40 of 44 kept; ...";
# MiniMax (maya-mm3): "maya mm3 serve: DSpark kept 40 of 44 drafts"; Hy4-Flash (maya-hy4): "maya hy4 serve: MTP kept 40
# of 44 drafts" (accepted first, then offered, in all four; one line a request that drafted)
MAYA_SPEC = re.compile(r"glm spec: (\d+) of (\d+) drafts accepted|maya ds4 serve: drafts (\d+) of (\d+) kept"
                       r"|maya mm3 serve: DSpark kept (\d+) of (\d+) drafts|maya hy4 serve: MTP kept (\d+) of (\d+) drafts")
# "glm fast: CUDA0 RAM tier 52.66 GB pinned, 4737 slots" — once per card at each start
MAYA_RAM_TIER = re.compile(r"\b(CUDA\d+) RAM tier ([0-9]+(?:\.[0-9]+)?) GB pinned")
# maya-ds4: "ds4: 11665 routed experts on the CPU lane (100.4 GB): 11665 copied into RAM ..." — once at each start
MAYA_DS4_CPU_LANE = re.compile(r"\bds4: \d+ routed experts on the CPU lane \(([0-9]+(?:\.[0-9]+)?) GB\)")
# maya-hy4: "INFO CPU lane: 11931 of 20992 routed experts in VRAM (57%), the rest on 25 CPU threads from the model's
# file" — once at each start; the rest are computed on the CPU from the model file (read through the page cache)
MAYA_HY4_CPU_LANE = re.compile(r"\bCPU lane: (\d+) of (\d+) routed experts in VRAM \(\d+%\)")


def maya_log_lines(data, source):
    """The engine log's per-request draft lines, oldest first, as (accepted, offered) pairs.
    source: ("container", name) reads the engine log there (maya_log_text); ("file", path) reads a file."""
    return [(int(m.group(1) or m.group(3) or m.group(5) or m.group(7)), int(m.group(2) or m.group(4) or m.group(6) or m.group(8)))
            for m in MAYA_SPEC.finditer(maya_log_text(data, source))]


def maya_host_ram(data, source):
    """The engine log's evidence of experts in host RAM, as host-ram prints it, or None. GLM: the RAM
    tier, the latest line per card, summed (the log is appended across restarts, so an earlier start's
    line for the same card is superseded). maya-ds4 has no tier line: the latest start's CPU lane, whose
    experts are in RAM (copied there, or read in place from the page cache). maya-hy4 logs neither: the
    latest start's "N of M routed experts in VRAM" line, the rest computed on the CPU from the model file
    (none when every expert is in VRAM). maya-mm3 logs no RAM line
    under the server: its evidence is in /metrics (main's host-ram)."""
    text = maya_log_text(data, source)
    latest = {}
    for m in MAYA_RAM_TIER.finditer(text):
        latest[m.group(1)] = float(m.group(2))
    if latest:
        gb = sum(latest.values())
        return f"engine log (RAM tier {gb:.2f} GB pinned)" if gb > 0 else None
    lane = MAYA_DS4_CPU_LANE.findall(text)
    gb = float(lane[-1]) if lane else 0.0
    if gb > 0:
        return f"engine log (CPU lane: {gb:.2f} GB of experts)"
    # maya-hy4 logs no GB figure: how many routed experts are NOT in VRAM, the latest start's line
    hy4 = MAYA_HY4_CPU_LANE.findall(text)
    if hy4:
        vram, total = (int(x) for x in hy4[-1])
        if total > vram:
            return f"engine log (CPU lane: {total - vram} of {total} routed experts on the CPU, from the model file)"
    return None


def maya_log_text(data, source):
    """The text of Project Maya's engine log. source as for maya_log_lines."""
    kind, where = source
    if kind == "file":
        try:
            with open(where, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            fail(f"could not read {where}: {e}")
    else:
        # The run config's `log` first: each Maya engine's config names its own (GLM maya.log, DeepSeek
        # maya-ds4.log, MiniMax maya-mm3.log). /metrics' storage.path is only the quant's folder, and from
        # v1.1.31 DeepSeek's and MiniMax's servers send it too, where <storage.path>/maya.log isn't their
        # log. <storage.path>/maya.log stays the fallback for a GLM server whose command line names no config.
        store = (data.get("storage") or {}).get("path") or ""
        log = config_log(where, required=not store.startswith("/"))
        text = container_cat(where, log or store.rstrip("/") + "/maya.log")
    return text


def container_read(where, path):
    """(a file's text in container `where` (docker exec), or None; why it couldn't be read)."""
    try:
        r = subprocess.run(["docker", "exec", where, "cat", path], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"could not read {path} in {where}: {e}"
    if r.returncode != 0:
        return None, f"could not read {path} in {where}: {r.stderr.strip() or 'docker exec failed'}"
    return r.stdout, ""


def container_cat(where, path):
    """A file's text in container `where` (docker exec), or exit 3."""
    text, why = container_read(where, path)
    if text is None:
        fail(why)
    return text


def config_log(where, required=True):
    """The engine log a Maya server writes: the `log` of the run config its command line names
    (serve/server.py --config FILE, the container's process 1 once the entrypoint has exec'd it). When it
    can't be found: exit 3, or None when not `required` (the caller has a fallback)."""
    def none(msg):
        if required:
            fail(msg)
        return None
    cmdline, why = container_read(where, "/proc/1/cmdline")
    if cmdline is None:
        return none(why)
    args = cmdline.split("\0")
    if "--config" not in args[:-1]:
        return none("Project Maya's /metrics names no storage.path and its server's command line no --config, "
                    "so its engine log can't be found")
    cfg = args[args.index("--config") + 1]
    text, why = container_read(where, cfg)
    if text is None:
        return none(why)
    try:
        log = json.loads(text).get("log") or ""
    except (ValueError, AttributeError):
        return none(f"{cfg} in {where} is not a JSON run config")
    if not str(log).startswith("/"):
        return none(f"{cfg} in {where} names no engine log")
    return str(log)


def log_source(args):
    """Pull --container C / --log-file F out of args (in place); None when neither is given."""
    for flag, kind in (("--container", "container"), ("--log-file", "file")):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                fail(f"{flag} needs a value", 2)
            value = args[i + 1]
            del args[i:i + 2]
            return (kind, value)
    return None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def rate(n, ms):
    return f"{n * 1000.0 / ms:.2f}" if ms > 0 else ""


def figures(output_tokens, decode_ms, prompt_read, prompt_ms, offered, accepted, requests):
    return {
        "requests": str(int(requests)),
        "output_tokens": str(int(output_tokens)),
        "decode_tok_s": rate(output_tokens, decode_ms),
        "prefill_tok_s": rate(prompt_read, prompt_ms),
        "drafts_offered": str(int(offered)),
        "drafts_accepted": str(int(accepted)),
        "accept_rate": f"{accepted / offered:.4f}" if offered > 0 else "",
    }


def totals_figures(t, base=None):
    base = base or {}
    d = {k: num(t.get(k)) - num(base.get(k)) for k in
         ("requests", "output_tokens", "decode_ms", "prompt_tokens", "reused", "prompt_ms",
          "drafts_offered", "drafts_accepted")}
    return figures(d["output_tokens"], d["decode_ms"], d["prompt_tokens"] - d["reused"], d["prompt_ms"],
                   d["drafts_offered"], d["drafts_accepted"], d["requests"])


def emit(pairs):
    for k, v in pairs.items():
        print(f"{k}={v}")


def main(argv):
    argv = list(argv)
    source = log_source(argv)
    if len(argv) < 2 or argv[1] not in ("snapshot", "delta", "recent", "host-ram"):
        print(__doc__.split("\n\n")[2], file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "snapshot":
        if len(argv) != 3:
            return main(argv[:1])
        data = fetch(argv[2])
        totals = dict(data["totals"])
        if source and data["engine"].get("maya_version") and "drafts_offered" not in totals:
            lines = maya_log_lines(data, source)
            totals["drafts_accepted"] = sum(a for a, _ in lines)
            totals["drafts_offered"] = sum(o for _, o in lines)
        json.dump({"totals": totals, "engine": data["engine"]}, sys.stdout)
        print()
        return 0
    if cmd == "delta":
        if len(argv) != 4:
            return main(argv[:1])
        snaps = []
        for path in argv[2:4]:
            try:
                with open(path, encoding="utf-8") as f:
                    snaps.append(parse(f.read(), path))
            except OSError as e:
                fail(f"{path}: {e}")
        before, after = snaps
        if num(after["totals"].get("requests")) < num(before["totals"].get("requests")):
            fail("the second snapshot has fewer requests than the first: the server restarted between them")
        emit(totals_figures(after["totals"], before["totals"]))
        return 0
    if cmd == "host-ram":
        if len(argv) != 3:
            return main(argv[:1])
        data = fetch(argv[2])
        slots = num(((data.get("tiers") or {}).get("now") or {}).get("ram_slots"))
        if slots > 0:
            print(f"/metrics ({int(slots)} expert slots in RAM)")
            return 0
        if source and data["engine"].get("maya_version"):
            evidence = maya_host_ram(data, source)
            if evidence:
                print(evidence)
                return 0
        # MiniMax's maya-mm3 logs no RAM line under the server; its /metrics carries the RAM tier: the engine's
        # INFO line (engine.ram_slots) and the tiers block (tiers.now.ram_gb)
        if data["engine"].get("maya_version"):
            now = (data.get("tiers") or {}).get("now") or {}
            gb, slots = num(now.get("ram_gb")), num(data["engine"].get("ram_slots"))
            if gb > 0 or slots > 0:
                print(f"/metrics ({int(slots)} expert slots in RAM, {gb:.2f} GB)")
                return 0
        return 1
    # recent
    if len(argv) not in (3, 4):
        return main(argv[:1])
    n = int(argv[3]) if len(argv) == 4 else 12
    data = fetch(argv[2])
    reqs = [r for r in (data.get("requests") or []) if isinstance(r, dict)][:max(n, 0)]
    out = figures(sum(num(r.get("output_tokens")) for r in reqs), sum(num(r.get("decode_ms")) for r in reqs),
                  # Maya's requests carry prompt_tokens and reused, not prompt_read
                  sum(num(r["prompt_read"]) if "prompt_read" in r else num(r.get("prompt_tokens")) - num(r.get("reused"))
                      for r in reqs), sum(num(r.get("prompt_ms")) for r in reqs),
                  sum(num(r.get("drafts_offered")) for r in reqs), sum(num(r.get("drafts_accepted")) for r in reqs),
                  len(reqs))
    if source and data["engine"].get("maya_version") and not any("drafts_offered" in r for r in reqs):
        lines = maya_log_lines(data, source)[-max(n, 0):] if n > 0 else []
        acc, off = sum(a for a, _ in lines), sum(o for _, o in lines)
        out.update(drafts_offered=str(off), drafts_accepted=str(acc),
                   accept_rate=f"{acc / off:.4f}" if off > 0 else "")
    hits = [num(r.get("hit_rate")) for r in reqs if r.get("hit_rate") is not None]
    out["hit_rate"] = f"{sum(hits) / len(hits):.3f}" if hits else ""
    eng = data["engine"]
    out.update(version=str(eng.get("version") or eng.get("maya_version") or ""), kv=str(eng.get("kv", "")),
               max_context=str(eng.get("max_context", "")), expert_slots=str(eng.get("expert_slots", "")))
    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
