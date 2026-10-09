#!/usr/bin/env python3
"""quality_line.py — the compose `Quality:` one-liner from a benchlocal results JSON.

    quality_line.py RESULTS_JSON MODE [BENCHLOCAL_ARG ...]

Prints `Quality:   <pack scores> (<provenance suffix>)`. MODE is the wrapper's
--quick/--medium/--full (or a pack id). The trailing arguments are the argv the
wrapper handed benchlocal-cli; they are read only for `--thinking-sampler` and
`--extra-body`, and only when the results JSON does not record them. benchlocal
records both since benchlocal-cli#188; before that a `--resume`d run, whose flags
come back from the journal rather than the argv, was stamped `pack:*`.

Provenance suffix (#983E): mode, thinking gate, sampler, topology, thinking
validity, pack versions, date. Each stamp appears only when it is known — a
missing field stays missing instead of lying.

THE SAMPLER STAMP (#1579)
    Until #1579 only `sampling=server` was stamped, so a run with explicit
    --temperature/--top-p overrides printed the same line as a canonical run.
    Every line now says which sampler produced it, using the precedence
    benchlocal-cli's runner.build_request applies:

      sampling=server            --sampling-from-server: no sampler sent, the
                                 server's defaults apply (benchlocal strips every
                                 sampler key, so nothing else can leak in)
      sampling=explicit k=v/...  --temperature/--top-p/... overrides; applied last,
                                 so they win on both thinking legs
      sampling=pack+extra-body k=v/...
                                 the pack contract with sampler keys merged from
                                 --extra-body (a thinking leg's sampler still
                                 overrides temperature/top_p/top_k/min_p)
      sampling=pack+thinking-sampler k=v/...
                                 the pack contract with --thinking-sampler
                                 replacing the thinking sampler (thinking legs)
      sampling=pack:greedy       canonical, thinking OFF: every pack's
                                 sampling_defaults is temperature 0 / top_p 1
      sampling=pack:thinking     canonical, thinking ON: benchlocal's thinking
                                 sampler (temperature 1.0 / top_p 0.95 / top_k 20 /
                                 min_p 0) unless a pack pins its own —
                                 hermesagent-20 stays at temperature 0
      sampling=pack              canonical, pack-default thinking (mixed legs)

    `max_tokens` rides in `sampling_overrides` on every default wrapper run
    (the 4,096 budget since 2026-10-02) but is a length budget, not a sampler, so
    it never makes a run "explicit".

    A JSON with no `thinking_mode` predates benchlocal recording its sampler
    (2026-05-24), so it gets no sampler stamp at all rather than a guessed one.
"""

from __future__ import annotations

import datetime
import json
import sys

# benchlocal-cli runner._SAMPLING_KEYS — what --sampling-from-server strips and
# what a sampler override can set. max_tokens is deliberately not here.
SAMPLER_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "dynatemp_range",
    "dynatemp_exponent", "typical_p", "seed", "mirostat",
    "mirostat_tau", "mirostat_eta",
)

PACK_SHORT = {"toolcall": "tc", "instructfollow": "if", "structoutput": "so",
              "dataextract": "de", "reasonmath": "rm", "bugfind": "bf",
              "hermesagent": "hm", "cli": "cli"}


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sampler_part(values: dict) -> str:
    """`k=v/k=v` in SAMPLER_KEYS order — comma-free, the suffix is comma-joined."""
    return "/".join(f"{k}={_fmt(values[k])}" for k in SAMPLER_KEYS if k in values)


def _last_flag(args: list[str], flag: str) -> str | None:
    """The value of the LAST `flag VALUE` / `flag=VALUE` — argparse last-wins."""
    value = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == flag and i + 1 < len(args):
            value = args[i + 1]
            i += 2
            continue
        if a.startswith(flag + "="):
            value = a[len(flag) + 1:]
        i += 1
    return value


def _json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def sampling_stamp(d: dict, cli_args: list[str]) -> str | None:
    if d.get("sampling_source") == "server":
        return "sampling=server"
    overrides = d.get("sampling_overrides") or {}
    explicit = {k: v for k, v in overrides.items() if k in SAMPLER_KEYS}
    if explicit:
        return "sampling=explicit " + _sampler_part(explicit)
    if "thinking_mode" not in d:
        return None
    extra = d.get("extra_body")
    if not isinstance(extra, dict):
        extra = _json_object(_last_flag(cli_args, "--extra-body"))
    extra_sampler = {k: v for k, v in extra.items() if k in SAMPLER_KEYS}
    if extra_sampler:
        return "sampling=pack+extra-body " + _sampler_part(extra_sampler)
    tm = d.get("thinking_mode")
    if tm != "force-off":
        thinking = d.get("thinking_sampler")
        if not isinstance(thinking, dict):
            thinking = _json_object(_last_flag(cli_args, "--thinking-sampler"))
        if thinking:
            return "sampling=pack+thinking-sampler " + _sampler_part(thinking)
    if tm == "force-off":
        return "sampling=pack:greedy"
    if tm == "force-on":
        return "sampling=pack:thinking"
    return "sampling=pack"


def quality_line(d: dict, mode: str, cli_args: list[str], date: str) -> str:
    parts = []
    versions = []
    for p in d.get("packs", []):
        if p.get("status") == "stubbed" and p.get("total", 0) == 0:
            continue
        pid = p["pack_id"]
        pa = p["passed"]
        pt = p["total"]
        pct = round(100 * p["score"]) if pt else 0
        parts.append(f"{pid} {pa}/{pt} ({pct}%)")
        # #981: per-pack version provenance — the same responses score 4/15 or 9/15
        # on dataextract depending only on pack version, so a Quality: line without
        # versions is untraceable. Compact id per #983E (tc·if·so·de·rm·bf·hm·cli);
        # unknown packs fall back to their full id. Old schema-v1 JSONs without a
        # version field omit the stamp rather than inventing one.
        ver = p.get("version")
        if ver:
            base = pid.split("-", 1)[0]
            versions.append(f"{PACK_SHORT.get(base, base)}{ver}")
    if not parts:
        return "Quality:   (no scoreable packs ran)"

    suffix_parts = [f"--{mode.lstrip('-')}"]
    tm = d.get("thinking_mode")
    if tm == "force-on":
        suffix_parts.append("thinking ON")
    elif tm == "force-off":
        suffix_parts.append("thinking OFF")
    stamp = sampling_stamp(d, cli_args)
    if stamp:
        suffix_parts.append(stamp)
    # #1396: the topology the scores were measured on (run_meta, from run_context.py).
    tp = (d.get("run_meta") or {}).get("tp")
    if tp:
        suffix_parts.append(f"tp={tp}")
    validity = d.get("thinking_validity") or {}
    if validity:
        statuses = {o.get("status") for o in validity.values()}
        suffix_parts.append("validity=valid" if statuses <= {"ok"} else "validity=CONTAMINATED")
    if versions:
        suffix_parts.append("packs " + "·".join(versions))
    suffix_parts.append(date)
    return "Quality:   " + " · ".join(parts) + f" ({', '.join(suffix_parts)})"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.split("\n\n", 2)[1], file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        d = json.load(fh)
    print(quality_line(d, argv[2], argv[3:], datetime.date.today().isoformat()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
