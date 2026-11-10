#!/usr/bin/env python3
"""club-3090's launch settings for MiniMax-M3.1-Flash on Project Maya, as mm3_prepare.py writes them into the run
config at every boot.

The same shape as the GLM recipe's maya_settings.py and DeepSeek's ds4_settings.py, with MiniMax's own defaults and
levels: one serve/server.py reads the same run-config keys for every engine, and for a MiniMax template it keeps the
client's own effort word (frontend.py effort_name). mm3_prepare.py rewrites the config at each start, so a setting
removed is gone at the next boot. Maya's serve/server.py (the strata-maya-club3090 image) reads:

  setting                config key            Maya's own rule
  REASONING_EFFORT       reasoning_effort      none | auto | low | medium | high | xhigh | max: MiniMax-M3.1's five
                                               levels reach its template as they are (frontend.py EFFORT_NAMES), auto
                                               names none (the template's adaptive default: the model decides whether
                                               to think), none turns thinking off. club default auto, as Maya's own
                                               setup writes it for MiniMax (maya.py MM3_EFFORT)
  ENABLE_THINKING=false  reasoning_effort      none: thinking off (wins over REASONING_EFFORT)
  THINKING_BUDGET        thinking_budget       tokens, 0 = none; unset = Maya's own
  TEMP TOP_P             sampling.temperature  >= 0 / 0 < x <= 1; club defaults 1.0 / 0.95 (the model card's,
                                               maya.py MM3_SAMPLING)
  TOP_K MIN_P            sampling.top_k        integer 1..64 / 0..1
  PRESENCE_PENALTY       sampling.presence_penalty   >= 0
  REPEAT_PENALTY         sampling.repetition_penalty > 0 (1 = off)
  SERVED_NAME            model_name            the name /v1/models reports; unset = minimax-m3.1-flash. One name,
                                               no spaces
  LOOP_GUARD             loop_guard            true | false: the server's loop guard (serve/loop_guard.py, v1.1.30
                                               on), which breaks an answer or a thinking block stuck repeating one
                                               block of tokens; unset = Maya's own (on)
  KV                     args: --kv-cache      int8 (the engine's default) | bf16 (twice the pinned RAM a token).
                                               maya-mm3 also takes fp8 and fp4, refused here: fp8 isn't validated on
                                               these cards, and fp4 is below the stack's KV floor
  DSPARK                 args: --dspark        on | off: the DSpark drafter in the pack (dspark/). Unset = on: Maya's
                                               own speed calibration (tools/calibrate_ds4.py) kept it on two 3090s
                                               (+20.5% sustained decode) and on one (+8.2%; v1.1.31, ~70% of drafts
                                               kept), though it costs expert slots in VRAM; a PC where it doesn't
                                               pay sets off
  THREADS                args: --cpu-threads   the CPU lane's threads (the experts VRAM doesn't hold are computed
                                               there, from RAM); whole number >= 1. Unset: half the CPUs this
                                               container may use, at least 1 (threads(), below)
  CONTEXT                args: --max-context   K/V capacity, a whole number >= 4096 (the same floor the Strata
                                               recipe checks). Unset: the entrypoint's default (204800 here);
                                               mm3_prepare.py writes it, this file only checks it

Nor SPEC_N: DSpark's block length comes from the drafter's own config, and maya-mm3 has no depth flag.

A new setting here needs its key's evidence in scripts/lib/profiles/engine-config-reads.json (where the pinned
server reads it): test-maya-mm3-config fails until it has some.

  mm3_settings.py validate     check the settings in the environment; exit 2 with the reason on a bad one
Python stdlib only, no Maya import: the launch-knob guard runs `validate` on the host.
"""
import os
import sys

DEFAULT_EFFORT = "auto"
DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 0.95}     # the model card's, as Maya's setup writes it
DEFAULT_NAME = "minimax-m3.1-flash"
EFFORTS = ("none", "auto", "low", "medium", "high", "xhigh", "max")
KV_FORMATS = ("int8", "bf16")
DSPARK = ("on", "off")
MIN_CONTEXT = 4096                   # the floor the Strata recipe checks for the same engine flag


class Bad(ValueError):
    pass


def _env(name):
    v = os.environ.get(name, "")
    return v if v != "" else None


def _int(name, v, lo, hi=None):
    if not v.isdigit():
        raise Bad(f"{name}={v!r}: expected a whole number")
    n = int(v)
    if n < lo or (hi is not None and n > hi):
        raise Bad(f"{name}={v!r}: expected {lo}..{hi}" if hi is not None else f"{name}={v!r}: expected >= {lo}")
    return n


def _float(name, v, ok, rule):
    try:
        x = float(v)
    except ValueError:
        raise Bad(f"{name}={v!r}: expected a number ({rule})") from None
    if not ok(x):
        raise Bad(f"{name}={v!r}: expected {rule}")
    return x


def _cpus():
    """The CPUs this process may use (its affinity: a container's or taskset's share)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def threads(cpus=None):
    """The CPU lane's default thread count: half the usable CPUs (nproc / 2, as the DeepSeek recipe's ds4_settings.py
    and resolve_offload_threads give), at least 1. A rig that has measured its own sets THREADS."""
    cpus = _cpus() if cpus is None else cpus
    return max(1, len(cpus) // 2)


def context():
    """CONTEXT from the environment, checked; None when unset (the entrypoint's own default applies).
    mm3_prepare.py passes the value to --max-context; `validate` proves the domain for the knob guard."""
    v = _env("CONTEXT")
    return None if v is None else _int("CONTEXT", v, MIN_CONTEXT)


def settings():
    """(config keys to set, sampling block, extra engine args, notes) from the environment, checked."""
    keys, notes, args = {}, [], []
    context()                                 # CONTEXT is checked here too: a bad value stops the boot
    effort = _env("REASONING_EFFORT")
    if effort is not None and effort not in EFFORTS:
        raise Bad(f"REASONING_EFFORT={effort!r}: expected one of {', '.join(EFFORTS)}")
    keys["reasoning_effort"] = effort if effort is not None else DEFAULT_EFFORT
    thinking = _env("ENABLE_THINKING")
    if thinking is not None:
        if thinking not in ("true", "false"):
            raise Bad(f"ENABLE_THINKING={thinking!r}: expected true or false")
        if thinking == "false":
            if effort is not None and effort != "none":
                notes.append(f"ENABLE_THINKING=false wins over REASONING_EFFORT={effort}: thinking off")
            keys["reasoning_effort"] = "none"
    budget = _env("THINKING_BUDGET")
    if budget is not None:
        keys["thinking_budget"] = _int("THINKING_BUDGET", budget, 0)
    sampling = dict(DEFAULT_SAMPLING)
    for name, key, ok, rule in (
            ("TEMP", "temperature", lambda x: x >= 0, ">= 0 (0 = greedy)"),
            ("TOP_P", "top_p", lambda x: 0 < x <= 1, "0 < top_p <= 1"),
            ("MIN_P", "min_p", lambda x: 0 <= x <= 1, "0 <= min_p <= 1"),
            ("PRESENCE_PENALTY", "presence_penalty", lambda x: x >= 0, ">= 0"),
            ("REPEAT_PENALTY", "repetition_penalty", lambda x: x > 0, "> 0 (1 = off)")):
        v = _env(name)
        if v is not None:
            sampling[key] = _float(name, v, ok, rule)
    top_k = _env("TOP_K")
    if top_k is not None:
        sampling["top_k"] = _int("TOP_K", top_k, 1, 64)
    served = _env("SERVED_NAME")
    if served is not None:
        if any(c.isspace() for c in served):
            raise Bad(f"SERVED_NAME={served!r}: expected one name, without spaces")
        keys["model_name"] = served
    guard = _env("LOOP_GUARD")
    if guard is not None:
        if guard not in ("true", "false"):
            raise Bad(f"LOOP_GUARD={guard!r}: expected true or false")
        keys["loop_guard"] = guard == "true"
    kv = _env("KV")
    if kv is not None:
        if kv not in KV_FORMATS:
            why = {"fp8": " (maya-mm3 takes it, but it isn't validated on these cards)",
                   "fp4": " (below the stack's KV floor: never under q4, q8 preferred)"}.get(kv, "")
            raise Bad(f"KV={kv!r}: expected one of {', '.join(KV_FORMATS)}{why}")
        args += ["--kv-cache", kv]
    dspark = _env("DSPARK")
    if dspark is not None and dspark not in DSPARK:
        raise Bad(f"DSPARK={dspark!r}: expected on or off")
    args += ["--dspark", dspark or "on"]
    n = _env("THREADS")
    if n is not None:
        args += ["--cpu-threads", str(_int("THREADS", n, 1))]
    else:
        n = threads()
        args += ["--cpu-threads", str(n)]
        notes.append(f"CPU lane: {n} threads (half the CPUs; THREADS sets it)")
    return keys, sampling, args, notes


def main(argv):
    if argv[1:] == ["validate"]:
        try:
            settings()
        except Bad as e:
            print(f"[maya-recipe] {e}", file=sys.stderr)
            return 2
        return 0
    print("usage: mm3_settings.py validate", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
