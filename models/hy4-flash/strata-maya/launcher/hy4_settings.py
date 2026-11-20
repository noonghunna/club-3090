#!/usr/bin/env python3
"""club-3090's launch settings for Hy4-Flash on Project Maya, as hy4_prepare.py writes them into the run config at
every boot.

The same shape as MiniMax's mm3_settings.py and DeepSeek's ds4_settings.py, with Hy4-Flash's own defaults and levels:
one serve/server.py reads the same run-config keys for every engine. hy4_prepare.py rewrites the config at each start,
so a setting removed is gone at the next boot. Maya's serve/server.py (the strata-maya-club3090 image) reads:

  setting                config key            Maya's own rule
  REASONING_EFFORT       reasoning_effort      Hy4-Flash thinks at one level, high, or not at all (frontend.py:
                                               "none" and "high", the model's no_think and high). none turns thinking
                                               off; every other level runs at high (the server's own fold), so this
                                               file folds it too and says so. club default high, as Maya's own setup
                                               writes it (maya_quants.MODEL_DEFAULTS["hy4"])
  ENABLE_THINKING=false  reasoning_effort      none: thinking off (wins over REASONING_EFFORT)
  THINKING_BUDGET        thinking_budget       tokens, 0 = none; unset = Maya's own
  TEMP TOP_P             sampling.temperature  >= 0 / 0 < x <= 1; club defaults 0.9 / 1.0 (Tencent's
                                               generation_config.json, maya_quants.MODEL_DEFAULTS["hy4"])
  TOP_K MIN_P            sampling.top_k        integer 1..64 / 0..1
  PRESENCE_PENALTY       sampling.presence_penalty   >= 0
  REPEAT_PENALTY         sampling.repetition_penalty > 0 (1 = off)
  SERVED_NAME            model_name            the name /v1/models reports; unset = hy4-flash. One name, no spaces
  LOOP_GUARD             loop_guard            true | false: the server's loop guard; unset = Maya's own (on)
  KV                     args: --kv            auto (the engine's default: F16 latent rows, 8-bit rows where F16 does
                                               not fit) | f16 | q8. The cache sits in VRAM: the engine's KV
                                               streaming (v1.1.69 on) is off and this launcher doesn't set it
  DRAFTING               args: --mtp           on | off: Tencent's MTP block (Hy4-Flash-mtp-BF16, its own file) drafts.
                                               Unset = on: about 4 GB of VRAM beside the model, and Maya's setup turns
                                               it on on cards from about 16 GB. A rig where it doesn't pay sets off
  VISION                 args: --vision off    on | off: the vision tower (the mmproj file in the pack). Unset = on,
                                               the engine's default when the pack has the tower. From v1.1.69 a picture
                                               borrows the prompt buffers, so off frees no VRAM
  THREADS                args: --cpu-threads   the CPU lane's threads (the experts VRAM doesn't hold are computed
                                               there, from RAM); whole number >= 1. Unset: half the CPUs this
                                               container may use, at least 1 (threads(), below)
  HY4_EG_POPULATE        env.HY4_EG_POPULATE   1 | 0: map the n-gram table (63 GB) into the page cache in the background
                                               after the load (1), or read each looked-up page from the disk at its
                                               first use (0, Maya's default: the table belongs on an SSD). Written into
                                               the config's env, which the server hands to the engine
  CONTEXT                args: --max-context   the cache's capacity, a whole number >= 4096 (the floor the other Maya
                                               recipes check). Unset: the entrypoint's default (262144 here; Maya
                                               has run its quants end to end up to 32K, maya.py
                                               HY4_TESTED_CONTEXT, and verify-stress recalled to 240,107 tokens); hy4_prepare.py writes it, this file only
                                               checks it

Nor SPEC_N: maya-hy4 sizes its drafts by itself (--draft is its bench's).

A new setting here needs its key's evidence in scripts/lib/profiles/engine-config-reads.json (where the pinned
server reads it): test-maya-hy4-config fails until it has some.

  hy4_settings.py validate     check the settings in the environment; exit 2 with the reason on a bad one
Python stdlib only, no Maya import: the launch-knob guard runs `validate` on the host.
"""
import os
import sys

DEFAULT_EFFORT = "high"
DEFAULT_SAMPLING = {"temperature": 0.9, "top_p": 1.0}     # Tencent's card, as Maya's setup writes it
DEFAULT_NAME = "hy4-flash"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")   # the club's level names: none, else Hy4's high
KV_FORMATS = ("auto", "f16", "q8")
ON_OFF = ("on", "off")
MIN_CONTEXT = 4096                   # the floor the other Maya recipes check for the same engine flag


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


def _choice(name, v, allowed):
    if v not in allowed:
        raise Bad(f"{name}={v!r}: expected one of {', '.join(allowed)}")
    return v


def _cpus():
    """The CPUs this process may use (its affinity: a container's or taskset's share)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def threads(cpus=None):
    """The CPU lane's default thread count: half the usable CPUs (as the other Maya recipes), at least 1."""
    cpus = _cpus() if cpus is None else cpus
    return max(1, len(cpus) // 2)


def context():
    """CONTEXT from the environment, checked; None when unset (the entrypoint's own default applies)."""
    v = _env("CONTEXT")
    return None if v is None else _int("CONTEXT", v, MIN_CONTEXT)


def settings():
    """(config keys to set, sampling block, extra engine args, config env, notes) from the environment, checked."""
    keys, notes, args, env = {}, [], [], {}
    context()                                 # CONTEXT is checked here too: a bad value stops the boot
    effort = _env("REASONING_EFFORT")
    if effort is not None:
        _choice("REASONING_EFFORT", effort, EFFORTS)
    level = "none" if effort == "none" else DEFAULT_EFFORT
    if effort not in (None, "none", "high"):
        notes.append(f"REASONING_EFFORT={effort}: Hy4-Flash thinks at high or not at all, so it runs at high")
    keys["reasoning_effort"] = level
    thinking = _env("ENABLE_THINKING")
    if thinking is not None:
        _choice("ENABLE_THINKING", thinking, ("true", "false"))
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
        keys["loop_guard"] = _choice("LOOP_GUARD", guard, ("true", "false")) == "true"
    kv = _env("KV")
    if kv is not None:
        args += ["--kv", _choice("KV", kv, KV_FORMATS)]
    drafting = _env("DRAFTING")
    if drafting is not None:
        _choice("DRAFTING", drafting, ON_OFF)
    if (drafting or "on") == "on":
        args += ["--mtp"]
    vision = _env("VISION")
    if vision is not None:
        _choice("VISION", vision, ON_OFF)
    if vision == "off":
        args += ["--vision", "off"]
    n = _env("THREADS")
    if n is not None:
        args += ["--cpu-threads", str(_int("THREADS", n, 1))]
    else:
        n = threads()
        args += ["--cpu-threads", str(n)]
        notes.append(f"CPU lane: {n} threads (half the CPUs; THREADS sets it)")
    pop = _env("HY4_EG_POPULATE")
    if pop is not None:
        env["HY4_EG_POPULATE"] = _choice("HY4_EG_POPULATE", pop, ("0", "1"))
    return keys, sampling, args, env, notes


def main(argv):
    if argv[1:] == ["validate"]:
        try:
            settings()
        except Bad as e:
            print(f"[maya-recipe] {e}", file=sys.stderr)
            return 2
        return 0
    print("usage: hy4_settings.py validate", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
