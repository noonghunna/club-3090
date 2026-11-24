#!/usr/bin/env python3
"""club-3090's launch settings for Inkling-2-Mini on Project Maya, as ik2_prepare.py writes them into the run config at
every boot.

The same shape as Hy4-Flash's hy4_settings.py, with Inkling-2-Mini's own defaults and levels: one serve/server.py reads
the same run-config keys for every engine. ik2_prepare.py rewrites the config at each start, so a setting removed is
gone at the next boot. Maya's serve/server.py (the strata-maya-club3090 image, v1.1.72 on) reads:

  setting                config key            Maya's own rule
  REASONING_EFFORT       reasoning_effort      the level for requests that name none: thinkingmachines' template maps
                                               a name onto its effort dial (effort_map: none 0.0, minimal 0.1, low 0.2,
                                               medium 0.7, high 0.9, xhigh / max 0.99) and writes it as "Thinking
                                               effort level: X"; none turns thinking off here (measured 2026-11-24:
                                               no reasoning at none, as on vllm-inkling it isn't). club default high,
                                               as Maya's own setup writes it
                                               (maya_quants.MODEL_DEFAULTS["ik2-mini"])
  TEMP TOP_P             sampling.temperature  >= 0 / 0 < x <= 1; club defaults 1.0 / 1.0 (Maya's, its author's values)
  TOP_K MIN_P            sampling.top_k        integer 1..64 / 0..1
  PRESENCE_PENALTY       sampling.presence_penalty   >= 0
  REPEAT_PENALTY         sampling.repetition_penalty > 0 (1 = off)
  SERVED_NAME            model_name            the name /v1/models reports; unset = inkling-2-mini (the name the
                                               vllm-inkling slugs serve too). One name, no spaces
  LOOP_GUARD             loop_guard            true | false: the server's loop guard; unset = Maya's own (on)
  KV                     env.IK2_KV            fp8 (E4M3, half the bytes: Maya's default, tools/ik2_fit.py) | f16
  DRAFTING               args: --drafting      auto (the default: maya-ik2 measures once which of no drafting, the
                                               prompt lookup, the drafter and both decodes fastest here, and keeps it;
                                               the record is on the data volume) | on | off. off leaves the drafter out
  THREADS                args: --threads       the CPU lane's threads (only the experts VRAM does not hold run there;
                                               on a 24 GB card every expert is in VRAM); whole number >= 1. Unset:
                                               the engine's own choice
  CONTEXT                args: --max-context   the cache's capacity, a whole number >= 4096 (the floor the other Maya
                                               recipes check). Unset: the entrypoint's default (262144, the
                                               maintainer's cap for Inkling-2-Mini; the model reaches 1M);
                                               ik2_prepare.py writes it, this file only checks it

Not ENABLE_THINKING (the template has no thinking switch: REASONING_EFFORT=none is the way to turn thinking off), not THINKING_BUDGET (the server
closes a reasoning block at a budget through a </think> token, which Inkling's typed segments don't have), not SPEC_N
(the drafter's window is the engine's), not VISION (pictures need Maya's separate maya-vision encoder, which this image
does not build).

A new setting here needs its key's evidence in scripts/lib/profiles/engine-config-reads.json (where the pinned
server reads it).

  ik2_settings.py validate     check the settings in the environment; exit 2 with the reason on a bad one
Python stdlib only, no Maya import: the launch-knob guard runs `validate` on the host.
"""
import os
import sys

DEFAULT_EFFORT = "high"
DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 1.0}     # maya_quants.MODEL_DEFAULTS["ik2-mini"]
DEFAULT_NAME = "inkling-2-mini"
DEFAULT_KV = "fp8"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")   # the template's effort_map names
KV_FORMATS = ("fp8", "f16")
DRAFTING = ("auto", "on", "off")
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


def context():
    """CONTEXT from the environment, checked; None when unset (the entrypoint's own default applies)."""
    v = _env("CONTEXT")
    return None if v is None else _int("CONTEXT", v, MIN_CONTEXT)


def settings():
    """(config keys to set, sampling block, extra engine args, config env, notes) from the environment, checked."""
    keys, notes, args, env = {}, [], [], {}
    context()                                 # CONTEXT is checked here too: a bad value stops the boot
    effort = _env("REASONING_EFFORT")
    keys["reasoning_effort"] = DEFAULT_EFFORT if effort is None else _choice("REASONING_EFFORT", effort, EFFORTS)
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
    env["IK2_KV"] = DEFAULT_KV if kv is None else _choice("KV", kv, KV_FORMATS)
    drafting = _env("DRAFTING")
    args += ["--drafting", "auto" if drafting is None else _choice("DRAFTING", drafting, DRAFTING)]
    n = _env("THREADS")
    if n is not None:
        args += ["--threads", str(_int("THREADS", n, 1))]
    return keys, sampling, args, env, notes


def main(argv):
    if argv[1:] == ["validate"]:
        try:
            settings()
        except Bad as e:
            print(f"[maya-recipe] {e}", file=sys.stderr)
            return 2
        return 0
    print("usage: ik2_settings.py validate", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
