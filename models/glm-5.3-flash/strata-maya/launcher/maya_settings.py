#!/usr/bin/env python3
"""club-3090's launch settings for Project Maya, as maya_prepare.py writes them into maya.json at every boot.

maya_prepare.py rewrites the config at each start, so a setting removed is gone at the next boot. Maya's
serve/server.py (the strata-maya-club3090 image) reads:

  setting                config key            Maya's own rule
  REASONING_EFFORT       reasoning_effort      none | low | high | max (frontend.py EFFORT: medium is high, xhigh is
                                               max); club default max, as the other GLM-5.3-Flash slugs
  ENABLE_THINKING=false  reasoning_effort      none: the server closes the thinking block (wins over REASONING_EFFORT)
  THINKING_BUDGET        thinking_budget       tokens, 0 = none; unset = Maya's own 32768
  TEMP TOP_P             sampling.temperature  >= 0 / 0 < x <= 1; club defaults 1.0 / 0.95 (GLM-5.3's vendor sampling)
  TOP_K MIN_P            sampling.top_k        integer 1..64 / 0..1
  PRESENCE_PENALTY       sampling.presence_penalty   >= 0
  REPEAT_PENALTY         sampling.repetition_penalty > 0 (1 = off)
  KV                     args: --kv            int8 (the engine's default from v1.1.55, MAYA_GLM_KV_INT8) | fp16
                                               (16 bits, the default before). The
                                               engine also parses q4_0 / k8v4, but only its Qwen path uses them:
                                               with --glm-pack they run fp16, so they are refused here
  SERVED_NAME            model_name            the name /v1/models reports and every answer carries (Maya answers a
                                               request whatever model it names); unset = glm-5.3-flash. One name,
                                               no spaces
  LOOP_GUARD             loop_guard            true | false: the server's loop guard (serve/loop_guard.py, v1.1.30
                                               on), which breaks an answer or a thinking block stuck repeating one
                                               block of tokens; unset = Maya's own (on)

SPEC_N is not one of them: Maya's --spec is documented for its DFlash2 drafter, not for GLM's MTP block.
Nor is MAX_TOKENS: the server takes a default output cap only from the Chat settings its web page saves
(maya.shared-settings.json, POST /settings), not from maya.json, whose max_tokens is the vision encoder's.

A new setting here needs its key's evidence in scripts/lib/profiles/engine-config-reads.json (where the pinned
server reads it): test-maya-config-knobs fails until it has some.

  maya_settings.py validate     check the settings in the environment; exit 2 with the reason on a bad one
Python stdlib only, no Maya import: the launch-knob guard runs `validate` on the host.
"""
import os
import sys

DEFAULT_EFFORT = "max"
DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 0.95}      # GLM-5.3's vendor sampling, as Maya's setup writes it
EFFORTS = {"none": "none", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}
KV_FORMATS = ("fp16", "int8")        # the GLM path (generate.cpp glm_pack_generate) reads only int8; q4_0 / k8v4 run fp16


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


def settings():
    """(config keys to set, sampling block, extra engine args, notes) from the environment, checked."""
    keys, notes, args = {}, [], []
    effort = _env("REASONING_EFFORT")
    if effort is not None and effort not in EFFORTS:
        raise Bad(f"REASONING_EFFORT={effort!r}: expected none, low, high, max (or medium / xhigh, which Maya runs "
                  f"as high / max)")
    keys["reasoning_effort"] = EFFORTS[effort] if effort is not None else DEFAULT_EFFORT
    thinking = _env("ENABLE_THINKING")
    if thinking is not None:
        if thinking not in ("true", "false"):
            raise Bad(f"ENABLE_THINKING={thinking!r}: expected true or false")
        if thinking == "false":
            if effort is not None and EFFORTS[effort] != "none":
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
            why = " (Strata's Qwen path only: a GLM pack would run it as fp16)" if kv in ("q4_0", "k8v4") else ""
            raise Bad(f"KV={kv!r}: expected one of {', '.join(KV_FORMATS)}{why}")
        args += ["--kv", kv]
    return keys, sampling, args, notes


def main(argv):
    if argv[1:] == ["validate"]:
        try:
            settings()
        except Bad as e:
            print(f"[maya-recipe] {e}", file=sys.stderr)
            return 2
        return 0
    print("usage: maya_settings.py validate", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
