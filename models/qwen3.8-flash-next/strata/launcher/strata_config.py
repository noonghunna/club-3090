#!/usr/bin/env python3
"""Apply club-3090's launch settings to Strata's run config at every boot (#1695 follow-up).

The entrypoint runs this inside the container (with the image's python), on the run config setup
writes, /data/install/strata-<tag>.json, before the server starts (Strata 0.2's launcher makes the
live launch config from it at every start, so an edit there is lost). Strata reads these keys from
its config (serve/server.py; engine_config_reads.py re-checks that on every bump), but setup writes none of them:

  knob                   config key                    Strata's own rule (server.py)
  REASONING_EFFORT       reasoning_effort              none | low | medium | high (high = the template's xhigh)
  ENABLE_THINKING=false  reasoning_effort: none        thinking off for requests that don't ask for it
  THINKING_BUDGET        reasoning_budget_tokens       whole tokens, 0 = no budget
  SPEC_N                 args: --spec N                the MTP drafter's verify window (setup writes 4; 1..8)
  ROPE_SCALING           args: --rope-scaling T        none | linear | yarn: extend past the model's trained 262144
                                                positions (llama.cpp's types; engine 0.1.21+). Past 262144 setup picks
                                                yarn and derives the factor itself; this knob overrides that at every
                                                boot, and inside the trained range it is a deliberate choice (the
                                                scaling applies at every position, so the run is a slightly different
                                                model from the stock one).
  ROPE_SCALE             args: --rope-scale F          the factor, >= 1: 2 -> 524288 positions. With linear/yarn and
                                                no ROPE_SCALE the factor is derived from CONTEXT (ctx / 262144, min 1)
                                                exactly as Strata's setup does past the trained length.
  TEMP TOP_P TOP_K       sampling.temperature / top_p  >= 0 / 0 < x <= 1 / integer 1..64
  MIN_P                  sampling.min_p                0..1
  PRESENCE_PENALTY       sampling.presence_penalty     >= 0
  REPEAT_PENALTY         sampling.repetition_penalty   > 0 (1 = off)
  SERVED_NAME            aliases: [name]               another name the model answers to (Strata #297, like
                                                llama-server's --alias): /v1/models lists it after setup's own
                                                name, and an answer to a request that names it carries it.
                                                One name, no spaces

A new setting here needs its key's evidence in scripts/lib/profiles/engine-config-reads.json (where the
pinned server reads it): test-strata-config-knobs fails until it has some.

MAX_TOKENS is not one of them: Strata reads no default output cap from the run config (its max_tokens
there is the vision encoder's); a default cap comes only from the Chat settings its web page saves.

A setting removed is a setting gone: `apply` first takes out every key it manages (Strata 0.2's setup
keeps keys it doesn't write itself across a new setup, so they would otherwise linger) and puts --spec
back to setup's own value, then writes the settings in the environment. SPEC_N goes into args; at its next
run setup keeps it as an option of the user's (user_args, folded into args and args_written), which is how
a later apply knows it is ours. Setup's own --spec is remembered in CLUB3090_STRATA_SETUP_SPEC_FILE, since
after an override setup writes ours instead. The rope pair is tricky the other way: past the trained 262144
setup resolves --rope-scaling/--rope-scale itself (yarn with a derived factor; it refuses no scaling there), so a
rope flag sitting in args_written with no matching user_args entry is setup's own resolution and strip LEAVES it --
deleting it would silently unscale a long-context boot. Ours are the ones in user_args or in args beyond
args_written; a removed rope setting deletes ours and setup re-derives its own at its next run (a CONTEXT change
re-runs it). When our flag sits in args_written too but with a different value, it is an override of setup's
resolution: strip restores setup's value rather than deleting, so the boot keeps its scaling. CONTEXT and KV
belong to setup (they size the KV and the expert cache): Strata's own entrypoint passes them to
`strata launch`, which runs setup again when they change; here they are only checked.

  strata_config.py validate               check the settings in the environment (exit 2 on a bad one)
  strata_config.py apply <in> <out>       write <out> = <in> with the settings (<in> = <out>: in place;
                                          an unchanged result is not written, so its bytes stay put)
  strata_config.py effective <config>     print what the server runs from (args and the managed keys;
                                          not setup's own bookkeeping, args_written and user_args)
"""
import json
import os
import sys

EFFORTS = {"none": "none", "low": "low", "medium": "medium", "high": "high",
           "xhigh": "high", "max": "high"}     # our other engines' names for Strata's "high" (the template's xhigh)
KV_FORMATS = ("fp16", "int8", "q4_0", "k8v4")  # `strata launch --kv` choices; strata-pin-bump.sh checks each new image still offers them
ROPE_METHODS = ("none", "linear", "yarn")      # `strata --help`: --rope-scaling choices (setup.py's resolve_rope set)
ROPE_FLAGS = ("--rope-scaling", "--rope-scale")
TRAINED_CTX = 262144                           # the model's trained length: setup.py picks yarn past it (factor ctx/262144, min 1) and refuses none; --yarn-orig-ctx's default
MANAGED = ("reasoning_effort", "reasoning_budget_tokens", "aliases")
MANAGED_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")


class Bad(ValueError):
    pass


def env(name):
    v = os.environ.get(name, "")
    return v if v != "" else None


def as_int(name, v, lo=None, hi=None):
    if not v.isdigit():
        raise Bad(f"{name}={v!r}: expected a whole number")
    n = int(v)
    if (lo is not None and n < lo) or (hi is not None and n > hi):
        rng = f"{lo}..{hi}" if hi is not None else f">= {lo}"
        raise Bad(f"{name}={v!r}: expected {rng}")
    return n


def as_float(name, v, ok, rule):
    try:
        x = float(v)
    except ValueError:
        raise Bad(f"{name}={v!r}: expected a number ({rule})") from None
    if not ok(x):
        raise Bad(f"{name}={v!r}: expected {rule}")
    return x


def settings():
    """The settings in the environment, checked: (config keys, sampling keys, spec, rope args, notes)."""
    keys, sampling, notes = {}, {}, []
    spec = None
    effort = env("REASONING_EFFORT")
    if effort is not None:
        if effort not in EFFORTS:
            raise Bad(f"REASONING_EFFORT={effort!r}: expected none, low, medium, high (or xhigh / max, "
                      f"which Strata runs as high)")
        keys["reasoning_effort"] = EFFORTS[effort]
    thinking = env("ENABLE_THINKING")
    if thinking is not None:
        if thinking not in ("true", "false"):
            raise Bad(f"ENABLE_THINKING={thinking!r}: expected true or false")
        if thinking == "false":
            if keys.get("reasoning_effort") not in (None, "none"):
                notes.append(f"ENABLE_THINKING=false wins over REASONING_EFFORT={effort}: thinking off")
            keys["reasoning_effort"] = "none"
    budget = env("THINKING_BUDGET")
    if budget is not None:
        keys["reasoning_budget_tokens"] = as_int("THINKING_BUDGET", budget, lo=0)
    spec_n = env("SPEC_N")
    if spec_n is not None:
        spec = as_int("SPEC_N", spec_n, lo=1, hi=8)
    for name, key, ok, rule in (
            ("TEMP", "temperature", lambda x: x >= 0, ">= 0 (0 = greedy)"),
            ("TOP_P", "top_p", lambda x: 0 < x <= 1, "0 < top_p <= 1"),
            ("MIN_P", "min_p", lambda x: 0 <= x <= 1, "0 <= min_p <= 1"),
            ("PRESENCE_PENALTY", "presence_penalty", lambda x: x >= 0, ">= 0"),
            ("REPEAT_PENALTY", "repetition_penalty", lambda x: x > 0, "> 0 (1 = off)")):
        v = env(name)
        if v is not None:
            sampling[key] = as_float(name, v, ok, rule)
    top_k = env("TOP_K")
    if top_k is not None:
        sampling["top_k"] = as_int("TOP_K", top_k, lo=1, hi=64)
    served = env("SERVED_NAME")
    if served is not None:
        if any(c.isspace() for c in served):
            raise Bad(f"SERVED_NAME={served!r}: expected one name, without spaces")
        keys["aliases"] = [served]
    kv = env("KV")
    if kv is not None and kv not in KV_FORMATS:
        raise Bad(f"KV={kv!r}: expected one of {', '.join(KV_FORMATS)}")
    context = env("CONTEXT")
    ctx = as_int("CONTEXT", context, lo=4096) if context is not None else None
    rope = {}
    scaling = env("ROPE_SCALING")
    if scaling is not None:
        if scaling not in ROPE_METHODS:
            raise Bad(f"ROPE_SCALING={scaling!r}: expected none, linear or yarn (Strata's --rope-scaling)")
        rope["--rope-scaling"] = scaling
    scale = env("ROPE_SCALE")
    if scale is not None:
        rope["--rope-scale"] = "%g" % as_float("ROPE_SCALE", scale, lambda x: x >= 1,
                                               ">= 1 (the extension factor: 2 -> 524288 positions)")
    if rope.get("--rope-scaling") == "none" and ctx is not None and ctx > TRAINED_CTX:
        raise Bad(f"ROPE_SCALING=none with CONTEXT={ctx}: past the trained {TRAINED_CTX} Strata's setup refuses no "
                  f"scaling (the stock angles break there)")
    if rope.get("--rope-scaling") in ("linear", "yarn"):
        if "--rope-scale" not in rope:
            if ctx is not None and ctx > TRAINED_CTX:
                rope["--rope-scale"] = "%g" % max(1.0, ctx / TRAINED_CTX)   # setup.py's own derivation past the trained length
                notes.append(f"ROPE_SCALE unset: derived {rope['--rope-scale']} from CONTEXT={ctx} / {TRAINED_CTX}, as setup does")
            else:
                notes.append(f"ROPE_SCALING={rope['--rope-scaling']} with no ROPE_SCALE: the factor comes from the model "
                             f"file, and this GGUF ships none (1 = no extension)")
        elif ctx is not None and ctx > TRAINED_CTX and float(rope["--rope-scale"]) * TRAINED_CTX < ctx:
            notes.append(f"ROPE_SCALE={rope['--rope-scale']} covers {int(float(rope['--rope-scale']) * TRAINED_CTX)} "
                         f"positions; CONTEXT={ctx} is past that (Strata keeps an explicit factor as given)")
        elif ctx is not None and ctx <= TRAINED_CTX:
            notes.append(f"ROPE_SCALING={rope['--rope-scaling']} inside the trained {TRAINED_CTX}: the scaling applies at "
                         f"every position, so the run is a slightly different model from the stock one")
    if "--rope-scale" in rope and "--rope-scaling" not in rope:
        notes.append("ROPE_SCALE set without ROPE_SCALING: Strata applies the factor only to linear/yarn scaling")
    key = env("API_KEY")                              # read by Strata's own entrypoint; here only checked
    if key is not None and len(key) < 16:
        notes.append(f"API_KEY is {len(key)} characters; Strata warns below 16 (openssl rand -hex 24 makes one)")
    return keys, sampling, spec, rope, notes


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


SETUP_SPEC = "4"   # `strata --help`: "--spec N ... (setup writes 4)", re-checked by strata-pin-bump.sh; used when setup's own was never seen


def setup_spec_record(value=None):
    """Setup's own --spec, kept beside the run config (CLUB3090_STRATA_SETUP_SPEC_FILE, set by the entrypoint):
    once we override it, setup writes ours instead, so its value is remembered from a run config we haven't."""
    path = os.environ.get("CLUB3090_STRATA_SETUP_SPEC_FILE", "")
    if not path:
        return None
    if value is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(value + "\n")
        return value
    try:
        return open(path, encoding="utf-8").read().strip() or None
    except OSError:
        return None


def set_arg(args, flag, value):
    args = list(args)
    if flag in args[:-1]:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    return args


def strip_managed(cfg):
    """Take out what an earlier apply wrote: the managed keys, --spec back to setup's own value, and our rope flags.

    Strata 0.2's setup keeps an option it doesn't choose itself as user_args (what args has beyond
    args_written) and folds it into both at its next run, so an --spec of ours is recognised by user_args.
    A rope flag is ours the same way, but the other direction matters too: past the trained 262144 setup writes
    --rope-scaling/--rope-scale itself, and a rope flag with no user_args entry is that resolution -- ours is
    deleted, setup's is left standing (stripping it would unscale a long-context boot into extrapolation)."""
    for k in MANAGED:
        cfg.pop(k, None)
    sampling = {k: v for k, v in (cfg.get("sampling") or {}).items() if k not in MANAGED_SAMPLING}
    if sampling:
        cfg["sampling"] = sampling
    else:
        cfg.pop("sampling", None)
    user = list(cfg.get("user_args") or [])
    written = cfg.get("args_written") or []
    for flag in ROPE_FLAGS:
        # ours = in user_args (setup folded it) or in args beyond args_written; a rope flag only in args_written is
        # setup's own resolution for this context (it auto-yarns past the trained length) and stays. A third state:
        # the flag in args AND in args_written with a DIFFERENT value — that is our override of setup's resolution,
        # and strip restores setup's value (deleting would unscale the boot; leaving ours is a stale override).
        args = cfg.get("args") or []
        ours = flag in user or (flag in args[:-1] and flag not in written)
        override = (not ours) and flag in args[:-1] and flag in written \
            and arg_value(args, flag) != arg_value(written, flag)
        if flag in user:
            i = user.index(flag)
            del user[i:i + 2]
        if ours and flag in (cfg.get("args") or [])[:-1]:
            args = list(cfg["args"])
            i = args.index(flag)
            del args[i:i + 2]
            cfg["args"] = args            # a removed rope setting is gone; setup re-derives its own at its next run
        elif override:
            cfg["args"] = set_arg(cfg["args"], flag, arg_value(written, flag))
    if user:
        cfg["user_args"] = user
    else:
        cfg.pop("user_args", None)
    if "--spec" not in user[:-1]:                      # no --spec of ours: setup's value is the one there
        theirs = arg_value(cfg.get("args_written") or [], "--spec") or arg_value(cfg.get("args") or [], "--spec")
        if theirs is not None:
            setup_spec_record(theirs)
            if "--spec" in (cfg.get("args") or [])[:-1]:
                cfg["args"] = set_arg(cfg["args"], "--spec", theirs)
        return cfg
    i = user.index("--spec")
    del user[i:i + 2]
    if user:
        cfg["user_args"] = user
    else:
        cfg.pop("user_args", None)
    theirs = setup_spec_record() or SETUP_SPEC
    for key in ("args", "args_written"):
        if key in cfg:
            cfg[key] = set_arg(cfg[key] or [], "--spec", theirs)
    return cfg


def apply(in_cfg):
    keys, sampling, spec, rope, notes = settings()
    cfg = strip_managed(json.loads(json.dumps(in_cfg)))   # a deep copy: the input is never touched
    cfg.update(keys)
    if sampling:
        cfg["sampling"] = {**(cfg.get("sampling") or {}), **sampling}
    if spec is not None:                              # setup records it as user_args itself at its next run
        cfg["args"] = set_arg(cfg.get("args") or [], "--spec", str(spec))
    for flag, value in rope.items():                  # ours: setup keeps them as user_args at its next run
        cfg["args"] = set_arg(cfg.get("args") or [], flag, value)
    context = env("CONTEXT")
    if context is not None and int(context) > TRAINED_CTX and "--rope-scaling" not in (cfg.get("args") or []):
        notes.append(f"CONTEXT={context} is past the trained {TRAINED_CTX} and the live args carry no rope scaling: "
                     f"a CONTEXT change re-runs setup, which derives yarn itself; until then this run extrapolates")
    shown = [f"{k}={v}" for k, v in keys.items()] + [f"sampling.{k}={v}" for k, v in sampling.items()]
    if spec is not None:
        shown.append(f"--spec {spec}")
    shown += [f"{flag} {value}" for flag, value in rope.items()]
    return cfg, shown, notes


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        if cmd == "validate":
            for n in settings()[4]:
                print(f"[strata-recipe] {n}")
            return 0
        if cmd == "effective" and len(argv) == 3:       # what the server runs from: args and our keys
            cfg = json.load(open(argv[2], encoding="utf-8"))
            print(json.dumps({k: cfg.get(k) for k in ("args", "sampling", *MANAGED)}, sort_keys=True))
            return 0
        if cmd == "apply" and len(argv) == 4:
            cur = json.load(open(argv[2], encoding="utf-8"))
            cfg, shown, notes = apply(cur)
            for n in notes:
                print(f"[strata-recipe] {n}")
            print("[strata-recipe] settings: " + (", ".join(shown) if shown else "none (Strata's defaults)"))
            if cfg == cur and os.path.realpath(argv[2]) == os.path.realpath(argv[3]):
                return 0                                      # unchanged, in place: leave the bytes alone
            tmp = argv[3] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, argv[3])
            return 0
    except Bad as e:
        print(f"[strata-recipe] {e}", file=sys.stderr)
        return 2
    print("usage: strata_config.py validate | apply <in.json> <out.json> | effective <config.json>", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
