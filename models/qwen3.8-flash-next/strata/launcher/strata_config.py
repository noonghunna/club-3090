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
after an override setup writes ours instead. The rope pair needs no record: setup is never given a rope option here
(Strata's launch runs it with the context alone), so its own rope flags follow from the run config's --max-context
(setup.py resolve_rope): none inside the trained 262144, and past it yarn with factor context / 262144 (setup refuses
no scaling there). strip puts both args and args_written back to that, whatever an earlier setup folded into
args_written: a carried flag of ours sits there too, and once its user_args entry is gone it can't be told from
setup's own any other way (#1958 follow-up). CONTEXT and KV belong to setup (they size the KV and the expert
cache): Strata's own entrypoint passes them to `strata launch`, which runs setup again when they change; here they
are only checked.

Strata 0.2.11's launch keeps a second copy, the launch config /data/state/strata/launch/<preset>.json, which is
what the server starts from (strata_control/arg_layers.py). Each launch builds it from the run config plus the
previous launch config's user_args and hand edits, which win. When setup folds our --spec or rope flags into the
run config's user_args, the launch copies them there, and from then on every launch kept them over the run config:
a changed or removed SPEC_N / ROPE_* never reached the engine, even across later setups. `forget` takes our flags
out of the launch config before every start, keeping what it starts unchanged, so the next launch takes them from
the run config again.

  strata_config.py validate               check the settings in the environment (exit 2 on a bad one)
  strata_config.py apply <in> <out>       write <out> = <in> with the settings (<in> = <out>: in place;
                                          an unchanged result is not written, so its bytes stay put)
  strata_config.py forget <launch config> take our flags out of the launch config's user_args (in place;
                                          the file keeps its mode, and an unchanged one its bytes)
  strata_config.py effective <config>     print what the server runs of our settings: --spec and the rope pair
                                          (user_args over args, as the server merges them; not args_written)
                                          and the managed keys; the entrypoint compares the launch config's
                                          with the run config's after the start
"""
import json
import os
import sys

EFFORTS = {"none": "none", "low": "low", "medium": "medium", "high": "high",
           "xhigh": "high", "max": "high"}     # our other engines' names for Strata's "high" (the template's xhigh)
KV_FORMATS = ("fp16", "int8", "q4_0", "k8v4")  # `strata launch --kv` choices; strata-pin-bump.sh checks each new image still offers them
ROPE_METHODS = ("none", "linear", "yarn")      # `strata --help`: --rope-scaling choices (setup.py's resolve_rope set)
ROPE_FLAGS = ("--rope-scaling", "--rope-scale")
MANAGED_FLAGS = ("--spec",) + ROPE_FLAGS       # the engine options our settings write (SPEC_N, ROPE_SCALING, ROPE_SCALE)
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


def del_arg(args, flag):
    """args without flag and its value, every time it appears."""
    args = list(args)
    while flag in args[:-1]:
        i = args.index(flag)
        del args[i:i + 2]
    return args


def set_arg(args, flag, value):
    """args with flag's value set where it first appears (appended when absent), and any later copy of it removed."""
    args = list(args)
    if flag in args[:-1]:
        i = args.index(flag)
        return args[:i] + [flag, value] + del_arg(args[i + 2:], flag)
    return args + [flag, value]


def groups(args):
    """An argument list as option groups (an option and the values after it), as Strata's user_args.groups splits them."""
    out = []
    for a in args if isinstance(args, list) else []:
        a = str(a)
        if a.startswith("--") or not out:
            out.append([a])
        else:
            out[-1].append(a)
    return out


def merged(args, user):
    """args with user_args over them: what serve/server.py engine_args (user_args.merge) starts the engine with."""
    ug = groups(user)
    mine = {g[0] for g in ug}
    return [t for g in [g for g in groups(args) if g[0] not in mine] + ug for t in g]


def setup_rope(cfg):
    """The rope flags setup itself writes for this run config's context ({flag: value}; {} for none): setup.py resolve_rope
    with no rope option given, which is how Strata's launch runs it -- nothing inside the trained 262144, past it yarn with
    factor context / 262144, written with %g. None when the config names no --max-context: then args_written's are taken
    as setup's."""
    ctx = arg_value(cfg.get("args_written") or cfg.get("args") or [], "--max-context")
    if ctx is None or not str(ctx).isdigit():
        return None
    ctx = int(ctx)
    if ctx <= TRAINED_CTX:
        return {}
    return {"--rope-scaling": "yarn", "--rope-scale": "%g" % max(1.0, ctx / TRAINED_CTX)}


def strip_managed(cfg):
    """Take out what an earlier apply wrote: the managed keys, --spec back to setup's own value, and the rope pair back
    to setup's own for the context.

    Strata 0.2's setup keeps an option it doesn't choose itself as user_args (what args has beyond
    args_written) and folds it into both at its next run, so an --spec of ours is recognised by user_args.
    Rope flags are put back to setup_rope in args and args_written alike: a folded flag of ours also sits in
    args_written, and after the first strip has taken its user_args entry out it looks like setup's own (#1958
    follow-up: a removed ROPE_SCALING stayed in force). Past the trained 262144 that keeps setup's yarn, so a
    removed setting never unscales a long-context boot."""
    for k in MANAGED:
        cfg.pop(k, None)
    sampling = {k: v for k, v in (cfg.get("sampling") or {}).items() if k not in MANAGED_SAMPLING}
    if sampling:
        cfg["sampling"] = sampling
    else:
        cfg.pop("sampling", None)
    user = list(cfg.get("user_args") or [])
    for flag in ROPE_FLAGS:
        user = del_arg(user, flag)                     # in user_args: setup folded ours
    theirs = setup_rope(cfg)
    if theirs is None:                                 # no context to derive from: args_written's are setup's
        written = cfg.get("args_written") or []
        theirs = {f: arg_value(written, f) for f in ROPE_FLAGS if arg_value(written, f) is not None}
    for key in ("args", "args_written"):
        if key in cfg:
            for flag in ROPE_FLAGS:
                cfg[key] = set_arg(cfg[key] or [], flag, theirs[flag]) if flag in theirs else del_arg(cfg[key] or [], flag)
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


def forget(cfg):
    """The launch config (/data/state/strata/launch/<preset>.json) without our flags in its user_args.

    Strata 0.2.11's launch builds this file from the run config plus the previous launch config's user_args and hand
    edits (args changed since args_written), and those win (strata_control/arg_layers.py). A --spec or rope flag of
    ours that setup folded into the run config's user_args lands here and would be kept by every later launch. Each
    one's value moves into args, so what this file starts is unchanged (user_args over args gives the same), and
    args_written agrees with args on our flags, so the next launch sees no hand edit and takes them from the run
    config. The person's other user_args stay."""
    ug = groups(cfg.get("user_args"))
    ours = [g for g in ug if g[0] in MANAGED_FLAGS]
    if ours:
        args = [g for g in groups(cfg.get("args")) if g[0] not in {g[0] for g in ours}] + ours
        cfg["args"] = [t for g in args for t in g]
        rest = [t for g in ug if g[0] not in MANAGED_FLAGS for t in g]
        if rest:
            cfg["user_args"] = rest
        else:
            cfg.pop("user_args", None)
    if isinstance(cfg.get("args_written"), list):    # our flags' groups exactly as in args: Strata compares each flag's whole list of groups
        cfg["args_written"] = [t for g in [g for g in groups(cfg["args_written"]) if g[0] not in MANAGED_FLAGS]
                               + [g for g in groups(cfg.get("args")) if g[0] in MANAGED_FLAGS] for t in g]
    return cfg


def effective(cfg):
    """What the server runs of our settings: --spec and the rope pair as it starts the engine (user_args over args), and
    the managed keys. A run config and the launch config made from it give the same when nothing kept a value of ours."""
    args = merged(cfg.get("args") or [], cfg.get("user_args"))
    return {"args": {f: arg_value(args, f) for f in MANAGED_FLAGS}, **{k: cfg.get(k) for k in ("sampling", *MANAGED)}}


def write_in_place(path, cfg):
    """Write cfg to path through a temporary sibling, keeping path's mode (the launch config holds the API key: 0600).
    The sibling is created 0600, so the key is never readable under the entrypoint's umask 0002 in between."""
    tmp = path + ".tmp"
    try:
        os.unlink(tmp)                                # a leftover keeps its own mode through O_CREAT
    except FileNotFoundError:
        pass
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    try:
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    except OSError:
        pass
    os.replace(tmp, path)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        if cmd == "validate":
            for n in settings()[4]:
                print(f"[strata-recipe] {n}")
            return 0
        if cmd == "effective" and len(argv) == 3:       # what the server runs of our settings
            cfg = json.load(open(argv[2], encoding="utf-8"))
            print(json.dumps(effective(cfg), sort_keys=True))
            return 0
        if cmd == "forget" and len(argv) == 3:
            cur = json.load(open(argv[2], encoding="utf-8"))
            cfg = forget(json.loads(json.dumps(cur)))
            if cfg != cur:
                write_in_place(argv[2], cfg)
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
    print("usage: strata_config.py validate | apply <in.json> <out.json> | forget <launch.json> | effective <config.json>",
          file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
