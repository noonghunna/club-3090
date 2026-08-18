#!/usr/bin/env bash
# Every compose that ships a speculative-decoding drafter must expose the SAME
# switch, and that switch must actually work.
#
# THE CONTRACT
# ------------
#   SPEC_N=<n>   drafter depth
#   SPEC_N=0     drafter off
#   SPEC=off     drafter off (back-compat alias; also 0/false/no)
#
# WHY THIS EXISTS
# ---------------
# 2026-08-18: all 27 drafter-shipping composes failed to honour both toggles,
# in four different ways, because the knob accreted across three generations
# and was never reconciled:
#
#   hardcoded (10)   no env escape at all — vllm/dual, the blessed qwen3.6 dual
#                    default, told users to "delete the --speculative-config
#                    line below", i.e. edit a tracked file
#   SPEC only (5)    SPEC_N=0 silently ignored
#   SPEC_N only (9)  SPEC=off silently ignored
#   n=0 passthru (3) flag still sent, with num_speculative_tokens:0
#
# The silent half is what makes this a safety bug rather than an annoyance:
# turning the drafter off is the documented mitigation for vllm#50021 (GDN
# spec-decode wild write). A user who learns `SPEC=off` on one slug, applies it
# to another, and sees the server come up healthy has NOT mitigated anything.
#
# HOW IT CHECKS
# -------------
# It does not grep. It extracts each compose's entrypoint script and `command:`
# list, applies docker-compose ${VAR:-default} interpolation, runs the script
# under bash with `vllm` replaced by a stub that prints its argv, and inspects
# the argv the engine would actually have received. No GPU, no weights, no image.

# SCOPE: vLLM (`--speculative-config`) and the llama.cpp lineage (mainline,
# ik-llama, beellama, llamacpp-club3090 — `--spec-type` / `--spec-draft-model` /
# `--spec-draft-n-max` / `-md`). SGLang (--speculative-algorithm) is not covered;
# its two composes ship no drafter today.

set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - "$ROOT_DIR" <<'PY'
import json, os, pathlib, re, subprocess, sys, tempfile

root = pathlib.Path(sys.argv[1])

FLAG = "--speculative-config"                      # vLLM
# llama.cpp lineage: four grammars (mainline `--spec-type draft-mtp
# --spec-draft-n-max N`, ik-llama `--spec-type mtp:n_max=N,...`, beellama
# `--spec-type dflash` + external GGUF, deepseek `-md <path>`). All that matters
# for the contract is that NONE of them survive SPEC_N=0.
LC_FLAGS = ("--spec-type", "--spec-draft-model", "--spec-draft-n-max",
            "--spec-draft-n-min", "-md", "--model-draft")


def _interp(v, env):
    """docker compose interpolation: $$ is a literal $, ${VAR:-default} resolves."""
    out, i = [], 0
    while i < len(v):
        if v[i:i + 2] == "$$":
            out.append("$"); i += 2; continue
        m = re.match(r"\$\{([A-Za-z_]\w*)(?::-([^}]*))?\}", v[i:])
        if m:
            val = env.get(m.group(1)) or (m.group(2) if m.group(2) is not None else "")
            out.append(val); i += m.end(); continue
        out.append(v[i]); i += 1
    return "".join(out)


def _block(lines, key):
    """Return (start, end, indent) of a `key:` block-scalar body, or None."""
    for i, l in enumerate(lines):
        if re.match(rf"^\s*{key}:\s*$", l):
            j = i + 1
            while j < len(lines) and not re.match(r"^\s*-\s*\|", lines[j]):
                if re.match(r"^\s*(command|image|volumes|environment):", lines[j]):
                    j = None; break
                j += 1
            if j is None:
                continue
            ind = len(lines[j]) - len(lines[j].lstrip()) + 2
            k = j + 1
            while k < len(lines):
                if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) < ind:
                    break
                k += 1
            return j + 1, k, ind
    return None


def container_env(text, host):
    """Reproduce docker's env filtering: a var reaches the container ONLY if the
    compose declares it under `environment:`. This is not a detail — the first
    version of this gate ran the entrypoint with the host env applied directly,
    passed 27/27, and every one of those composes was in fact DEAD because the
    knob was never forwarded. A live boot caught it; this function is why the
    gate would have."""
    lines, out = text.split("\n"), {}
    for i, l in enumerate(lines):
        if not re.match(r"^\s*environment:\s*$", l):
            continue
        ind = len(l) - len(l.lstrip())
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if cur.strip() and (len(cur) - len(cur.lstrip())) <= ind:
                break
            m = re.match(r"^\s*-\s*([A-Za-z_]\w*)(?:=(.*))?$", cur)
            if m:
                name, val = m.group(1), m.group(2)
                if val is None:                       # bare `- NAME` passthrough
                    if name in host:
                        out[name] = host[name]
                else:
                    out[name] = _interp(val.strip().strip("'\""), host)
            j += 1
        break
    return out


def entrypoint_of(text):
    lines = text.split("\n")
    b = _block(lines, "entrypoint")
    if b is None:
        return None
    s, e, ind = b
    return "\n".join(l[ind:] if len(l) > ind else "" for l in lines[s:e])


def command_of(text, env):
    """`command:` as a list, or as a folded/literal scalar that docker splits."""
    lines = text.split("\n")
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)command:\s*(>-|>|\||\|-)\s*$", l)
        if not m:
            continue
        ind = len(m.group(1))
        j, buf = i + 1, []
        while j < len(lines):
            cur = lines[j]
            if cur.strip() and (len(cur) - len(cur.lstrip())) <= ind:
                break
            buf.append(cur.strip())
            j += 1
        import shlex
        return shlex.split(_interp(" ".join(buf), env))
    return _command_list(text, env)


def _command_list(text, env):
    lines, args = text.split("\n"), []
    for i, l in enumerate(lines):
        if not re.match(r"^\s*command:\s*$", l):
            continue
        ind = len(l) - len(l.lstrip())
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if not cur.strip():
                j += 1; continue
            if (len(cur) - len(cur.lstrip())) <= ind:
                break
            m = re.match(r"^\s*-\s*(.*)$", cur)
            if m:
                v = m.group(1).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                args.append(_interp(v, env))
            j += 1
        break
    return args


def argv_under(text, env):
    """Run the compose's entrypoint with `vllm` stubbed, under the env docker
    would actually give the container; return the argv string."""
    body = entrypoint_of(text)
    if body is None:
        return None, ("no block-scalar entrypoint found, so the drafter cannot be "
                      "switched off at runtime. If this compose uses a flow-style "
                      "`entrypoint: [...]`, this gate cannot execute it — convert it "
                      "to `- |` block form (every shipped vLLM compose uses that)")
    with tempfile.TemporaryDirectory() as d:
        dp = pathlib.Path(d)
        (dp / "app").mkdir(exist_ok=True)
        srv = dp / "app" / "llama-server"
        srv.write_text('#!/bin/bash\nfor a in "$@"; do printf "ARG:%s\\n" "$a"; done\nexit 0\n')
        srv.chmod(0o755)
        stub = dp / "vllm"
        stub.write_text('#!/bin/bash\nfor a in "$@"; do printf "ARG:%s\\n" "$a"; done\nexit 0\n')
        stub.chmod(0o755)
        etc = dp / "etc" / "club3090"
        etc.mkdir(parents=True, exist_ok=True)
        (etc / "detect_nvlink.sh").write_text("_NVLINK_ENABLED=0\n")
        # any `bash /etc/club3090/<x>/install.sh` the entrypoint runs
        for sub in set(re.findall(r"/etc/club3090/([\w.-]+)/install\.sh", body)):
            (etc / sub).mkdir(parents=True, exist_ok=True)
            (etc / sub / "install.sh").write_text("#!/bin/bash\nexit 0\n")
        script = dp / "ep.sh"
        script.write_text(body.replace("$$", "$")
                              .replace("/etc/club3090", str(etc))
                              .replace("/app/llama-server", str(srv)))
        # only what the compose declares crosses into the container
        e = {k: v for k, v in os.environ.items() if not k.startswith(("SPEC", "NUM_SPEC"))}
        e.update(container_env(text, env)); e["PATH"] = f"{d}:{os.environ['PATH']}"
        r = subprocess.run(["bash", str(script), "--", *command_of(text, env)],
                           capture_output=True, text=True, env=e, timeout=60)
        args = [l[4:] for l in r.stdout.split("\n") if l.startswith("ARG:")]
        return (args or None), (r.stderr.strip()[-300:] or "no argv emitted")


def spec_payload(args):
    """The --speculative-config value the engine would receive, or None."""
    if not args or FLAG not in args:
        return None
    i = args.index(FLAG)
    return args[i + 1] if i + 1 < len(args) else ""


def depth(args):
    """None = no drafter; int = the depth the engine would receive."""
    p = spec_payload(args)
    if p is None:
        return None
    m = re.search(r"num_speculative_tokens\D*(\d+)", p)
    return int(m.group(1)) if m else -1


def payload_problem(args):
    """vLLM parses this value as JSON. A payload that only LOOKS right in a regex
    is how a stray list-item prefix (`- '{...}`) survived review once — it kept a
    readable token count while being unparseable. Assert the real thing."""
    p = spec_payload(args)
    if p is None:
        return None
    try:
        obj = json.loads(p)
    except Exception as e:
        return f"--speculative-config is not valid JSON ({e}): {p!r}"
    if not isinstance(obj, dict):
        return f"--speculative-config must be a JSON object, got {type(obj).__name__}"
    if "num_speculative_tokens" not in obj:
        return f"--speculative-config has no num_speculative_tokens: {p!r}"
    if not ({"method", "model"} & set(obj)):
        return f"--speculative-config names neither a method nor a draft model: {p!r}"
    return None


def lc_drafter_args(args):
    return [a for a in (args or []) if a in LC_FLAGS]


def check_llamacpp(text, label, sink):
    """Same contract, llama.cpp lineage: SPEC_N=0 must REMOVE the drafter flags.
    Zeroing the count is not enough — llama.cpp still builds the draft context
    (and still loads an external draft model) at --spec-draft-n-max 0."""
    base, err = argv_under(text, {})
    if base is None:
        sink.append(f"{label}: {err}"); return
    if not lc_drafter_args(base):
        sink.append(f"{label}: ships drafter flags but the default resolves to none")
        return
    for env, desc in [({"SPEC_N": "0"}, "SPEC_N=0"), ({"SPEC": "off"}, "SPEC=off")]:
        a, _ = argv_under(text, env)
        left = lc_drafter_args(a)
        if left:
            sink.append(f"{label}: {desc} does not disable the drafter — "
                        f"{' '.join(sorted(set(left)))} still passed to the engine")


def check(text, label, sink):
    """Assert the full contract for one compose's text. Appends to sink."""
    base, err = argv_under(text, {})
    if base is None:
        sink.append(f"{label}: {err}"); return
    bad = payload_problem(base)
    if bad:
        sink.append(f"{label}: {bad}")
    d0 = depth(base)
    if d0 is None or d0 <= 0:
        sink.append(f"{label}: ships {FLAG} but the default resolves to no drafter")
        return
    for env, desc in [({"SPEC_N": "0"}, "SPEC_N=0"),
                      ({"SPEC": "off"}, "SPEC=off")]:
        a, _ = argv_under(text, env)
        d = depth(a)
        if d is not None:
            got = "still passes the flag with n=0" if d == 0 else f"drafter still on at n={d}"
            sink.append(f"{label}: {desc} does not disable the drafter — {got}")
    pick = 7 if d0 != 7 else 5
    a, _ = argv_under(text, {"SPEC_N": str(pick)})
    if depth(a) != pick:
        sink.append(f"{label}: SPEC_N={pick} ignored — engine would get n={depth(a)}")


# ---------------------------------------------------------------------------
# SELF-TEST — runs BEFORE the real scan. A gate that only ever passes is
# indistinguishable from a gate that does nothing. Each case is a real shape
# this repo shipped.
# ---------------------------------------------------------------------------
def _compose(entry, cmd="", env="      - SPEC_N=${SPEC_N:-}\n      - SPEC=${SPEC:-}\n"):
    return ("services:\n  x:\n    environment:\n" + env
            + "    entrypoint:\n      - bash\n      - -c\n      - |\n"
            + "".join(f"        {l}\n" for l in entry.split("\n"))
            + "      - --\n" + (f"    command:\n{cmd}" if cmd else ""))

GOOD = _compose(
    '_spec_n="$${SPEC_N:-3}"\n'
    'case "$${SPEC:-}" in off|0) _spec_n=0 ;; esac\n'
    'SPEC_ARGS=()\n'
    '[ "$$_spec_n" -gt 0 ] && SPEC_ARGS=(--speculative-config '
    '"{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":$$_spec_n}")\n'
    'exec vllm serve "$$@" "$${SPEC_ARGS[@]}"')

HARDCODED = _compose('exec vllm serve "$$@"',
                     "      - --speculative-config\n"
                     "      - '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'\n")

SPEC_ONLY = _compose(
    'SPEC_ARGS=()\n'
    'if [ "$${SPEC:-on}" != "off" ]; then\n'
    '  SPEC_ARGS=(--speculative-config '
    '"{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":3}")\n'
    'fi\n'
    'exec vllm serve "$$@" "$${SPEC_ARGS[@]}"')

SPECN_ONLY = _compose(
    'SPEC_ARGS=()\n'
    '[ "$${SPEC_N:-3}" -gt 0 ] && SPEC_ARGS=(--speculative-config '
    '"{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":$${SPEC_N:-3}}")\n'
    'exec vllm serve "$$@" "$${SPEC_ARGS[@]}"')

N_ZERO = _compose('exec vllm serve "$$@"',
                  "      - --speculative-config\n"
                  "      - '{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_N:-3}}'\n")

# entrypoint logic is perfect, but `environment:` declares nothing — the exact
# bug that shipped past the first version of this gate.
UNDECLARED = _compose(
    '_spec_n="$${SPEC_N:-3}"\n'
    'case "$${SPEC:-}" in off|0) _spec_n=0 ;; esac\n'
    'SPEC_ARGS=()\n'
    '[ "$$_spec_n" -gt 0 ] && SPEC_ARGS=(--speculative-config '
    '"{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":$$_spec_n}")\n'
    'exec vllm serve "$$@" "$${SPEC_ARGS[@]}"',
    env="      - OTHER=1\n")

MALFORMED = _compose(
    '_spec_n="$${SPEC_N:-3}"\n'
    'case "$${SPEC:-}" in off|0) _spec_n=0 ;; esac\n'
    'SPEC_ARGS=()\n'
    '[ "$$_spec_n" -gt 0 ] && SPEC_ARGS=(--speculative-config '
    '"- \'{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":$$_spec_n}")\n'
    'exec vllm serve "$$@" "$${SPEC_ARGS[@]}"')

CASES = [
    ("honours the full contract",      GOOD,       False),
    ("payload is not valid JSON",      MALFORMED,  True),
    ("knob never declared in environment:", UNDECLARED, True),
    ("hardcoded drafter, no escape",   HARDCODED,  True),
    ("SPEC only — SPEC_N=0 ignored",   SPEC_ONLY,  True),
    ("SPEC_N only — SPEC=off ignored", SPECN_ONLY, True),
    ("n=0 passthrough, flag not removed", N_ZERO,   True),
]

selffail = []
for name, text, should_flag in CASES:
    sink = []
    check(text, "<self-test>", sink)
    if bool(sink) != should_flag:
        selffail.append(f"self-test {name!r}: expected "
                        f"{'a problem' if should_flag else 'no problem'}, got {sink or 'none'}")
if selffail:
    print("test-spec-toggle-contract: FAIL (the gate itself is broken)")
    for e in selffail:
        print(f"  ⛔ {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# The real scan.
# ---------------------------------------------------------------------------
errors, checked = [], 0
for f in sorted(root.glob("models/*/*/compose/**/*.yml")):
    if "_archive" in str(f):
        continue
    text = f.read_text(encoding="utf-8")
    live = [l for l in text.split("\n") if not l.lstrip().startswith("#")]
    if any(FLAG in l for l in live):
        checked += 1
        check(text, str(f.relative_to(root)), errors)
    elif any(re.search(r"(?<![\w-])" + re.escape(fl) + r"(?![\w-])", l)
             for l in live for fl in LC_FLAGS):
        checked += 1
        check_llamacpp(text, str(f.relative_to(root)), errors)

if errors:
    print("test-spec-toggle-contract: FAIL")
    for e in errors:
        print(f"  ⛔ {e}")
    print()
    print(f"  {len(errors)} problem(s) across {checked} drafter-shipping compose(s).")
    print("  Fix: read SPEC_N (with SPEC=off as an alias) in the entrypoint and build")
    print("  the flag into a SPEC_ARGS array — see any compose in models/ for the shape.")
    sys.exit(1)

print(f"test-spec-toggle-contract: ok "
      f"(self-test {len(CASES)}/{len(CASES)} · {checked} drafter composes honour "
      f"SPEC_N=<n> / SPEC_N=0 / SPEC=off)")
PY
