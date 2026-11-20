#!/usr/bin/env python3
"""Does the pinned engine READ every key a launch setting writes into its config? (gate: test-engine-config-reads)

Why this exists: #1723 wrote Strata's MAX_TOKENS into the run config as `max_tokens`. Strata reads that key only
in its vision encoder class, so the setting did nothing, and every gate passed: test-launch-knobs proves a value
reaches the container, not that the engine uses it. Measured 2026-10-17: MAX_TOKENS=64, and a request with no
max_tokens of its own ran 1,514 tokens.

The manifest (engine-config-reads.json) has one entry per engine whose launcher writes the engine's own config.

  check [--engine E]  no docker (also run by test-strata-config-knobs / test-maya-config-knobs, the tests a
                   launcher edit runs). Runs the real launcher code with no settings, then once per setting, and diffs the
                   configs (args compared as flag -> value). Fails when a setting writes a key the entry has no
                   evidence for; when the entry has evidence for a key no setting writes; when a setting the
                   program reads writes nothing and isn't declared setup-only; when the entry's verified image
                   isn't the engine profile's pin (a bump needs `verify --record`); or when a MOUNTED_READERS
                   program (launch_knobs.py) is neither covered by an entry nor exempted.
  verify ENGINE    reads the server source and the engine's --help out of the pinned image (docker, --network
                   none) and checks every evidence key:
                     key            read from the config object (`cfg.get("key")` / `cfg["key"]`) inside one of
                                    the entry's config_scopes, a short APPROVED list: a key read only elsewhere
                                    (MAX_TOKENS: only in Vision.__init__) fails, and the message says where it is
                                    read, so a pin bump that moves a read fails until someone looks;
                     sampling.x     a config scope reads cfg "sampling" and compares a field name against "x";
                     args:--flag    --help lists the flag on a boundary that excludes "-" (--spec-min-p is not
                                    --spec).
  verify ENGINE --record           the same, then stores the profile's pin as the verified image (after a bump;
                                   refuses when the image isn't local)
  selftest         the checks against synthetic sources and help text: positive and negative controls
  format           rewrite the manifest in its canonical layout (check requires it)
  engines / pin ENGINE             for the test script

Not covered:
  - values (Maya's `--kv q4_0` is in --help, and its GLM path runs it as fp16: the launcher refuses it);
  - settings the entrypoint hands to the engine's setup rather than writing into the config (Strata's CONTEXT /
    KV -> setup.py), which an entry declares as setup_only;
  - keys a launcher writes whatever the settings say: only what a setting CHANGES is diffed. Strata's launcher
    only overlays settings on setup's output, so that is all of it; Maya's writes the whole maya.json, and its
    fixed keys (exe, port, installer, ...) are not checked;
  - a setting the program reads other than through env("X") / _env("X") (os.environ directly, or a value the
    shell entrypoint passes as an argument, like Maya's CONTEXT, which its entry lists by hand): the coverage
    check can't see it, so a new one goes unnoticed until someone adds it to `settings`.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).resolve().with_name("engine-config-reads.json")
ENV_READ = re.compile(r"\b_?env\(\s*\"([A-Z][A-Z0-9_]*)\"")   # env("X") (strata_config.py), _env("X") (maya_settings.py)
MISSING = object()


class Fail(Exception):
    pass


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(manifest: dict) -> str:
    """The one layout the file is kept in, so a `verify --record` diff is the one line it changed."""
    return json.dumps(manifest, indent=1, ensure_ascii=True) + "\n"


def write_manifest(manifest: dict) -> None:
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(canonical(manifest), encoding="utf-8")
    os.replace(tmp, MANIFEST)


def profile_pin(entry: dict) -> str:
    """install.spec of the engine profile (the image the launchers run)."""
    text = (ROOT / entry["engine_profile"]).read_text(encoding="utf-8")
    m = re.search(r"^install:\s*\n(?:[ \t]+.*\n|\s*\n)*?[ \t]+spec:\s*(\S+)", text, re.M)
    if not m:
        raise Fail(f"{entry['engine_profile']}: no install.spec")
    return m.group(1).strip("'\"")


# ---- the launchers, run for real -------------------------------------------------------------------------

def _env_for(entry: dict, settings: dict) -> dict:
    env = dict(os.environ)
    for name in set(entry["settings"]) | set(entry.get("setup_only", {})) | program_settings(entry):
        env.pop(name, None)
    env.update(settings)
    env["PYTHONDONTWRITEBYTECODE"] = "1"      # the launcher folders are mounted into the containers
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run(cmd: list[str], env: dict, what: str) -> None:
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise Fail(f"{what} failed (rc={r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")


def render_strata_config(entry: dict, settings: dict) -> dict:
    prog = ROOT / entry["programs"][0]
    with tempfile.TemporaryDirectory() as d:
        setup, cfg = Path(d) / "setup.json", Path(d) / "cfg.json"
        setup.write_text(json.dumps(entry["fixture"]), encoding="utf-8")
        _run([sys.executable, str(prog), "apply", str(setup), str(cfg)], _env_for(entry, settings),
             f"{prog.name} apply {settings or '(no settings)'}")
        return json.loads(cfg.read_text(encoding="utf-8"))


MAYA_RENDER = r"""
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import maya_prepare
d = Path(sys.argv[2])
maya_prepare.config("maya-s", d, d / "pack", types.SimpleNamespace(gpus="0,1", context=int(sys.argv[3])))
"""


def render_maya_prepare(entry: dict, settings: dict) -> dict:
    launcher = ROOT / Path(entry["programs"][-1]).parent
    settings = dict(settings)
    context = settings.pop("CONTEXT", "131072")   # the entrypoint passes CONTEXT as maya_prepare.py --context
    with tempfile.TemporaryDirectory() as d:
        _run([sys.executable, "-c", MAYA_RENDER, str(launcher), d, context], _env_for(entry, settings),
             f"maya_prepare.config() {settings or '(no settings)'}")
        # the paths config() writes (pack, log, tokenizer) are the scratch folder's: give every render the
        # container's own, so only what a setting changed differs between two renders
        text = (Path(d) / "maya.json").read_text(encoding="utf-8").replace(json.dumps(d)[1:-1], "/data/maya-s")
        return json.loads(text)


MAYA_DS4_RENDER = r"""
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ds4_prepare
d = Path(sys.argv[2])
ds4_prepare.config("ista-iq2-xxs", d, d / "pack", types.SimpleNamespace(gpus="0,1", tp=1, context=int(sys.argv[3])))
"""


def render_maya_ds4_prepare(entry: dict, settings: dict) -> dict:
    """DeepSeek-V4.1-Flash on Maya's maya-ds4 (models/deepseek-v4.1-flash-1017/strata-maya/launcher/ds4_prepare.py):
    the same shape as render_maya_prepare, its own config file (maya-ds4.json) and quant folder."""
    launcher = ROOT / Path(entry["programs"][-1]).parent
    settings = dict(settings)
    context = settings.pop("CONTEXT", "65536")    # the entrypoint passes CONTEXT as ds4_prepare.py --context
    with tempfile.TemporaryDirectory() as d:
        _run([sys.executable, "-c", MAYA_DS4_RENDER, str(launcher), d, context], _env_for(entry, settings),
             f"ds4_prepare.config() {settings or '(no settings)'}")
        text = (Path(d) / "maya-ds4.json").read_text(encoding="utf-8").replace(json.dumps(d)[1:-1], "/data/ista-iq2-xxs")
        return json.loads(text)


MAYA_MM3_RENDER = r"""
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mm3_prepare
d = Path(sys.argv[2])
mm3_prepare.config("minimax-maya-s", d, d / "pack", types.SimpleNamespace(gpus="0,1", context=int(sys.argv[3])))
"""


def render_maya_mm3_prepare(entry: dict, settings: dict) -> dict:
    """MiniMax-M3.1-Flash on Maya's maya-mm3 (models/minimax-m3.1-flash/strata-maya/launcher/mm3_prepare.py): the
    same shape as render_maya_ds4_prepare, its own config file (maya-mm3.json) and quant folder."""
    launcher = ROOT / Path(entry["programs"][-1]).parent
    settings = dict(settings)
    context = settings.pop("CONTEXT", "204800")   # the entrypoint passes CONTEXT as mm3_prepare.py --context
    with tempfile.TemporaryDirectory() as d:
        _run([sys.executable, "-c", MAYA_MM3_RENDER, str(launcher), d, context], _env_for(entry, settings),
             f"mm3_prepare.config() {settings or '(no settings)'}")
        text = (Path(d) / "maya-mm3.json").read_text(encoding="utf-8").replace(json.dumps(d)[1:-1], "/data/minimax-maya-s")
        return json.loads(text)


MAYA_HY4_RENDER = r"""
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hy4_prepare
d = Path(sys.argv[2])
hy4_prepare.config("hy4-maya-l", d, d / "pack", types.SimpleNamespace(gpus="0,1", context=int(sys.argv[3])))
"""


def render_maya_hy4_prepare(entry: dict, settings: dict) -> dict:
    """Hy4-Flash on Maya's maya-hy4 (models/hy4-flash/strata-maya/launcher/hy4_prepare.py): the same shape as
    render_maya_mm3_prepare, its own config file (maya-hy4.json) and quant folder."""
    launcher = ROOT / Path(entry["programs"][-1]).parent
    settings = dict(settings)
    context = settings.pop("CONTEXT", "262144")   # the entrypoint passes CONTEXT as hy4_prepare.py --context
    with tempfile.TemporaryDirectory() as d:
        _run([sys.executable, "-c", MAYA_HY4_RENDER, str(launcher), d, context], _env_for(entry, settings),
             f"hy4_prepare.config() {settings or '(no settings)'}")
        text = (Path(d) / "maya-hy4.json").read_text(encoding="utf-8").replace(json.dumps(d)[1:-1], "/data/hy4-maya-l")
        return json.loads(text)


RENDERERS = {"strata_config": render_strata_config, "maya_prepare": render_maya_prepare,
             "maya_ds4_prepare": render_maya_ds4_prepare, "maya_mm3_prepare": render_maya_mm3_prepare,
             "maya_hy4_prepare": render_maya_hy4_prepare}


def arg_pairs(args) -> dict:
    out, i = {}, 0
    args = [str(a) for a in (args or [])]
    while i < len(args):
        if args[i].startswith("--"):
            val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else None
            out[args[i]] = val
            i += 2 if val is not None else 1
        else:
            i += 1
    return out


def flatten(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if k == "args":
            out.update({f"args:{f}": val for f, val in arg_pairs(v).items()})
        elif k == "sampling" and isinstance(v, dict):
            out.update({f"sampling.{s}": sv for s, sv in v.items()})
        else:
            out[k] = v
    return out


def written(base: dict, cfg: dict) -> set[str]:
    a, b = flatten(base), flatten(cfg)
    return {p for p in set(a) | set(b) if a.get(p, MISSING) != b.get(p, MISSING)}


def program_settings(entry: dict) -> set[str]:
    names = set()
    for p in entry["programs"]:
        names |= set(ENV_READ.findall((ROOT / p).read_text(encoding="utf-8")))
    return names


def check(manifest: dict, only: str | None = None) -> list[str]:
    problems = []
    if only is None and MANIFEST.read_text(encoding="utf-8") != canonical(manifest):
        problems.append(f"{MANIFEST.name} isn't in its canonical layout. Fix: python3 scripts/lib/profiles/engine_config_reads.py format")
    sys.path.insert(0, str(ROOT / "scripts/lib/profiles"))
    import launch_knobs   # noqa: E402 — the knob scanner's MOUNTED_READERS
    covered = {p for e in manifest["engines"].values() for p in e["programs"]} | set(manifest.get("not_config_writers", {}))
    for reader in launch_knobs.MOUNTED_READERS if only is None else ():
        if reader["file"] not in covered:
            problems.append(f"{reader['file']} (a MOUNTED_READERS program) has no engine-config-reads entry: add one, or "
                            f"list it under not_config_writers with the reason it writes no engine config")
    for name, e in manifest["engines"].items():
        if only is not None and name != only:
            continue
        try:
            pin = profile_pin(e)
        except Fail as exc:
            problems.append(f"{name}: {exc}"); continue
        if e["verified_image"] != pin:
            problems.append(f"{name}: evidence was verified on {e['verified_image']}, but {e['engine_profile']} pins {pin}. "
                            f"Fix: docker pull {pin} && python3 scripts/lib/profiles/engine_config_reads.py verify {name} --record")
        render = RENDERERS.get(e["render"])
        if render is None:
            problems.append(f"{name}: unknown render {e['render']!r} (one of {', '.join(RENDERERS)})"); continue
        unmapped = program_settings(e) - set(e["settings"]) - set(e.get("setup_only", {}))
        for n in sorted(unmapped):
            problems.append(f"{name}: {n} is read by {', '.join(Path(p).name for p in e['programs'])} but the entry gives "
                            f"it no probe value (settings) and doesn't declare it setup_only")
        try:
            base = render(e, {})
        except Fail as exc:
            problems.append(f"{name}: {exc}"); continue
        seen = set()
        for setting, value in e["settings"].items():
            try:
                keys = written(base, render(e, {setting: value}))
            except Fail as exc:
                problems.append(f"{name}: {exc}"); continue
            if not keys:
                problems.append(f"{name}: {setting}={value} writes nothing into the config: the launcher drops it, or "
                                f"the probe value equals the default (if setup owns it, declare it setup_only)")
            for k in sorted(keys - set(e["reads"])):
                problems.append(f"{name}: {setting}={value} writes {k}, and the entry has no evidence that the engine reads "
                                f"it. Find where the pinned server reads it (from the config, in a config scope) and add it "
                                f"to reads, or stop writing it")
            seen |= keys
        for k in sorted(set(e["reads"]) - seen):
            problems.append(f"{name}: reads has evidence for {k}, which no setting writes any more: remove the entry")
    return problems


# ---- the evidence, read out of the image ------------------------------------------------------------------

def scan(src: str):
    """(reads, compared): reads = [(key, qualname, receiver, line)] for every `<x>.get("k")` / `<x>["k"]`;
    compared = {qualname: string literals in its comparisons} (direct body only, not nested defs)."""
    tree = ast.parse(src)
    reads, compared = [], {}

    def visit(node, scope):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(ch, scope + [ch.name])
                continue
            q = ".".join(scope) or "<module>"
            if isinstance(ch, ast.Call) and isinstance(ch.func, ast.Attribute) and ch.func.attr == "get" and ch.args \
                    and isinstance(ch.args[0], ast.Constant) and isinstance(ch.args[0].value, str):
                reads.append((ch.args[0].value, q, ast.unparse(ch.func.value), ch.lineno))
            elif isinstance(ch, ast.Subscript) and isinstance(ch.ctx, ast.Load) and isinstance(ch.slice, ast.Constant) \
                    and isinstance(ch.slice.value, str):
                reads.append((ch.slice.value, q, ast.unparse(ch.value), ch.lineno))
            elif isinstance(ch, ast.Compare):
                for side in [ch.left, *ch.comparators]:
                    for c in ([side] if not isinstance(side, (ast.Tuple, ast.List, ast.Set)) else side.elts):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            compared.setdefault(q, set()).add(c.value)
            visit(ch, scope)

    visit(tree, [])
    return reads, compared


def help_has(help_text: str, flag: str) -> bool:
    # one flag a line ("  --kv fp16|int8  ..."), or an option of a usage synopsis ("[--max-context N]": maya-ds4
    # prints only that); a flag merely named in a description doesn't count
    return re.search(rf"(?:^\s*|\[){re.escape(flag)}(?=[\s=,\]|]|$)", help_text, re.M) is not None


def _where(reads, key, limit=6) -> str:
    sites = [f"{q} (from {r}, line {ln})" for k, q, r, ln in reads if k == key]
    if not sites:
        return "not read anywhere"
    return "read in " + "; ".join(sites[:limit]) + (f"; and {len(sites) - limit} more" if len(sites) > limit else "")


def server_files(entry: dict) -> list[str]:
    """The entry's server source: one path, or a list (Strata 0.2.17 split server.py into serve/server_parts/, and
    the sampling block is read in server_parts/http_api.py)."""
    s = entry["server"]
    return [s] if isinstance(s, str) else list(s)


def verify_sources(entry: dict, src: str | list[str], help_text: str) -> list[str]:
    problems = []
    reads, compared = [], {}
    for one in ([src] if isinstance(src, str) else src):   # each file parsed on its own, the evidence merged
        r, c = scan(one)
        reads += r
        for q, lits in c.items():
            compared.setdefault(q, set()).update(lits)
    scopes, recv = set(entry["config_scopes"]), set(entry["config_receivers"])
    in_cfg = {(k, q) for k, q, r, _ in reads if r in recv}
    for key in sorted(entry["reads"]):
        if key.startswith("args:"):
            flag = key[5:]
            if not help_has(help_text, flag):
                near = sorted(set(re.findall(rf"(?:^\s*|\[)({re.escape(flag)}[-\w]*)", help_text, re.M)))
                problems.append(f"{key}: the engine's --help doesn't list {flag}" + (f" (only {', '.join(near)})" if near else ""))
        elif key.startswith("sampling."):
            field = key.split(".", 1)[1]
            ok = [q for q in scopes if ("sampling", q) in in_cfg and field in compared.get(q, set())]
            if not ok:
                problems.append(f"{key}: no config scope ({', '.join(sorted(scopes))}) both reads cfg \"sampling\" and "
                                f"compares a field against {field!r}; \"sampling\" is {_where(reads, 'sampling')}")
        elif not any((key, q) in in_cfg for q in scopes):
            problems.append(f"{key}: not read from the config ({', '.join(sorted(recv))}) in a config scope "
                            f"({', '.join(sorted(scopes))}); it is {_where(reads, key)}")
    return problems


def docker_out(args: list[str]) -> tuple[str, int]:
    r = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout + r.stderr, r.returncode


def image_local(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def verify(manifest: dict, name: str, record: bool) -> int:
    e = manifest["engines"][name]
    image = profile_pin(e)
    if not image_local(image):
        print(f"[engine-config-reads] {name}: {image} is not local. Fix: docker pull {image}", file=sys.stderr)
        return 2
    srcs = []
    for path in server_files(e):
        src, rc = docker_out(["run", "--rm", "--network", "none", "--entrypoint", "cat", image, path])
        if rc != 0:
            print(f"[engine-config-reads] {name}: can't read {path} from {image}: {src.strip()[-300:]}", file=sys.stderr)
            return 1
        srcs.append(src)
    help_text, _ = docker_out(["run", "--rm", "--network", "none", "--entrypoint", e["engine_help"][0], image,
                               *e["engine_help"][1:]])
    if not help_text.strip():
        print(f"[engine-config-reads] {name}: {' '.join(e['engine_help'])} printed nothing", file=sys.stderr)
        return 1
    try:
        problems = verify_sources(e, srcs, help_text)
    except SyntaxError as exc:
        print(f"[engine-config-reads] {name}: {', '.join(server_files(e))} doesn't parse with this python ({exc}); the check can't "
              f"run, which is a failure, not a skip", file=sys.stderr)
        return 1
    for p in problems:
        print(f"  ✗ {name}: {p}")
    if problems:
        return 1
    print(f"  ✓ {name}: the pinned image reads all {len(e['reads'])} keys ({image.split('@')[-1][:19]})")
    if record and e["verified_image"] != image:
        e["verified_image"] = image
        write_manifest(manifest)
        print(f"  recorded {name}: verified_image = {image}")
    return 0


# ---- controls ---------------------------------------------------------------------------------------------

SELFTEST_SRC = '''
class Vision:
    def __init__(self, cfg):
        if cfg.get("max_tokens"):
            pass

def sampling_defaults_from_config(cfg):
    for key, value in (cfg.get("sampling") or {}).items():
        if key == "temperature":
            pass

def main():
    cfg = {}
    effort = cfg.get("reasoning_effort")
    budget = req.get("thinking_budget")
'''
SELFTEST_HELP = """usage
  --kv fp16|int8       KV storage
  --spec-min-p P       not the verify window
  --ngram N            drafts like --draft-depth does
prog --pack DIR [--max-context N] [--threads-cap N] [--vision on|off]
"""


def selftest() -> int:
    entry = {"config_scopes": ["main", "sampling_defaults_from_config"], "config_receivers": ["cfg"]}
    cases = [  # (reads, must fail?, what)
        ({"reasoning_effort": ""}, False, "a key read from cfg in main passes"),
        ({"max_tokens": ""}, True, "a key read from cfg only in Vision.__init__ fails"),
        ({"thinking_budget": ""}, True, "a key read in main but from req, not cfg, fails"),
        ({"sampling.temperature": ""}, False, "a sampling field the config scope compares passes"),
        ({"sampling.top_k": ""}, True, "a sampling field nothing compares fails"),
        ({"args:--kv": ""}, False, "a flag --help lists passes"),
        ({"args:--spec": ""}, True, "--spec fails when --help lists only --spec-min-p"),
        ({"args:--max-context": ""}, False, "a usage synopsis's bracketed option ([--max-context N]) passes"),
        ({"args:--vision": ""}, False, "a bracketed option with choices ([--vision on|off]) passes"),
        ({"args:--threads": ""}, True, "--threads fails when the synopsis has only [--threads-cap N]"),
        ({"args:--draft-depth": ""}, True, "a flag only named inside a description fails"),
    ]
    bad = 0
    for reads, must_fail, what in cases:
        problems = verify_sources({**entry, "reads": reads}, SELFTEST_SRC, SELFTEST_HELP)
        good = bool(problems) == must_fail
        print(f"  {'✓' if good else '✗'} selftest: {what}" + (f" ({problems[0]})" if problems and good else ""))
        bad += not good
    # a server split over files (Strata 0.2.17: sampling_defaults_from_config moved to server_parts/http_api.py)
    head, rest = SELFTEST_SRC.split("def sampling_defaults_from_config")
    samp_src, tail = ("def sampling_defaults_from_config" + rest).split("\ndef main")
    main_src = head + "\ndef main" + tail                 # Vision and main(), without the sampling scope
    for srcs, must_fail, what in [([main_src, samp_src], False, "a sampling scope in the second server file passes"),
                                  ([main_src], True, "the same field with only the first server file fails")]:
        problems = verify_sources({**entry, "reads": {"sampling.temperature": ""}}, srcs, SELFTEST_HELP)
        good = bool(problems) == must_fail
        print(f"  {'✓' if good else '✗'} selftest: {what}" + (f" ({problems[0]})" if problems and good else ""))
        bad += not good
    loc = verify_sources({**entry, "reads": {"max_tokens": ""}}, SELFTEST_SRC, SELFTEST_HELP)
    good = bool(loc) and "Vision.__init__" in loc[0]
    print(f"  {'✓' if good else '✗'} selftest: the failure says where the key IS read")
    bad += not good
    good = written({"args": ["--spec", "4"]}, {"args": ["--spec", "3"]}) == {"args:--spec"}
    print(f"  {'✓' if good else '✗'} selftest: a changed flag value (--spec 4 -> 3) counts as written")
    bad += not good
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    cmd = argv[1:2]
    manifest = load_manifest()
    if cmd == ["check"]:
        only = argv[3] if argv[2:3] == ["--engine"] and len(argv) == 4 else None
        if len(argv) > 2 and (only is None or only not in manifest["engines"]):
            print(f"usage: engine_config_reads.py check [--engine {'|'.join(manifest['engines'])}]", file=sys.stderr)
            return 64
        problems = check(manifest, only)
        for p in problems:
            print(f"  ✗ {p}")
        if not problems:
            engines = [n for n in manifest["engines"] if only in (None, n)]
            n = sum(len(manifest["engines"][e]["reads"]) for e in engines)
            print(f"  ✓ check: {', '.join(engines)}: {n} keys the settings write, each with evidence; pin matches the verified image")
        return 1 if problems else 0
    if cmd == ["verify"] and len(argv) >= 3 and argv[2] in manifest["engines"]:
        # an engine whose image carries a second launcher's engine (strata-maya: maya-ds4, entry "strata-maya:ds4")
        # has one entry per launcher: verifying the engine verifies them all, so a pin bump re-records every one
        names = [argv[2]] + [n for n in manifest["engines"] if ":" not in argv[2] and n.startswith(argv[2] + ":")]
        return max(verify(manifest, n, "--record" in argv[3:]) for n in names)
    if cmd == ["selftest"]:
        return selftest()
    if cmd == ["format"]:
        write_manifest(manifest)
        return 0
    if cmd == ["engines"]:
        print("\n".join(manifest["engines"]))
        return 0
    if cmd == ["pin"] and len(argv) == 3 and argv[2] in manifest["engines"]:
        print(profile_pin(manifest["engines"][argv[2]]))
        return 0
    print("usage: engine_config_reads.py check [--engine E] | verify ENGINE [--record] | selftest | format | engines | pin ENGINE",
          file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
