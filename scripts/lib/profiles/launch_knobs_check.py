"""launch_knobs_check — the checks behind scripts/tests/test-launch-knobs.sh (#1465, phase 3a).

STDLIB ONLY. Every function takes the catalogue as a dict, so the test can hand
it a deliberately broken copy and prove each check can fail.

  check_coverage   every slug that reads a knob is described by exactly one
                   variant; the knob's `engines` list, each variant's `default` and
                   `every_consumer_contains` agree with the composes; no compose
                   reads a knob it never receives (dead read), pins it to a
                   constant, or forwards it to nothing.
  check_domains    each variant's value domain is run through the REAL compose
                   code it names (a shell fragment, a JSON splice, or a plain
                   pass-through): accepted values must pass both the catalogue and
                   the compose, rejected ones must fail the catalogue — and the
                   compose too when the variant says the compose enforces at boot.
  check_delivery   `docker compose config` with every catalogued knob set to a
                   sentinel, through BOTH delivery channels the launchers use (the
                   process environment: switch.sh / launch.sh; an --env-file:
                   gpu-mode.sh). A claimed knob must reach the environment /
                   command / entrypoint; an unclaimed one must not.
  check_empty_forwards
                   a knob forwarded as `NAME=${NAME:-}` arrives EMPTY rather than
                   absent; that is harmless only while every container-side read
                   treats empty like unset (`${NAME:-x}`), so a colon-less read
                   (`${NAME-x}`, `${NAME+x}`, `${NAME?}`) of such a knob fails.

Engine KIND (vllm / sglang / llamacpp / exllamav3) always comes from
scripts/lib/engine-kind.sh — this module never classifies an engine itself (#1282).
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:  # imported as scripts.lib.profiles.launch_knobs_check (the test) …
    from . import launch_knobs as lk
except ImportError:  # … or run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import launch_knobs as lk  # type: ignore

SENTINEL = "lkprobe_{}_7c1e"


@dataclass
class Consumer:
    slug: str
    compose: str            # repo-relative compose path
    engine_kind: str
    model: str
    text: str
    uses: dict              # knob -> lk.KnobUse (every catalogued knob)


def engine_kinds(root: Path, ids) -> dict[str, str]:
    """engine id -> kind, decided by scripts/lib/engine-kind.sh (the one classifier)."""
    ids = sorted(set(ids))
    lib = root / "scripts" / "lib" / "engine-kind.sh"
    out = subprocess.run(
        ["bash", "-c", 'source "$1"; shift; for i in "$@"; do printf "%s\\t%s\\n" "$i" '
         '"$(engine_kind_from_engine_id "$i")"; done', "_", str(lib), *ids],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    kinds = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)
    missing = set(ids) - set(kinds)
    if missing:
        raise RuntimeError(f"engine-kind.sh gave no answer for {sorted(missing)}")
    return kinds


def load_consumers(root: Path, cat: dict, registry: dict,
                   unreadable: list[str] | None = None) -> list[Consumer]:
    """One Consumer per registry slug (registry = the CURATED catalog in tests —
    a rig's gitignored local layer must not decide whether the suite is green).
    A compose the scan refuses is left out and described in `unreadable`."""
    names = lk.knob_names(cat)
    unreadable = [] if unreadable is None else unreadable
    kinds = engine_kinds(root, (e["engine"] for e in registry.values()))
    scans: dict[str, tuple[str, dict]] = {}
    out = []
    for slug, e in sorted(registry.items()):
        cp = e["compose_path"]
        if cp not in scans:
            text = (root / cp).read_text(encoding="utf-8")
            try:
                scans[cp] = (text, lk.scan_compose_text(text, names))
            except ValueError as exc:
                unreadable.append(f"{cp}: the knob scan cannot read it: {exc}")
                scans[cp] = (text, None)
        text, uses = scans[cp]
        if uses is not None:
            out.append(Consumer(slug, cp, kinds[e["engine"]], e["model"], text, uses))
    return out


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def check_coverage(cat: dict, consumers: list[Consumer]) -> list[str]:
    errs: list[str] = []
    for name, knob in sorted(cat["knobs"].items()):
        readers = [c for c in consumers if c.uses[name].consumed]
        if not readers:
            errs.append(f"{name}: no registered compose reads it — a catalogued knob no slug consumes "
                        "is a dead setting; drop it or wire a compose")
            continue
        kinds = sorted({c.engine_kind for c in readers})
        if sorted(knob["engines"]) != kinds:
            errs.append(f"{name}: engines {sorted(knob['engines'])} != the engines whose composes read it {kinds}")
        hit = [0] * len(knob["variants"])
        for c in readers:
            vs = lk.variant_for(knob, c.engine_kind, c.model, c.text)
            if len(vs) != 1:
                what = "no variant" if not vs else f"{len(vs)} variants"
                errs.append(f"{name}: {c.slug} ({c.engine_kind}, {c.model}) is covered by {what} — "
                            "each reader needs exactly one domain")
                continue
            v = vs[0]
            hit[knob["variants"].index(v)] += 1
            u = c.uses[name]
            if "default" in v:
                want = "" if v["default"] is None else v["default"]
                got = u.compose_default()
                if got != want:
                    errs.append(f"{name}: {c.slug} defaults to {got!r} in its compose, the catalogue says {want!r}")
            for needle in v.get("every_consumer_contains", []):
                if needle not in c.text:
                    errs.append(f"{name}: {c.compose} lacks {needle!r}, which this variant's domain relies on")
        for i, n in enumerate(hit):
            if n == 0:
                errs.append(f"{name}: variants[{i}] ({knob['variants'][i]['match']}) matches no reader — stale")
    seen = set()
    for c in consumers:
        if c.compose in seen:
            continue
        seen.add(c.compose)
        for name, u in sorted(c.uses.items()):
            if u.dead_read:
                errs.append(f"{name}: {c.compose} reads $${{{name}}} (lines {u.shell[:3]}) but never forwards it "
                            "(no environment: entry, no ${...} interpolation) — the container never sees the value")
            if "pinned" in u.env_forms():
                errs.append(f"{name}: {c.compose} pins it to a constant in environment: — a saved value "
                            "would silently do nothing")
            if u.dead_env:
                errs.append(f"{name}: {c.compose} forwards it but nothing in the compose reads it")
    return errs


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------

_STUBS = ("rm() { :; }; curl() { return 0; }; sleep() { :; }; mkdir() { :; }; chmod() { :; }; "
          "chown() { :; }; ln() { :; }; mv() { :; }; cp() { :; }; touch() { :; }; docker() { :; }; "
          "nvidia-smi() { return 1; }")
# The script's own `set -e` / `set -euo pipefail` line: a fragment runs under the
# same shell options as the real entrypoint (`set -u` changes what an unset knob does).
_SET_RX = re.compile(r"^\s*set\s+-[a-z]+(?:\s+(?:-o\s+)?[a-z]+)?\s*$")


def _fragment(text: str, check: dict) -> str:
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.search(check["from"], ln)), None)
    if start is None:
        raise LookupError(f"fragment start /{check['from']}/ not found")
    end = next((i for i in range(start, len(lines)) if re.search(check["to"], lines[i])), None)
    if end is None:
        raise LookupError(f"fragment end /{check['to']}/ not found after /{check['from']}/")
    end = min(len(lines) - 1, end + int(check.get("to_plus", 0)))
    body = "\n".join(lines[start:end + 1])
    prelude = next((ln.strip() for ln in lines if _SET_RX.match(ln)), "")
    return prelude + "\n" + body.replace("$$", "$")


def _run_shell(text: str, check: dict, env: dict) -> tuple[bool, str]:
    frag = _fragment(text, check)
    emit = check.get("emit")
    tail = f'\nprintf "\\n__EMIT__%s\\n" "{emit}"' if emit else ""
    script = f"{_STUBS}\n( {frag}{tail}\n)\necho \"__RC__$?\""
    penv = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/nonexistent", **env}
    p = subprocess.run(["bash", "-c", script], env=penv, capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    rc_line = [ln for ln in p.stdout.splitlines() if ln.startswith("__RC__")]
    rc = int(rc_line[-1][6:]) if rc_line else 99
    if rc != 0:
        return False, f"rc={rc}: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''}"
    if emit and check.get("json"):
        em = [ln[8:] for ln in p.stdout.splitlines() if ln.startswith("__EMIT__")]
        try:
            json.loads(em[-1])
        except (IndexError, ValueError) as exc:
            return False, f"emitted {em[-1:]!r} is not JSON ({exc})"
    return True, "rc=0"


def interpolate(s: str, env: dict) -> str:
    """Compose-style interpolation of one string: $$ -> $, ${N}, ${N:-d}, ${N-d},
    ${N:+a}, ${N+a}, ${N:?e}, ${N?e} (nested defaults allowed), $N."""
    out, i = [], 0
    while i < len(s):
        if s.startswith("$$", i):
            out.append("$")
            i += 2
            continue
        if s[i] == "$" and i + 1 < len(s) and s[i + 1] == "{":
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i + 2:])
            if not m:
                raise ValueError(f"bad interpolation at {s[i:i + 20]!r}")
            name, j = m.group(0), i + 2 + len(m.group(0))
            op = next((o for o in (":-", ":+", ":?", "-", "+", "?") if s.startswith(o, j)), None)
            j += len(op or "")
            depth, k = 0, j
            while k < len(s) and not (s[k] == "}" and depth == 0):
                depth += {"{": 1, "}": -1}.get(s[k], 0)
                k += 1
            arg, val, i = s[j:k], env.get(name), k + 1
            if op is None:
                out.append(val or "")
            elif op == ":-":
                out.append(val if val else interpolate(arg, env))
            elif op == "-":
                out.append(val if val is not None else interpolate(arg, env))
            elif op == ":+":
                out.append(interpolate(arg, env) if val else "")
            elif op == "+":
                out.append(interpolate(arg, env) if val is not None else "")
            else:
                if not val and (op == ":?" or val is None):
                    raise ValueError(f"required variable {name} is missing: {arg}")
                out.append(val)
            continue
        if s[i] == "$":
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i + 1:])
            if m:
                out.append(env.get(m.group(0)) or "")
                i += 1 + len(m.group(0))
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _yaml_scalar(item: str) -> str:
    """The value of a one-line YAML list item / scalar (`- '...'`, `- "..."`, `- x`)."""
    s = item.strip()
    if s.startswith("- "):
        s = s[2:].strip()
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return json.loads(s)
    return s


def _find_line(text: str, rx: str) -> str:
    for ln in text.splitlines():
        code = lk.strip_comment(ln)
        if code.strip() and re.search(rx, code):
            return code
    raise LookupError(f"no line matches /{rx}/")


def _run_json_line(text: str, check: dict, env: dict) -> tuple[bool, str]:
    rendered = interpolate(_yaml_scalar(_find_line(text, check["line"])), env)
    pre = check.get("strip_prefix", "")
    if pre:
        if not rendered.startswith(pre):
            return False, f"rendered line does not start with {pre!r}: {rendered[:80]!r}"
        rendered = rendered[len(pre):]
    try:
        json.loads(rendered)
    except ValueError as exc:
        return False, f"not JSON after interpolation: {rendered[:80]!r} ({exc})"
    return True, "json ok"


def compose_accepts(root: Path, knob_name: str, check: dict, env: dict) -> tuple[bool, str]:
    text = (root / check["compose"]).read_text(encoding="utf-8")
    kind = check["kind"]
    if kind == "shell":
        return _run_shell(text, check, env)
    if kind == "json_line":
        return _run_json_line(text, check, env)
    # passthrough: the compose interpolates the value into this line verbatim and
    # validates nothing — record that the line exists and really carries the knob.
    line = _find_line(text, check["line"])
    if not lk._interp_refs(knob_name, line):
        return False, f"line /{check['line']}/ does not interpolate {knob_name}"
    return True, "passed through unvalidated"


def _probe_env(knob_name: str, check: dict, probe) -> dict:
    env = dict(check.get("env") or {})
    if isinstance(probe, dict):
        env.update(probe)
    else:
        env[knob_name] = probe
    return env


def catalogue_error(cat: dict, consumer: Consumer, env: dict) -> str | None:
    """Why the catalogue refuses this set of values for this slug (None = accepted)."""
    for k, v in env.items():
        knob = cat["knobs"].get(k)
        if knob is None or not consumer.uses[k].consumed:
            continue
        vs = lk.variant_for(knob, consumer.engine_kind, consumer.model, consumer.text)
        if len(vs) != 1:
            return f"{k}: no single variant for {consumer.slug}"
        err = lk.value_error(knob, vs[0], v)
        if err:
            return f"{k}: {err}"
    deps = lk.dependency_errors(cat, env)
    return deps[0] if deps else None


def check_domains(root: Path, cat: dict, consumers: list[Consumer]) -> tuple[list[str], int]:
    errs: list[str] = []
    n = 0
    by_compose: dict[str, Consumer] = {}
    for c in consumers:
        by_compose.setdefault(c.compose, c)
    for name, knob in sorted(cat["knobs"].items()):
        cited = [("variants", v["source"]) for v in knob["variants"]]
        cited += [(sec, x["source"]) for sec in ("requires", "interacts", "caveats") for x in knob.get(sec, []) or []]
        for sec, src in cited:
            if not (root / src["file"]).is_file():
                errs.append(f"{name}.{sec}: cited source {src['file']} does not exist")
        for vi, v in enumerate(knob["variants"]):
            w = f"{name}.variants[{vi}]"
            for ev in v.get("evidence", []) or []:
                try:
                    body = (root / ev["file"]).read_text(encoding="utf-8")
                except OSError as exc:
                    errs.append(f"{w}: evidence file {ev['file']}: {exc}")
                    continue
                if ev["contains"] not in body:
                    errs.append(f"{w}: {ev['file']} no longer contains {ev['contains']!r} — the domain may have moved")
            for ci, chk in enumerate(v["checks"]):
                cw = f"{w}.checks[{ci}] ({chk['compose']})"
                c = by_compose.get(chk["compose"])
                if c is None or not c.uses[name].consumed:
                    errs.append(f"{cw}: not a registered compose that reads {name}")
                    continue
                if lk.variant_for(knob, c.engine_kind, c.model, c.text) != [v]:
                    errs.append(f"{cw}: that compose ({c.engine_kind}, {c.model}) is not covered by this variant")
                    continue
                boot = v["enforced"] == "boot"
                for label, probes in (("accept", chk["accept"]), ("reject", chk["reject"]),
                                      ("stricter", chk.get("stricter") or [])):
                    for probe in probes:
                        env = _probe_env(name, chk, probe)
                        n += 1
                        cat_err = catalogue_error(cat, c, env)
                        try:
                            ok, detail = compose_accepts(root, name, chk, env)
                        except (LookupError, ValueError, subprocess.TimeoutExpired) as exc:
                            errs.append(f"{cw}: cannot run the compose check: {exc}")
                            break
                        tag = f"{cw} {label} {probe!r}"
                        if label == "accept":
                            if cat_err:
                                errs.append(f"{tag}: the catalogue refuses it ({cat_err})")
                            if not ok:
                                errs.append(f"{tag}: the compose refuses it ({detail})")
                        else:
                            if not cat_err:
                                errs.append(f"{tag}: the catalogue accepts it")
                            if label == "reject" and boot and ok:
                                errs.append(f"{tag}: enforced=boot but the compose accepts it — "
                                            "drop the value from reject, or the compose lost its check")
                            if label == "reject" and not boot and not ok:
                                errs.append(f"{tag}: enforced={v['enforced']} but the compose refuses it "
                                            f"({detail}) — the compose validates now; say enforced=boot")
                            if label == "stricter" and not ok:
                                errs.append(f"{tag}: listed as stricter-than-the-compose, but the compose "
                                            f"refuses it too ({detail}) — move it to reject")
    return errs, n


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------

def docker_compose_available() -> bool:
    try:
        return subprocess.run(["docker", "compose", "version"], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def render(compose: Path, knob_env: dict, channel: str, tmp: Path) -> dict:
    """`docker compose config --format json` of one compose (read-only).

    channel "env": knob values in the process environment (switch.sh / launch.sh
    export them via club_config_load). channel "env-file": knob values only in an
    --env-file (gpu-mode.sh passes the resolved settings that way). Never reads a
    real .env or settings file: the env starts empty and --env-file is ours."""
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/nonexistent"),
            "MODEL_DIR": "/nonexistent-model-dir"}
    for k in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "XDG_RUNTIME_DIR"):
        if k in os.environ:
            base[k] = os.environ[k]
    fd, envfile = tempfile.mkstemp(dir=tmp, suffix="-probe.env")  # not a settings file: sentinels only
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        if channel == "env-file":
            fh.write("".join(f"{k}={v}\n" for k, v in knob_env.items()))
    penv = {**base, **knob_env} if channel == "env" else base
    p = subprocess.run(["docker", "compose", "--env-file", envfile, "-f", str(compose), "config",
                        "--format", "json"], env=penv, capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    os.unlink(envfile)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-400:])
    return json.loads(p.stdout)


def _strings(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(i) for i in x]
    return [json.dumps(x)]


def delivery_errors(rendered: dict, uses: dict, where: str) -> list[str]:
    """Compare what docker rendered with what the scan claims, per knob."""
    errs = []
    services = rendered.get("services") or {}
    for name, u in sorted(uses.items()):
        s = SENTINEL.format(name)
        env_hit = interp_hit = False
        for svc in services.values():
            env = svc.get("environment") or {}
            if isinstance(env, list):
                env = dict(e.split("=", 1) if "=" in e else (e, None) for e in env)
            if env.get(name) == s:
                env_hit = True
            other = [str(v) for k, v in env.items() if k != name and v is not None]
            if any(s in t for t in other + _strings(svc.get("command")) + _strings(svc.get("entrypoint"))):
                interp_hit = True
        if u.forwarded and not env_hit:
            errs.append(f"{where}: {name} is declared under environment: but docker did not deliver it")
        if u.interp and not interp_hit:
            errs.append(f"{where}: {name} is interpolated (lines {u.interp[:3]}) but the value reaches no "
                        "environment/command/entrypoint — it lands somewhere that never reaches the process")
        if not u.consumed and (env_hit or interp_hit):
            errs.append(f"{where}: docker delivers {name} but the scan does not claim it — the extractor missed "
                        "a consumption path")
    return errs


def check_delivery(root: Path, cat: dict, targets: list[tuple[str, dict]],
                   channels=("env", "env-file"), workers: int = 8) -> tuple[list[str], int]:
    """targets: (compose path relative to root or absolute, uses) — every compose is
    rendered with EVERY catalogued knob set, so unclaimed knobs are checked too."""
    knob_env = {n: SENTINEL.format(n) for n in lk.knob_names(cat)}
    errs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="launch-knobs-") as td:
        tmp = Path(td)

        def one(job):
            compose, uses, channel = job
            path = Path(compose) if Path(compose).is_absolute() else root / compose
            try:
                return delivery_errors(render(path, knob_env, channel, tmp), uses, f"{compose} [{channel}]")
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                return [f"{compose} [{channel}]: docker compose config failed: {exc}"]

        jobs = [(c, u, ch) for c, u in targets for ch in channels]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(one, jobs):
                errs += res
    return errs, len(jobs)


# ---------------------------------------------------------------------------
# empty forwards
# ---------------------------------------------------------------------------

def check_empty_forwards(consumers: list[Consumer]) -> tuple[list[str], dict[str, int]]:
    """Every `NAME=${NAME:-}` forward of a catalogued knob, and whether it is safe."""
    errs, counts, seen = [], {}, set()
    for c in consumers:
        if c.compose in seen:
            continue
        seen.add(c.compose)
        code = "\n".join(lk.strip_comment(ln) for ln in c.text.splitlines())
        for name, u in sorted(c.uses.items()):
            if "empty-default" not in u.env_forms():
                continue
            counts[name] = counts.get(name, 0) + 1
            risky = re.findall(r"(?<!\$)\$\$\{" + re.escape(name) + r"(?:[-+?]|\})", code)
            risky = [r for r in risky if not r.endswith("}")]  # ${NAME} alone: empty == unset
            if risky:
                errs.append(f"{name}: {c.compose} forwards it as {name}=${{{name}:-}} (EMPTY when unset) and reads "
                            f"it colon-less ({sorted(set(risky))}) — there empty and unset differ; forward it "
                            f"bare (- {name}) instead")
    return errs, counts


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--catalogue")
    ap.add_argument("--no-delivery", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    sys.path.insert(0, str(root))
    from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402

    fail = 0
    try:
        cat = lk.load_catalogue(a.catalogue)
    except lk.CatalogueError as exc:
        print(f"  ✗ catalogue: {exc}")
        return 1
    print(f"  ✓ catalogue shape ok ({len(cat['knobs'])} knobs)")
    unreadable: list[str] = []
    consumers = load_consumers(root, cat, COMPOSE_REGISTRY, unreadable)
    for label, (errs, extra) in (
        ("coverage", (unreadable + check_coverage(cat, consumers), None)),
        ("domains", check_domains(root, cat, consumers)),
        ("empty forwards", check_empty_forwards(consumers)),
    ):
        if errs:
            fail = 1
            for e in errs:
                print(f"  ✗ {label}: {e}")
        else:
            print(f"  ✓ {label} ok" + (f" ({extra} probes)" if isinstance(extra, int) else ""))
    if not a.no_delivery:
        if not docker_compose_available():
            print("  - delivery: SKIP (docker compose unavailable)")
        else:
            # EVERY registered compose, not only the ones the scan found a reference
            # in: an unclaimed knob that docker nevertheless delivers is a scan miss.
            targets, seen = [], set()
            for c in consumers:
                if c.compose not in seen:
                    seen.add(c.compose)
                    targets.append((c.compose, c.uses))
            errs, n = check_delivery(root, cat, targets)
            for e in errs:
                print(f"  ✗ delivery: {e}")
            fail |= bool(errs)
            if not errs:
                print(f"  ✓ delivery ok ({n} renders: {len(targets)} composes x process env + --env-file)")
    return fail


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(main(sys.argv[1:]))
