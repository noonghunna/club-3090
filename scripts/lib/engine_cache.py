#!/usr/bin/env python3
"""engine_cache.py — the host directories behind a compose's compile caches and its
KV-offload disk tier (club-3090#1466, phases 4a and 4b).

WHY THIS EXISTS
---------------
The vLLM composes bind-mounted their torch.compile and Triton caches from
``models/<model>/<engine>/cache/`` inside the checkout, and the KV-offload disk tier
from the repo's ``kv-offload/``. So every worktree compiled from cold, the same
kernels were stored once per model and again per checkout (22 GiB in one checkout
on the reference rig), and files the container wrote as root sat in the repo tree,
some of them impossible to delete without sudo.

The launchers now put them in per-user directories (club_config.cache_dir/data_dir):

    compile caches   <cache dir>/<image key>/<subdir>
                     cache dir = $CLUB3090_CACHE_DIR, else ${XDG_CACHE_HOME:-~/.cache}/club-3090
    KV disk tier     <data dir>/kv-offload            (an explicit KV_OFFLOAD_DIR still wins)
                     data dir  = $CLUB3090_DATA_DIR,  else ${XDG_DATA_HOME:-~/.local/share}/club-3090

THE CONTRACT WITH A COMPOSE
---------------------------
    - ${CLUB3090_ENGINE_CACHE_DIR:-../../../cache}/triton:/root/.triton/cache
    - ${KV_OFFLOAD_DIR:-${CLUB3090_DATA_DIR:-../../../../../..}/kv-offload}:/kv-offload

A launcher sets the two variables. With neither set — a plain ``docker compose up`` —
the defaults are the old in-repo paths, so that keeps working exactly as before.
CLUB3090_ENGINE_CACHE_DIR is recomputed on every launch; to move the caches, set
CLUB3090_CACHE_DIR instead. test-compose-cache-ownership.sh holds the composes to
this shape.

THE KEY
-------
``<image reference, sanitised>-<first 12 hex digits of the local image ID>``, e.g.
``vllm-vllm-openai-v0.30.0-8a69ffad015f``. Two checkouts, or two models, running the
same image share one directory: the caches key their own entries by config and kernel
hash inside it, which is how the composes of one model already shared a directory.
The ID, not just the tag: a re-pulled tag (``gemma4-unified`` moves) is different
engine code and must not be handed another build's compiled graphs. The reference is
there for people reading ``settings.sh caches``.

The image is whatever ``docker compose config`` renders in the launch's own
environment, so an engine-profile image a launcher injects (VLLM_IMAGE) and a compose
default both resolve exactly as they will at ``up``. An image that isn't on the
machine yet is pulled first — ``up`` would pull it anyway — so its ID is known.

WHY THE LAUNCHER CREATES THE DIRECTORIES
----------------------------------------
Docker creates a missing bind-mount source itself, as root:root 0755, and under $HOME
that is a directory the user can't clean up without sudo. So prepare() creates every
directory a compose mounts, as the calling user, before compose runs. The containers
still run as root; ``user: "0:${DOCKER_GID:-1000}"`` gives their files the user's group.

prepare() never fails a launch. When something is missing (docker compose v2, the
image, a writable directory) it says why and returns nothing, and the compose uses its
in-repo default for that launch.

USER-FACING: ``bash scripts/settings.sh caches [--remove-legacy]`` (caches_main) shows
where everything is and how big, and removes the old in-repo caches after asking.

Standard library only: the launcher path runs on a bare python3.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib import club_config  # noqa: E402

CACHE_VAR = "CLUB3090_ENGINE_CACHE_DIR"
DATA_VAR = "CLUB3090_DATA_DIR"
OUTPUT_VARS = (CACHE_VAR, DATA_VAR)
# Absolute stand-ins rendered in place of the two variables, so the rendered compose
# shows which mounts hang off them (docker compose leaves absolute paths alone).
SENTINELS = {CACHE_VAR: "/__club3090_engine_cache__", DATA_VAR: "/__club3090_data__"}
_MARKER_RE = re.compile(r"\$\{(?:CLUB3090_ENGINE_CACHE_DIR|CLUB3090_DATA_DIR)\b")
_FALLBACK_RE = re.compile(r"\$\{CLUB3090_ENGINE_CACHE_DIR:-([^}]*)\}")
_NOOP_BINS = {":", "true"}            # COMPOSE_BIN=: — the tests' "never launch" switch
LEGACY_KEEP = {".gitignore", "README.md"}
SETTINGS_CMD = "bash scripts/settings.sh"


class PrepareError(Exception):
    pass


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit in ("B", "KiB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# ── the key ──────────────────────────────────────────────────────────────────
def image_key(ref: str, image_id: str) -> str:
    """``vllm/vllm-openai:v0.30.0`` + ``sha256:8a69ffad015f…`` → ``vllm-vllm-openai-v0.30.0-8a69ffad015f``."""
    hexid = image_id.strip().split(":", 1)[-1].lower()
    if not re.fullmatch(r"[0-9a-f]{12,}", hexid):
        raise PrepareError(f"unexpected image ID {image_id!r}")
    name = ref.strip()
    for prefix in ("docker.io/library/", "docker.io/", "index.docker.io/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.split("@", 1)[0]                      # a digest ref: the ID already says it
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:80].strip("-.") or "image"
    return f"{name}-{hexid[:12]}"


def image_id(ref: str, docker_cmd, pull: bool = True):
    """The local image ID of `ref`, pulling it first when it's missing and `pull`. None
    when there is no such image (or no docker)."""
    def inspect():
        try:
            p = subprocess.run([*docker_cmd, "image", "inspect", "--format", "{{.Id}}", ref],
                               capture_output=True, encoding="utf-8", errors="replace", timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        out = p.stdout.strip().splitlines()
        return out[0].strip() if p.returncode == 0 and out else None

    iid = inspect()
    if iid is None and pull:
        _say(f"[cache] pulling {ref} now (compose would pull it at 'up'), so its compile cache can be keyed by the image ID")
        try:
            # progress to stderr: stdout carries only the KEY=VALUE result
            rc = subprocess.run([*docker_cmd, "pull", ref], stdout=sys.stderr, stderr=sys.stderr).returncode
        except (OSError, subprocess.SubprocessError):
            rc = 1
        if rc == 0:
            iid = inspect()
    return iid


# ── reading a compose ────────────────────────────────────────────────────────
def mounts_shared_dirs(compose_text: str) -> bool:
    return bool(_MARKER_RE.search(compose_text))


def render(compose_files, compose_cmd, env, env_file=None, cwd=None) -> dict:
    """``docker compose [--env-file F] -f … config --format json`` in `env`. Read-only:
    nothing is started, and docker compose doesn't even need the daemon for it."""
    cmd = [*compose_cmd]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    for f in compose_files:
        cmd += ["-f", str(f)]
    cmd += ["config", "--format", "json"]
    try:
        p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, encoding="utf-8",
                           errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise PrepareError(f"`{' '.join(cmd[:2])} config` could not run ({e})") from None
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        raise PrepareError(f"`{' '.join(cmd[:2])} config` failed: {tail[-1] if tail else 'exit %d' % p.returncode}")
    try:
        return json.loads(p.stdout)
    except ValueError:
        raise PrepareError(f"`{' '.join(cmd[:2])} config --format json` did not print JSON "
                           "(docker compose v2 is needed for the shared caches)") from None


def scan(cfg: dict):
    """The mounts under the two variables in a rendered compose:
    ({image: {cache subdir, …}}, {data subdir, …})."""
    cache: dict[str, set[str]] = {}
    data: set[str] = set()
    for svc in (cfg.get("services") or {}).values():
        for v in svc.get("volumes") or []:
            if not isinstance(v, dict) or v.get("type", "bind") != "bind":
                continue
            src = str(v.get("source") or "")
            for var, sentinel in SENTINELS.items():
                if src != sentinel and not src.startswith(sentinel + "/"):
                    continue
                rel = src[len(sentinel):].strip("/")
                if ".." in Path(rel).parts:
                    raise PrepareError(f"mount source {src!r} climbs out of ${{{var}}}")
                if var == CACHE_VAR:
                    cache.setdefault(str(svc.get("image") or ""), set()).add(rel)
                else:
                    data.add(rel)
    return cache, data


# ── the directories ──────────────────────────────────────────────────────────
def _absolute(p: Path, var: str) -> Path:
    p = Path(os.path.expanduser(str(p)))
    if not p.is_absolute():
        raise PrepareError(f"{var} must be an absolute path (got {str(p)!r})")
    if "\n" in str(p):
        raise PrepareError(f"{var} contains a newline")
    return p


def _make(paths) -> None:
    """mkdir -p each, as us — before docker gets the chance to create them as root."""
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PrepareError(f"couldn't create {p}: {e.strerror or e}") from None
        if not p.is_dir():
            raise PrepareError(f"{p} exists and is not a directory")


def _foreign_owner_notes(paths) -> list[str]:
    me = os.geteuid()
    if me == 0:
        return []
    notes = []
    for p in paths:
        try:
            uid = p.stat().st_uid
        except OSError:
            continue
        if uid != me:
            notes.append(f"[cache] {p} belongs to uid {uid}, not you: docker created it before a launcher "
                         f"could, so you can't clean up inside it. Fix once: sudo chown -R \"$(id -u):$(id -g)\" '{p}'")
    return notes


def _rel_to(path: Path, root) -> str:
    try:
        return str(path.relative_to(root)) if root else str(path)
    except ValueError:
        return str(path)


def _legacy_note(first: Path, text: str, root) -> str | None:
    """The old in-repo cache dir this compose used to mount, when it still holds a cache."""
    m = _FALLBACK_RE.search(text)
    if not m:
        return None
    old = Path(os.path.normpath(first.parent / m.group(1)))
    try:
        leftovers = [e for e in old.iterdir() if e.name not in LEGACY_KEEP]
    except OSError:
        return None
    if not leftovers:
        return None
    return (f"[cache] {_rel_to(old, root)} still holds this checkout's old compile cache, which is no "
            f"longer used — `{SETTINGS_CMD} caches` shows its size and can remove it")


def _free(path: Path) -> str:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return _human(shutil.disk_usage(probe).free)
    except OSError:
        return "unknown"


def prepare(compose_files, **kw) -> dict[str, str]:
    """Create the directories a compose mounts under the two variables, as the calling
    user, and return {CLUB3090_ENGINE_CACHE_DIR: …, CLUB3090_DATA_DIR: …} (only the ones
    it mounts) to add to that compose's environment.

    Keywords: `env` resolves the per-user directories and should already hold the saved
    settings (club_config.load); `render_env` is the environment compose will interpolate
    with at `up` (default: env); `env_file`, `compose_cmd`, `docker_cmd`, `pull`, `cwd`,
    `repo_root`. Never raises: a launch must not fail over a cache directory, so any
    problem is reported and {} returned, and the compose keeps its in-repo default."""
    try:
        return _prepare(compose_files, **kw)
    except Exception as e:                      # noqa: BLE001 — see the docstring
        _say(f"[cache] unexpected error ({type(e).__name__}: {e}); this launch uses the compose's in-repo paths")
        return {}


def _prepare(compose_files, *, env=None, render_env=None, env_file=None, compose_cmd=("docker", "compose"),
             docker_cmd=("docker",), pull=True, cwd=None, repo_root=None) -> dict[str, str]:
    env = dict(os.environ if env is None else env)
    files = [Path(f) for f in compose_files]
    if not files or not files[0].is_file():
        return {}
    try:
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files if f.is_file())
    except OSError:
        return {}
    if not mounts_shared_dirs(text):
        return {}
    compose_cmd = list(compose_cmd)
    if not compose_cmd or compose_cmd[0] in _NOOP_BINS:
        return {}
    renv = dict(env if render_env is None else render_env)
    renv.update(SENTINELS)
    try:
        cfg = render(files, compose_cmd, renv, env_file=env_file, cwd=cwd or files[0].parent)
        cache_mounts, data_mounts = scan(cfg)
    except PrepareError as e:
        _say(f"[cache] {e}; this launch uses the compose's in-repo cache paths")
        return {}

    out: dict[str, str] = {}
    notes: list[str] = []
    if cache_mounts:
        try:
            if len(cache_mounts) != 1:
                raise PrepareError(f"services on different images ({', '.join(sorted(cache_mounts))}) "
                                   "mount one cache directory")
            (ref, subdirs), = cache_mounts.items()
            if not ref:
                raise PrepareError("the service mounting the cache names no image")
            iid = image_id(ref, list(docker_cmd), pull)
            if iid is None:
                raise PrepareError(f"{ref} is not on this machine" + (" and could not be pulled" if pull else ""))
            keyed = _absolute(club_config.cache_dir(env), "CLUB3090_CACHE_DIR") / image_key(ref, iid)
            made = [keyed / s for s in sorted(subdirs)]
            _make(made)
            notes += _foreign_owner_notes([keyed, *made])
            out[CACHE_VAR] = str(keyed)
            notes.insert(0, f"[cache] compile cache: {keyed}  (shared by every checkout that runs this image)")
            legacy = _legacy_note(files[0], text, repo_root)
            if legacy:
                notes.append(legacy)
        except PrepareError as e:
            notes.append(f"[cache] {e}; this launch uses the compose's in-repo compile cache")
    if data_mounts:
        try:
            base = _absolute(club_config.data_dir(env), "CLUB3090_DATA_DIR")
            made = [base / s for s in sorted(data_mounts)]
            _make(made)
            notes += _foreign_owner_notes(made)
            out[DATA_VAR] = str(base)
            disk_on = "1" in (env.get("KV_OFFLOAD_DISK"), renv.get("KV_OFFLOAD_DISK"))
            if disk_on and "kv-offload" in data_mounts:
                notes.append(f"[kv-offload] disk tier: {base / 'kv-offload'} ({_free(base)} free on that filesystem). "
                             "vLLM's tier has no size cap; KV_OFFLOAD_DIR puts it on another disk.")
        except PrepareError as e:
            notes.append(f"[kv-offload] {e}; this launch uses the compose's in-repo default")
    for n in notes:
        _say(n)
    return out


# ── settings.sh caches ───────────────────────────────────────────────────────
def _walk(path: Path):
    """(bytes on disk, newest mtime, needs root to delete, fully readable) for a file or tree."""
    try:
        st = path.lstat()
    except OSError:
        return 0, 0.0, False, False
    total, newest = st.st_blocks * 512, st.st_mtime
    needs_root, complete = not os.access(path.parent, os.W_OK | os.X_OK), True
    if not path.is_dir() or path.is_symlink():
        return total, newest, needs_root, complete
    stack = [path]
    while stack:
        d = stack.pop()
        if not os.access(d, os.W_OK | os.X_OK):
            needs_root = True
        try:
            it = os.scandir(d)
        except OSError:
            needs_root, complete = True, False
            continue
        with it:
            for e in it:
                try:
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    complete = False
                    continue
                total += s.st_blocks * 512
                newest = max(newest, s.st_mtime)
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
    return total, newest, needs_root, complete


def legacy_items(root, env) -> tuple[list[Path], str | None]:
    """The old in-repo caches of checkout `root`: everything in models/*/*/cache/ but the
    tracked .gitignore/README.md, and the repo kv-offload/'s contents — unless
    KV_OFFLOAD_DIR points there (then it's in use; the note says so)."""
    root = Path(root)
    items: list[Path] = []
    for cache in sorted(root.glob("models/*/*/cache")):
        if cache.is_dir():
            items += [e for e in sorted(cache.iterdir()) if e.name not in LEGACY_KEEP]
    kv = root / "kv-offload"
    note = None
    explicit = env.get("KV_OFFLOAD_DIR")
    if explicit and os.path.realpath(os.path.expanduser(explicit)) == os.path.realpath(kv):
        note = f"{kv} is your KV_OFFLOAD_DIR, so it is in use and not listed here."
    elif kv.is_dir():
        items += [e for e in sorted(kv.iterdir()) if e.name != ".gitignore"]
    return items, note


def _cache_source(env, var, xdg) -> str:
    return f"${var}" if env.get(var) else f"${xdg}" if env.get(xdg) else "the default"


def caches_report(root, env) -> tuple[str, list[tuple[Path, int, bool]]]:
    lines = []
    croot = Path(os.path.expanduser(str(club_config.cache_dir(env))))
    lines.append("Compile caches — shared by every checkout, one folder per engine image")
    lines.append(f"  {croot}   ({_cache_source(env, 'CLUB3090_CACHE_DIR', 'XDG_CACHE_HOME')})")
    dirs = sorted(p for p in croot.iterdir() if p.is_dir()) if croot.is_dir() else []
    if not dirs:
        lines.append("    nothing yet — a launcher creates one the first time it starts a vLLM compose")
    for p in dirs:
        size, newest, needs_root, complete = _walk(p)
        when = time.strftime("%Y-%m-%d", time.localtime(newest)) if newest else "?"
        flag = "   some files need sudo to delete" if needs_root else ""
        lines.append(f"    {p.name:<48} {('≥' if not complete else '') + _human(size):>11}   last written {when}{flag}")
    if dirs:
        lines.append("  Deleting one is safe: the next start on that image compiles from cold (a few minutes).")

    droot = Path(os.path.expanduser(str(club_config.data_dir(env))))
    kv = droot / "kv-offload"
    lines += ["", "KV-offload disk tier (used only with KV_OFFLOAD_DISK=1)"]
    size = _walk(kv)[0] if kv.exists() else 0
    lines.append(f"  {kv}   {_human(size)}, {_free(kv)} free on that filesystem   "
                 f"({_cache_source(env, 'CLUB3090_DATA_DIR', 'XDG_DATA_HOME')})")
    if env.get("KV_OFFLOAD_DIR"):
        lines.append(f"  KV_OFFLOAD_DIR is set, so the tier is at {env['KV_OFFLOAD_DIR']} instead.")
    lines.append("  vLLM's disk tier has no size cap. SGLang's can be capped with KV_OFFLOAD_DISK_GB.")

    items, note = legacy_items(root, env)
    rows = []
    lines += ["", f"Old caches inside this checkout ({root}) — no longer used"]
    for p in items:
        size, _newest, needs_root, complete = _walk(p)
        rows.append((p, size, needs_root))
        flag = "   root-owned: needs sudo" if needs_root else ""
        lines.append(f"    {_rel_to(p, Path(root)):<56} {('≥' if not complete else '') + _human(size):>11}{flag}")
    if not items:
        lines.append("    none")
    else:
        lines.append(f"  total {_human(sum(r[1] for r in rows))} — remove with: {SETTINGS_CMD} caches --remove-legacy")
    if note:
        lines.append(f"  {note}")
    return "\n".join(lines) + "\n", rows


def remove_legacy(rows, assume_yes: bool) -> int:
    doable = [r for r in rows if not r[2]]
    rooted = [r for r in rows if r[2]]
    if not rows:
        print("Nothing to remove.")
        return 0
    if doable:
        total = _human(sum(r[1] for r in doable))
        if not assume_yes:
            if not sys.stdin.isatty():
                print(f"[settings] refused: not asking on a non-terminal; re-run with --yes to remove {total}. "
                      "Nothing was changed.", file=sys.stderr)
                return 2
            ans = input(f"Remove {len(doable)} old cache folder(s), {total}, from this checkout? [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                print("Nothing was removed.")
                return 0
        freed = 0
        for p, size, _ in doable:
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                freed += size
            except OSError as e:
                print(f"  couldn't remove {p}: {e.strerror or e}", file=sys.stderr)
                rooted.append((p, size, True))
        print(f"Removed {_human(freed)}.")
    if rooted:
        print("These are root's: docker created the folder itself (a missing bind-mount source), or a container "
              "wrote into it before the composes gave their files your group. Only root can delete them:")
        print("  sudo rm -rf -- " + " ".join(shlex.quote(str(p)) for p, _s, _r in rooted))
    return 0


def caches_main(argv) -> int:
    ap = argparse.ArgumentParser(prog=f"{SETTINGS_CMD} caches",
                                 description="Where the compile caches and the KV-offload disk tier are, how big "
                                             "they are, and the old caches this checkout kept inside the repo.")
    ap.add_argument("--root", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--remove-legacy", action="store_true",
                    help="delete this checkout's old in-repo caches, after asking")
    ap.add_argument("--yes", action="store_true", help="don't ask (with --remove-legacy)")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    club_config.load(a.root, env, warn=False)
    text, rows = caches_report(a.root, env)
    sys.stdout.write(text)
    if a.remove_legacy:
        print()
        return remove_legacy(rows, a.yes)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["caches"]:
        return caches_main(argv[1:])
    ap = argparse.ArgumentParser(prog="engine_cache.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="create the dirs a compose mounts; print KEY=VALUE lines for its environment")
    p.add_argument("--compose", action="append", required=True, help="compose file (repeat for overrides)")
    p.add_argument("--root", help="repo checkout whose saved settings (and legacy .env) to read")
    p.add_argument("--compose-bin", default=os.environ.get("COMPOSE_BIN", "docker compose"))
    p.add_argument("--docker", default="docker", help='docker command, e.g. "sudo docker"')
    p.add_argument("--env-file", help="the --env-file the compose call will get")
    p.add_argument("--clean-env", action="store_true",
                   help="render as `sudo docker compose` will see it: only PATH, HOME, --env-file and the VAR=val args")
    p.add_argument("--no-pull", action="store_true", help="don't pull a missing image (fall back instead)")
    p.add_argument("assign", nargs="*", metavar="VAR=val", help="assignments the compose call will get (after --)")
    k = sub.add_parser("key", help="print the cache key for an image reference and ID")
    k.add_argument("--image", required=True)
    k.add_argument("--id", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "key":
        try:
            print(image_key(a.image, a.id))
        except PrepareError as e:
            print(f"[cache] {e}", file=sys.stderr)
            return 2
        return 0
    assigns = {}
    for s in a.assign:
        if "=" in s:
            k_, _, v_ = s.partition("=")
            assigns[k_] = v_
    env = dict(os.environ)
    env.update(assigns)
    if a.root:
        club_config.load(a.root, env, warn=False)
    if a.clean_env:
        render_env = {k_: os.environ[k_] for k_ in ("PATH", "HOME") if k_ in os.environ}
        render_env.update(assigns)
    else:
        render_env = env
    files = [Path(f).resolve() for f in a.compose]
    out = prepare(files, env=env, render_env=render_env, env_file=a.env_file,
                  compose_cmd=shlex.split(a.compose_bin), docker_cmd=shlex.split(a.docker),
                  pull=not a.no_pull, repo_root=Path(a.root).resolve() if a.root else None)
    for k_ in OUTPUT_VARS:
        if k_ in out:
            print(f"{k_}={out[k_]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
