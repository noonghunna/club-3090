"""Prepare one Project Maya quant on the data volume, the way maya.py's setup does on a PC, for maya-entrypoint.sh.

The model folder is mounted read-only at /gguf, but Maya's pack (its index of the model files) is written next to
them. So each quant gets a folder on /data with a symlink per shard (the engine reads the experts from the shard named
beside the pack) and the pack inside it:

    /data/<tag>/<shard>.gguf -> /gguf/<shard>.gguf      one per shard
    /data/<tag>/pack/                                   tools/iq_pack.py's output + sources.json
    /data/<tag>/verified.json                           the shards' sizes and times when their SHA-256 matched
    /data/<tag>/maya.json                               the config serve/server.py starts from

Steps, each skipped when already done: link the shards; check their SHA-256 against maya.py's MODELS (a few minutes,
once; again only when a file's size or time changes); build the pack (a few minutes, once, or when a shard changed after
it was built: Maya's pack_stale, #155); write the config. Python stdlib only, plus the image's /opt/maya."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import maya_settings                                  # beside this file: club-3090's launch settings

MAYA = Path("/opt/maya")
# our quant tags -> maya.py's MODELS names
MODELS = {"maya-s": "Maya-S-v2-IQ2_XXS", "maya-m": "Maya-M", "maya-l": "Maya-L"}
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def say(msg: str) -> None:
    print(f"[maya-recipe] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[maya-recipe] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(p: Path, data: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def stamp(files: list[Path]) -> list[dict]:
    return [{"name": f.name, "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in files]


def shards_in(src: Path) -> list[Path]:
    firsts = sorted(src.glob("*-00001-of-*.gguf"))
    if len(firsts) != 1:
        die(f"want exactly one first shard (*-00001-of-NNNNN.gguf) in {src}, found {[f.name for f in firsts]}")
    m = SHARD.search(firsts[0].name)
    n, stem = int(m.group(2)), firsts[0].name[:m.start()]
    files = [src / f"{stem}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        die(f"{src} lacks {', '.join(missing)} (a download still running, or another quant's folder?)")
    return files


def link(files: list[Path], d: Path) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    names = {f.name for f in files}
    for old in d.glob("*.gguf"):                      # dangling links, or links to another folder's shards
        if old.is_symlink() and (not old.exists() or old.name not in names):
            old.unlink()
    out = []
    for f in files:
        dst = d / f.name
        if not (dst.is_symlink() and os.readlink(dst) == str(f)):
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(f)
        out.append(dst)
    return out


def verify(tag: str, links: list[Path], d: Path) -> None:
    rec = d / "verified.json"
    now = stamp(links)
    if read_json(rec).get("files") == now:
        say("shards already verified")
        return
    sys.path.insert(0, str(MAYA))
    try:
        import maya                                   # noqa: E402  (nothing runs on import)
        want = maya.MODELS[MODELS[tag]]["sha256"]
    except Exception as e:                           # an upstream rename: say so, don't serve unverified quietly
        die(f"could not read the shards' SHA-256 from maya.py's MODELS[{MODELS[tag]!r}] ({e!r}); "
            "set VERIFY=0 to start without the check")
    for f in links:
        ok = want.get(f.name)
        if ok is None:
            die(f"maya.py lists no SHA-256 for {f.name}: is this {MODELS[tag]}'s folder?")
        ok = [ok] if isinstance(ok, str) else list(ok)
        say(f"checking {f.name}'s SHA-256 (once) ...")
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(16 << 20), b""):
                h.update(block)
        if h.hexdigest() not in ok:
            die(f"{f.name}: SHA-256 {h.hexdigest()} is none of Maya's ({', '.join(ok)}): a damaged or "
                "partial download; download it again")
    write_json(rec, {"model": MODELS[tag], "files": now})
    say("every shard matches Maya's SHA-256")


def pack(links: list[Path], d: Path, repack: bool) -> Path:
    p = d / "pack"
    rec = read_json(p / "sources.json").get("files")
    if not repack and (p / "index.txt").is_file() and rec == stamp(links):
        say(f"pack up to date: {p}")
        return p
    why = "REPACK=1" if repack else ("no pack yet" if not (p / "index.txt").is_file() else
                                     "a shard changed after the pack was built")
    say(f"building the pack ({why}; a few minutes) ...")
    tmp = d / "pack.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    r = subprocess.run([str(MAYA / ".venv/bin/python"), str(MAYA / "tools/iq_pack.py"), "--gguf", str(links[0]),
                        "--out", str(tmp), "--compat-bf16"], cwd=str(MAYA))
    if r.returncode != 0:
        die(f"tools/iq_pack.py stopped (exit {r.returncode}); the old pack, if any, is unchanged")
    write_json(tmp / "sources.json", {"files": stamp(links)})      # maya.py's stamp_pack format
    if p.exists():
        slots = p / "slots"                           # saved conversations: keep them across a repack
        if slots.is_dir():
            shutil.move(str(slots), str(tmp / "slots"))
        shutil.rmtree(p)
    os.replace(tmp, p)
    say(f"pack built: {p}")
    return p


def engine() -> Path:
    """The image's engine: build/maya from Maya v1.1.16 on (that release renamed its programs), build/strata before.
    With neither present (the config rendered outside the image, as the tests do) it names build/maya."""
    for name in ("maya", "strata"):
        if (MAYA / "build" / name).is_file():
            return MAYA / "build" / name
    return MAYA / "build" / "maya"


def config(tag: str, d: Path, p: Path, a) -> Path:
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    keys, sampling, args, notes = maya_settings.settings()   # checked once already at the start of main()
    cfg = {
        "exe": str(engine()),
        "args": ["--glm-pack", str(p), "--max-context", str(a.context), "--prefill", "auto",
                 "--kv-streaming", "on", "--expert-profile", str(MAYA / "data/expert-profile-glm.bin")],
        "cwd": str(MAYA),
        "tokenizer": str(p / "tokenizer"),
        "model_name": "glm-5.3-flash",               # SERVED_NAME replaces it (maya_settings.py)
        "gpu": gpus,
        # GLM-5.3's vendor sampling (1.0 / 0.95, as Maya's setup writes it) unless TEMP / TOP_P / ... say otherwise
        "sampling": sampling,
        "lib_dirs": [],
        "port": 8080,
        "host": "0.0.0.0",
        "log": str(d / "maya.log"),
        "installer": {"quant": MODELS[tag], "written_by": "club-3090 maya_prepare.py"},
    }
    cfg.update(keys)                                  # reasoning_effort (max unless set), thinking_budget
    cfg["args"] += args                               # --kv
    for n in notes:
        say(n)
    out = d / "maya.json"
    write_json(out, cfg)
    shown = ", ".join([f"{k}={v}" for k, v in keys.items()] + [f"sampling.{k}={v}" for k, v in sampling.items()]
                      + [" ".join(args)] * bool(args))
    say(f"config: {out} (GPUs {gpus}, context {a.context}; {shown})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--src", default="/gguf")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--verify", default="1")
    ap.add_argument("--repack", default="0")
    a = ap.parse_args()
    try:
        maya_settings.settings()                      # a bad setting stops here, before the SHA-256 check and the pack
    except maya_settings.Bad as e:
        die(str(e))
    d = Path(a.data) / a.tag
    links = link(shards_in(Path(a.src)), d)
    if a.verify != "0":
        verify(a.tag, links, d)
    else:
        say("VERIFY=0: the shards' SHA-256 is not checked")
    p = pack(links, d, a.repack == "1")
    config(a.tag, d, p, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
