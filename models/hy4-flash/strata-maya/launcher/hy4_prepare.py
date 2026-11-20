"""Prepare Hy4-Flash for Project Maya's maya-hy4 engine on the data volume, for hy4-entrypoint.sh.

The compose mounts a quant's files read-only into /gguf under the names Maya's own setup gives them: the model (shard
00001), the n-gram table (shard 00002: ONE file for every size, Maya's repo carries it once a folder), Tencent's MTP
block (on the Hub as mtp/Hy4-Flash-mtp-BF16.gguf_file, saved as .gguf) and the vision tower (Maya's one mmproj file for
every size, Maya-XS/mmproj-Hy4-Flash-Maya-XS.gguf). Maya's tools/hy4/pack_gguf.py writes its pack (an index of those
files, no weights copied) wherever --out says, naming the files by their path from the model folder, so the pack lives
on /data and the files stay where they are:

    /data/<tag>/pack/            tools/hy4/pack_gguf.py's output + sources.json
    /data/<tag>/verified.json    the files' sizes and times when their SHA-256 matched
    /data/<tag>/maya-hy4.json    the config serve/server.py starts from
    /data/<tag>/maya-hy4.log     the engine's log

Steps, each skipped when already done: check the files' SHA-256 against WEIGHTS (Maya's own manifest's; minutes, once,
again only when a file's size or time changes); build the pack (seconds; again when a file changed after it was built);
write the config. Python stdlib only, plus the image's /opt/maya."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hy4_settings                                   # beside this file: club-3090's launch settings

MAYA = Path("/opt/maya")
PROFILE = MAYA / "data/expert-profile-hy4.bin"     # (none in v1.1.68; maya.py passes it only when it exists)
# The files every size shares, then each size's model shard, with their SHA-256 as Maya's own manifest pins them
# (tools/hy4/download.json at v1.1.68: peasantsmith/Hy4-Flash-Maya-GGUF @9d73294d). The names are the ones in /gguf.
SHARED = {
    "mtp": ("Hy4-Flash-mtp-BF16.gguf", "a49e5ae439de9a51c680b9110563106743e8d579d91e4d4cb514a527fa908908"),
    "vision": ("mmproj-Hy4-Flash-Maya-XS.gguf", "6884c0317a4e3e781eef4a535706e47ae1423742ffe8341adf6e50497a7338be"),
}
NGRAM_SHA = "5819578b52eb1cda0945a716e0575331406d976b6eebc74db8544b3d478ad244"   # shard 00002, the same for every size
# The tag is the quant's folder on the data volume, which the GLM-5.3-Flash, DeepSeek and MiniMax quants share: "hy4-",
# so it never meets another model's folder. Its key in Maya's own setup (tools/maya_quants.py QUANTS: hy4 maya-l ...)
# names the files on the dashboard, /health and /metrics (the installer record's hy4_quant / quant).
WEIGHTS = {
    "hy4-maya-xs": ("maya-xs", "Hy4-Flash-Maya-XS", "Hy4-Flash-Maya-XS-IQ2_XXS",
                    "18fcc5224d5f882ed3f6e6619b508b9eeec87e88cb714557091bf6baf64e7d4d"),
    "hy4-maya-s": ("maya-s", "Hy4-Flash-Maya-S", "Hy4-Flash-Maya-S-IQ2_S",
                   "80843b8df8e6ce4c1f4d768b42b717a6862ef1d91c06ae9083411b537d6b816e"),
    "hy4-maya-m": ("maya-m", "Hy4-Flash-Maya-M", "Hy4-Flash-Maya-M-IQ3_XXS",
                   "9ef12ab340a06b631069f99b9faccc9e99b686cf9d98b74bb872c4923e4732af"),
    "hy4-maya-l": ("maya-l", "Hy4-Flash-Maya-L", "Hy4-Flash-Maya-L-IQ4_XS",
                   "5f47ce10c0cb6186ec34036cd5eae47b662b780933abca9f1db0db2770799c17"),
}


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


def want(tag: str) -> dict[str, str]:
    """{name in /gguf: SHA-256} of every file the tag reads: its two shards, the MTP block, the vision tower."""
    _, _, stem, sha = WEIGHTS[tag]
    return {f"{stem}-00001-of-00002.gguf": sha, f"{stem}-00002-of-00002.gguf": NGRAM_SHA,
            **{name: s for name, s in SHARED.values()}}


def files_of(tag: str, src: Path) -> dict[str, Path]:
    """The tag's files in /gguf by role, all present, else a stop naming what's missing."""
    names = list(want(tag))
    f = {"model": src / names[0], "ngram": src / names[1], "mtp": src / SHARED["mtp"][0],
         "vision": src / SHARED["vision"][0]}
    missing = [str(p) for p in f.values() if not p.is_file()]
    if missing:
        die(f"missing: {', '.join(missing)} (a download still running, or the wrong folder mounted? "
            "bash scripts/setup.sh <slug> downloads them)")
    return f


def sha256(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for block in iter(lambda: fh.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(tag: str, files: list[Path], d: Path) -> None:
    rec = d / "verified.json"
    now = stamp(files)
    if read_json(rec).get("files") == now:
        say("files already verified")
        return
    w = want(tag)
    gb = sum(f.stat().st_size for f in files) / 1e9
    say(f"checking {len(files)} files' SHA-256 (once; minutes for {gb:.0f} GB) ...")
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:   # hashlib frees the GIL: the disk sets the pace
        got = dict(zip(files, pool.map(sha256, files)))
    bad = [f"{f.name}: {h}, want {w[f.name]}" for f, h in got.items() if h != w[f.name]]
    if bad:
        die("SHA-256 mismatch (a damaged or partial download; download it again): " + "; ".join(bad))
    write_json(rec, {"tag": tag, "files": now})
    say("every file matches its SHA-256")


def pack(f: dict[str, Path], d: Path, repack: bool) -> Path:
    p = d / "pack"
    files = list(f.values())
    need = ("hy4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja")
    rec = read_json(p / "sources.json").get("files")
    if not repack and all((p / n).exists() for n in need) and rec == stamp(files):
        say(f"pack up to date: {p}")
        return p
    why = "REPACK=1" if repack else ("no pack yet" if not (p / "tensors.txt").is_file() else
                                     "a file changed after the pack was built")
    say(f"building the pack ({why}; seconds: it indexes the files, no weights copied) ...")
    tmp = d / "pack.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    # the two shards by name, the vision tower's and the MTP block's files beside them, as Maya's setup passes them
    # (maya.py hy4_pack_step)
    r = subprocess.run([str(MAYA / ".venv/bin/python"), str(MAYA / "tools/hy4/pack_gguf.py"),
                        "--model", str(f["model"].parent), "--out", str(tmp),
                        "--mmproj", f["vision"].name, "--mtp", f["mtp"].name,
                        "--files", f["model"].name, f["ngram"].name], cwd=str(MAYA))
    if r.returncode != 0:
        die(f"tools/hy4/pack_gguf.py stopped (exit {r.returncode}); the old pack, if any, is unchanged")
    missing = [n for n in need if not (tmp / n).exists()]
    if missing:
        die(f"the pack builder did not write {tmp / missing[0]}")
    write_json(tmp / "sources.json", {"files": stamp(files)})
    if p.exists():
        shutil.rmtree(p)
    os.replace(tmp, p)
    say(f"pack built: {p}")
    return p


def config(tag: str, d: Path, p: Path, a) -> Path:
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    keys, sampling, args, env, notes = hy4_settings.settings()   # checked once already at the start of main()
    quant, full, _, _ = WEIGHTS[tag]
    cfg = {
        "exe": str(MAYA / "build/maya-hy4"),
        # Maya's setup for its sizes' standard layout: the pack, the context and an expert profile where the release
        # ships one (v1.1.68 has none: maya.py passes it only when data/expert-profile-hy4.bin exists); no
        # --gpus (the server shows the engine only the config's "gpu" cards and it takes all it sees), no --tp (auto:
        # tensor parallel over the largest group of equal GPUs, the layers split where no group fits), no --chunk /
        # --reserve-gb (the engine picks them at every start). Drafting (--mtp), the CPU lane, KV rows and pictures
        # come from hy4_settings.py
        "args": ["--pack", str(p), "--max-context", str(a.context)] +
                (["--profile", str(PROFILE)] if PROFILE.is_file() else []),
        "cwd": str(MAYA),
        "tokenizer": str(p / "tokenizer"),
        "model_name": hy4_settings.DEFAULT_NAME,     # SERVED_NAME replaces it (hy4_settings.py)
        "gpu": gpus,
        "sampling": sampling,                         # Tencent's 0.9 / 1.0 unless TEMP / TOP_P / ... say otherwise
        "lib_dirs": [],
        "port": 8080,
        "host": "0.0.0.0",
        "log": str(d / "maya-hy4.log"),
        # (Maya's installer record: family / hy4_quant / quant name the files on the dashboard, /health and /metrics;
        #  its "drafting" is the setup's own bookkeeping, which the server doesn't read - drafting is --mtp below)
        "installer": {"family": "hy4", "hy4_quant": quant, "quant": full, "layout": "standard", "draft_block": True,
                      "written_by": "club-3090 hy4_prepare.py"},
    }
    if env:
        cfg["env"] = env                              # HY4_EG_POPULATE, handed to the engine by the server
    cfg.update(keys)                                  # reasoning_effort (high unless set), thinking_budget, model_name
    cfg["args"] += args                               # --mtp, --kv, --vision off, --cpu-threads
    for n in notes:
        say(n)
    out = d / "maya-hy4.json"
    write_json(out, cfg)
    shown = ", ".join([f"{k}={v}" for k, v in keys.items()] + [f"sampling.{k}={v}" for k, v in sampling.items()]
                      + [f"env.{k}={v}" for k, v in env.items()] + [" ".join(args)])
    say(f"config: {out} (GPUs {gpus}, context {a.context}; {shown})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(WEIGHTS))
    ap.add_argument("--src", default="/gguf")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--verify", default="1")
    ap.add_argument("--repack", default="0")
    a = ap.parse_args()
    try:
        hy4_settings.settings()                       # a bad setting stops here, before the SHA-256 check and the pack
    except hy4_settings.Bad as e:
        die(str(e))
    d = Path(a.data) / a.tag
    d.mkdir(parents=True, exist_ok=True)
    f = files_of(a.tag, Path(a.src))
    if a.verify != "0":
        verify(a.tag, list(f.values()), d)
    else:
        say("VERIFY=0: the files' SHA-256 is not checked")
    p = pack(f, d, a.repack == "1")
    config(a.tag, d, p, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
