"""Prepare DeepSeek-V4.1-Flash-1017 for Project Maya's maya-ds4 engine on the data volume, for ds4-entrypoint.sh.

The weights are mounted read-only at /gguf (one quant's GGUF shards: ISTA-DASLab's GSQ-RCO IQ2_XXS with the shared
Engram file beside them, or unsloth's UD-IQ3_S, whose last shard holds the Engram tables) and a tag's DSpark drafter,
when it has one, at /dspark (peasantsmith's dspark-xs2: a .bin and its .json). Maya's tools/ds4/pack_gguf.py writes its
pack (an index of those files, no weights copied) wherever --out says, and the pack names the model folder it indexes
(ds4.txt model_dir), so the pack lives on /data and the files stay where they are:

    /data/<tag>/pack/            tools/ds4/pack_gguf.py's output + sources.json; expert-profile.bin (the engine's
                                 routing counts, seeded from the image's data/expert-profile-ds4.bin, then its own)
    /data/<tag>/verified.json    the files' sizes and times when their SHA-256 matched
    /data/<tag>/maya-ds4.json    the config serve/server.py starts from
    /data/<tag>/maya-ds4.log     the engine's log

Steps, each skipped when already done: check the files' SHA-256 against WEIGHTS (Maya's own tables don't list ISTA's
quant; UD-IQ3_S's are the ones Maya pins; minutes, once, again only when a file's size or time changes); build the
pack (seconds; again when a file
changed after it was built; Maya's pack builder fills in the DSpark geometry ISTA's GGUF lacks, from v1.1.30); seed the
expert profile; write the config. Python stdlib only, plus the image's /opt/maya."""
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

import ds4_settings                                   # beside this file: club-3090's launch settings

MAYA = Path("/opt/maya")
# Every file a tag reads, with its SHA-256 (the Hugging Face LFS oid at the pinned revision; the .json is a plain git
# file: its SHA-256 measured from that revision's copy). Folder: "gguf" = /gguf, "engram" = /engram, "dspark" = /dspark.
WEIGHTS = {
    "ista-iq2-xxs": {
        "gguf": {   # ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF @85d6710f (files unchanged since cb02422c)
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf":
                "d46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050",
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf":
                "cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70",
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf":
                "9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3",
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf":
                "9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059",
            "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf":
                "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a",
        },
        "dspark": {   # peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF @d1fc0100, dspark/
            "dspark-xs2.bin": "8dfe21082a6cb320f334fddb454630c241c6d44e0463eef4f8b0b07334511779",
            "dspark-xs2.json": "56d47ad12d6548a8d3d50a79be96451411f18d5b77064de7b8544bb2da22988f",
        },
    },
    "ud-iq3-s": {
        "gguf": {   # unsloth/DeepSeek-V4.1-Flash-1017-GGUF @8a6ec9cf, UD-IQ3_S/ (the SHA-256s Maya v1.1.48's
                    # tools/ds4/download_gguf.json pins; shard 8 holds the Engram tables, Q6_K)
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00001-of-00008.gguf":
                "1d3027d28a1c749a5a185420640d2849849e5d42e7719df5d852572e1771de25",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00002-of-00008.gguf":
                "a5c5b3a1d56fc64e2018442170371348683a6ca31d805040946635c508698c89",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00003-of-00008.gguf":
                "0182203a6278cd29702003589b11635a8b6c58a17613270a94532a7487760092",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00004-of-00008.gguf":
                "5e1aed1952f95ada91abb3332adb8d77eff6a98281a09c246e827c6031ba0a93",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00005-of-00008.gguf":
                "69dd858b60744b9430be3b69684120d1bcb33ba4b5a740b60ddf8fc5a2875dee",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00006-of-00008.gguf":
                "962c493b4f41f2dedfb31ebbfe725d14b55d97a442f0cd14296a42ac51345ff7",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00007-of-00008.gguf":
                "c96d4a18a86dcdb3ed10e3e367bc9f3070d61a1da0144926519d2987c4e41b26",
            "DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00008-of-00008.gguf":
                "e70b4bf3ac5b26aa2addb048cff61a6be5490e99ba88cfbc29f7fb7279000141",
        },
        "dspark": {},   # no drafter: neither DSpark build sped it up on two 3090s (compose header)
    },
    "maya-xs": {
        "gguf": {   # peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF @c33d87b8, Maya-XS/ (the SHA-256s Maya v1.1.56's
                    # tools/ds4/download_gguf.json pins as the quant's model files: its three v2 shards)
            "DeepSeek-V4.1-Flash-1017-Maya-XS-v2-IQ1_S-00001-of-00003.gguf":
                "9d92dd13d4603dbf07720845419963693f4c07ae098e55a5aae2f207dd44d04d",
            "DeepSeek-V4.1-Flash-1017-Maya-XS-v2-IQ1_S-00002-of-00003.gguf":
                "b7f26bae6d6fccfb256c911f645289c0aec8459766ef83f6a04d7ed47c7e06c8",
            "DeepSeek-V4.1-Flash-1017-Maya-XS-v2-IQ1_S-00003-of-00003.gguf":
                "62b98172562543126204fa2ff18f7d72d75353445ff93f4135d037355b409f35",
        },
        "engram": {   # the Q6_K n-gram-table pair Maya-XS shares with Maya-S, as Maya's own setup reads it: a GGUF set
                      # of its own, NOT shards of the model's split. The Hugging Face repo keeps the two files inside
                      # Maya-XS/ (they are v1's files 4-5; v2 replaced the model files only), and pack_gguf.py refuses
                      # a folder holding two splits - "the shards of more than one GGUF split (-of-00003, -of-00005):
                      # name the model's own with --shards" - so the compose mounts just these two at /engram and the
                      # pack indexes them as the set, recording each by its path relative to the model folder the way
                      # Maya's own Engram-Q6_K/ folder does. Their own metadata is only split.count / split.no /
                      # split.tensors.count and one tensor each: the tables, never the model's.
            "DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S-00004-of-00005.gguf":
                "0d93315942c523d82f6dbc0a6e0e1751548e309f3963f96b82f92e8f9bc52e36",
            "DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S-00005-of-00005.gguf":
                "39bbc8cde8b17604fed9a7ba7fc6bc9c5e72e34733657943ac049b4b9ca2d340",
        },
        "dspark": {   # the quant's own drafter (the manifest's maya/quants/Maya-XS/drafter): the same two files the
                      # ista tag reads
            "dspark-xs2.bin": "8dfe21082a6cb320f334fddb454630c241c6d44e0463eef4f8b0b07334511779",
            "dspark-xs2.json": "56d47ad12d6548a8d3d50a79be96451411f18d5b77064de7b8544bb2da22988f",
        },
    },
}
# Each tag's files as Maya shows them (its dashboard, /health and /metrics' model_label: tools/maya_quants.config_label,
# from the installer record's ds4_quant - without one it names Maya's own EXL3 2.9 bpw quant). ISTA's quant is not one
# of Maya's, so the name is shown as given; UD-IQ3_S is in Maya's setup list, so its key is recorded and Maya shows it
# as "UD-IQ3_S" (maya_quants.py's table), as its own setup would.
SHOWN = {"ista-iq2-xxs": "ISTA IQ2_XXS", "ud-iq3-s": "gguf-iq3_s", "maya-xs": "maya-xs"}


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


def files_of(tag: str, src: Path, dspark: Path, engram: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """(the model's files in /gguf, its Engram set in /engram, the drafter's in /dspark), all present, else a stop
    naming what's missing. A tag whose tables are among its own shards has no "engram" key and gets an empty list."""
    w = WEIGHTS[tag]
    model = [src / n for n in w["gguf"]]
    tables = [engram / n for n in w.get("engram", {})]
    drafter = [dspark / n for n in w["dspark"]]
    missing = [str(f) for f in model + tables + drafter if not f.is_file()]
    if missing:
        die(f"missing: {', '.join(missing)} (a download still running, or the wrong folder mounted? "
            "bash scripts/setup.sh <slug> downloads both)")
    return model, tables, drafter


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
    want = {**WEIGHTS[tag]["gguf"], **WEIGHTS[tag].get("engram", {}), **WEIGHTS[tag]["dspark"]}
    gb = sum(f.stat().st_size for f in files) / 1e9
    say(f"checking {len(files)} files' SHA-256 (once; minutes for {gb:.0f} GB) ...")
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:   # hashlib frees the GIL: the disk sets the pace
        got = dict(zip(files, pool.map(sha256, files)))
    bad = [f"{f.name}: {h}, want {want[f.name]}" for f, h in got.items() if h != want[f.name]]
    if bad:
        die("SHA-256 mismatch (a damaged or partial download; download it again): " + "; ".join(bad))
    write_json(rec, {"tag": tag, "files": now})
    say("every file matches its SHA-256")


def pack(model: list[Path], tables: list[Path], drafter: list[Path], d: Path, repack: bool) -> Path:
    p = d / "pack"
    files = model + tables + drafter
    rec = read_json(p / "sources.json").get("files")
    if not repack and (p / "tensors.txt").is_file() and rec == stamp(files):
        say(f"pack up to date: {p}")
        return p
    why = "REPACK=1" if repack else ("no pack yet" if not (p / "tensors.txt").is_file() else
                                     "a file changed after the pack was built")
    say(f"building the pack ({why}; seconds: it indexes the files, no weights copied) ...")
    tmp = d / "pack.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    # a tag without a drafter passes none: pack_gguf.py then looks for <model>/../dspark/dspark.json, which the compose
    # of such a tag doesn't mount, so the pack holds no drafter
    dspark_json = next((f for f in drafter if f.suffix == ".json"), None)
    want = ["--drafter", str(dspark_json)] if dspark_json else []
    # a quant whose Engram tables are a GGUF set of their own names the folder: the pack indexes those files by their
    # path relative to the model folder, which is how Maya's own setup's Engram-Q6_K/ beside the quant reaches them
    eg = ["--engram", str(tables[0].parent)] if tables else []
    r = subprocess.run([str(MAYA / ".venv/bin/python"), str(MAYA / "tools/ds4/pack_gguf.py"),
                        "--model", str(model[0].parent), "--out", str(tmp), *want, *eg,
                        "--shards", *[f.name for f in model]], cwd=str(MAYA))
    if r.returncode != 0:
        die(f"tools/ds4/pack_gguf.py stopped (exit {r.returncode}); the old pack, if any, is unchanged")
    write_json(tmp / "sources.json", {"files": stamp(files)})
    if p.exists():
        for keep in ("expert-profile.bin", "slots"):   # the learned routing counts, saved conversations
            if (p / keep).exists():
                shutil.move(str(p / keep), str(tmp / keep))
        shutil.rmtree(p)
    os.replace(tmp, p)
    say(f"pack built: {p}")
    return p


def seed_profile(p: Path) -> None:
    """The engine places experts by the routing counts in <pack>/expert-profile.bin and keeps adding to them; a new
    pack starts from DeepSeek's shipped profile, as Maya's setup does."""
    prof = p / "expert-profile.bin"
    if not prof.exists():
        shutil.copyfile(MAYA / "data/expert-profile-ds4.bin", prof)
        say("expert profile seeded from data/expert-profile-ds4.bin")


def config(tag: str, d: Path, p: Path, a) -> Path:
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    keys, sampling, args, notes = ds4_settings.settings()   # checked once already at the start of main()
    cfg = {
        "exe": str(MAYA / "build/maya-ds4"),
        # --tp 1 on two GPUs: a layer split (each GPU its own layers, whole experts), not tensor parallel; DSpark on
        # for a tag with a drafter (Maya's setup writes none until its calibration says drafting pays), off without one
        "args": ["--pack", str(p), "--max-context", str(a.context), "--tp", str(a.tp), "--vision", "off",
                 "--dspark", "on" if WEIGHTS[tag]["dspark"] else "off"],
        "cwd": str(MAYA),
        "tokenizer": str(p / "tokenizer"),
        "model_name": ds4_settings.DEFAULT_NAME,     # SERVED_NAME replaces it (ds4_settings.py)
        "gpu": gpus,
        "sampling": sampling,                         # DeepSeek's 1.0 / 1.0 unless TEMP / TOP_P / ... say otherwise
        "lib_dirs": [],
        "port": 8080,
        "host": "0.0.0.0",
        "log": str(d / "maya-ds4.log"),
        "installer": {"family": "deepseek", "ds4_quant": SHOWN[tag], "quant": tag,
                      "written_by": "club-3090 ds4_prepare.py"},
    }
    cfg.update(keys)                                  # reasoning_effort (high unless set), thinking_budget, model_name
    cfg["args"] += args                               # --cpu-threads
    for n in notes:
        say(n)
    out = d / "maya-ds4.json"
    write_json(out, cfg)
    shown = ", ".join([f"{k}={v}" for k, v in keys.items()] + [f"sampling.{k}={v}" for k, v in sampling.items()]
                      + [" ".join(args)])
    say(f"config: {out} (GPUs {gpus}, --tp {a.tp}, context {a.context}; {shown})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(WEIGHTS))
    ap.add_argument("--src", default="/gguf")
    ap.add_argument("--dspark", default="/dspark")
    ap.add_argument("--engram", default="/engram")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--verify", default="1")
    ap.add_argument("--repack", default="0")
    a = ap.parse_args()
    try:
        ds4_settings.settings()                       # a bad setting stops here, before the SHA-256 check and the pack
    except ds4_settings.Bad as e:
        die(str(e))
    d = Path(a.data) / a.tag
    d.mkdir(parents=True, exist_ok=True)
    model, tables, drafter = files_of(a.tag, Path(a.src), Path(a.dspark), Path(a.engram))
    if a.verify != "0":
        verify(a.tag, model + tables + drafter, d)
    else:
        say("VERIFY=0: the files' SHA-256 is not checked")
    p = pack(model, tables, drafter, d, a.repack == "1")
    seed_profile(p)
    config(a.tag, d, p, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
