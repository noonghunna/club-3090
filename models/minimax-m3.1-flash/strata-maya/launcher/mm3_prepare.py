"""Prepare MiniMax-M3.1-Flash for Project Maya's maya-mm3 engine on the data volume, for mm3-entrypoint.sh.

The weights are mounted read-only at /gguf (Project Maya's own GGUF quant: three shards and the dense layers'
indexers side file, <name>-maya-indexers.gguf) and the DSpark drafter at /companions/dspark (the EXL3 repo's dspark/
folder, which Maya's GGUF repo carries byte for byte). Maya's tools/mm3/pack_gguf.py writes its pack (an index of
those files, no weights copied) wherever --out says, and the pack names the files by their path from the model
folder, so the pack lives on /data and the files stay where they are:

    /data/<tag>/pack/            tools/mm3/pack_gguf.py's output + sources.json; prefill-tune.txt (the engine's
                                 prefill calibration for this PC, kept across a repack)
    /data/<tag>/verified.json    the files' sizes and times when their SHA-256 matched
    /data/<tag>/maya-mm3.json    the config serve/server.py starts from
    /data/<tag>/maya-mm3.log     the engine's log

Steps, each skipped when already done: check the files' SHA-256 against WEIGHTS (Maya's own manifests'; minutes,
once, again only when a file's size or time changes); build the pack (seconds; again when a file changed after it was
built); write the config. Text only: the pack takes the drafter from /companions and no vision tower (none is
mounted). Python stdlib only, plus the image's /opt/maya."""
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

import mm3_settings                                   # beside this file: club-3090's launch settings

MAYA = Path("/opt/maya")
# Every file a tag reads, with its SHA-256, as Maya's own manifests pin them: tools/mm3/download_gguf.json (the Maya
# GGUF quants, peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF @4b40cc79) and tools/mm3/download.json's dspark (the EXL3
# repo's drafter, Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw @fe796c40; the GGUF repo's dspark/ is the same
# files, every one hashed equal 2026-10-27). Folder: "gguf" = /gguf, "dspark" = /companions/dspark.
# The tag is the quant's folder on the data volume, which the GLM-5.3-Flash and DeepSeek quants share: "minimax-",
# so it never meets GLM's maya-s folder.
WEIGHTS = {
    "minimax-maya-s": {
        "gguf": {
            "MiniMax-M3.1-Flash-Maya-S-IQ2_XXS-00001-of-00003.gguf":
                "bce9fee035af23dfeaebfb47beb7985d5888e594274e35585c4fb00109573be4",
            "MiniMax-M3.1-Flash-Maya-S-IQ2_XXS-00002-of-00003.gguf":
                "bec5a2f99c7932515537c740ccd745cf33b32f3a7580bc64ce3f1eab99b33827",
            "MiniMax-M3.1-Flash-Maya-S-IQ2_XXS-00003-of-00003.gguf":
                "6629e9152fcb3ec3a99b68438994679abbe36b4509e602c32cf74ef5923c8544",
            "MiniMax-M3.1-Flash-Maya-S-maya-indexers.gguf":
                "7282c4a836fd436b213249c493182bf17d027a0e28fe43617c22fd6ee0dd6b86",
        },
        "dspark": {
            "added_tokens.json": "63f277741fb061e615cb604286a845d971178ba4716732a59cf42b3752c0ce4e",
            "chat_template.jinja": "01a13fa194313b925b4f7b69b7aa315aa13ee68a47d4ddb4a8e73e2cf8768ff8",
            "config.json": "d3f2ef9a308be0c5689e98b348d8cff73b90487db22ba79e9269f9b1f25cfa3a",
            "merges.txt": "ca47ea60cf5bd48832586adc264f439d0ea4254176b24a95f85eaa9750e2b5f9",
            "model-dspark-00001.safetensors": "a71a16355b4e772383a433f92c8c452dc24b5bb82046142b8221576c6c5fe23c",
            "model.safetensors.index.json": "593f6315b17044d94d5a7e1d231669bcc4dbf9efb06657816d945f1f45c82a6a",
            "special_tokens_map.json": "401b624ec44507183d1dc65c74f7f3387699a77971aa19fa5a0cc57a3989ba97",
            "tokenizer.json": "bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e",
            "tokenizer_config.json": "15caca9f74fbce9b86dc014e9324cbefa56703d81ccd6d56020faa1c41d2eae5",
            "vocab.json": "b44c066b5dc34c800c4e3ecbd85f3e95ce3bfdbf8a5fe30223e005175103578a",
        },
    },
}
# Each tag's key in Maya's own setup (tools/maya_quants.py QUANTS: --mm3-quant gguf-s is Maya-S). Its server names the
# files from the installer record's mm3_quant (the dashboard, /health and /metrics' model_label: config_label) - without
# one it names Maya's EXL3 3.5 bpw quant.
MM3_QUANT = {"minimax-maya-s": "gguf-s"}


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


def files_of(tag: str, src: Path, dspark: Path) -> tuple[list[Path], list[Path]]:
    """(the model's files in /gguf, the drafter's in /companions/dspark), all present, else a stop naming what's
    missing."""
    w = WEIGHTS[tag]
    model = [src / n for n in w["gguf"]]
    drafter = [dspark / n for n in w["dspark"]]
    missing = [str(f) for f in model + drafter if not f.is_file()]
    if missing:
        die(f"missing: {', '.join(missing)} (a download still running, or the wrong folder mounted? "
            "bash scripts/setup.sh <slug> downloads both)")
    return model, drafter


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
    want = {**WEIGHTS[tag]["gguf"], **WEIGHTS[tag]["dspark"]}
    say(f"checking {len(files)} files' SHA-256 (once; minutes for ~125 GB) ...")
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:   # hashlib frees the GIL: the disk sets the pace
        got = dict(zip(files, pool.map(sha256, files)))
    bad = [f"{f.name}: {h}, want {want[f.name]}" for f, h in got.items() if h != want[f.name]]
    if bad:
        die("SHA-256 mismatch (a damaged or partial download; download it again): " + "; ".join(bad))
    write_json(rec, {"tag": tag, "files": now})
    say("every file matches its SHA-256")


def pack(model: list[Path], drafter: list[Path], companions: Path, d: Path, repack: bool) -> Path:
    p = d / "pack"
    files = model + drafter
    rec = read_json(p / "sources.json").get("files")
    if not repack and (p / "tensors.txt").is_file() and rec == stamp(files):
        say(f"pack up to date: {p}")
        return p
    why = "REPACK=1" if repack else ("no pack yet" if not (p / "tensors.txt").is_file() else
                                     "a file changed after the pack was built")
    say(f"building the pack ({why}; seconds: it indexes the files, no weights copied) ...")
    tmp = d / "pack.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    # --shards names the indexers side file too (pack_gguf.py picks it out by its -maya-indexers.gguf name), as
    # Maya's setup passes them; --companions: the folder holding dspark/ (no vision tower there: text only)
    r = subprocess.run([str(MAYA / ".venv/bin/python"), str(MAYA / "tools/mm3/pack_gguf.py"),
                        "--model", str(model[0].parent), "--out", str(tmp), "--companions", str(companions),
                        "--shards", *[f.name for f in model]], cwd=str(MAYA))
    if r.returncode != 0:
        die(f"tools/mm3/pack_gguf.py stopped (exit {r.returncode}); the old pack, if any, is unchanged")
    write_json(tmp / "sources.json", {"files": stamp(files)})
    if p.exists():
        for keep in ("prefill-tune.txt", "slots"):   # the engine's prefill calibration for this PC, saved conversations
            if (p / keep).exists():
                shutil.move(str(p / keep), str(tmp / keep))
        shutil.rmtree(p)
    os.replace(tmp, p)
    say(f"pack built: {p}")
    return p


def config(tag: str, d: Path, p: Path, a) -> Path:
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    keys, sampling, args, notes = mm3_settings.settings()   # checked once already at the start of main()
    cfg = {
        "exe": str(MAYA / "build/maya-mm3"),
        # a GGUF pack runs as a layer split over the GPUs (maya-mm3's tensor parallel is its EXL3 path's); experts
        # placed by the shipped profile, as Maya's setup does; K / V streamed (32K positions of each layer in VRAM,
        # every one in pinned RAM), Maya's setup default
        "args": ["--pack", str(p), "--max-context", str(a.context),
                 "--profile", str(MAYA / "data/expert-profile-mm3.bin"), "--kv-streaming", "on", "--vision", "off"],
        "cwd": str(MAYA),
        "tokenizer": str(p / "tokenizer"),
        "model_name": mm3_settings.DEFAULT_NAME,     # SERVED_NAME replaces it (mm3_settings.py)
        "gpu": gpus,
        "sampling": sampling,                         # the card's 1.0 / 0.95 unless TEMP / TOP_P / ... say otherwise
        "lib_dirs": [],
        "port": 8080,
        "host": "0.0.0.0",
        "log": str(d / "maya-mm3.log"),
        "installer": {"family": "minimax", "mm3_quant": MM3_QUANT[tag], "quant": tag,
                      "written_by": "club-3090 mm3_prepare.py"},
    }
    cfg.update(keys)                                  # reasoning_effort (auto unless set), thinking_budget, model_name
    cfg["args"] += args                               # --kv-cache, --dspark, --cpu-threads
    for n in notes:
        say(n)
    out = d / "maya-mm3.json"
    write_json(out, cfg)
    shown = ", ".join([f"{k}={v}" for k, v in keys.items()] + [f"sampling.{k}={v}" for k, v in sampling.items()]
                      + [" ".join(args)])
    say(f"config: {out} (GPUs {gpus}, context {a.context}; {shown})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(WEIGHTS))
    ap.add_argument("--src", default="/gguf")
    ap.add_argument("--companions", default="/companions")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--context", type=int, default=204800)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--verify", default="1")
    ap.add_argument("--repack", default="0")
    a = ap.parse_args()
    try:
        mm3_settings.settings()                       # a bad setting stops here, before the SHA-256 check and the pack
    except mm3_settings.Bad as e:
        die(str(e))
    d = Path(a.data) / a.tag
    d.mkdir(parents=True, exist_ok=True)
    companions = Path(a.companions)
    model, drafter = files_of(a.tag, Path(a.src), companions / "dspark")
    if a.verify != "0":
        verify(a.tag, model + drafter, d)
    else:
        say("VERIFY=0: the files' SHA-256 is not checked")
    p = pack(model, drafter, companions, d, a.repack == "1")
    config(a.tag, d, p, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
