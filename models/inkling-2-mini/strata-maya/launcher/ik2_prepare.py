"""Prepare Inkling-2-Mini for Project Maya's maya-ik2 engine on the data volume, for ik2-entrypoint.sh.

The compose mounts the files read-only into /gguf under the names Maya's own setup gives them (tools/ik2/download.json
at v1.1.75): unsloth's UD-Q4_K_XL GGUF of the model (Unsloth's looped layout, which maya-ik2 reads in place) and the
model's own drafter as unsloth's GGUF of it. maya-ik2 needs no pack: it reads the GGUF itself. What Maya's setup
writes next to the GGUF (maya.py ik2_tokenizer_step: the tokenizer, from the GGUF, with thinkingmachines' own chat
template in place of the conversion's) goes to /data instead, since the model folder is read-only:

    /data/<tag>/maya-ik2/tokenizer/  tools/strata_tokenizer.py's export, round-trip checked (--check)
    /data/<tag>/maya-ik2/sources.json  what it was built from (the GGUF's size, the two tools' SHA-256)
    /data/<tag>/verified.json        the files' sizes and times when their SHA-256 matched
    /data/<tag>/ik2-drafting.txt     maya-ik2's --drafting auto record (measured once on the PC; IK2_DRAFT_FILE)
    /data/<tag>/maya-ik2.json        the config serve/server.py starts from
    /data/<tag>/maya-ik2.log         the engine's log

Steps, each skipped when already done: check the files' SHA-256 against WEIGHTS (Maya's own manifest's; a minute, once,
again only when a file's size or time changes); export the tokenizer (seconds; again when the GGUF or the tools
changed); write the config. Python stdlib only, plus the image's /opt/maya."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import ik2_settings                                   # beside this file: club-3090's launch settings

MAYA = Path("/opt/maya")
TOKENIZER_TOOL = MAYA / "tools/strata_tokenizer.py"
TEMPLATE = MAYA / "tools/ik2/chat_template.jinja"     # thinkingmachines' own (maya.py IK2_TEMPLATE)
# The files, as Maya's own manifest pins them (tools/ik2/download.json at v1.1.75: the model from
# unsloth/Inkling-2-Mini-GGUF @ae9327c5, the drafter from peasantsmith/Inkling-2-Mini-Maya-GGUF @37027719). The names
# are the ones in /gguf. The tag is the quant's folder on the data volume, which the other Maya models' quants share:
# "ik2-", so it never meets another model's folder. Maya's own key for the quant (tools/maya_quants.py IK2_QUANT) and
# its name name the files on the dashboard, /health and /metrics (the installer record).
WEIGHTS = {
    "ik2-ud-q4_k_xl": {
        "model": ("Inkling-2-Mini-UD-Q4_K_XL.gguf",
                  "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15"),
        "drafter": ("dspark-Inkling-2-Mini-Q8_0.gguf",
                    "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"),
        "quant": "ud-q4_k_xl",                       # maya_quants.IK2_QUANT
        "name": "unsloth UD-Q4_K_XL",                # maya_quants.OTHERS[("ik2", "ud-q4_k_xl")]
    },
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


def files_of(tag: str, src: Path) -> dict[str, Path]:
    """The tag's files in /gguf by role, all present, else a stop naming what's missing."""
    w = WEIGHTS[tag]
    f = {role: src / w[role][0] for role in ("model", "drafter")}
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


def verify(tag: str, f: dict[str, Path], d: Path) -> None:
    rec = d / "verified.json"
    files = list(f.values())
    now = stamp(files)
    if read_json(rec).get("files") == now:
        say("files already verified")
        return
    gb = sum(p.stat().st_size for p in files) / 1e9
    say(f"checking {len(files)} files' SHA-256 (once; about a minute for {gb:.0f} GB) ...")
    bad = []
    for role, p in f.items():
        want = WEIGHTS[tag][role][1]
        got = sha256(p)
        if got != want:
            bad.append(f"{p.name}: {got}, want {want}")
    if bad:
        die("SHA-256 mismatch (a damaged or partial download; download it again): " + "; ".join(bad))
    write_json(rec, {"tag": tag, "files": now})
    say("every file matches its SHA-256")


def tokenizer(model: Path, d: Path, rebuild: bool) -> Path:
    """The tokenizer from the GGUF with thinkingmachines' chat template (maya.py ik2_tokenizer_step), on /data."""
    out = d / "maya-ik2"
    tok = out / "tokenizer"
    tools = {p.name: sha256(p) for p in (TOKENIZER_TOOL, TEMPLATE)}
    want = {"tools": tools, "gguf": {"name": model.name, "size": model.stat().st_size}}
    rec = read_json(out / "sources.json")
    have = all((tok / n).exists() for n in ("vocab.json", "chat_template.jinja"))
    if not rebuild and have and {k: rec.get(k) for k in want} == want:
        say(f"tokenizer up to date: {tok}")
        return tok
    why = "REPACK=1" if rebuild else ("no tokenizer yet" if not have else "the GGUF or the tools changed")
    say(f"exporting the tokenizer ({why}; seconds) ...")
    tmp = d / "maya-ik2.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    r = subprocess.run([str(MAYA / ".venv/bin/python"), str(TOKENIZER_TOOL), "--gguf", str(model), "--out", str(tmp),
                        "--chat-template", str(TEMPLATE), "--check"], cwd=str(MAYA))
    if r.returncode != 0 or not (tmp / "tokenizer" / "vocab.json").exists():
        die(f"tools/strata_tokenizer.py stopped (exit {r.returncode}); the old tokenizer, if any, is unchanged")
    write_json(tmp / "sources.json", {**want, "files": [{"name": model.name, "size": model.stat().st_size}]})
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    say(f"tokenizer: {tok} (thinkingmachines' chat template)")
    return tok


def config(tag: str, d: Path, f: dict[str, Path], tok: Path, a) -> Path:
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    keys, sampling, args, env, notes = ik2_settings.settings()   # checked once already at the start of main()
    w = WEIGHTS[tag]
    drafting = args[args.index("--drafting") + 1]
    # Maya's setup for one card (maya.py write_config_ik2): the GGUF, the context, the drafter beside it with
    # --drafting (auto: measured once at the engine's first start), no --gpus (one card), the cache in FP8 (IK2_KV).
    # Not IK2_WARM: on a 24 GB card every expert is in VRAM (the profile seeds only the offload path's tier)
    cfg = {
        "exe": str(MAYA / "build/maya-ik2"),
        "args": ["--model", str(f["model"]), "--max-context", str(a.context)] +
                (["--draft", str(f["drafter"])] if drafting != "off" else []) + args,
        "cwd": str(MAYA),
        "tokenizer": str(tok),
        "model_name": ik2_settings.DEFAULT_NAME,      # SERVED_NAME replaces it (ik2_settings.py)
        "gpu": gpus,
        "sampling": sampling,                         # 1.0 / 1.0 unless TEMP / TOP_P / ... say otherwise
        "lib_dirs": [],
        "port": 8080,
        "host": "0.0.0.0",
        "log": str(d / "maya-ik2.log"),
        # IK2_KV (the KV setting) and where --drafting auto keeps its choice: the server's own default is inside the
        # image (/opt/maya/data), which a new container doesn't have, so every start would measure again
        "env": {**env, "IK2_DRAFT_FILE": str(d / "ik2-drafting.txt")},
        # (Maya's installer record: family / ik2_quant / quant name the files on the dashboard, /health and /metrics,
        #  and pick the model's defaults - maya_quants.is_ik2_mini; its "drafting" is the setup's own bookkeeping,
        #  which the server doesn't read - drafting is --drafting above)
        "installer": {"family": "ik2", "quant": w["name"], "ik2_quant": w["quant"],
                      "experimental": True, "written_by": "club-3090 ik2_prepare.py"},
    }
    cfg.update(keys)                                  # reasoning_effort (high unless set), model_name, loop_guard
    for n in notes:
        say(n)
    out = d / "maya-ik2.json"
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
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--verify", default="1")
    ap.add_argument("--repack", default="0")
    a = ap.parse_args()
    try:
        ik2_settings.settings()                       # a bad setting stops here, before the SHA-256 check
    except ik2_settings.Bad as e:
        die(str(e))
    d = Path(a.data) / a.tag
    d.mkdir(parents=True, exist_ok=True)
    f = files_of(a.tag, Path(a.src))
    if a.verify != "0":
        verify(a.tag, f, d)
    else:
        say("VERIFY=0: the files' SHA-256 is not checked")
    tok = tokenizer(f["model"], d, a.repack == "1")
    config(a.tag, d, f, tok, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
