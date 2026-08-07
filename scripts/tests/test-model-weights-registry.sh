#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
READER="${ROOT_DIR}/scripts/lib/profiles/weights.py"

load_entry() {
  local key="$1"
  local env_lines
  env_lines="$(python3 "$READER" entry "$key")"
  eval "$env_lines"
}

load_lookup() {
  local path="$1"
  local env_lines
  env_lines="$(python3 "$READER" lookup "$path")"
  eval "$env_lines"
}

assert_entry() {
  local key="$1" repo="$2" subdir="$3"
  load_entry "$key"
  if [[ "$WEIGHT_REPO" != "$repo" || "$WEIGHT_SUBDIR" != "$subdir" ]]; then
    echo "ASSERTION FAILED: bad entry for $key" >&2
    echo "  repo:   got '$WEIGHT_REPO' expected '$repo'" >&2
    echo "  subdir: got '$WEIGHT_SUBDIR' expected '$subdir'" >&2
    exit 1
  fi
}

assert_lookup() {
  local path="$1" expected="$2"
  load_lookup "$path"
  if [[ "$WEIGHT_KEY" != "$expected" ]]; then
    echo "ASSERTION FAILED: lookup for $path -> $WEIGHT_KEY, expected $expected" >&2
    exit 1
  fi
}

assert_entry qwen3.6-27b:autoround-int4 Lorbus/Qwen3.6-27B-int4-AutoRound qwen3.6-27b-autoround-int4
assert_entry qwen3.6-27b:dflash z-lab/Qwen3.6-27B-DFlash qwen3.6-27b-dflash
assert_entry qwen3.6-27b:prism_eagle3 Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 qwen3.6-27b-prism-eagle3
assert_entry qwen3.6-27b:unsloth-q4km unsloth/Qwen3.6-27B-MTP-GGUF qwen3.6-27b-gguf/unsloth-mtp-q4km
assert_entry qwen3.6-27b:gguf_mmproj_f16 unsloth/Qwen3.6-27B-GGUF qwen3.6-27b-gguf
assert_entry qwen3.6-27b:ubergarm-iq4ks ubergarm/Qwen3.6-27B-GGUF qwen3.6-27b-gguf/ubergarm-mtp-iq4ks
assert_entry qwen3.6-35b-a3b:autoround-int4 Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound qwen3.6-35b-a3b-autoround-int4
assert_entry gemma-4-31b:autoround-int4 Intel/gemma-4-31B-it-int4-AutoRound gemma-4-31b-autoround-int4
assert_entry gemma-4-31b:awq cyankiwi/gemma-4-31B-it-AWQ-4bit gemma-4-31b-it-AWQ-4bit
assert_entry gemma-4-31b:google-qat-w4a16 google/gemma-4-31B-it-qat-w4a16-ct gemma-4-31b-google-qat-w4a16
assert_entry gemma-4-31b:assistant google/gemma-4-31B-it-assistant gemma-4-31b-it-assistant
assert_entry gemma-4-31b:dflash z-lab/gemma-4-31b-it-dflash gemma-4-31b-it-dflash
assert_entry gemma-4-26b-a4b:autoround-int4-mixed Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound gemma-4-26b-a4b-autoround-int4-mixed
assert_entry gemma-4-26b-a4b:awq cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit gemma-4-26b-a4b-awq-4bit
assert_entry gemma-4-26b-a4b:assistant google/gemma-4-26B-A4B-it-assistant gemma-4-26b-a4b-it-assistant
assert_entry qwen3.6-27b:carnice-bf16mtp wasifb/Carnice_V2_27B_INT4_BF16MTP carnice-v2-27b-int4-recipe-d-bf16mtp

load_entry qwen3.6-27b:qwopus-bf16mtp
[[ -z "$WEIGHT_REPO" ]]
[[ "$WEIGHT_SUBDIR" == "qwopus3.6-27b-int4-recipe-d-bf16mtp" ]]
[[ -n "$WEIGHT_MANUAL_NOTE" ]]

# Legacy preflight aliases remain accepted for user-facing setup hints.
load_entry qwen3.6-27b-gguf-iq4ks
[[ "$WEIGHT_KEY" == "qwen3.6-27b:ubergarm-iq4ks" ]]

assert_lookup qwen3.6-27b-autoround-int4/config.json qwen3.6-27b:autoround-int4
assert_lookup qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf qwen3.6-27b:unsloth-q4km
assert_lookup qwen3.6-27b-gguf/mmproj-F16.gguf qwen3.6-27b:gguf_mmproj_f16
assert_lookup qwen3.6-27b-gguf/ubergarm-mtp-iq4ks/Qwen3.6-27B-MTP-IQ4_KS.gguf qwen3.6-27b:ubergarm-iq4ks
assert_lookup qwen3.6-27b-prism-eagle3/compressed qwen3.6-27b:prism_eagle3
assert_lookup carnice-v2-27b-int4-recipe-d-bf16mtp/chat_template.jinja qwen3.6-27b:carnice-bf16mtp
assert_lookup qwopus3.6-27b-int4-recipe-d-bf16mtp/config.json qwen3.6-27b:qwopus-bf16mtp
assert_lookup gemma-4-31b-google-qat-w4a16/config.json gemma-4-31b:google-qat-w4a16

# `verify_glob` must match the entry's declared `format`.
#
# It DEFAULTS to `*.safetensors` (weights.py), so a GGUF entry that simply omits
# it gets a glob matching nothing — and both consumers fail SILENTLY-ish in ways
# that blame the wrong thing: c3 reports the weights absent while they sit on
# disk, and setup.sh's post-download verify counts zero files and exits with
# "download may have failed" AFTER a successful multi-hundred-GB pull. Nothing
# else catches it, because a mismatched glob is valid YAML and a valid glob.
python3 - <<'PY'
import pathlib, sys, yaml

bad = []
for f in sorted(pathlib.Path("scripts/lib/profiles/models").glob("*.yml")):
    doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    for key, meta in (doc.get("weights") or {}).items():
        if not isinstance(meta, dict):
            continue
        glob = meta.get("verify_glob") or "*.safetensors"   # weights.py default
        is_gguf = str(meta.get("format", "")).lower() == "gguf"
        if is_gguf != glob.endswith(".gguf"):
            bad.append(f"  {f.name}::{key}  format={meta.get('format')}  "
                       f"verify_glob={glob}"
                       f"{'  (DEFAULTED — declare it)' if not meta.get('verify_glob') else ''}")

if bad:
    print("verify_glob does not match declared format:", *bad, sep="\n", file=sys.stderr)
    sys.exit(1)
PY

# `files:` scopes the fetch, and `verify_glob` must agree with it.
#
# An entry with no `files:` downloads the ENTIRE hf_repo. That is correct for a
# whole-repo bucket, and catastrophic for one quant of a multi-quant repo — the
# DeepSeek repo holds 13 quants / 1,537 GB, so the unscoped form would pull all
# of it to serve one. Two entries sharing a local_subdir is the tell: they are
# slices of one repo, so each MUST name its files or they fetch the same superset
# into the same directory.
#
# And because `hf download --local-dir` PRESERVES repo folder structure, a file
# in a repo subfolder lands under that subfolder — so verify_glob has to carry
# the same prefix the files do. If the two ever drift, verification looks in a
# directory the download never wrote to and reports a good pull as a failure.
python3 - <<'PY'
import collections, fnmatch, pathlib, sys, yaml


def gmatch(name: str, pattern: str) -> bool:
    """Glob semantics as the CONSUMERS apply them, not fnmatch's.

    Both readers (`for f in $verify_glob` in bash, `Path.glob()` in c3) treat
    `*` as NOT crossing a `/`. Plain fnmatch does cross it, so `*.gguf` would
    falsely appear to match `UD-IQ2_XXS/x.gguf` and this guard would pass a
    prefix drift it exists to catch. Match segment-wise instead.
    """
    n, p = name.split("/"), pattern.split("/")
    return len(n) == len(p) and all(fnmatch.fnmatch(a, b) for a, b in zip(n, p))


bad = []
for f in sorted(pathlib.Path("scripts/lib/profiles/models").glob("*.yml")):
    doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    # Only FETCHABLE entries can misfetch. An entry with no `hf_repo` is a local
    # bucket placeholder (e.g. qwen3.6-27b::gguf) — it downloads nothing, so it
    # neither needs a files: filter nor makes a shared subdir dangerous.
    weights = {
        k: m for k, m in (doc.get("weights") or {}).items()
        if isinstance(m, dict) and m.get("hf_repo")
    }

    shared = collections.Counter(
        str(m.get("local_subdir") or m.get("path") or "") for m in weights.values()
    )
    for key, meta in weights.items():
        files = meta.get("files") or []
        if isinstance(files, str):
            files = [files]
        subdir = str(meta.get("local_subdir") or meta.get("path") or "")

        if not files and shared[subdir] > 1:
            bad.append(f"  {f.name}::{key}  shares local_subdir '{subdir}' with "
                       f"{shared[subdir] - 1} other entry(s) but declares no files: "
                       f"— it would fetch the whole repo into a shared dir")
            continue

        glob = meta.get("verify_glob") or "*.safetensors"
        if files and not any(gmatch(name, glob) for name in files):
            bad.append(f"  {f.name}::{key}  verify_glob={glob} matches NONE of its "
                       f"{len(files)} declared file(s) (e.g. {files[0]})")

if bad:
    print("files:/verify_glob inconsistency:", *bad, sep="\n", file=sys.stderr)
    sys.exit(1)
PY

echo "test-model-weights-registry: ok"
