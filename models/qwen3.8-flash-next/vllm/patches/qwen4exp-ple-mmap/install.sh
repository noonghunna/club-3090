#!/usr/bin/env bash
# qwen4exp-ple-mmap installer: runs in the compose entrypoint before `vllm serve`.
# Applies the mmap PLE port (ple-mmap-v<version>.diff) to the image's vllm package.
# Only active when VLLM_PLE_MMAP_PATH is set; the patched code falls back to stock
# behaviour otherwise, but we skip the patch then to keep the RAM arm pure stock.
# Idempotent: a second run is a verified no-op. Wrong vLLM version or a failed hunk -> exit 1.
set -u
TAG="[qwen4exp-ple-mmap]"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${VLLM_PLE_MMAP_PATH:-}" ]; then
  echo "$TAG VLLM_PLE_MMAP_PATH unset: not applying (PLE stays on the stock path)"
  exit 0
fi
site="$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin)))')" || exit 1
ver="$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')" || exit 1
diff_file="$here/ple-mmap-v${ver%%+*}.diff"
if [ ! -f "$diff_file" ]; then
  echo "$TAG REFUSE: no port for vLLM $ver (have: $(cd "$here" && ls ple-mmap-v*.diff | tr '\n' ' '))" >&2
  exit 1
fi
target="$site/vllm/models/qwen4_exp/nvidia/ngram_embedding.py"
if grep -q "class Qwen4ExpPLEMmapHostEmbedding" "$target"; then
  echo "$TAG already applied (vLLM $ver)"
  exit 0
fi
if ! (cd "$site" && patch -p1 --batch --forward --dry-run < "$diff_file" >/dev/null); then
  echo "$TAG REFUSE: $diff_file does not apply cleanly to vLLM $ver" >&2
  (cd "$site" && patch -p1 --batch --forward --dry-run < "$diff_file" | grep -E "FAILED|fuzz") >&2
  exit 1
fi
(cd "$site" && patch -p1 --batch --forward --no-backup-if-mismatch < "$diff_file" >/dev/null) || exit 1
python3 -m py_compile "$target" "$site/vllm/models/qwen4_exp/nvidia/model_state.py" || exit 1
echo "$TAG applied (vLLM $ver): PLE table mmap'd from $VLLM_PLE_MMAP_PATH"
