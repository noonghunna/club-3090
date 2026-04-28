"""
Disk-edit patch for vllm/v1/attention/backends/turboquant_attn.py: fix CUDA
graph capture crashes at `.tolist()` calls inside the backend.

There are two sites that force GPU->CPU syncs via `.tolist()` inside code
paths that can execute under active CUDA stream capture:

  A) forward() mixed-batch branch:
       prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
     Hit when the engine builds a mixed batch (decodes + prefills) during
     warmup/capture.

  B) _prefill_attention() continuation branch:
       qsl = query_start_loc.tolist()
       seq_lens_list = attn_metadata.seq_lens.tolist()
     Hit when max_query_len != max_seq_len during warmup/capture (which is
     what happens with spec-decode + chunked-prefill: warmup's dummy batch
     simulates continuation chunks).

Fix:
  A) During capture, use `attn_metadata.max_seq_len` (Python int, batch-level
     upper bound — safe overestimate for the prefill portion; flash_attn
     just uses it as a grid sizing bound).
  B) During capture, early-return the graph-safe fast path
     (flash_attn_varlen_func with cu_seqlens == query_start_loc). Attention
     is a splitting_op in V1 PIECEWISE mode, so capture-time values
     only drive memory profiling, not graph content. At inference (non-
     capture), the original correct continuation path runs.

This patcher discovers vLLM's install path via `import vllm` so it works on
docker (`dist-packages`) AND host venvs (`site-packages`). Uses small,
whitespace-tolerant regex anchors so it survives upstream nightly-to-nightly
churn around the patched lines.

Genesis v7.x ships an attribution-credited backport as P78
(`patch_78_tolist_capture_guard.py`); if you're already loading the Genesis
tree with `GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1` you can skip this
patcher. They patch the same line — running both is harmless (idempotent
guard), running just one is sufficient.

Runs AFTER Genesis patches (Site B's anchor survives Genesis's disk edits;
Site A is in a different function Genesis doesn't touch).
"""

import logging
import os
import re

log = logging.getLogger("tolist_cudagraph_fix")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

PATCH_TAG = "[tolist_cudagraph_fix]"


def _find_target():
    """Auto-discover turboquant_attn.py via vllm import.

    Falls back to common docker path if import fails (e.g. when this script
    is invoked outside the vLLM venv but the file path is known).
    """
    try:
        import vllm
        vllm_dir = os.path.dirname(vllm.__file__)
        path = os.path.join(vllm_dir, "v1/attention/backends/turboquant_attn.py")
        if os.path.exists(path):
            return path
    except ImportError:
        pass

    # Fallbacks for non-imported invocation
    fallbacks = [
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py",
        "/usr/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py",
    ]
    for path in fallbacks:
        if os.path.exists(path):
            return path
    return None


# Site B: insert capture-guard early-return at the START of the
# `_prefill_attention` function body. Anchor on `N, Hq, D = query.shape`
# which is the first executable line in the function body (and unique in
# the file). We insert immediately AFTER this line.
SITE_B_ANCHOR_RE = re.compile(
    r"^(        N, Hq, D = query\.shape\n)",
    re.MULTILINE,
)

SITE_B_INSERT = """
        # %s During CUDA graph capture, the continuation branch below calls
        # .tolist() which forces a GPU->CPU sync — illegal under
        # torch.cuda.graph(). vLLM V1 PIECEWISE mode lists
        # unified_attention_with_output as a splitting_op, so the captured
        # piece does not include attention outputs; capture-time values only
        # need to drive memory profiling. Fall back to the graph-safe fast
        # path. At inference (non-capture), is_current_stream_capturing()
        # returns False and the original continuation path runs unchanged.
        if torch.cuda.is_current_stream_capturing():
            if _HAS_FLASH_ATTN:
                return flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.max_query_len,
                    softmax_scale=self.scale,
                    causal=True,
                )
            return torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
""" % PATCH_TAG

# Site A: forward() mixed-batch prefill_max_seq tolist.
# Single-line anchor — `prefill_max_seq = max(...)` is unique in the file.
SITE_A_OLD_RE = re.compile(
    r"^(            prefill_max_seq = max\(attn_metadata\.seq_lens\[num_decodes:\]\.tolist\(\)\))$",
    re.MULTILINE,
)

SITE_A_NEW_TEMPLATE = (
    "            # %s During CUDA graph capture, substitute a safe upper\n"
    "            # bound (batch-level max_seq_len, a Python int) to avoid the\n"
    "            # tolist() sync. Overestimates prefill_max_seq but flash_attn\n"
    "            # uses it only as a grid upper bound; non-capture takes else.\n"
    "            if torch.cuda.is_current_stream_capturing():\n"
    "                prefill_max_seq = attn_metadata.max_seq_len\n"
    "            else:\n"
    r"    \1"
) % PATCH_TAG


def _apply_site_b(src):
    if PATCH_TAG in src:
        return src, "skip-already-applied"
    m = SITE_B_ANCHOR_RE.search(src)
    if not m:
        return src, "anchor-not-found"
    return src[: m.end()] + SITE_B_INSERT + src[m.end():], "applied"


def _apply_site_a(src):
    if SITE_A_OLD_RE.search(src) is None:
        return src, "anchor-not-found"
    if PATCH_TAG in src and "prefill_max_seq = attn_metadata.max_seq_len" in src:
        return src, "skip-already-applied"
    return SITE_A_OLD_RE.sub(SITE_A_NEW_TEMPLATE, src, count=1), "applied"


def main():
    target = _find_target()
    if target is None:
        log.error("%s vLLM turboquant_attn.py not found via import or fallbacks", PATCH_TAG)
        return 1
    log.info("%s target: %s", PATCH_TAG, target)

    with open(target, "r") as f:
        src = f.read()
    original_src = src

    if PATCH_TAG in src:
        log.info("%s already applied (idempotent)", PATCH_TAG)
        return 0

    src, status_b = _apply_site_b(src)
    log.info("%s Site B (_prefill_attention): %s", PATCH_TAG, status_b)
    src, status_a = _apply_site_a(src)
    log.info("%s Site A (forward mixed-batch): %s", PATCH_TAG, status_a)

    if src == original_src:
        log.error(
            "%s NO sites patched — file unchanged. "
            "Sites: A=%s B=%s. Either upstream changed beyond our anchor "
            "tolerance, or Genesis P78 already supplies the equivalent guard.",
            PATCH_TAG, status_a, status_b,
        )
        return 1

    with open(target, "w") as f:
        f.write(src)
    log.info(
        "%s Patched %s. Site A=%s, Site B=%s",
        PATCH_TAG, target, status_a, status_b,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
