#!/usr/bin/env python3
"""Vendored vllm#48375 — honor `drop_eagle_block` in MambaManager.find_longest_cache_hit.

Upstream bug (vllm#43559): with MTP/EAGLE spec-decode + prefix caching on a
Qwen3-Next hybrid, MambaManager receives drop_eagle_block but ignores it, so the
final matched page's recurrent-state snapshot can reflect draft tokens that
verification later rejected. Fix (upstream PR #48375, open): lower the hit-search
ceiling by one page.

Inert under the shipped prefix-off default (the patched path only runs when
prefix caching is enabled). Idempotent; anchor-checked; exits non-zero on drift
so the entrypoint can refuse to boot a silently-unpatched configuration.
"""
import io
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py"
MARKER = "drop_eagle_block and max_num_blocks > 0"
ANCHOR = "        max_num_blocks = max_length // block_size\n"
INSERT = """        if drop_eagle_block and max_num_blocks > 0:
            # club-3090 vendored vllm#48375 (see patches.yml). EAGLE/MTP: drop the
            # final matched page -- its recurrent-state snapshot may reflect draft
            # tokens that verification later rejected. Mamba keeps only the
            # rightmost real block, so lower the search ceiling by one page
            # instead of popping (a pop would delete the state block itself).
            max_num_blocks -= 1
"""

def main() -> int:
    try:
        src = io.open(TARGET, encoding="utf-8").read()
    except OSError as e:
        print(f"[pr48375] REFUSE: cannot read target: {e}")
        return 1
    if MARKER in src:
        print("[pr48375] already patched (idempotent no-op)")
        return 0
    cls = src.find("class MambaManager")
    if cls < 0:
        print("[pr48375] REFUSE: anchor drift - class MambaManager not found")
        return 1
    i = src.find(ANCHOR, cls)
    if i < 0:
        print("[pr48375] REFUSE: anchor drift - max_num_blocks line not found in MambaManager")
        return 1
    j = i + len(ANCHOR)
    io.open(TARGET, "w", encoding="utf-8").write(src[:j] + INSERT + src[j:])
    print("[pr48375] applied: MambaManager now honors drop_eagle_block")
    return 0

if __name__ == "__main__":
    sys.exit(main())
