"""Genesis PN30 DS conv-state dst-shaped temp fix (setup-time patch).

This patches Sandermage's Genesis PN30 wiring in our local checkout.
PN30 originally handled DS layout + speculative decode by materializing:

    state[src_block_id, :, offset:].contiguous()

and then raw-memcpying that compact buffer into `state[dest_block_id]`.
That is not layout-correct for DS. The source tail is compact, but the
destination block is still strided by the full conv state length, so row 1+
land at the wrong address and corrupt the conv state.

The corrected path lives in `collect_mamba_copy_meta`, where both source and
destination block ids are known. For DS conv offset > 0, it builds a full
destination-shaped temporary block, copies the source tail into the destination
prefix columns, then gives batch_memcpy a normal contiguous full-block copy:

    tmp = state[dest_block_id].clone()
    tmp[..., :tail].copy_(state[src_block_id, ..., offset:offset + tail])

PN30's existing temp tensor list and post-batch stream sync/clear are reused.
The old compact path in `get_conv_copy_spec` is changed to fail closed; if the
collect-time bypass ever misses, we crash with a clear error instead of
silently corrupting the DS conv state.
"""
from __future__ import annotations

import os
import sys


TARGET = (
    "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/wiring/"
    "spec_decode/patch_N30_ds_layout_spec_decode_align.py"
)

PATCH_NAME = "pn30_dst_shaped_temp_fix"
MARKER = "club-3090: PN30 dst-shaped DS temp wiring v1"
RUNTIME_MARKER = "# club-3090: PN30 dst-shaped DS temp v1"


PART1_COMPACT_SNIPPET = '''    "        # [Genesis PN30 issue #17 fix] Make non-contiguous slice contiguous\\n"
    "        # and retain reference until next batch (cleared by patched\\n"
    "        # do_mamba_copy_block after stream sync). Replaces the upstream\\n"
    "        # NotImplementedError that blocked all spec-decode AL>1 + DS configs.\\n"
    "        if offset > 0:\\n"
    "            src_state = state[src_block_id, :, offset:].contiguous()\\n"
    "            try:\\n"
    "                _GENESIS_PN30_TEMP_TENSORS.append(src_state)\\n"
    "                _GENESIS_PN30_FLAG[0] = True\\n"
    "            except NameError:\\n"
    "                pass  # PN30 not loaded — defensive fallback\\n"
    "        else:\\n"
    "            src_state = state[src_block_id]\\n"
'''

PART1_FAIL_CLOSED_SNIPPET = f'''    "        # [Genesis PN30 issue #17 fix] {RUNTIME_MARKER}.\\n"
    "        # DS offset>0 is handled in collect_mamba_copy_meta, where the\\n"
    "        # destination block id is available and a dst-shaped temp can be\\n"
    "        # built without corrupting row strides. If this path is reached,\\n"
    "        # fail closed rather than using PN30's old compact-temp copy.\\n"
    "        if offset > 0:\\n"
    "            raise RuntimeError(\\n"
    "                \\"[Genesis PN30 club-3090] DS conv state offset>0 \\"\\n"
    "                \\"must be handled by collect_mamba_copy_meta's \\"\\n"
    "                \\"dst-shaped temp path; refusing compact copy.\\"\\n"
    "            )\\n"
    "        src_state = state[src_block_id]\\n"
'''


PART3_ANCHOR = '''def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset
'''

PART3_REPLACEMENT = f'''def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset
    num_accepted_tokens = accept_token_bias + 1

    try:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            _GENESIS_PN30_FLAG,
            _GENESIS_PN30_TEMP_TENSORS,
            get_conv_copy_spec as _GENESIS_PN30_GET_CONV_COPY_SPEC,
            is_conv_state_dim_first as _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST,
        )
    except (ImportError, AttributeError):
        _GENESIS_PN30_FLAG = None
        _GENESIS_PN30_TEMP_TENSORS = None
        _GENESIS_PN30_GET_CONV_COPY_SPEC = None
        _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST = None

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                is_conv_copy_func = (
                    state_copy_func is _GENESIS_PN30_GET_CONV_COPY_SPEC
                    or getattr(state_copy_func, "__name__", "")
                    == "get_conv_copy_spec"
                )
                if (
                    num_accepted_tokens > 1
                    and is_conv_copy_func
                    and _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST is not None
                    and _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST()
                    and state.dim() >= 3
                ):
                    # {RUNTIME_MARKER}
                    # DS layout stores each block as (..., state_len). The
                    # source tail is strided by the full state_len, and the
                    # destination prefix must keep that same row stride. Build
                    # a full dst-shaped temp, patch in the source tail, then
                    # memcpy the whole block as one contiguous copy entry.
                    src_block_id = block_ids[src_block_idx]
                    token_offset = num_accepted_tokens - 1
                    state_len = int(state.shape[-1])
                    tail = max(state_len - int(token_offset), 0)
                    tmp_state = state[dest_block_id].clone()
                    if tail > 0:
                        tmp_state[..., :tail].copy_(
                            state[src_block_id, ..., token_offset:token_offset + tail]
                        )
                    if _GENESIS_PN30_TEMP_TENSORS is not None:
                        _GENESIS_PN30_TEMP_TENSORS.append(tmp_state)
                    if _GENESIS_PN30_FLAG is not None:
                        _GENESIS_PN30_FLAG[0] = True

                    src_ptrs_np[offset] = tmp_state.data_ptr()
                    dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                    sizes_np[offset] = tmp_state.numel() * state.element_size()
                    offset += 1
                    continue

                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, num_accepted_tokens
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset
'''

CONSTANTS_BLOCK = f'''
# {MARKER}
# Sub-patch 3: v1/worker/mamba_utils.py:collect_mamba_copy_meta.
# Corrects PN30's DS offset>0 path by materializing a dst-shaped temp
# instead of compacting only the source tail.
PN30_PART3_ANCHOR = {PART3_ANCHOR!r}

PN30_PART3_REPLACEMENT = {PART3_REPLACEMENT!r}
'''

CONSTANTS_ANCHOR = "\n\ndef _make_patcher_part1() -> TextPatcher | None:\n"

PART3_FUNCTION_ANCHOR = "\n\ndef apply() -> tuple[str, str]:\n"
PART3_FUNCTION_BLOCK = '''

def _make_patcher_part3() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN30 v1/worker/mamba_utils.py — collect_mamba_copy_meta "
            "dst-shaped DS temp (issue #17)"
        ),
        target_file=str(target),
        marker=GENESIS_PN30_MARKER + " part3 dst-shaped-temp",
        sub_patches=[
            TextPatch(
                name="pN30_collect_mamba_copy_meta_dst_shaped_temp",
                anchor=PN30_PART3_ANCHOR,
                replacement=PN30_PART3_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "club-3090: PN30 dst-shaped DS temp",
        ],
    )
'''

OLD_APPLY = '''def apply() -> tuple[str, str]:
    """Apply PN30 — DS layout spec-decode AL>1 fix (two-file text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN30")
    log_decision("PN30", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Both files must patch successfully — partial application would
    # leave the system in inconsistent state (one half of the
    # coordinated fix without the other).
    p1 = _make_patcher_part1()
    p2 = _make_patcher_part2()
    if p1 is None or p2 is None:
        return "skipped", (
            "target file(s) not resolvable — vllm tree may differ "
            "from expected layout"
        )

    r1, f1 = p1.apply()
    if r1 == TextPatchResult.FAILED:
        return "failed", (
            f"PN30 part1 (mamba_utils.py:get_conv_copy_spec) failed: "
            f"{f1.detail if f1 else 'unknown'}"
        )

    r2, f2 = p2.apply()
    if r2 == TextPatchResult.FAILED:
        # Partial patch state — log warning. Part1 stays applied;
        # cleanup will not run but the contiguous() fix itself is
        # correct (just leaks a small list of tensors per batch).
        log.warning(
            "[PN30] part2 (do_mamba_copy_block) failed: %s — part1 "
            "applied but cleanup will not fire. Tensor list will grow "
            "per batch until process restart. Recommend disabling PN30 "
            "until both halves can apply.",
            f2.detail if f2 else "unknown",
        )
        return "failed", "PN30 partial application — see warning"

    # Both halves applied (or skipped if anchors missing — drift-safe)
    return result_to_wiring_status(
        r1 if r1 != TextPatchResult.APPLIED else r2,
        f1 if r1 != TextPatchResult.APPLIED else f2,
        applied_message=(
            "PN30 applied: DS conv state layout + spec-decode AL>1 path "
            "now uses contiguous-copy + delayed cleanup. Two-file patch — "
            "mamba_utils.py:get_conv_copy_spec replaces NotImplementedError "
            "with .contiguous() copy + temp-tensor list; "
            "v1/worker/mamba_utils.py:do_mamba_copy_block adds stream sync "
            "+ list clear after batch_memcpy when DS+offset>0 path used. "
            "Closes issue #17. Cost: ~10-50us per batch when path active."
        ),
        patch_name="PN30 DS layout + spec-decode AL>1",
    )
'''

NEW_APPLY = '''def apply() -> tuple[str, str]:
    """Apply PN30 — DS layout spec-decode AL>1 fix (three-file text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN30")
    log_decision("PN30", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # All three coordinated patches must be present. In particular, part3
    # bypasses PN30's original compact-temp path; without it, DS offset>0 can
    # corrupt row strides. Treat skipped required anchors as failed when PN30
    # is explicitly enabled.
    p1 = _make_patcher_part1()
    p2 = _make_patcher_part2()
    p3 = _make_patcher_part3()
    if p1 is None or p2 is None or p3 is None:
        return "skipped", (
            "target file(s) not resolvable — vllm tree may differ "
            "from expected layout"
        )

    patch_results = [
        ("part1 mamba_utils.py:get_conv_copy_spec", *p1.apply()),
        ("part2 v1/worker/mamba_utils.py:do_mamba_copy_block", *p2.apply()),
        ("part3 v1/worker/mamba_utils.py:collect_mamba_copy_meta", *p3.apply()),
    ]
    for label, result, failure in patch_results:
        if result not in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
            reason = failure.reason if failure else "unknown"
            detail = failure.detail if failure and failure.detail else "unknown"
            return "failed", f"PN30 {label} did not apply safely: {reason} — {detail}"

    status_result = (
        TextPatchResult.APPLIED
        if any(r == TextPatchResult.APPLIED for _, r, _ in patch_results)
        else TextPatchResult.IDEMPOTENT
    )
    return result_to_wiring_status(
        status_result,
        None,
        applied_message=(
            "PN30 applied: DS conv state layout + spec-decode AL>1 now "
            "uses collect_mamba_copy_meta dst-shaped temp blocks for DS "
            "conv offset>0, preserving destination row stride. "
            "get_conv_copy_spec fails closed if the collect-time bypass is "
            "missed; do_mamba_copy_block keeps PN30's stream sync + temp "
            "clear lifecycle."
        ),
        patch_name="PN30 DS layout + spec-decode AL>1",
    )
'''


def _read(path: str) -> str | None:
    if not os.path.isfile(path):
        print(f"[{PATCH_NAME}] target not found: {path}", file=sys.stderr)
        return None
    with open(path, "r") as f:
        return f.read()


def _write(path: str, src: str) -> None:
    with open(path, "w") as f:
        f.write(src)


def _replace_once(src: str, old: str, new: str, label: str) -> tuple[str, bool, bool]:
    if old not in src:
        print(f"[{PATCH_NAME}] {label} anchor not found", file=sys.stderr)
        return src, False, False
    return src.replace(old, new, 1), True, True


def main() -> int:
    src = _read(TARGET)
    if src is None:
        return 1

    changed = False

    if PART1_FAIL_CLOSED_SNIPPET not in src:
        src, ok, did_change = _replace_once(
            src,
            PART1_COMPACT_SNIPPET,
            PART1_FAIL_CLOSED_SNIPPET,
            "part1 fail-closed compact-copy replacement",
        )
        if not ok:
            return 1
        changed = changed or did_change

    if MARKER not in src:
        src, ok, did_change = _replace_once(
            src,
            CONSTANTS_ANCHOR,
            "\n" + CONSTANTS_BLOCK + CONSTANTS_ANCHOR,
            "part3 constants insertion",
        )
        if not ok:
            return 1
        changed = changed or did_change

    if "_make_patcher_part3" not in src:
        src, ok, did_change = _replace_once(
            src,
            PART3_FUNCTION_ANCHOR,
            PART3_FUNCTION_BLOCK + PART3_FUNCTION_ANCHOR,
            "part3 patcher insertion",
        )
        if not ok:
            return 1
        changed = changed or did_change

    if "p3 = _make_patcher_part3()" not in src:
        src, ok, did_change = _replace_once(
            src,
            OLD_APPLY,
            NEW_APPLY,
            "apply() three-part coordination replacement",
        )
        if not ok:
            return 1
        changed = changed or did_change

    if changed:
        _write(TARGET, src)
        print(f"[{PATCH_NAME}] patched Genesis PN30 with dst-shaped DS temp")
    else:
        print(f"[{PATCH_NAME}] already applied")
    print(
        f"[{PATCH_NAME}] done — PN30 now bypasses compact DS tail copies "
        "and fails closed if the bypass is missed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
