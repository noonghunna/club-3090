"""
Disk-edit patch — backports vllm-project/vllm#40798 onto the pinned
nightly to test whether it fixes #40831 (TurboQuant + spec-decode +
cudagraph degenerate token loops).

⚠ RESEARCH ARTIFACT — does NOT fix #40831.

We tested this hypothesis: PR #40798 moves _tq_mid_o_buf / _tq_output_buf
/ _tq_lse_buf from per-layer register_buffer (B=max_num_seqs=1) to
WorkspaceManager.get_simultaneous() (persistent base-buffer with stable
data_ptr). Hypothesis: this stabilizes the captured-cudagraph addresses
across warmup-vs-runtime shape mismatch.

Result on our pinned nightly + this backport + cudagraph ON: bug persists.
Same shape as the originally-failing config — tool calls produce inline
<tool_call> cascade, long-context recall produces first-token loops,
streaming has token duplication. TPS ~96 confirms cudagraph + compile
genuinely active. So either (a) #40798 is necessary but not sufficient,
(b) a companion change in main we haven't backported is also required,
or (c) our backport has a subtle anchor mismatch that a clean
main + #40798 CI build would reveal.

Full data: vllm-project/vllm#40831 comment 4317503179.

Kept in the patches/ dir as documentation of the research path, NOT as a
shipped patch. The shipped workaround for #40831 is the
--compilation-config '{"cudagraph_mode":"NONE"}' flag in
docker-compose.longctx-experimental.yml.

Idempotent. Run AFTER patch_genesis_unified.py and patch_tolist_cudagraph.py.
"""

import logging
import os

log = logging.getLogger("patch_pr40798_workspace")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"

# ----------------------------------------------------------------------
# Site 1: attention.py — remove the per-layer register_buffer block
# ----------------------------------------------------------------------
SITE1_FILE = f"{VLLM}/model_executor/layers/attention/attention.py"

SITE1_OLD = """        # Pre-allocate decode intermediate buffers so model.to(device) moves
        # them to GPU *before* the memory profiler runs.  Without this the
        # profiler gives all free memory to KV cache blocks and the first
        # decode OOMs when these buffers are lazily allocated.
        _vllm_cfg = get_current_vllm_config()
        B = _vllm_cfg.scheduler_config.max_num_seqs
        Hq = self.num_heads
        S = _vllm_cfg.attention_config.tq_max_kv_splits_for_cuda_graph
        D = head_size
        self.register_buffer(
            "_tq_mid_o_buf",
            torch.empty(B, Hq, S, D + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_output_buf",
            torch.empty(B, Hq, D, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_lse_buf",
            torch.empty(B, Hq, dtype=torch.float32),
            persistent=False,
        )"""

SITE1_NEW = """        # [patch_pr40798] TQ decode scratch space is allocated through the v1
        # workspace manager at runtime. It is shared across layers, rather
        # than registered once per attention layer, so large max_num_seqs
        # values do not multiply the scratch memory by num_layers."""

# ----------------------------------------------------------------------
# Site 2: turboquant_attn.py — remove getattr block from _decode_attention
# ----------------------------------------------------------------------
SITE2_FILE = f"{VLLM}/v1/attention/backends/turboquant_attn.py"

SITE2_OLD = """    ) -> torch.Tensor:
        # Grab cached decode buffers from the layer (lazily allocated).
        mid_o_buf = output_buf = lse_buf = None
        if layer is not None:
            mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
            output_buf = getattr(layer, "_tq_output_buf", None)
            lse_buf = getattr(layer, "_tq_lse_buf", None)

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        return result"""

SITE2_NEW = """    ) -> torch.Tensor:
        # [patch_pr40798] buffers acquired from WorkspaceManager inside the
        # triton kernel launcher; no per-layer caching here.
        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        return result"""

# ----------------------------------------------------------------------
# Site 3: triton_turboquant_decode.py — three sub-edits
#   3a) drop `buf_holder: Any = None,` from the launcher signature
#   3b) drop `from typing import Any` (unused after 3a)
#   3c) insert workspace-manager fetch before the mid_o slice/alloc check
#   3d) drop the three `if buf_holder is not None: ... = ...` writes
# ----------------------------------------------------------------------
SITE3_FILE = f"{VLLM}/v1/attention/ops/triton_turboquant_decode.py"

SITE3A_OLD = """    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)"""
SITE3A_NEW = """    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)"""

SITE3B_OLD = """import math
from typing import Any

import torch"""
SITE3B_NEW = """import math

import torch"""

# Insert workspace fetch immediately AFTER `NUM_KV_SPLITS = max_num_kv_splits`
SITE3C_OLD = """    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o"""
SITE3C_NEW = """    NUM_KV_SPLITS = max_num_kv_splits

    if mid_o_buf is None or output_buf is None or lse_buf is None:
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        if is_workspace_manager_initialized():
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, NUM_KV_SPLITS, D + 1), torch.float32),
                    ((B, Hq, D), torch.float32),
                    ((B, Hq), torch.float32),
                )
            )

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )"""

SITE3D_OLD = """        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse"""
SITE3D_NEW = """        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)"""

# ----------------------------------------------------------------------
# Site 4: gpu_model_runner.py — add workspace reservation in capture_model
# ----------------------------------------------------------------------
SITE4_FILE = f"{VLLM}/v1/worker/gpu_model_runner.py"

SITE4A_OLD = "from vllm.v1.worker.workspace import lock_workspace"
SITE4A_NEW = "from vllm.v1.worker.workspace import current_workspace_manager, lock_workspace"

# Inject reservation call + helper method. Anchor: the @instrument decorator
# above capture_model in our pinned image, plus the function signature.
SITE4B_OLD = """    @instrument(span_name=\"Capture model\")
    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:"""
SITE4B_NEW = """    @instrument(span_name=\"Capture model\")
    def capture_model(self) -> int:
        # [patch_pr40798] reserve TQ decode scratch through workspace manager
        # so the captured cudagraph holds a stable data_ptr (fixes #40831).
        self._reserve_turboquant_decode_workspace()

        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:"""

# Helper method — append before _warmup_and_capture
SITE4C_OLD = """    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,"""
SITE4C_NEW = """    def _reserve_turboquant_decode_workspace(self) -> None:
        # [patch_pr40798]
        if not self.cache_config.cache_dtype.startswith(\"turboquant_\"):
            return
        if not self.attn_groups:
            return

        max_num_reqs = self.scheduler_config.max_num_seqs
        num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        max_num_splits = (
            self.vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        for groups in self.attn_groups:
            for group in groups:
                if group.backend.get_name() != \"TURBOQUANT\":
                    continue

                current_workspace_manager().get_simultaneous(
                    (
                        (max_num_reqs, num_heads, max_num_splits, head_size + 1),
                        torch.float32,
                    ),
                    ((max_num_reqs, num_heads, head_size), torch.float32),
                    ((max_num_reqs, num_heads), torch.float32),
                )
                return

    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,"""


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

EDITS = [
    ("attention.py", SITE1_FILE, SITE1_OLD, SITE1_NEW),
    ("turboquant_attn.py _decode_attention", SITE2_FILE, SITE2_OLD, SITE2_NEW),
    ("triton_decode signature", SITE3_FILE, SITE3A_OLD, SITE3A_NEW),
    ("triton_decode imports", SITE3_FILE, SITE3B_OLD, SITE3B_NEW),
    ("triton_decode workspace fetch + mid_o cleanup", SITE3_FILE, SITE3C_OLD, SITE3C_NEW),
    ("triton_decode output/lse cleanup", SITE3_FILE, SITE3D_OLD, SITE3D_NEW),
    ("gpu_model_runner imports", SITE4_FILE, SITE4A_OLD, SITE4A_NEW),
    ("gpu_model_runner capture_model call", SITE4_FILE, SITE4B_OLD, SITE4B_NEW),
    ("gpu_model_runner reserve helper", SITE4_FILE, SITE4C_OLD, SITE4C_NEW),
]


def main():
    files_done: set[str] = set()
    failures = []
    successes = []

    file_state: dict[str, str] = {}

    for label, path, old, new in EDITS:
        if not os.path.exists(path):
            failures.append((label, "TARGET MISSING"))
            continue

        if path not in file_state:
            with open(path, "r") as f:
                file_state[path] = f.read()

        src = file_state[path]
        if "[patch_pr40798]" in src and label.endswith("imports"):
            # Mid-edit re-entry would be confusing, only check imports
            pass

        if old not in src:
            # already applied?
            if (
                ("[patch_pr40798]" in src)
                or (label == "triton_decode imports" and "from typing import Any" not in src)
            ):
                successes.append((label, "already applied"))
                continue
            failures.append((label, f"ANCHOR NOT FOUND in {path}"))
            continue

        file_state[path] = src.replace(old, new)
        successes.append((label, "patched"))

    # Write back changed files
    if not failures:
        for path, new_content in file_state.items():
            with open(path, "w") as f:
                f.write(new_content)
                files_done.add(path)

    log.info("[patch_pr40798] %d edits OK, %d failed", len(successes), len(failures))
    for label, msg in successes:
        log.info("[patch_pr40798]   ok    %s  (%s)", label, msg)
    for label, msg in failures:
        log.error("[patch_pr40798]   FAIL  %s  (%s)", label, msg)
    if failures:
        log.error("[patch_pr40798] NOT WRITTEN due to failures")
    else:
        log.info("[patch_pr40798] Wrote %d files", len(files_done))


if __name__ == "__main__":
    main()
