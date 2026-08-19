# club-3090 fix — order async input-prep after the prev-step spec-decode postprocess

**Bug:** our [club-3090#1052](https://github.com/noonghunna/club-3090/issues/1052)
· upstream [vllm#52873](https://github.com/vllm-project/vllm/issues/52873) +
the [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) thread. On a
Qwen3-Next hybrid (GDN linear attention) with the MTP drafter **+** prefix caching
**+** async scheduling (vLLM v0.27.1 default), the engine dies with a CUDA illegal
memory access in `gdn_attn.py` (Xid 31 MMU fault, `VIRT_WRITE`) within 6–13k
generated tokens of agent-shaped traffic. Drafter-off **or** `--no-async-scheduling`
avoids it; `CUDA_LAUNCH_BLOCKING=1` suppresses it — i.e. a cross-stream race, not
an index bug. **Distinct from vllm#50021** (whose in-kernel bounds do NOT fix this —
built its head, crashed 2/2) and from vllm#43559/#48375 (the corruption pair).

**Cause:** in `gpu_model_runner.py::synchronize_input_prep`, the next step waits
`prepare_inputs_event` before touching reused CPU tensors — but that event is
recorded at the END of input-prep, BEFORE the forward + spec-decode fused-align
postprocess (`_update_states_after_model_execute`, which records
`num_accepted_tokens_event`). So under async scheduling the next step's
`_update_states` block-table mutation raced the still-in-flight postprocess reading
those buffers → stale state-block index → wild GPU write.

**Fix (2 lines):** also wait `num_accepted_tokens_event` (recorded AFTER the
postprocess) at the top of `synchronize_input_prep`, closing the write-after-read
window. It relocates a wait that align-mode already performed later inside
`_prepare_inputs`, so there is no net GPU stall. Scoped to spec decode (the event
is `None` otherwise) and async (the whole method is a no-op without
`prepare_inputs_event`).

**Delivery:** idempotent, anchor-checked `install.sh` → `patch_gdn_mtp_async_spec_order.py`,
mounted at `/etc/club3090/gdn-async-order/` and invoked from the compose entrypoint
before serve. Anchor drift → the container **refuses to boot** (exit 1) rather than
serving unpatched.

**⚠️ Attribution / status:** this is **our own fix, NOT an upstream PR** — no
upstream commit vendored here. Validated on the reference 2×3090 (v0.27.1) only,
not community-blessed. Evidence: async-ON survived **3/3 boots** to 33–36k generated
tokens each (crossing ctx 32k + a compaction) where **5/5 unpatched arms died at
6–13k**; TPS neutral (79.1 narr / 108.5 code / 1756 prefill ≥ baseline). It does
**not** address the separate acceptance-collapse bug (vllm#52873 / Bug A).

**Drop trigger:** an upstream fix for this race lands AND the `vllm-stable` pin
moves past it → remove the mounts + entrypoint call + this dir; `patches.yml` row
retires. If we file it upstream, update `upstream.status` from `ours` to `open`
with the PR link.
