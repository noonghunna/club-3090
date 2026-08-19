#!/usr/bin/env python3
"""club-3090 fix — order async input-prep AFTER the previous step's spec-decode postprocess.

Bug (our club-3090#1052, upstream vllm#52873 + the #50021 thread): on a
Qwen3-Next hybrid (GDN linear attention) with the MTP drafter + prefix caching +
async scheduling (vLLM v0.27.1 default), the engine dies with a CUDA illegal
memory access in gdn_attn.py (Xid 31 MMU fault, VIRT_WRITE) within 6-13k
generated tokens of agent-shaped traffic.

Cause: in gpu_model_runner.py, `synchronize_input_prep` waits `prepare_inputs_event`
before the next step touches reused CPU tensors -- but that event is recorded at
the END of input-prep, BEFORE the forward + spec-decode fused-align postprocess.
So under async scheduling the next step's `_update_states` block-table mutation
raced the still-in-flight postprocess reading those buffers -> stale state-block
index -> wild GPU write. This waits `num_accepted_tokens_event` (recorded AFTER
the postprocess) as well, closing the write-after-read window.

Scoped to spec decode (the event is None otherwise) and async scheduling (the
whole method is a no-op without prepare_inputs_event). An unrecorded event is
treated complete, so the first step is unaffected. NOT an upstream PR -- our own
fix; validated 3/3 boots on the reference 2x3090 (async-ON survived 33-36k
generated incl. a compaction where 5/5 unpatched died 6-13k), TPS-neutral.

Idempotent; anchor-checked; exits non-zero on drift so the entrypoint can refuse
to boot a silently-unpatched configuration.
"""
import io
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
MARKER = "club-3090 GDN+MTP async spec-order fix"
DEF = "    def synchronize_input_prep(self):"
ANCHOR = "        self.prepare_inputs_event.synchronize()\n"
INSERT = """        # club-3090 GDN+MTP async spec-order fix (our #1052 / upstream #52873 +
        # PR#50021 thread). prepare_inputs_event above is recorded BEFORE the
        # forward + spec-decode fused-align postprocess, so under async scheduling
        # the next step's _update_states block-table mutation raced the
        # still-in-flight postprocess -> stale state-block index -> wild GPU write
        # (CUDA illegal access / Xid 31 VIRT_WRITE). Wait the postprocess event
        # (recorded AFTER it) to close that write-after-read window. Scoped to spec
        # decode (event None otherwise); an unrecorded event is complete, so the
        # first step is unaffected.
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
"""

def main() -> int:
    try:
        src = io.open(TARGET, encoding="utf-8").read()
    except OSError as e:
        print(f"[gdn-async-order] REFUSE: cannot read target: {e}")
        return 1
    if MARKER in src:
        print("[gdn-async-order] already patched (idempotent no-op)")
        return 0
    d = src.find(DEF)
    if d < 0:
        print("[gdn-async-order] REFUSE: anchor drift - def synchronize_input_prep not found")
        return 1
    i = src.find(ANCHOR, d)
    if i < 0:
        print("[gdn-async-order] REFUSE: anchor drift - prepare_inputs_event.synchronize() not found in method")
        return 1
    j = i + len(ANCHOR)
    io.open(TARGET, "w", encoding="utf-8").write(src[:j] + INSERT + src[j:])
    print("[gdn-async-order] applied: input-prep now waits the prev-step spec-decode postprocess")
    return 0

if __name__ == "__main__":
    sys.exit(main())
