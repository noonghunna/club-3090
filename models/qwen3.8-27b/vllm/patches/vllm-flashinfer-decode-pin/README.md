# vllm-flashinfer-decode-pin

Unpins the FlashInfer **decode** workspace buffer to fix the MTP **c≥4 Xid 31
VIRT_READ** crash on this Qwen3.8-27B W4A8+MTP (hybrid GDN) config.

## What it fixes

At `MAX_NUM_SEQS >= 4` with the MTP drafter (SPEC_N=4) this stack crashed the
engine in ~15s: `NVRM: Xid 31, MMU Fault ... FAULT_PDE ACCESS_TYPE_VIRT_READ`,
`torch.AcceleratorError: illegal memory access`, both TP ranks. Root cause is
[vllm-project/vllm#40756](https://github.com/vllm-project/vllm/issues/40756)
(thread comments 2026-08-17): FlashInfer `plan()` reuses one **pinned** host
buffer per wrapper and copies it out asynchronously; the MTP drafter re-plans
the **decode** wrapper K−1×/step, so a stale plan feeds the split-KV merge a
garbage row → the Xid. It needs ≥4 genuinely concurrent in-flight MTP requests
to lose the race.

The prior documented mitigations were `SPEC_N=0` (kill the drafter — slow) or
`MAX_NUM_SEQS=1` (no concurrency). This patch fixes the crash at **full async
speed** and lets you keep `MAX_NUM_SEQS=4`.

## The change

`patch_flashinfer_decode_pin.py` flips `pin_memory=True,` → `pin_memory=False,`
in the `_pin_memory_int_workspace_buffer` allocations of
`flashinfer/decode.py` (the decode wrapper — the one MTP re-plans). A
pinned buffer is safe against the async copy-out only when the re-plan can't
overlap it; the drafter breaks that. `pin_memory=False` forces a synchronous
host→device copy of the plan each step, closing the window.

- **decode.py alone is necessary and sufficient** (A/B: unpinning
  prefill/sparse/pod *without* decode.py still crashed).
- `FI_PINQ_LIB_ALL=1` also unpins prefill/sparse/pod (validated full mirror;
  defense in depth). Default is decode-only.

## Safety

- Auto-detects flashinfer via `import flashinfer` (no hardcoded path); **no-ops
  if not installed** (a non-FlashInfer backend config boots clean).
- Guard: in each file it touches, `count(pin_memory=True,)` must equal the
  number of `_pin_memory_int_workspace_buffer* = torch.empty(` allocs. A mismatch
  means a pin exists outside a workspace buffer → hard-fail (exit 2) so the
  compose refuses to serve a half-patched state.
- Idempotent (marker-gated), py_compiles every file it writes.

## Validation (v0.27.1 / flashinfer 0.6.16.post3 / 2× RTX 3090 SM86)

| Config | c=4, 120s | result |
|--------|-----------|--------|
| stock (pinned) | — | **crash ~15s** |
| decode.py unpin only (this patch) | OK=128, ERR=0, 0 new Xid | **survive** |
| full mirror (+ prefill/sparse/pod) | OK=126 (120s) + OK=345 (300s soak) | **survive, 425s clean** |

Perf impact (c=1 stock-vs-patched, the clean A/B): decode narrative −2.1%, code
−1.3%, prefill ~0% — all within CV, i.e. **cost-neutral**. At c=4 the patch
unlocks a point stock cannot reach: 72.4 narrative / 85.0 code tok/s
per-request, ~3.4× the c=1 aggregate request rate. Throughput only — no quality
gate run.

## Wiring

Mounted `ro` and invoked from the compose entrypoint before `vllm serve`
(`bash /etc/club3090/flashinfer-decode-pin/install.sh`), gated on the FlashInfer
backend. See `models/qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml`.
