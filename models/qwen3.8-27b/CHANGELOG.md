# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo.

## 2026-08-18 — dual-fast: FlashInfer decode-buffer unpin bounds the MTP c≥4 Xid 31 crash

`vllm/qwen38-27b-dual-fast` (W4A8 + MTP n=4 + fp8 KV + prefix caching, the hybrid GDN family) crashed the engine in ~15 s at `MAX_NUM_SEQS >= 4` with the drafter on: `NVRM: Xid 31, MMU Fault ... FAULT_PDE ACCESS_TYPE_VIRT_READ` (unmapped page), `torch.AcceleratorError: illegal memory access`, both TP ranks. The prior documented mitigations were `SPEC_N=0` (kill the drafter) or keeping `MAX_NUM_SEQS=1` (no concurrency). This adds a patch that fixes the crash at **full async speed**, so the concurrency ceiling is no longer bounded by the crash.

**Root cause** — [vllm-project/vllm#40756](https://github.com/vllm-project/vllm/issues/40756), thread comments 2026-08-17 (brasrox retraction + sempai SM86 corroboration). FlashInfer `plan()` reuses one **pinned** host buffer per wrapper and copies it out asynchronously with nothing guarding it; the MTP drafter re-plans the **decode** wrapper K−1×/step, so a stale plan feeds the split-KV merge (`PersistentVariableLengthMergeStatesKernel`) a garbage row (observed: row 33,621 of an 85-row buffer) → the Xid. It needs **≥4 genuinely concurrent in-flight MTP requests** to lose the race — c≤3 and the `SPEC_N=0` path never crash. (This is the same bug family the compose header already flagged as "exposed to open vllm#50021 / GDN spec-decode"; the 2026-08-17 thread comments re-identify it as the FlashInfer stale-plan async-copy race, and the buffer that must be unpinned is the **decode** wrapper.)

**The fix** — new patch `models/qwen3.8-27b/vllm/patches/vllm-flashinfer-decode-pin/` (idempotent, marker-gated Python patcher + `install.sh`). Flips `pin_memory=True,` → `pin_memory=False,` on the `_pin_memory_int_workspace_buffer` allocations in `flashinfer/decode.py` — the decode wrapper MTP re-plans. `pin_memory=False` forces a synchronous host→device copy of the plan each step, closing the stale-read window. Auto-detects flashinfer via `import flashinfer` and **no-ops if it isn't installed** (a non-FlashInfer backend config boots clean); hard-fails (exit 2) on drift so the compose never serves a half-patched state. `FI_PINQ_LIB_ALL=1` also unpins prefill/sparse/pod (the validated full mirror; default is decode-only).

**Validation** (v0.27.1 / flashinfer 0.6.16.post3 / 2× RTX 3090 SM86, TP=2):

| Config | c=4, 120 s window | result |
|--------|-------------------|--------|
| stock (pinned) | — | **crash ~15 s** (Xid 31) |
| decode.py unpin only (this patch) | OK=128, ERR=0, **0 new Xid** | **survive** |
| full mirror (+ prefill/sparse/pod) | OK=126 (120 s) + OK=345 (300 s soak) | **survive, 425 s clean** |

A/B isolates **decode.py as necessary and sufficient**: unpinning prefill/sparse/pod *without* decode.py still crashed; decode.py alone is the fix. End-to-end re-verified inside the pinned v0.27.1 image (`install.sh`: `decode.py: 4 pin_memory True->False (py_compile OK)`, idempotent re-run `already patched`, decode 0/4 and the other three 4/0).

**Performance impact** (c=1 stock-vs-patched — the clean A/B, since stock crashes at c=4): decode narrative 86.31 → 84.47 tok/s (−2.1%), code 102.32 → 101.02 (−1.3%), prefill ~0% — **all within CV, i.e. cost-neutral**. At c=4 the patch unlocks a point stock cannot reach: 72.4 narrative / 85.0 code tok/s per-request, ~3.4× the c=1 aggregate request rate. **Throughput only — no quality gate run** (the 8-pack is the separate follow-up).

**What this does and does not change — deliberately.**
- **Bounds the crash, not VRAM.** The compose still ships `MAX_NUM_SEQS=1` because of the *separate* W4A8 16K-prompt OOM documented in the header (peak VRAM 23,872 MiB at N=2, 1.75 GB over the 0.90 budget). That is a different constraint and this patch does **not** touch it. At the shipped N=1 default the patch is inert (no c≥4, so no crash). It becomes load-bearing the moment someone raises `MAX_NUM_SEQS` — then the drafter and concurrency coexist without the Xid.
- **Status unchanged**: the slug stays 🧪 Experimental. A patch that fixes a crash does not by itself promote a tier that still has no 8-pack / NIAH / soak.

**Wiring:** mounted `ro` and invoked from the `dual-fast` compose entrypoint before `vllm serve` (`bash /etc/club3090/flashinfer-decode-pin/install.sh`), registered in `scripts/lib/profiles/patches.yml` as `vllm-flashinfer-decode-pin` (`delivery_mechanism: install_script`). `test-patch-attribution.sh` passes (71 entries) and the compose parses clean.

**Drop when:** flashinfer lands a fix for the pinned-buffer plan reuse (a `pin_memory=False` default on the decode wrapper, or an explicit sync guard on the async copy-out) and the pin moves past it → remove the compose mount + entrypoint call + this patch dir and retire the `patches.yml` row.
