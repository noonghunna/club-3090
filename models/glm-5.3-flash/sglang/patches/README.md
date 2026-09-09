# Patches for GLM-5.3-Flash on SGLang

**Nothing is vendored here yet, and that is deliberate.** This file defines the patch
contract for the `sglang` engine type so the first real patch has somewhere to land, and
records the one attempt already made so it is not repeated.

## The contract (mirrors the vLLM one, not the old sglang one)

Two patch styles exist in this repo. Use the first.

| style | where | delivery |
|---|---|---|
| ✅ **install_script** | `models/<model>/vllm/patches/<patch-id>/` | `install.sh` mounted into the container and invoked before the server starts |
| 🗑️ bare `.py` overlay | `models/qwen3.6-27b/sglang/patches/patch_*.py` | undefined — deprecated 2026-06-05 (#254), no functional compose uses them |

A new sglang patch is a directory `models/<model>/sglang/patches/<patch-id>/` containing an
idempotent `install.sh`, plus a `patches.yml` entry:

```yaml
- id: <model>-sglang-<what>
  model: [<model>]
  files: [models/<model>/sglang/patches/<patch-id>]
  delivery: {dockerfile_bake: false, entrypoint_invoke: true, genesis: false}
  delivery_mechanism: install_script
  delivery_spec:
    script: models/<model>/sglang/patches/<patch-id>/install.sh
    mounted_at: /etc/club3090/<patch-id>.sh
    invoke: bash /etc/club3090/<patch-id>.sh
    invoked_before: sglang-launch
    wired_at: [volumes, entrypoint]
  drift_guard: {kind: behavioral, check: "<observable that fails if the patch silently no-ops>", on_fail: capability-degraded}
  upstream: {ref: sgl-project/sglang#NNNNN, status: <open|merged|ours>, drop_when: "<condition>"}
```

⚠️ `test-patch-attribution` enforces both directions: every `files:` path must exist, and
every `.py`/`.sh` under a `patches/` tree must be claimed by an entry. A `.md` is not an
artifact, which is why this README needs no entry. Adding code here without the entry
reds the guard.

⚠️ `install.sh` must be **anchor-based and refuse on drift** (grep for the exact upstream
line it patches; `exit 1` if absent) rather than applying blind. An overlay that silently
no-ops against a bumped image is the failure mode the `drift_guard` field exists for.

## Recorded dead end: `glm5-next-sideload` (2026-09-09)

Do not retry this shape.

Someone attempted to side-load `glm5_next` support onto the **v0.5.19 release image** by
copying individual files from sglang main: `srt/models/glm5_next.py`,
`srt/models/glm5_next_nextn.py`, `srt/configs/glm5_next.py`, a newer
`kernels/ops/layernorm/mhc.py` (v0.5.19's lacks the `hc_contract`/`hc_pre`/`hc_post` that
glm5_next needs), plus appending `AutoConfig.register('glm5_next'/'glm5_next_text', …)` to
`configs/__init__.py`.

It got further than expected and still failed:

1. ✅ config registration worked — `model_type=glm5_next` resolved, flashinfer backend announced;
2. ⛔ the model import then died on `is_dsa_prefill_cp_interleave` missing from
   `layers/attention/dsa/utils.py`.

**sglang#36507 touches ~50 files.** Per-file backporting onto a release image is a dead
end, not an unfinished job — each fixed import reveals the next. The supported route is a
main-built image (`lmsysorg/sglang:dev-cu12`) carrying the whole change natively, declared
as its own engine profile.

The sideload files are NOT vendored here on purpose: they are copies of upstream sources
that can never work on the pinned image, and keeping them would oblige `patches.yml` to
carry an entry for code with no path to working. This record is the useful part.

⚠️ Even with a main-built image, the Ampere question is open: sparse-MLA/DSA backends are
SM90-gated (see `LEARNINGS.md` 2026-09-09), and SGLang's Triton DSA path is the only
plausible route on sm_86 — **unvalidated**.

## Likely first real patch

Not glm5_next. Discussion #1178 (@jb-seo) reports Qwen3.8-27B running on this engine with
**3 in-container patches** on v0.5.18. Those are the first concrete candidates for this
contract, and reproducing them is what would let `sglang-stable` carry a validated compose.
