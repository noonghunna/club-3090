"""club-3090 compat shim for the DFlash2 backport on stock vLLM v0.27.1.

The syv-ai DFlash2 backport calls the module-level `apply_top_k_top_p(logits, k, p, top_k_max)`
with a 4th arg (a sort-free small-k path HINT) that exists in syv-ai's vLLM build but NOT in
stock v0.27.1 (`topk_topp_sampler.py` defines only 3 params). Add the 4th param as OPTIONAL +
IGNORED — the call then resolves, and we fall through to the standard (sort-based) path.
Correctness-preserving: top_k_max only selects an implementation, it does not change the result.
Idempotent; exits 1 on anchor drift so the boot refuses rather than silently mis-serving.
"""
import io, sys
T = "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/topk_topp_sampler.py"
OLD = ("def apply_top_k_top_p(\n"
       "    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None\n"
       ") -> torch.Tensor:")
NEW = ("def apply_top_k_top_p(\n"
       "    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None,\n"
       "    top_k_max: int | None = None,  # club-3090 DFlash2 compat: sort-free hint, ignored on v0.27.1\n"
       ") -> torch.Tensor:")
src = io.open(T, encoding="utf-8").read()
if "top_k_max: int | None = None,  # club-3090 DFlash2 compat" in src:
    print("[dflash2-compat] apply_top_k_top_p already 4-arg — skipping"); sys.exit(0)
if OLD not in src:
    print("[dflash2-compat] ANCHOR DRIFT: 3-arg apply_top_k_top_p not found — refusing.", file=sys.stderr); sys.exit(1)
io.open(T, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
print("[dflash2-compat] apply_top_k_top_p -> optional 4th param (top_k_max, ignored)")
