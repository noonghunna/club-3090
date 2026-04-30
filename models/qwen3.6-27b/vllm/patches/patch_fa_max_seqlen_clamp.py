"""
Local P104-style FlashAttention max_seqlen_k runtime clamp.

Cliff 1 mechanism A is an FA2 softmax_lse cap leak: the varlen kernel sizes
workspace from max_seqlen_k, not from the actual cu_seqlens_k span. On long
TurboQuant contexts, vLLM can pass a conservative upper bound, so a modest
chunk allocates as if it were the full model length.

This sidecar patches TurboQuantAttention._flash_attn_varlen to clamp
max_seqlen_k to the runtime maximum derived from cu_seqlens_k. It is:

- env-gated: GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP=1
- runtime-only: skips while torch.cuda.is_current_stream_capturing()
- conservative: never below max(max actual K span, max_seqlen_q)

It lives in club-3090 rather than Sandermage's Genesis tree while Cliff 1
mechanism B work is still being validated locally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("fa_max_seqlen_clamp")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

PATCH_TAG = "[fa_max_seqlen_clamp]"
PATCH_MARKER = "LOCAL P104 FA max_seqlen_k runtime clamp v1"


def _find_target() -> Path | None:
    try:
        import vllm

        candidate = (
            Path(vllm.__file__).resolve().parent
            / "v1"
            / "attention"
            / "backends"
            / "turboquant_attn.py"
        )
        if candidate.exists():
            return candidate
    except ImportError:
        pass

    for raw_path in (
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py",
        "/usr/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py",
    ):
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
    return None


FLASH_ATTN_VARLEN_OLD = (
    "    def _flash_attn_varlen(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        cu_seqlens_q: torch.Tensor,\n"
    "        cu_seqlens_k: torch.Tensor,\n"
    "        max_seqlen_q: int,\n"
    "        max_seqlen_k: int,\n"
    "    ) -> torch.Tensor:\n"
    "        # fa_utils.get_flash_attn_version() returns None on backends that\n"
    "        # should not pass an explicit fa_version kwarg.\n"
    "        if self.fa_version is None:\n"
    "            return flash_attn_varlen_func(\n"
    "                q=q,\n"
    "                k=k,\n"
    "                v=v,\n"
    "                cu_seqlens_q=cu_seqlens_q,\n"
    "                cu_seqlens_k=cu_seqlens_k,\n"
    "                max_seqlen_q=max_seqlen_q,\n"
    "                max_seqlen_k=max_seqlen_k,\n"
    "                softmax_scale=self.scale,\n"
    "                causal=True,\n"
    "            )\n"
    "        return flash_attn_varlen_func(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            cu_seqlens_q=cu_seqlens_q,\n"
    "            cu_seqlens_k=cu_seqlens_k,\n"
    "            max_seqlen_q=max_seqlen_q,\n"
    "            max_seqlen_k=max_seqlen_k,\n"
    "            softmax_scale=self.scale,\n"
    "            causal=True,\n"
    "            fa_version=self.fa_version,\n"
    "        )\n"
)


FLASH_ATTN_VARLEN_NEW = (
    "    def _flash_attn_varlen(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        cu_seqlens_q: torch.Tensor,\n"
    "        cu_seqlens_k: torch.Tensor,\n"
    "        max_seqlen_q: int,\n"
    "        max_seqlen_k: int,\n"
    "    ) -> torch.Tensor:\n"
    f"        # [{PATCH_MARKER}] Clamp FA2's max_seqlen_k at runtime to\n"
    "        # avoid softmax_lse over-allocation from conservative metadata\n"
    "        # bounds. Do not run while CUDA graph capture is active: capture\n"
    "        # metadata intentionally keeps model-length upper bounds for\n"
    "        # shape stability.\n"
    "        _local_p104_max_seqlen_k = max_seqlen_k\n"
    "        _local_p104_actual_max_k = None\n"
    "        _local_p104_capturing = False\n"
    "        try:\n"
    "            import os as _local_p104_os\n"
    "            _local_p104_enabled = (\n"
    "                _local_p104_os.environ.get(\n"
    "                    'GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP', ''\n"
    "                ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            )\n"
    "            _local_p104_debug = (\n"
    "                _local_p104_os.environ.get(\n"
    "                    'GENESIS_FA_CLAMP_DEBUG', ''\n"
    "                ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            )\n"
    "            _local_p104_capturing = torch.cuda.is_current_stream_capturing()\n"
    "            if (\n"
    "                _local_p104_enabled\n"
    "                and not _local_p104_capturing\n"
    "                and cu_seqlens_k.numel() >= 2\n"
    "                and (max_seqlen_k > max_seqlen_q or _local_p104_debug)\n"
    "            ):\n"
    "                _local_p104_k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]\n"
    "                _local_p104_actual_max_k = int(_local_p104_k_lens.max().item())\n"
    "                if 0 < _local_p104_actual_max_k < max_seqlen_k:\n"
    "                    _local_p104_max_seqlen_k = max(\n"
    "                        _local_p104_actual_max_k, max_seqlen_q\n"
    "                    )\n"
    "            if _local_p104_debug and not getattr(\n"
    "                self, '_local_p104_debug_logged', False\n"
    "            ):\n"
    "                import logging as _local_p104_logging\n"
    "                _local_p104_logging.getLogger(\n"
    "                    'local.p104_fa_max_seqlen_clamp'\n"
    "                ).warning(\n"
    "                    'num_actual_tokens=%s max_query_len=%s '\n"
    "                    'attn_metadata.max_seq_len=%s seq_lens.max()=%s '\n"
    "                    'max_seqlen_k_effective=%s capture=%s',\n"
    "                    q.shape[0],\n"
    "                    max_seqlen_q,\n"
    "                    max_seqlen_k,\n"
    "                    _local_p104_actual_max_k,\n"
    "                    _local_p104_max_seqlen_k,\n"
    "                    _local_p104_capturing,\n"
    "                )\n"
    "                self._local_p104_debug_logged = True\n"
    "        except Exception:\n"
    "            _local_p104_max_seqlen_k = max_seqlen_k\n"
    "\n"
    "        # fa_utils.get_flash_attn_version() returns None on backends that\n"
    "        # should not pass an explicit fa_version kwarg.\n"
    "        if self.fa_version is None:\n"
    "            return flash_attn_varlen_func(\n"
    "                q=q,\n"
    "                k=k,\n"
    "                v=v,\n"
    "                cu_seqlens_q=cu_seqlens_q,\n"
    "                cu_seqlens_k=cu_seqlens_k,\n"
    "                max_seqlen_q=max_seqlen_q,\n"
    "                max_seqlen_k=_local_p104_max_seqlen_k,\n"
    "                softmax_scale=self.scale,\n"
    "                causal=True,\n"
    "            )\n"
    "        return flash_attn_varlen_func(\n"
    "            q=q,\n"
    "            k=k,\n"
    "            v=v,\n"
    "            cu_seqlens_q=cu_seqlens_q,\n"
    "            cu_seqlens_k=cu_seqlens_k,\n"
    "            max_seqlen_q=max_seqlen_q,\n"
    "            max_seqlen_k=_local_p104_max_seqlen_k,\n"
    "            softmax_scale=self.scale,\n"
    "            causal=True,\n"
    "            fa_version=self.fa_version,\n"
    "        )\n"
)


def _apply(src: str) -> tuple[str, str]:
    if PATCH_MARKER in src:
        return src, "skip-already-applied"
    if "GENESIS_ENABLE_FA_MAX_SEQLEN_CLAMP" in src:
        return src, "skip-equivalent-clamp-present"
    if FLASH_ATTN_VARLEN_OLD not in src:
        return src, "anchor-not-found"
    return src.replace(FLASH_ATTN_VARLEN_OLD, FLASH_ATTN_VARLEN_NEW, 1), "applied"


def main() -> int:
    target = _find_target()
    if target is None:
        log.error("%s vLLM turboquant_attn.py not found", PATCH_TAG)
        return 1

    log.info("%s target: %s", PATCH_TAG, target)
    src = target.read_text()
    patched, status = _apply(src)
    log.info("%s _flash_attn_varlen: %s", PATCH_TAG, status)

    if patched == src:
        if status.startswith("skip-"):
            return 0
        log.error("%s no changes written; %s", PATCH_TAG, status)
        return 1

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(patched)
    os.replace(tmp_path, target)
    log.info("%s patched %s", PATCH_TAG, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
