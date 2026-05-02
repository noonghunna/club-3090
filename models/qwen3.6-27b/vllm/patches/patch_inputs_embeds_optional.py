#!/usr/bin/env python3
"""
patch_inputs_embeds_optional.py — backport of vllm-project/vllm#35975

Skips inputs_embeds GPU buffer allocation for text-only models. Saves
~64 MiB GPU + ~64 MiB pinned CPU on a config that doesn't use multimodal
inputs or prompt embeddings.

Why we ship it locally:
  PR #35975 is open upstream as of 2026-05-02 (awaiting code-owner approval
  after author addressed reviewer feedback). On club-3090's TP=1 + 24GB
  Qwen3.6-27B + MTP K=3 + 0.95 mem-util config, Cliff 2 at 60K fires with
  ~24.5 MiB free at the time of the failing 50 MiB allocation. PR #35975
  frees ~64 MiB GPU + ~64 MiB pinned CPU on text-only models — sufficient
  margin to potentially close Cliff 2 at 60K MTP-on without reducing
  context or mem-util.

  This sidecar exists because we don't want to fork the docker image just
  for one PR. When PR #35975 merges upstream and ships in our nightly tag,
  this sidecar can be deleted.

References:
  - https://github.com/vllm-project/vllm/pull/35975
  - club-3090 results/v0.20-migration/v769-codex-r1-test.summary

Idempotent. Safe to re-run.
"""
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("inputs_embeds_optional")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

PATCH_TAG = "[inputs_embeds_optional]"


def patch_gpu_model_runner() -> bool:
    """
    File: vllm/v1/worker/gpu_model_runner.py

    Before:
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )

    After:
        # vllm#35975: skip inputs_embeds buffer for text-only models.
        self.inputs_embeds = None
        if self.supports_mm_inputs or self.enable_prompt_embeds:
            self.inputs_embeds = self._make_buffer(
                self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
            )
    """
    target = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
    )
    if not target.exists():
        log.error(f"{PATCH_TAG} target not found: {target}")
        return False

    src = target.read_text()
    if PATCH_TAG in src:
        log.info(f"{PATCH_TAG} gpu_model_runner.py already patched, skipping.")
        return True

    # Find the exact allocation site. Use a regex that matches multi-line.
    # Keep the `numpy=False` argument literal.
    pattern = re.compile(
        r"^( *)self\.inputs_embeds = self\._make_buffer\(\n"
        r"\1    self\.max_num_tokens, self\.inputs_embeds_size, dtype=self\.dtype, numpy=False\n"
        r"\1\)$",
        re.MULTILINE,
    )
    match = pattern.search(src)
    if not match:
        log.error(
            f"{PATCH_TAG} gpu_model_runner.py: anchor not found "
            f"(self.inputs_embeds = self._make_buffer block). vLLM may have "
            f"changed; review PR #35975 manually."
        )
        return False

    indent = match.group(1)
    replacement = (
        f"{indent}# {PATCH_TAG} vllm#35975: skip inputs_embeds buffer for text-only models.\n"
        f"{indent}self.inputs_embeds = None\n"
        f"{indent}if self.supports_mm_inputs or self.enable_prompt_embeds:\n"
        f"{indent}    self.inputs_embeds = self._make_buffer(\n"
        f"{indent}        self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False\n"
        f"{indent}    )"
    )
    new_src = pattern.sub(replacement, src, count=1)
    target.write_text(new_src)
    log.info(f"{PATCH_TAG} gpu_model_runner.py: applied")
    return True


def patch_llm_base_proposer() -> bool:
    """
    File: vllm/v1/spec_decode/llm_base_proposer.py
    (PR #35975 originally targeted eagle.py; in this nightly the code lives
     in the shared base class llm_base_proposer.py.)

    Before:
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

    After:
        # vllm#35975: skip inputs_embeds tensor for text-only proposers.
        self.inputs_embeds = None
        if self.supports_mm_inputs:
            self.inputs_embeds = torch.zeros(
                (self.max_num_tokens, self.inputs_embeds_size),
                dtype=self.dtype,
                device=device,
            )
    """
    target = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
    )
    if not target.exists():
        log.error(f"{PATCH_TAG} target not found: {target}")
        return False

    src = target.read_text()
    if PATCH_TAG in src:
        log.info(f"{PATCH_TAG} llm_base_proposer.py already patched, skipping.")
        return True

    pattern = re.compile(
        r"^( *)self\.inputs_embeds = torch\.zeros\(\n"
        r"\1    \(self\.max_num_tokens, self\.inputs_embeds_size\),\n"
        r"\1    dtype=self\.dtype,\n"
        r"\1    device=device,\n"
        r"\1\)$",
        re.MULTILINE,
    )
    match = pattern.search(src)
    if not match:
        log.error(
            f"{PATCH_TAG} llm_base_proposer.py: anchor not found "
            f"(self.inputs_embeds = torch.zeros block). vLLM may have changed."
        )
        return False

    indent = match.group(1)
    replacement = (
        f"{indent}# {PATCH_TAG} vllm#35975: skip inputs_embeds tensor for text-only proposers.\n"
        f"{indent}self.inputs_embeds = None\n"
        f"{indent}if self.supports_mm_inputs:\n"
        f"{indent}    self.inputs_embeds = torch.zeros(\n"
        f"{indent}        (self.max_num_tokens, self.inputs_embeds_size),\n"
        f"{indent}        dtype=self.dtype,\n"
        f"{indent}        device=device,\n"
        f"{indent}    )"
    )
    new_src = pattern.sub(replacement, src, count=1)
    target.write_text(new_src)
    log.info(f"{PATCH_TAG} llm_base_proposer.py: applied")
    return True


def main() -> int:
    ok = True
    ok &= patch_gpu_model_runner()
    ok &= patch_llm_base_proposer()
    if ok:
        log.info(f"{PATCH_TAG} all patches applied successfully")
        return 0
    log.error(f"{PATCH_TAG} one or more patches failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
