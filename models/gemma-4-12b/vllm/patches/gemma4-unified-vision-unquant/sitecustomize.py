# EVAL/experimental workaround for vLLM #44494 (gemma4_unified compressed-tensors).
# The QAT-W4A16 gemma-4-12B checkpoint trips TWO bugs in the gemma4_unified loader;
# this auto-imported sitecustomize (on PYTHONPATH, incl. spawn workers) patches both
# so the compose boots self-contained — NO checkpoint edit required. NOT for prod.
# Proper upstream fixes: plumb prefix= to the vision embedder + ship/derive
# num_soft_tokens. See vLLM #44494.
#
#   Bug B — vision embedder force-quantized: Gemma4UnifiedVisionEmbedder builds
#     patch_dense as ColumnParallelLinear(quant_config=...) with NO prefix=, so
#     compressed-tensors can't match the checkpoint `ignore` list and quantizes the
#     BF16 vision embedder -> "no parameter vision_embedder.patch_dense.weight".
#     Fix: drop quant_config so the embedder stays unquantized (it's BF16 on disk).
#   Bug A — missing num_soft_tokens: the QAT config omits
#     vision_config.num_soft_tokens, but Gemma4UnifiedProcessingInfo
#     .get_mm_max_tokens_per_item reads it unconditionally -> AttributeError.
#     Fix: default it to 280 (the value google/gemma-4-12B-it ships) on the config
#     returned by get_hf_config().
import importlib.abc
import importlib.machinery
import sys

_TARGET = "vllm.model_executor.models.gemma4_unified"
_NUM_SOFT_TOKENS = 280


class _Gemma4UnifiedW4A16Workaround(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)  # one-shot
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        _exec = spec.loader.exec_module

        def exec_module(module):
            _exec(module)
            done = []

            # Bug B: keep the vision embedder unquantized.
            emb = getattr(module, "Gemma4UnifiedVisionEmbedder", None)
            if emb is not None:
                _orig_init = emb.__init__

                def __init__(self, config, quant_config=None):
                    _orig_init(self, config, quant_config=None)

                emb.__init__ = __init__
                done.append("vision-embedder-unquant")

            # Bug A: default vision_config.num_soft_tokens.
            pi = getattr(module, "Gemma4UnifiedProcessingInfo", None)
            if pi is not None and hasattr(pi, "get_hf_config"):
                _orig_cfg = pi.get_hf_config

                def get_hf_config(self):
                    cfg = _orig_cfg(self)
                    vc = getattr(cfg, "vision_config", None)
                    if vc is not None and not hasattr(vc, "num_soft_tokens"):
                        vc.num_soft_tokens = _NUM_SOFT_TOKENS
                    return cfg

                pi.get_hf_config = get_hf_config
                done.append("num_soft_tokens=%d" % _NUM_SOFT_TOKENS)

            print("[g4-unified-patch] applied (vLLM #44494): " + ", ".join(done or ["nothing"]), flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Gemma4UnifiedW4A16Workaround())
