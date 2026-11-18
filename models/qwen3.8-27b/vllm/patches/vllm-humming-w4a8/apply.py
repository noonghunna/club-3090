"""W4A8 on vLLM v0.34.0's humming backend, for AutoRound INT4 checkpoints.

v0.34.0 removed Marlin's 8-bit activations (vllm#65542). W4A8 is now
`--quantization humming` with VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8"}'.
On our AutoRound checkpoints that path needs two fixes, and one log line proves
int8 is really in use. This script applies all three to the installed vLLM.

1. Fallback init (vllm#68194, fix = open upstream PR vllm#68202).
   HummingConfig ignores AutoRound's extra_config, so the checkpoint's BF16
   layers (vision tower, linear_attn.in_proj_ba, mtp.fc) reach the weight
   loader's unquantized fallback. It swaps `self.__class__` to
   UnquantizedLinearMethod without running its __init__, so `_gemm_impl` is
   missing and model load fails. Fix: run __init__ after the swap. This is the
   upstream PR's change, line for line.

2. Placeholder weight (vllm#68195; no upstream PR, so this is ours).
   create_weights registers a scalar placeholder `weight`. In
   process_weights_after_loading, replace_layer_parameters then treats
   humming's converted `weight` as an existing name and keeps the checkpoint
   `qweight` storage alive beside it: about 2x weight memory (15.17 vs
   9.19 GiB a card on Qwen3.8-27B), and 262,144 context no longer fits on
   24 GB cards. Fix: drop the scalar placeholder before that call.

3. Activation log (diagnostic, the positive control).
   Nothing in vLLM's own log says which activation dtype humming settled on,
   and a worker's /proc/<pid>/environ is overwritten by setproctitle. After
   check_and_fallback_input_schema, print the effective input and weight
   dtypes once per worker process.

Idempotent: each change carries a marker and is skipped when present. A fix
already present upstream is reported and skipped. A missing anchor for fix 1
or fix 2 exits 2, because serving without them either fails to load or
doubles the weight memory. A missing anchor for the log only warns.
"""

import importlib.util
import os
import sys

MARK = "# club3090 humming-w4a8"

SWAP = "                    self.__class__ = UnquantizedLinearMethod  # type: ignore\n"
INIT = (
    "                    # The class swap skips __init__, which sets _gemm_impl.\n"
    f"                    UnquantizedLinearMethod.__init__(self)  # type: ignore {MARK}\n"
)

MEM = (
    "            tensors.update(input_tensors)\n"
    "            replace_layer_parameters(layer, tensors, list(layer._parameters))\n"
)
MEM_FIX = (
    "            tensors.update(input_tensors)\n"
    f"            # The scalar placeholder from create_weights would make the converted {MARK}\n"
    f"            # `weight` look like an existing parameter and keep qweight alive (#68195). {MARK}\n"
    f"            _ph = layer._parameters.get(\"weight\")  {MARK}\n"
    f"            if _ph is not None and _ph.ndim == 0 and \"weight\" in tensors:  {MARK}\n"
    f"                del layer._parameters[\"weight\"]  {MARK}\n"
    "            replace_layer_parameters(layer, tensors, list(layer._parameters))\n"
)
PLACEHOLDER = 'prepare_param(torch.tensor(0), "weight", extra_weight_attrs)'

LOG_ANCHOR = (
    "        self.input_schema = check_and_fallback_input_schema(\n"
    "            weight_schema=self.weight_schema,\n"
    "            input_schema=self.input_schema,\n"
    "            param_dtype=layer.param_dtype,\n"
    "            allow_fallback=self.quant_config.allow_input_schema_fallback,\n"
    "        )\n"
)
LOG = (
    f"        if not globals().get(\"_CLUB3090_HUMMING_LOGGED\"):  {MARK}\n"
    f"            globals()[\"_CLUB3090_HUMMING_LOGGED\"] = True  {MARK}\n"
    f"            import sys as _sys  {MARK}\n"
    f"            print(\"[humming-w4a8] effective activations a_dtype=%s, weights b_dtype=%s \"  {MARK}\n"
    f"                  \"(first humming linear layer of this worker)\"  {MARK}\n"
    f"                  % (getattr(self.input_schema, \"a_dtype\", None),  {MARK}\n"
    f"                     getattr(self.weight_schema, \"b_dtype\", None)),  {MARK}\n"
    f"                  file=_sys.stderr, flush=True)  {MARK}\n"
)


def humming_path() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("[humming-w4a8] FATAL: vllm is not importable in this container")
    root = list(spec.submodule_search_locations)[0]
    return os.path.join(root, "model_executor", "layers", "quantization", "humming.py")


def main() -> int:
    path = humming_path()
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as e:
        print(f"[humming-w4a8] FATAL: cannot read {path}: {e}", file=sys.stderr)
        return 2
    out = src
    fatal = False

    # 1. fallback init (vllm#68202)
    if INIT in out:
        print("[humming-w4a8] fallback init: already applied")
    elif out.count(SWAP) == 1:
        tail = out.split(SWAP, 1)[1].split("\n", 3)
        if any("UnquantizedLinearMethod.__init__(self)" in line for line in tail[:3]):
            print("[humming-w4a8] fallback init: already fixed upstream, skipped")
        else:
            out = out.replace(SWAP, SWAP + INIT, 1)
            print("[humming-w4a8] fallback init: applied (vllm#68202)")
    else:
        print(f"[humming-w4a8] FATAL: fallback-swap anchor found {out.count(SWAP)} times, "
              "expected 1 (drifted vLLM?)", file=sys.stderr)
        fatal = True

    # 2. placeholder weight (vllm#68195)
    if MEM_FIX in out:
        print("[humming-w4a8] placeholder weight: already applied")
    elif out.count(MEM) == 1:
        out = out.replace(MEM, MEM_FIX, 1)
        print("[humming-w4a8] placeholder weight: applied (vllm#68195)")
    elif PLACEHOLDER not in out:
        print("[humming-w4a8] placeholder weight: create_weights no longer registers the "
              "scalar placeholder, so treated as fixed upstream, skipped")
    else:
        print(f"[humming-w4a8] FATAL: placeholder-weight anchor found {out.count(MEM)} times, "
              "expected 1 (drifted vLLM?)", file=sys.stderr)
        fatal = True

    # 3. activation log (diagnostic)
    if LOG in out:
        print("[humming-w4a8] activation log: already applied")
    elif out.count(LOG_ANCHOR) == 1:
        out = out.replace(LOG_ANCHOR, LOG_ANCHOR + LOG, 1)
        print("[humming-w4a8] activation log: applied")
    else:
        print("[humming-w4a8] WARN: activation-log anchor not found; int8 cannot be "
              "confirmed from the log on this boot", file=sys.stderr)

    if fatal:
        return 2
    if out != src:
        tmp = path + ".club3090.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(out)
        os.replace(tmp, path)
        compile(out, path, "exec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
