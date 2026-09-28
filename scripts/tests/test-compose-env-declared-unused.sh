#!/usr/bin/env bash
# test-compose-env-declared-unused.sh — the INVERSE of
# test-compose-entrypoint-env-declared.sh: a compose may not declare a
# SCRIPT-LEVEL knob that nothing reads.
#
# THE CLASS: that sibling gate catches "read but not declared" — a knob the
# compose documents that docker never forwards. This one catches the mirror
# image: a knob docker DOES forward that no script ever looks at. Both end the
# same way — the user sets it, nothing errors, nothing changes.
#
# ⚠️ FOUND IN THE WILD 2026-09-18 (club-3090 exl3/prism onboarding): the
# llama-cpp-prism compose declared INSTRUCT / THINKING / REASONING / TEMPERATURE
# because its `environment:` block was copied from a sibling that HAS a bash
# entrypoint reading them to flip the sampler row. The prism compose had no
# wrapper at all, so `INSTRUCT=1` was accepted, forwarded, and silently ignored —
# the user would get the thinking row while believing they had selected instruct.
#
# ⚠️⚠️ WHY THIS NEEDS AN ALLOWLIST AND THE SIBLING GATE DOES NOT.
# Most declared vars are consumed by the ENGINE BINARY or the CUDA/NCCL runtime,
# never by the YAML: CUDA_VISIBLE_DEVICES, VLLM_*, GGML_*, LLAMA_ARG_*, NCCL_*,
# OMP_NUM_THREADS, TRITON_CACHE_DIR … A naive "declared but not referenced" check
# flags all of them. Measured on this repo before the allowlist: 33 distinct names
# across 54 composes, of which exactly ONE was a real defect. So the rule is
# narrow BY DESIGN — it only polices names that look like script knobs.
#
# The contract, per service:
#   declared in `environment:`  AND
#   not read at runtime as `$${VAR}` (entrypoint/command, evaluated IN-container) AND
#   not substituted host-side as `${VAR}` (compose render time) AND
#   not matched by ENGINE_PREFIXES / ENGINE_EXACT
#   ⇒ FAIL — it is a knob that does nothing.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUTF8="${PYTHONUTF8:-1}"

python3 - <<'PY'
import io, re, sys, glob

# Consumed by the engine binary / CUDA / NCCL / torch — NOT by any script, so
# their absence from the YAML body is expected and correct.
ENGINE_PREFIXES = (
    "VLLM_", "GGML_", "LLAMA_", "NCCL_", "CUDA_", "TRITON_", "PYTORCH_",
    "TORCH", "SGLANG_", "LMCACHE_", "HF_", "HUGGING_FACE_", "NVIDIA_",
    "OMP_", "MKL_", "EXL3_", "TABBY_", "SAFETENSORS_", "XDG_",
)
ENGINE_EXACT = {"PATH", "HOME", "HOSTNAME", "LD_LIBRARY_PATH", "PWD", "IFS",
                "DEBIAN_FRONTEND", "PYTHONUNBUFFERED", "PYTHONUTF8"}

# DEBT REGISTER — each entry must name WHY the dead knob is deliberate.
# Shrink it; never grow it silently. An entry here means a documented knob is
# inert in production.
KNOWN = {
    # The qwen3.8-27b llama.cpp composes keep THINKING declared as DOCUMENTED
    # back-compat: their sampler flip keys on INSTRUCT (thinking is already the
    # default since the 2026-09-01 flip), and their headers say so explicitly —
    # "THINKING=1 is now a back-compat no-op". Deliberate, and stated to users.
    ("models/qwen3.8-27b/llama-cpp/compose/single/unsloth-iq4xs/q4kv-vision.yml", "THINKING"),
    ("models/qwen3.8-27b/llama-cpp/compose/dual/unsloth-q8kxl/q8kv.yml", "THINKING"),
}

def engine_owned(name: str) -> bool:
    return name in ENGINE_EXACT or name.startswith(ENGINE_PREFIXES)

bad = []
checked = 0
for path in sorted(glob.glob("models/*/*/compose/*/*/*.yml")):
    raw = io.open(path, encoding="utf-8").read()
    try:
        import yaml
        doc = yaml.safe_load(raw) or {}
    except Exception:
        continue
    # $${VAR} — evaluated INSIDE the container at runtime (needs a declaration)
    runtime = set(re.findall(r'\$\$\{?([A-Z_][A-Z0-9_]*)', raw))
    # ${VAR}  — substituted by compose on the HOST at render time (no declaration needed)
    hostsub = set(re.findall(r'(?<!\$)\$\{([A-Z_][A-Z0-9_]*)', raw))
    for svc in (doc.get("services") or {}).values():
        env = svc.get("environment") or []
        names = [e.split("=")[0] for e in env] if isinstance(env, list) else list(env)
        for n in names:
            checked += 1
            if engine_owned(n) or n in runtime or n in hostsub:
                continue
            if (path, n) in KNOWN:
                continue
            bad.append((path, n))

if bad:
    print("FAIL: compose declares env var(s) that NOTHING reads — a knob the user can")
    print("      set, that docker forwards, and that changes nothing:")
    for p, n in bad:
        print(f"  ⛔ {p}")
        print(f"       ${n} declared in environment: but never read as $${{{n}}} (runtime)")
        print(f"       nor substituted as ${{{n}}} (host). Either WIRE it (entrypoint/command)")
        print(f"       or REMOVE the declaration. If it is engine-consumed, add its prefix")
        print(f"       to ENGINE_PREFIXES in this test with a one-line reason.")
    sys.exit(1)

print(f"  ✓ no dead env declarations ({checked} declarations checked across "
      f"{len(glob.glob('models/*/*/compose/*/*/*.yml'))} composes, "
      f"{len(KNOWN)} documented back-compat exception(s))")
PY

echo "test-compose-env-declared-unused: ok"
