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
#   not matched by ENGINE_PREFIXES / ENGINE_EXACT AND
#   not read by a mounted program the compose's CODE runs (launch_knobs.MOUNTED_READERS —
#     e.g. `python3 /etc/club3090/effort_budget.py shell-env` reads THINKING_BUDGET_*;
#     the same table the launch-knobs scan uses, so the two can't disagree)
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
    # QEngine: its image's entrypoint, server/app.py and libqengine.so / libqflash.so read QE_*
    "QE_",
)
ENGINE_EXACT = {"PATH", "HOME", "HOSTNAME", "LD_LIBRARY_PATH", "PWD", "IFS",
                "DEBIAN_FRONTEND", "PYTHONUNBUFFERED", "PYTHONUTF8",
                # QEngine's Flash-Next engine (libqflash.so) reads these; named one by one,
                # because a FLASH_ prefix would also excuse FLASH_ATTN on the vLLM composes
                "FLASH_MAX_SEQS", "FLASH_KV_ARCHIVE_GB", "FLASH_NGRAM_PIN",
                "FLASH_EC_MARGIN_MB", "FLASH_LOOKUP", "FLASH_COMM",
                # from 1.3.1
                "FLASH_KV", "FLASH_KV_RAM_TOKENS", "FLASH_PFB_BIG", "FLASH_EC_RECLAIM_S",
                "FLASH_MAX_PREFILLS", "FLASH_GUARD_MAX",
                # Strata's engine (components/engine/<version>/strata) reads these; named one by one, because a
                # STRATA_ prefix would also excuse our own STRATA_START_GPUS / STRATA_API_KEY settings
                "STRATA_GDN_TC", "STRATA_GDN_CONVL2",
                # the engine's PLE residency policy: 0 keeps the n-gram table on direct disk reads
                "STRATA_PLE_AUTO_RAM",
                # from 1.3.2
                "FLASH_PFB_ALIAS", "FLASH_PREFIX_GB", "FLASH_YARN", "FLASH_LEND_CHECK",
                # 1.3.2's copy-engine prefill pipeline switch (#1841)
                "FLASH_QSA_PIPE_CE",
                # 1.3.3's engine-step watchdog
                "FLASH_RUN_TIMEOUT_S",
                # 1.3.4's parallel mode (tensor parallel / layer pipeline)
                "FLASH_PAR",
                # Project Maya's serve/server.py reads it (model load/unload endpoints, on by default from v1.1.47)
                "MAYA_MODEL_MANAGEMENT"}

# DEBT REGISTER — each entry must name WHY the dead knob is deliberate.
# Shrink it; never grow it silently. An entry here means a documented knob is
# inert in production.
KNOWN = {
    # (empty since 2026-10-17: the qwen3.8-27b llama.cpp composes' THINKING, once a documented
    # back-compat no-op, now wins over ENABLE_THINKING=false, so it is read)
}

def engine_owned(name: str) -> bool:
    return name in ENGINE_EXACT or name.startswith(ENGINE_PREFIXES)

# A mounted program reads knobs from the environment, so no `$${VAR}` appears in the
# YAML. Only a line of CODE that runs it counts — a comment mentioning it does not.
sys.path.insert(0, "scripts/lib/profiles")
from launch_knobs import MOUNTED_READERS

# Every name the program itself reads from the environment counts, not only the catalogued knobs
# MOUNTED_READERS lists: strata_config.py and maya_settings.py also read the sampler settings,
# THINKING_BUDGET and KV, which the launch-knob catalogue doesn't cover (until 2026-10-17 this test
# flagged them, red on master since #1723 / #1725).
# The names come from the program's string constants (an exact NAME literal: both launchers read the
# sampler settings in a loop over ("TEMP", ...) tuples, so no env("TEMP") call names them); a name only
# mentioned inside a longer string (a docstring, a message) doesn't count.
import ast
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]+")

def program_reads(reader) -> set:
    try:
        tree = ast.parse(io.open(reader["file"], encoding="utf-8").read())
    except (OSError, SyntaxError):
        return set()
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and ENV_NAME.fullmatch(n.value)}

def mounted_reads(raw: str) -> set:
    code = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")]
    names = set()
    for reader in MOUNTED_READERS:
        if any(reader["invoke"] in ln for ln in code):
            names.update(reader["knobs"])
            names.update(program_reads(reader))
    return names

# Negative controls: the rule must not excuse a forward with no (or only a commented) invoke.
_inv = MOUNTED_READERS[0]["invoke"]
assert mounted_reads("x:\n  - |\n    " + _inv + " --engine vllm\n") >= set(MOUNTED_READERS[0]["knobs"])
assert not mounted_reads("x:\n  - |\n    # " + _inv + " --engine vllm\n"), "a commented invoke must not count"
assert not mounted_reads("x:\n  - |\n    exec vllm serve\n"), "no invoke must not count"
_maya = next(r for r in MOUNTED_READERS if r["file"].endswith("maya_settings.py"))
_got = mounted_reads("x:\n  - |\n    " + _maya["invoke"] + " maya-s\n")
assert {"TEMP", "THINKING_BUDGET", "KV"} <= _got, f"the program's own reads must count: {sorted(_got)}"
assert "NOT_A_MAYA_SETTING" not in _got and "MAX_TOKENS" not in _got, "a name the program doesn't read must not count"

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
    runtime = set(re.findall(r'\$\$\{?([A-Z_][A-Z0-9_]*)', raw)) | mounted_reads(raw)
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
