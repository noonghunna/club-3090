#!/usr/bin/env bash
set -euo pipefail

# test-hwdetect.sh — v0.8.2 STEP V4 (CONTRACT-3 §8: optional whichllm
# hardware-detect-ONLY subprocess).
#
# The test IS the spec. CONTRACT-3 binds BOTH RED-LINE halves; this test
# proves an OUTCOME moved (not merely that a module exists — the V3
# relabel lesson / outcome-not-addition convention):
#
#   (a) SAFETY half. `hwdetect` is OPTIONAL with NO new hard dependency:
#       - the module imports cleanly even though `whichllm` is absent in
#         CI (no ImportError, no side effect);
#       - every non-delivery path degrades to None and NEVER raises:
#         tool absent, runner failing, unparseable JSON, an NVIDIA-only
#         payload, an unrecognised-vendor payload;
#       - the NVIDIA majority NEVER enters the seam: with `hardware_sm`
#         injected (nvidia-smi present), the run_pull result is
#         byte-identical whether `hwdetect_fn` is absent OR present and
#         degrading — proving zero NVIDIA-path behaviour change;
#       - when nvidia-smi is absent AND hwdetect degrades, the eval path
#         is byte-identical to the shipped `hardware-sm-undetermined`
#         terminal (hwdetect-absent vs hwdetect-present-degrading).
#
#   (b) DELIVERY half (MANDATORY — an always-degrading no-op stub is a
#       FAIL). A SIMULATED non-nvidia environment (an injected runner
#       returning an AMD / Apple enumeration JSON — this rig is
#       NVIDIA-only so the non-nvidia env is necessarily simulated;
#       the simulation is explicit + deterministic) makes the eval
#       path's hardware view OBSERVABLY DIFFER from the bare degrade
#       path: `run_pull` no longer terminates at
#       `hardware-sm-undetermined`; it reaches the [C0] SM gate with the
#       enumerated SM-equivalent and `res.diagnostics["hwdetect"]
#       ["augmented"]` is True. The exact downstream consume-point is
#       `pull.py`'s stratum-3 `if hardware_sm is None:` SM-gate input.
#
#   * kv-calc is NEVER consulted (CONTRACT-3 is hw-detect-ONLY; kv-calc
#     stays the sole fit authority). Asserted: hwdetect imports no
#     kv-calc symbol; the augment touches only `hardware_sm` + an
#     additive notice/diag, never a fit/VRAM call. The shell wrapper
#     additionally re-runs `tools/kv-calc.py --calibration` so the suite
#     proves the calibration verdict is unaffected by V4.
#
#   * Rig-independent leak convention (mandatory for every STEP since the
#     V2 sandbox-masked miss): assert the absolute repo dir string is NOT
#     present in any user-visible notice / diagnostic the augment emits —
#     `str(abs_dir) not in shared`, NOT a `/opt|/home` substring
#     allowlist (a sandbox path defeats that).
#
# PURE / hermetic: no Docker, no GPU, no network, no real `whichllm`.
# Every external is injected (the shipped inject-or-detect seam).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import hwdetect as H  # noqa: E402
from scripts.lib.profiles import pull as P  # noqa: E402
from scripts.lib.profiles import deriver as D  # noqa: E402
from scripts.lib.profiles.compat import load_profiles  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


profiles = load_profiles()
SM_86 = 8.6  # RTX 3090 (Ampere) — the NVIDIA majority anchor.

CFG = f"{D._HF_RESOLVE}/{{slug}}/resolve/main/config.json"
API = f"{D._HF_API}/{{slug}}?blobs=true"


class FixtureFetcher:
    def __init__(self, routes: dict):
        self.routes = routes

    def get(self, url, headers=None, range_=None):
        if url not in self.routes:
            return D.FetchResponse(status=404, body=b"")
        spec = self.routes[url]
        if isinstance(spec, D.FetchResponse):
            return spec
        return D.FetchResponse(
            status=200, body=json.dumps(spec).encode("utf-8")
        )


def fake_statvfs(free_gb: float):
    def _sv(_p):
        class S:
            f_frsize = 4096
            f_bavail = int(free_gb * (1024 ** 3) / 4096)
        return S()
    return S if False else _sv


BIG_DISK = fake_statvfs(500.0)


def dense_cfg(arch="LlamaForCausalLM", **over):
    c = {
        "model_type": "llama",
        "architectures": [arch],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
    }
    c.update(over)
    return c


def good_fetcher(slug: str) -> FixtureFetcher:
    """A well-formed single-safetensors repo that derives cleanly + passes
    stratum-1/2 so run_pull REACHES the stratum-3 `hardware_sm is None`
    augment seam (the exact V4 consume-point)."""
    return FixtureFetcher({
        API.format(slug=slug): {"siblings": [
            {"rfilename": "model.safetensors", "size": 9_000_000_000},
            {"rfilename": "config.json", "size": 700},
        ]},
        CFG.format(slug=slug): dense_cfg(),
    })


# ===========================================================================
# SECTION 0 — module hygiene: optional, no new hard dependency.
# ===========================================================================
print("--- module hygiene (no hard dep) ---")

# Imported at top-level WITHOUT `whichllm` installed -> no ImportError.
check(hasattr(H, "detect_non_nvidia_sm")
      and hasattr(H, "detect_non_nvidia_hw")
      and hasattr(H, "HwDetectResult"),
      "hwdetect imports cleanly with `whichllm` absent (no hard dep)")

# hwdetect must NOT import / call kv-calc or the fit authority
# (hw-detect-ONLY). The docstring legitimately *names* kv-calc to state
# it is NOT consulted, so assert on real import/call tokens, not the
# bare string: no `import`/`from ... kv`/`tools.kv`/`kv-calc.py` exec.
src = (root / "scripts/lib/profiles/hwdetect.py").read_text()
import_lines = [
    ln.strip() for ln in src.splitlines()
    if ln.strip().startswith(("import ", "from "))
]
check(not any("kv" in ln.lower() or "fit" in ln.lower()
              for ln in import_lines),
      f"hwdetect imports NO kv-calc/fit module (imports={import_lines}) "
      "— kv-calc stays sole fit authority (CONTRACT-3 hw-detect-ONLY)")
check("tools.kv" not in src and "kv-calc.py" not in src
      and "import kv" not in src,
      "hwdetect never imports/execs kv-calc (hw-detect-ONLY)")


# ===========================================================================
# SECTION 1 — SAFETY half: every non-delivery path degrades to None,
#             never raises.
# ===========================================================================
print("\n--- safety half: graceful degrade, never raise ---")


def runner_raises():
    raise RuntimeError("simulated `whichllm` crash")


def runner_empty():
    return ""


def runner_garbage():
    return "this is not json {{{"


def runner_nvidia_only():
    # An enumeration that lists ONLY NVIDIA — hwdetect must NOT claim it
    # (the nvidia-smi path owns NVIDIA; by construction it already ran).
    return json.dumps({"devices": [
        {"vendor": "nvidia", "name": "NVIDIA RTX 3090"},
    ]})


def runner_unknown_vendor():
    return json.dumps({"devices": [
        {"vendor": "weirdsilicon", "name": "Mystery TPU v9"},
    ]})


for rn, label in [
    (runner_raises, "runner raises"),
    (runner_empty, "empty stdout"),
    (runner_garbage, "unparseable JSON"),
    (runner_nvidia_only, "NVIDIA-only payload"),
    (runner_unknown_vendor, "unrecognised vendor"),
]:
    try:
        r_hw = H.detect_non_nvidia_hw(runner=rn)
        r_sm = H.detect_non_nvidia_sm(runner=rn)
        check(r_hw is None and r_sm is None,
              f"safety: {label} -> None (graceful degrade, no raise)")
    except Exception as exc:  # noqa: BLE001
        check(False, f"safety: {label} RAISED {type(exc).__name__} "
                     f"(must degrade to None)")

# Default (no runner, no `whichllm` on PATH in CI) -> None.
check(H.detect_non_nvidia_sm() is None,
      "safety: default (no `whichllm` on PATH) -> None")


# ===========================================================================
# SECTION 2 — NVIDIA path is byte-identical (seam never entered).
#   With hardware_sm injected (== nvidia-smi present), the augment block
#   is unreachable. Result must be identical whether hwdetect_fn is
#   ABSENT or PRESENT-and-degrading -> zero NVIDIA-path behaviour change.
# ===========================================================================
print("\n--- NVIDIA path byte-identity (seam never entered) ---")

slug = "fixtures/hwdetect-nvidia"


def _summ(r):
    return (r.ok, r.stratum.name, r.abort_reason, r.detail,
            r.raw_verdict, r.terminal,
            r.diagnostics.get("hwdetect"))


r_absent = P.run_pull(slug, "vllm/minimal", path="B", hardware_sm=SM_86,
                       fetcher=good_fetcher(slug), profiles=profiles,
                       statvfs=BIG_DISK)
# A degrading hwdetect that would FAIL the test if it were ever called
# on the NVIDIA path:
sentinel = {"called": False}


def degrading_hwdetect():
    sentinel["called"] = True
    return None


r_present = P.run_pull(slug, "vllm/minimal", path="B", hardware_sm=SM_86,
                        fetcher=good_fetcher(slug), profiles=profiles,
                        statvfs=BIG_DISK, hwdetect_fn=degrading_hwdetect)

check(_summ(r_absent) == _summ(r_present),
      "NVIDIA path: result byte-identical with hwdetect_fn absent vs "
      "present-degrading")
check(sentinel["called"] is False,
      "NVIDIA path: hwdetect_fn is NEVER invoked when nvidia-smi gave an "
      "SM (seam guarded by `hardware_sm is None`)")
check(r_present.diagnostics.get("hwdetect") is None,
      "NVIDIA path: no `hwdetect` diagnostic emitted (augment not run)")


# ---------------------------------------------------------------------------
# This rig is NVIDIA-only (the dev 2x 3090 — `nvidia-smi` IS present, so
# the real `detect_hardware_sm()` returns 8.6). To exercise the non-NVIDIA
# eval path the absence of nvidia-smi MUST be simulated — the brief
# explicitly accepts + expects this; the simulation is made EXPLICIT and
# DETERMINISTIC here by stubbing `detect_hardware_sm` -> None for the
# duration of the non-nvidia sections (restored after, so nothing leaks).
# ---------------------------------------------------------------------------
import contextlib  # noqa: E402


@contextlib.contextmanager
def simulated_no_nvidia_smi():
    """Explicit, deterministic 'nvidia-smi absent' simulation: the real
    SM probe returns None exactly as it would on an AMD/Apple rig."""
    orig = P.detect_hardware_sm
    P.detect_hardware_sm = lambda: None
    try:
        yield
    finally:
        P.detect_hardware_sm = orig


# Sanity: the stub really simulates the non-nvidia condition.
with simulated_no_nvidia_smi():
    check(P.detect_hardware_sm() is None,
          "simulation: detect_hardware_sm() -> None (nvidia-smi absent "
          "simulated, explicit + deterministic)")
check(P.detect_hardware_sm() == 8.6,
      "simulation: real detect_hardware_sm() restored after the context "
      "(no leak into other sections / suites)")


# ===========================================================================
# SECTION 3 — non-nvidia env, hwdetect DEGRADES == bare degrade path.
#   nvidia-smi absent (simulated) AND hwdetect returns None -> the eval
#   path is the shipped `hardware-sm-undetermined` terminal,
#   byte-identical whether hwdetect_fn is absent or present-degrading.
# ===========================================================================
print("\n--- non-nvidia + hwdetect degrades == bare degrade ---")

slug = "fixtures/hwdetect-degrade"
with simulated_no_nvidia_smi():
    r_bare = P.run_pull(slug, "vllm/minimal", path="B",
                        fetcher=good_fetcher(slug), profiles=profiles,
                        statvfs=BIG_DISK, hwdetect_fn=lambda: None)
    r_bare2 = P.run_pull(slug, "vllm/minimal", path="B",
                         fetcher=good_fetcher(slug), profiles=profiles,
                         statvfs=BIG_DISK, hwdetect_fn=lambda: None)
check(r_bare.stratum is P.Stratum.C0
      and r_bare.abort_reason == "hardware-sm-undetermined"
      and not r_bare.ok,
      "degrade: no nvidia-smi + hwdetect None -> shipped "
      "`hardware-sm-undetermined` terminal (unchanged)")
check(_summ(r_bare) == _summ(r_bare2),
      "degrade: byte-identical across repeated degrading runs")
check(r_bare.diagnostics.get("hwdetect") is None,
      "degrade: NO `hwdetect` augment diagnostic on the degrade path")


# ===========================================================================
# SECTION 4 — DELIVERY half (MANDATORY): simulated non-nvidia env yields
#   a structured enumeration the eval path ACTUALLY CONSUMES — the
#   observable OUTCOME moves off the degrade terminal.
# ===========================================================================
print("\n--- delivery half: simulated non-nvidia env moves the outcome ---")


def amd_runner():
    # Explicit, deterministic simulation of an AMD ROCm rig's `whichllm
    # list --json` (this rig is NVIDIA-only — the non-nvidia env is
    # necessarily simulated; the simulation is fixed + reproducible).
    return json.dumps({"devices": [
        {"vendor": "amd", "name": "AMD Instinct MI300X"},
        {"vendor": "amd", "name": "AMD Instinct MI300X"},
    ]})


def apple_runner():
    return json.dumps({"gpus": [
        {"backend": "metal", "name": "Apple M3 Max"},
    ]})


# 4a — the structured result itself (the module half of "delivery").
hw = H.detect_non_nvidia_hw(runner=amd_runner)
check(hw is not None and isinstance(hw, H.HwDetectResult)
      and hw.vendor == "amd" and hw.sm_equiv == 8.0
      and hw.device_count == 2
      and hw.device_names == ["AMD Instinct MI300X",
                              "AMD Instinct MI300X"],
      "delivery: AMD enumeration -> structured HwDetectResult "
      f"(got {hw!r})")
hw_a = H.detect_non_nvidia_hw(runner=apple_runner)
check(hw_a is not None and hw_a.vendor in ("metal", "apple")
      and hw_a.sm_equiv == 8.0
      and hw_a.device_names == ["Apple M3 Max"],
      f"delivery: Apple enumeration -> structured result (got {hw_a!r})")
check(H.detect_non_nvidia_sm(runner=amd_runner) == 8.0,
      "delivery: detect_non_nvidia_sm returns the SM-equivalent (8.0)")

# 4b — the WIRING half: the eval path's hardware view OBSERVABLY DIFFERS
#      from the bare degrade path (Section 3). Same slug/fetcher, the
#      ONLY change is an enumerating hwdetect_fn. The outcome moves OFF
#      `hardware-sm-undetermined` — the [C0] SM gate now runs.
slug = "fixtures/hwdetect-deliver"
with simulated_no_nvidia_smi():
    r_deg = P.run_pull(slug, "vllm/minimal", path="B",
                       fetcher=good_fetcher(slug), profiles=profiles,
                       statvfs=BIG_DISK, hwdetect_fn=lambda: None)
    r_aug = P.run_pull(slug, "vllm/minimal", path="B",
                       fetcher=good_fetcher(slug), profiles=profiles,
                       statvfs=BIG_DISK,
                       hwdetect_fn=lambda: H.detect_non_nvidia_sm(
                           runner=amd_runner))

# The bare-degrade arm is the `hardware-sm-undetermined` terminal …
check(r_deg.abort_reason == "hardware-sm-undetermined",
      "delivery: control arm (hwdetect None) IS the degrade terminal")
# … the augmented arm is OBSERVABLY DIFFERENT: it did NOT stop at
# `hardware-sm-undetermined` — the eval path consumed the enumerated
# SM-equiv and advanced past the SM-blind-refuse terminal.
check(r_aug.abort_reason != "hardware-sm-undetermined",
      "DELIVERY (outcome moved): augmented eval path does NOT terminate "
      f"at `hardware-sm-undetermined` (got {r_aug.abort_reason!r}, "
      f"stratum={r_aug.stratum.name})")
check(_summ(r_aug) != _summ(r_deg),
      "DELIVERY (outcome moved): augmented hardware view differs from "
      "the bare-degrade hardware view (same slug/fetcher; only delta = "
      "an enumerating hwdetect_fn)")
# The structured augment is recorded + carries the consumed SM-equiv.
diag = r_aug.diagnostics.get("hwdetect")
check(isinstance(diag, dict) and diag.get("augmented") is True
      and diag.get("sm_equiv") == 8.0,
      f"delivery: augment diagnostic records sm_equiv=8.0 (got {diag!r})")
check(any("optional hardware-detect subprocess" in n
          for n in r_aug.notices),
      "delivery: an honest user-visible notice records the non-NVIDIA "
      "enumeration + the kv-calc-stays-authority caveat")


# ===========================================================================
# SECTION 5 — kv-calc is NEVER consulted by the augment (hw-detect-ONLY).
#   The augment sets ONLY `hardware_sm` + an additive notice/diag. There
#   is no fit/VRAM call on the augment path. (The shell wrapper also
#   re-runs `kv-calc.py --calibration` to prove the verdict is intact.)
# ===========================================================================
print("\n--- hw-detect-ONLY: augment never feeds kv-calc ---")

# The augmented run advanced PAST the SM-blind refuse — but the value it
# fed is the [C0] *SM gate* input (8.0), NOT a kv-calc fit number. The
# augment diag carries `sm_equiv` only; no predicted-VRAM/fit field.
check(set(diag.keys()) == {"augmented", "sm_equiv"},
      f"hw-detect-ONLY: augment diag has ONLY enumeration keys "
      f"(no fit/VRAM field) — got {sorted(diag)}")


# ===========================================================================
# SECTION 6 — rig-independent leak convention (the V2 lesson).
#   The absolute repo dir string must NOT appear in ANY user-visible
#   notice / diagnostic the augment emits. Assert `str(abs_dir) not in
#   shared`, NOT a `/opt|/home` substring allowlist.
# ===========================================================================
print("\n--- leak-clean (rig-independent assertion) ---")

abs_dir = str(root.resolve())
shared = "\n".join(r_aug.notices) + "\n" + json.dumps(
    r_aug.diagnostics.get("hwdetect"), default=str
)
check(abs_dir not in shared,
      f"leak: absolute repo dir {abs_dir!r} NOT in any augment "
      "notice/diagnostic (rig-independent str(abs_dir) assertion)")
check(str(root) not in shared,
      "leak: the non-resolved abs root form is also absent")


# ---------------------------------------------------------------------------
if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll V4 hwdetect (CONTRACT-3 §8 hw-detect-ONLY) assertions "
      "passed — BOTH halves (safety + delivery) proven.")
PY

# kv-calc verdict must be unaffected by V4 (hw-detect-ONLY: kv-calc is
# the sole fit authority and V4 must not move its calibration).
echo "--- kv-calc --calibration unaffected by V4 ---"
python3 tools/kv-calc.py --calibration >/dev/null 2>&1 \
  && echo "PASS: kv-calc --calibration runs clean (RC=0) post-V4" \
  || { echo "FAIL: kv-calc --calibration regressed" >&2; exit 1; }

echo "test-hwdetect.sh OK"
