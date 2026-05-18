"""v0.8.2 CONTRACT-3 — optional `whichllm` hardware-detect subprocess.

§8 posture: `whichllm`-as-a-VRAM/fit-engine is OUT (kv-calc stays the sole
fit authority). The ONLY in-scope slice is an **optional, bounded subprocess
that augments hardware *enumeration*** for the eval path on environments
where `nvidia-smi` does not apply (AMD ROCm / Apple Silicon / other-vendor
accelerators). Strictly detect-only: this module enumerates accelerators and
maps a non-NVIDIA device to an SM-equivalent so the eval path can proceed
honestly instead of bare-degrading to `hardware-sm-undetermined`. It NEVER
performs fit/VRAM math and NEVER feeds kv-calc.

Design invariants (the STEP V4 RED-LINE — BOTH halves):

  (a) Safety half. This module is OPTIONAL. There is no new hard
      dependency: the `whichllm` tool being absent, failing, timing out,
      or producing an unrecognised payload all degrade *gracefully* to
      ``None`` (the caller then keeps today's exact `nvidia-smi`-absent
      behaviour). It is consulted ONLY when `nvidia-smi` already returned
      nothing — so for the NVIDIA majority this code path is never
      entered and the shipped detection is byte-identical. It never
      raises out at the caller (every failure mode is caught here and
      returns ``None``), and it never touches kv-calc.

  (b) Delivery half (MANDATORY — an always-degrading no-op stub is a
      FAIL). Given a non-`nvidia-smi` environment in which the optional
      tool DOES enumerate an accelerator, `detect_non_nvidia_sm()`
      returns a real structured `HwDetectResult` carrying an
      SM-equivalent the eval path actually consumes — so the eval path's
      hardware view in that environment differs observably from the bare
      `nvidia-smi`-absent degrade path (it can reach the [C0] SM gate
      instead of terminating at `hardware-sm-undetermined`).

PURE-STDLIB (json / shutil / subprocess), matching the no-external-deps
style of its `scripts/lib/profiles/` siblings. The `whichllm` enumeration
JSON is parsed defensively; nothing in this module imports the rest of the
pull pipeline (no decision-logic coupling — it is a leaf).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# The external enumeration tool. Absence is the COMMON case (NVIDIA rigs do
# not install it) and MUST degrade silently — it is never a hard dependency.
_WHICHLLM_BIN = "whichllm"

# Bounded subprocess (mirrors detect_hardware_sm()'s 15s nvidia-smi bound).
_SUBPROCESS_TIMEOUT_S = 15

# Non-NVIDIA vendor -> the SM-equivalent the eval path's [C0] SM gate keys
# on. These are deliberately CONSERVATIVE enumeration anchors, NOT a fit
# claim: the value only lets the SM gate run on a recognised device class
# instead of refusing blind. kv-calc is NEVER consulted with these — the
# fit authority is untouched (CONTRACT-3 hw-detect-only).
#
# Mapping rationale (enumeration-only, documented constraint):
#   * AMD ROCm CDNA/RDNA -> 8.0  (Ampere-class capability tier; the eval
#     path's SM gate is the only consumer — this is not a perf claim).
#   * Apple Silicon (Metal/MPS) -> 8.0 (same: lets the gate proceed; the
#     §7 boot-fit≠runtime caveat + kv-calc still own the real verdict).
# An unrecognised vendor stays None -> graceful degrade (safety half).
_VENDOR_SM_EQUIV = {
    "amd": 8.0,
    "rocm": 8.0,
    "apple": 8.0,
    "metal": 8.0,
    "mps": 8.0,
}


@dataclass(frozen=True)
class HwDetectResult:
    """A structured non-NVIDIA hardware-enumeration result.

    `sm_equiv` is the ONLY value the eval path consumes (it feeds the [C0]
    SM gate exactly where `detect_hardware_sm()` would have). The vendor /
    device-name / count fields are diagnostics-only enumeration metadata —
    they are NEVER passed to kv-calc.
    """

    sm_equiv: float
    vendor: str
    device_names: list = field(default_factory=list)
    device_count: int = 0
    # The raw enumeration source, for diagnostics/audit only.
    source: str = _WHICHLLM_BIN


def _tool_available() -> bool:
    """True iff the optional enumeration tool is on PATH. Absence is the
    common NVIDIA-rig case and is NOT an error — the caller degrades."""
    return shutil.which(_WHICHLLM_BIN) is not None


def _run_whichllm(runner: Optional[object] = None) -> Optional[str]:
    """Run the bounded enumeration subprocess; return its stdout, or None
    on ANY failure (absent / non-zero / timeout / OSError). `runner` is an
    injection seam so tests are hermetic (CI has no `whichllm` and no
    non-NVIDIA GPU) — None uses the real bounded subprocess."""
    if runner is not None:
        try:
            return runner()
        except Exception:  # noqa: BLE001 - any runner failure -> degrade
            return None
    if not _tool_available():  # pragma: no cover - env dependent
        return None
    try:  # pragma: no cover - exercised on non-NVIDIA rigs, injected in CI
        out = subprocess.run(
            [_WHICHLLM_BIN, "list", "--json"],
            capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:  # pragma: no cover
        return None
    return out.stdout


def _parse_enumeration(stdout: Optional[str]) -> Optional[HwDetectResult]:
    """Parse the enumeration JSON defensively into a HwDetectResult, or
    None when it is absent / unparseable / has no recognised non-NVIDIA
    device. Tolerant to a few plausible `whichllm`-style shapes; an
    unrecognised shape degrades (never raises)."""
    if not stdout:
        return None
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError):
        return None

    # Accept either a bare list of device dicts or a {"devices": [...]} /
    # {"gpus": [...]} envelope — all degrade to None if no list is found.
    devices = None
    if isinstance(payload, list):
        devices = payload
    elif isinstance(payload, dict):
        for key in ("devices", "gpus", "accelerators"):
            if isinstance(payload.get(key), list):
                devices = payload[key]
                break
    if not devices:
        return None

    names: list = []
    matched_vendor: Optional[str] = None
    matched_sm: Optional[float] = None
    for dev in devices:
        if not isinstance(dev, dict):
            continue
        vendor_raw = str(
            dev.get("vendor") or dev.get("backend") or dev.get("type") or ""
        ).strip().lower()
        name = str(
            dev.get("name") or dev.get("device") or dev.get("model") or ""
        ).strip()
        # Skip NVIDIA entries — those belong to the nvidia-smi path which,
        # by construction, already ran and returned nothing if we are here.
        if vendor_raw in ("nvidia", "cuda"):
            continue
        sm_equiv = _VENDOR_SM_EQUIV.get(vendor_raw)
        if sm_equiv is None:
            # Also probe the device name for a known vendor token (e.g.
            # "Apple M3 Max", "AMD Instinct MI300X").
            low = name.lower()
            for token, val in _VENDOR_SM_EQUIV.items():
                if token in low:
                    sm_equiv = val
                    vendor_raw = token
                    break
        if sm_equiv is None:
            continue
        names.append(name or vendor_raw)
        if matched_sm is None:
            matched_sm = sm_equiv
            matched_vendor = vendor_raw

    if matched_sm is None or matched_vendor is None:
        return None
    return HwDetectResult(
        sm_equiv=float(matched_sm),
        vendor=matched_vendor,
        device_names=names,
        device_count=len(names),
        source=_WHICHLLM_BIN,
    )


def detect_non_nvidia_hw(
    runner: Optional[object] = None,
) -> Optional[HwDetectResult]:
    """The structured entry point: enumerate non-NVIDIA accelerators via
    the optional bounded subprocess, returning a HwDetectResult or None.

    None on EVERY non-delivery path (tool absent / failed / timed out /
    payload unparseable / no recognised non-NVIDIA device) — the caller
    then keeps today's exact behaviour (graceful degrade; safety half).
    A real recognised device yields a structured result (delivery half).
    Never raises (every failure is caught -> None)."""
    try:
        return _parse_enumeration(_run_whichllm(runner=runner))
    except Exception:  # noqa: BLE001 - leaf module: NEVER raise at caller
        return None


def detect_non_nvidia_sm(
    runner: Optional[object] = None,
) -> Optional[float]:
    """Convenience: just the SM-equivalent the eval path's [C0] SM gate
    consumes (None -> graceful degrade). This is the ONLY value that
    reaches a decision gate; it NEVER reaches kv-calc."""
    res = detect_non_nvidia_hw(runner=runner)
    return res.sm_equiv if res is not None else None
