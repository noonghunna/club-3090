#!/usr/bin/env bash
#
# pull.sh — v0.8.0 Pull-Gate orchestrator (PR #147, STEP P4).
#
# Derives an HF repo via [A], gates it through the LOCKED 6-stratum abort
# taxonomy (deriver-errors → --profile-like → [C0] → [C2a] → eligibility →
# [B]→[C1] → Path-A [D] dry-run), and on a curated, [D]-emittable, gate-
# passing Path A run hands the validated registry key to the existing #141
# generator for real emission. Path B (--dry-run / any non-curated slug)
# prints a §7-caveated verdict and NEVER calls [D] / downloads.
#
# Honest by construction (design §1): every non-eligible / non-pass outcome
# hard-stops with a precise structured reason; only `exact × fits-clean`
# reaches `proceed` silently (§4.1). `--force-download` is a no-op + notice
# this phase (download/telemetry deferred to the Loop phase).
#
# Usage:
#   scripts/pull.sh <hf-slug> --profile-like <COMPOSE_REGISTRY-key> [opts]
#
#   # Path A — curated pull-and-emit:
#   scripts/pull.sh Lorbus/Qwen3.6-27B-int4-AutoRound \
#       --profile-like vllm/minimal --out /tmp/qwen.yml
#
#   # Path B — universal evaluate (never emits/downloads):
#   scripts/pull.sh some-org/Some-Llama-7B --profile-like vllm/minimal --dry-run
#
#   # Failure on-ramp — submit a captured failed pull (a SEPARATE, consented
#   # verb: the ONLY step that touches the network, and only after an
#   # explicit y; reuses the shipped dedup; needs no slug/--profile-like):
#   scripts/pull.sh --submit-last            # the most-recent capture
#   scripts/pull.sh --submit <capture-dir>   # an explicit bundle dir
#
# Opts: --yes  --force-download  --experimental-arch  --trust-remote-code
#       --hf-home DIR  --out FILE (Path A)  --hardware SM (override nvidia-smi)
#
# All decision logic lives in scripts/lib/profiles/pull.py (this is a thin
# argv pass-through, matching the generate-compose.sh / diagnose-profile.sh
# pattern). Exit: 0 = download-eligible / clean verdict; 3 = needs a flag
# (confirm→proceed / advisory); 2 = honest hard-stop; 64 = usage.
#
# --profile-like ... --dry-run --json (structured swap_path) — TUI contract
# -------------------------------------------------------------------------
# ADDITIVE: a machine-readable form of the `--profile-like ... --dry-run`
# gate verdict. Emits a single JSON object:
#
#   {"fit_verdict": <raw_verdict|terminal|abort_reason>,
#    "arch": <config architectures[0] | null>,
#    "eligible": <bool — the gate said download-eligible / clean verdict>,
#    "note": <the human detail line, verbatim>,
#    "swap_path": {"route": "B"|"C"|null,
#                  "sibling_slug": <curated model slug | null>,
#                  "quant_match": <quant the user must match | null>,
#                  "drop_spec_config": <bool>}}
#
# CRITICAL (per the contract): the BRING_YOUR_OWN swap path that the human
# gate bakes into the NOTE message *string* is surfaced here as STRUCTURED
# `swap_path` FIELDS — derived from the SAME `arch_model_xref` registry the
# gate's own `_hint` consults (read-only reuse, never a sentence parse), so
# the TUI reads fields instead of parsing prose.
#
# STRICTLY ADDITIVE: `--json` is intercepted HERE in the wrapper and stripped
# before the gate would ever see it; WITHOUT `--json` this script's behaviour
# (and pull.py's argv) is byte-for-byte unchanged. `--json` forces the
# evaluate-only Path-B gate (it never emits/downloads), matching the
# `--dry-run` contract surface.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Intercept --json (strictly additive). Absent -> the original exec path,
# byte-identical. Present -> strip it and hand the remaining argv to the
# structured-emit helper (which forces the evaluate-only gate).
_pull_json=0
_pull_apply_swap=0
_pull_emit_only=0
_pull_args=()
for _a in "$@"; do
    if [ "$_a" = "--json" ]; then
        _pull_json=1
    elif [ "$_a" = "--apply-swap" ]; then
        _pull_apply_swap=1
    elif [ "$_a" = "--emit-only" ]; then
        _pull_emit_only=1
    else
        _pull_args+=("$_a")
    fi
done

# --apply-swap: the BYO Route-C weight-swap ACTION (strictly additive, like
# --json). It NEVER runs the locked 6-stratum gate — a curated-arch fine-tune
# hard-stops there at stratum-5 `no-fit-model`, which is correct (nothing to
# price). Instead it downloads the brought weights SHA-verified and emits a
# serve-locally compose that CLONES the --profile-like sibling with --model
# re-pointed at those weights. The c3 [D] press on a Route-C fit-check is the
# explicit opt-in; without --apply-swap this script's behaviour is unchanged.
#
# --emit-only (with --apply-swap): skip the SHA-verified download and JUST emit
# the serve-locally compose (do_download=False). For when the weights are
# already on disk — c3's ② Serve uses this so a present-weights serve needs no
# [D] download step (the mount points at the existing pull dir).
if [ "$_pull_apply_swap" -eq 1 ]; then
    export _PULL_EMIT_ONLY="$_pull_emit_only"
    exec python3 - "${ROOT_DIR}" "${_pull_args[@]+"${_pull_args[@]}"}" <<'PY'
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.lib.profiles import swap_apply as SA   # the apply-swap action

ap = argparse.ArgumentParser(prog="pull.sh --apply-swap", add_help=False)
ap.add_argument("slug")
ap.add_argument("--profile-like", required=True, dest="profile_like")
ap.add_argument("--hf-home", default=None)
# tolerate (ignore) the flags a caller may carry over from the fit-check line
for _f in ("--dry-run", "--yes", "--force-download", "--experimental-arch",
           "--trust-remote-code", "--recommend"):
    ap.add_argument(_f, action="store_true")
ap.add_argument("--hardware", default=None)
ap.add_argument("--hardware-gpus", default=None)
ap.add_argument("--out", default=None)
args, _unknown = ap.parse_known_args(sys.argv[2:])

_emit_only = os.environ.get("_PULL_EMIT_ONLY") == "1"
print(f"[apply-swap] resolving Route-C swap for {args.slug} "
      f"(profile-like={args.profile_like})"
      + (" [emit-only — weights on disk, no download]" if _emit_only else ""),
      flush=True)
res = SA.apply_swap(root, args.slug, args.profile_like, hf_home=args.hf_home,
                    do_download=not _emit_only)
if not res.get("ok"):
    print(f"[apply-swap] ERROR: {res.get('error')}", file=sys.stderr, flush=True)
    if res.get("detail"):
        print(f"[apply-swap]   detail: {res['detail']}", file=sys.stderr, flush=True)
    sys.exit(1)

mtp = "MTP kept" if res.get("has_mtp_head") else "MTP dropped (no head)"
print(f"[apply-swap] sibling={res['sibling_slug']}  served-as={res['served_model_name']}"
      f"  ({mtp})", flush=True)
print(f"[apply-swap] weights: {res['weights_dir']}", flush=True)
# The final, machine-parseable line the c3 lane greps for → serve this compose.
print(f"[apply-swap] compose: {res['compose_path']}", flush=True)
print("[apply-swap] ok", flush=True)
PY
fi

if [ "$_pull_json" -eq 0 ]; then
    exec python3 "${ROOT_DIR}/scripts/lib/profiles/pull.py" "$@"
fi

# --json path: run the SHIPPED gate (run_pull, evaluate-only) and serialize
# its verdict + a structurally-derived swap_path. pull.py / deriver /
# generate_compose are imported READ-ONLY (no edit) — this reuses their
# logic, never reimplements it.
exec python3 - "${ROOT_DIR}" "${_pull_args[@]+"${_pull_args[@]}"}" <<'PY'
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.lib.profiles import pull as P            # P4 gate — READ-ONLY
from scripts.lib.profiles import deriver as D         # P2 — READ-ONLY
from scripts.lib import generate_compose as gc        # [D] — READ-ONLY

# Mirror the gate parser's REQUIRED surface (slug + --profile-like) plus the
# evaluate-only flags that are meaningful for the JSON verdict. argparse
# error() defaults to exit 2; the JSON contract wants a clean usage code, so
# fall back to 64 (matching pull.py's _UsageExit64Parser intent).
class _Ap(argparse.ArgumentParser):
    def error(self, message):
        self.exit(64, f"pull.sh --json: error: {message}\n")


ap = _Ap(prog="pull.sh --json")
ap.add_argument("slug")
ap.add_argument("--profile-like", required=True, dest="profile_like")
ap.add_argument("--dry-run", action="store_true")        # implied; accepted
ap.add_argument("--yes", action="store_true")
ap.add_argument("--force-download", action="store_true")
ap.add_argument("--experimental-arch", action="store_true")
ap.add_argument("--trust-remote-code", action="store_true")
ap.add_argument("--hf-home")
ap.add_argument("--hardware", type=float, default=None)
ap.add_argument("--hardware-gpus", default=None)
# Tolerate (and ignore for the verdict) Path-A-only / presentation flags so a
# caller can append --json to an existing command line without a usage error.
ap.add_argument("--out")
ap.add_argument("--recommend", action="store_true")
args = ap.parse_args(sys.argv[2:])

gpu_topology = None
if args.hardware_gpus:
    vram: list[int] = []
    names: list[str] = []
    for tok in args.hardware_gpus.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v, n = (tok.split(":", 1) if ":" in tok else (tok, "GPU"))
        vram.append(int(float(v)))
        names.append(n.strip() or "GPU")
    if vram:
        gpu_topology = (len(vram), vram, names)


def _swap_path(slug, der, eligible):
    """Surface the BRING_YOUR_OWN swap path as STRUCTURED fields, derived
    from the SAME `arch_model_xref` registry the gate's `_hint` consults
    (read-only) — never by parsing the human NOTE sentence.

      route "C": the uncurated arch maps to a curated hybrid/MoE model we
                 serve — reuse that model's compose (point --model at the
                 weights). sibling_slug/quant_match populated.
      route "B": no curated sibling — the self-contained GGUF fallback
                 (copy the closest llama.cpp/ik compose).
      route null: not a swap situation (curated hit, or the verdict is not a
                  no-fit-model eligibility stop).
    """
    blank = {"route": None, "sibling_slug": None,
             "quant_match": None, "drop_spec_config": False}
    if der is None or der.error is not None:
        return blank
    # A curated Tier-1 hit needs no swap (its own compose serves it).
    if der.tier1 is not None:
        return blank
    # Only the no-fit-model eligibility stop carries a swap path.
    if eligible or der.generic_dense_eligible:
        return blank
    arch = (der.profile or {}).get("arch")
    try:
        rt = gc._load_yaml(root, "scripts/lib/profiles/profile_runtime.yml")
        canon, row = gc.resolve_arch_from_config(rt, gc.load_arches(root), arch)
    except Exception:
        canon, row = None, None
    sibling = None
    family = None
    if row is not None:
        family = row.get("family")
        slugs = (((rt.get("arch_model_xref") or {}).get(canon) or {})
                 .get("model_slugs") or [])
        if slugs:
            sibling = slugs[0]
    # Route C iff the arch resolves to a curated hybrid/MoE sibling we serve
    # (the families pull.py prices via the curated path, NOT generic-dense).
    if sibling is not None and family in P._FAMILY_WEIGHT_FIELDS:
        quant_match = None
        try:
            from scripts.lib.profiles.compat import load_profiles
            model = load_profiles().models.get(sibling)
            if model is not None and getattr(model, "weights", None):
                # Prefer the quant FORMAT the curated compose loads (what the
                # user must match on their own repo); fall back to the slug.
                first_slug = next(iter(model.weights))
                meta = model.weights[first_slug] or {}
                quant_match = meta.get("format") or first_slug
        except Exception:
            quant_match = None
        # Keep --speculative-config IFF the brought checkpoint actually carries
        # an MTP head (deriver.detect_mtp_head: config declares MTP layers + a
        # dedicated mtp weights file). The old blanket drop served head-
        # preserving fine-tunes (e.g. ThinkingCap AutoRound) MTP-off; a plain
        # re-quant without the head still drops.
        has_mtp = bool((der.profile or {}).get("has_mtp_head"))
        return {"route": "C", "sibling_slug": sibling,
                "quant_match": quant_match,
                "drop_spec_config": not has_mtp}
    # No curated sibling -> the self-contained GGUF fallback (route B).
    return {"route": "B", "sibling_slug": None,
            "quant_match": "gguf", "drop_spec_config": False}


# Derive once (read-only) for arch + the swap-path xref; run the SHIPPED gate
# (evaluate-only: --json never emits/downloads) for the authoritative verdict.
der = D.derive(args.slug, hf_home=args.hf_home)
res = P.run_pull(
    args.slug, args.profile_like,
    dry_run=True,                       # --json is evaluate-only by contract
    yes=args.yes, force_download=args.force_download,
    experimental_arch=args.experimental_arch,
    trust_remote_code=args.trust_remote_code,
    hf_home=args.hf_home,
    hardware_sm=args.hardware, gpu_topology=gpu_topology,
)

# fit_verdict: the most specific machine signal the gate produced — the raw
# [B] verdict when [B] was reached, else the terminal, else the abort reason.
fit_verdict = res.raw_verdict or res.terminal or res.abort_reason

obj = {
    "fit_verdict": fit_verdict,
    "arch": (der.profile or {}).get("arch") if der.error is None else None,
    "eligible": bool(res.ok),
    "note": res.detail or "",
    "swap_path": _swap_path(args.slug, der, bool(res.ok)),
}
print(json.dumps(obj, sort_keys=True))
sys.exit(0 if res.ok else 0)   # JSON emit always exits 0 (verdict is in-band)
PY
