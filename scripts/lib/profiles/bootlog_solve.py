"""Boot-log KV back-solve — automates ADDING_MODELS.md Step 5's manual math.

A vLLM boot log reports how much KV-cache pool the engine actually allocated:

    Available KV cache / card = X GiB

Back-solving per-token bytes from that number against the served config,

    measured_per_token_bytes = (Available_KV * 1024^3) / (max_ctx * max_num_seqs / TP)

and comparing against the kv-calc prediction for the same slug classifies the
boot into one of four verdicts:

    match             measured within ±10% of predicted — the calibration row
                      can ship
    half              ~2x delta — K=V tying suspect (either the model ties and
                      the prediction missed it, or vice versa)
    mismatch          anything else — growing-layer miscount suspect (hybrid
                      num_growing/full-attention layer count wrong)
    insufficient-log  the log (or registry) could not supply the required
                      fields — listed honestly, never guessed

WHY A NEW MODULE, NOT A kv-calc SUBCOMMAND:
kv-calc.py is the PREDICTOR — its fit/calibration CLI contract is load-bearing
(scripts/tests/test-kv-calc-*.sh + report.sh + the cockpit all consume it).
Parsing a measured boot log is MEASUREMENT domain and lives with the other
profile tooling under scripts/lib/profiles/.  Keeping it separate also keeps
the parse core STDLIB-ONLY: the registry / kv-calc imports are lazy, so
`parse_bootlog()` works on any box (a contributor's laptop, a CI grep) without
PyYAML or a repo checkout.  The cockpit consumes the CLI below through its
standard ``--json`` subprocess contract — the same seam it uses for
``tools/kv-calc.py --fit`` — so the packaged TUI never imports repo internals.

The PREDICTED side is NOT reimplemented here: kv-calc is loaded by file
location (its filename has a dash) and its own ``kv_pool_per_card_bytes``
evaluated at ctx=1/seqs=1/TP=1 yields exactly the KV_MATH formula's
predicted_per_token_bytes (num_growing_layers x num_kv_heads x head_dim x
k_v_tensors x bytes-per-element), including the measured-calibrated override
for gemma4_unified.  One source of truth.

Usage:
    docker logs <container> 2>&1 | \\
        python3 scripts/lib/profiles/bootlog_solve.py --slug <engine>/<variant> --json
    # or: ... --log-file /tmp/boot.log --json

Field precedence (honesty rules):
  - ``available_kv_gib`` / ``kv_pool_tokens`` MUST come from the log.
  - ``max_ctx`` / ``max_num_seqs`` / ``tp`` come from the LOG when present,
    else fall back to the slug's registry compose config (marked "registry" in
    ``field_sources``) — an env-overridden serve is still solved correctly when
    the log prints the values, and flagged as registry-derived when it doesn't.
  - When the log carries BOTH the available-GiB line and the
    ``GPU KV cache size: N tokens`` line, the direct bytes/tokens solve wins —
    it needs no ctx/seqs/TP assumptions at all.

Known method noise (documented, not hidden): the back-solve attributes the
whole reported pool to context-growing tokens, so models with a fixed
sliding-window term (Gemma family) read slightly ABOVE the pure growing-term
prediction; the ±10% match band absorbs small pools, and a HALF verdict is
still unambiguous at 2x.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Classification bands ──────────────────────────────────────────────────────
MATCH_BAND = 0.10        # |measured/predicted - 1| <= 0.10 → match
HALF_LO, HALF_HI = 1.75, 2.25   # ratio (or inverse) within → half (K=V tying)

VERDICT_MATCH = "match"
VERDICT_HALF = "half"
VERDICT_MISMATCH = "mismatch"
VERDICT_INSUFFICIENT = "insufficient-log"
VERDICT_UNSUPPORTED = "unsupported"


# =============================================================================
# Log parsing (STDLIB ONLY — no repo imports at module scope)
# =============================================================================

# The repo-documented form (ADDING_MODELS.md / KV_MATH.md) plus the real vLLM
# phrasings ("Available KV cache memory: X GiB").  Everything between the words
# "KV cache" and the number ("/ card =", "memory:", ":") is tolerated.
_KV_GIB_RE = re.compile(
    r"Available\s+KV\s+cache[^=\n:]*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*GiB", re.IGNORECASE
)
_KV_GIB_BARE_RE = re.compile(
    r"Available\s+KV\s+cache[^=\n:]*\s([0-9]+\.[0-9]+)\s*GiB", re.IGNORECASE
)
# vLLM's token-count line — enables the direct bytes/tokens solve.
_POOL_TOKENS_RE = re.compile(r"GPU\s+KV\s+cache\s+size:\s*([0-9][0-9,]*)\s*tokens", re.IGNORECASE)
_MAX_CTX_RE = re.compile(r"max_model_len\s*[=:]\s*([0-9]+)")
_MAX_CTX_CONCURRENCY_RE = re.compile(r"Maximum concurrency for\s+([0-9]+)\s+tokens", re.IGNORECASE)
_SEQS_RE = re.compile(r"max_num_seqs\s*[=:]\s*([0-9]+)")
_TP_RE = re.compile(r"tensor_parallel_size\s*[=:]\s*([0-9]+)")
# Context lines kept as evidence (weights / non-KV gpu-memory accounting).
_GPU_MEM_LINE_RE = re.compile(
    r"(model weights take|Non-KV memory|non-torch|torch\.cuda|GPU memory|GiB\))",
    re.IGNORECASE,
)


def _first_int(text: str, *patterns: re.Pattern[str]) -> tuple[Optional[int], Optional[str]]:
    """First matching pattern wins; returns (value, matched-line)."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            line = text[m.start():].splitlines()[0].strip()
            return int(m.group(1)), line
    return None, None


def parse_bootlog(text: str) -> dict[str, Any]:
    """Extract the back-solve fields from a container boot log.

    Returns a dict of nullable fields plus the raw evidence lines that produced
    them (so the UI can SHOW its work and failures stay diagnosable)."""
    gib_match = _KV_GIB_RE.search(text) or _KV_GIB_BARE_RE.search(text)
    available_kv_gib: Optional[float] = None
    kv_line: Optional[str] = None
    if gib_match:
        available_kv_gib = float(gib_match.group(1))
        kv_line = text[gib_match.start():].splitlines()[0].strip()

    tokens_raw = _POOL_TOKENS_RE.search(text)
    kv_pool_tokens: Optional[int] = None
    if tokens_raw:
        kv_pool_tokens = int(tokens_raw.group(1).replace(",", ""))
        tok_line = text[tokens_raw.start():].splitlines()[0].strip()
    else:
        tok_line = None

    max_ctx, ctx_line = _first_int(text, _MAX_CTX_RE, _MAX_CTX_CONCURRENCY_RE)
    seqs_val, seqs_line = _first_int(text, _SEQS_RE)
    tp, tp_line = _first_int(text, _TP_RE)

    gpu_memory_lines = [
        ln.strip() for ln in text.splitlines() if _GPU_MEM_LINE_RE.search(ln)
    ][:6]

    return {
        "available_kv_gib": available_kv_gib,
        "kv_pool_tokens": kv_pool_tokens,
        "max_ctx": max_ctx,
        "max_num_seqs": seqs_val,
        "tp": tp,
        "evidence": {
            "available_kv_line": kv_line,
            "pool_tokens_line": tok_line,
            "model_length_line": ctx_line,
            "max_num_seqs_line": seqs_line,
            "tensor_parallel_line": tp_line,
            "gpu_memory_lines": gpu_memory_lines,
        },
    }


# =============================================================================
# Classification (pure — unit-testable without any registry)
# =============================================================================

def classify(measured: float, predicted: float) -> tuple[str, str]:
    """Classify measured vs predicted per-card per-token KV bytes.

    Returns (verdict, suspected_cause).  Bands: ±10% → match; ratio (or its
    inverse) within [1.75, 2.25] → half (K=V tying suspect); else mismatch
    (growing-layer miscount suspect)."""
    if predicted <= 0:
        return VERDICT_INSUFFICIENT, "non-positive predicted per-token bytes — cannot compare"
    ratio = measured / predicted
    if abs(ratio - 1.0) <= MATCH_BAND + 1e-9:  # 1e-9: 110/100 lands at 0.1000…09 in binary
        return (
            VERDICT_MATCH,
            "measured per-token KV matches the kv-calc prediction within ±10% "
            "— calibration row can ship",
        )
    factor = ratio if ratio > 1 else 1 / ratio
    if HALF_LO <= factor <= HALF_HI:
        if ratio < 1:
            return (
                VERDICT_HALF,
                f"measured is ~{factor:.2f}x BELOW predicted — K==V tying likely ACTIVE "
                "in the engine but not modelled (set k_v_tensors=1)",
            )
        return (
            VERDICT_HALF,
            f"measured is ~{factor:.2f}x ABOVE predicted — K==V tying assumed but apparently "
            "NOT active (check attention_k_eq_v / k_v_tensors)",
        )
    return (
        VERDICT_MISMATCH,
        f"delta is {factor:.2f}x — not a 2x tying signature; growing-layer miscount suspect "
        "(hybrid num_growing/full-attention layer count wrong for this arch)",
    )


def back_solve(
    parsed: dict[str, Any],
    *,
    predicted_tp1: Optional[float],
    registry_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Turn parsed log fields (+ optional registry fallback config and the
    kv-calc TP=1 per-token prediction) into the verdict card payload."""
    cfg = registry_cfg or {}
    field_sources: dict[str, Optional[str]] = {}
    missing: list[str] = []

    def pick(name: str) -> Optional[int]:
        val = parsed.get(name)
        if val is not None:
            field_sources[name] = "log"
            return int(val)
        if cfg.get(name) is not None:
            field_sources[name] = "registry"
            return int(cfg[name])
        field_sources[name] = None
        return None

    available_kv_gib = parsed.get("available_kv_gib")
    kv_pool_tokens = parsed.get("kv_pool_tokens")
    if available_kv_gib is None:
        missing.append("available-kv-gib ('Available KV cache / card = X GiB')")

    max_ctx = pick("max_ctx")
    max_num_seqs = pick("max_num_seqs")
    tp = pick("tp")

    # The measured side is NORMALIZED TO TP1-EQUIVALENT per-token bytes so it
    # compares against ``predicted_tp1`` under one convention (the per-card
    # view is reported alongside, divided by TP):
    #   back-solve path: bytes / (max_ctx * seqs / tp)      → b_tp1  (KV_MATH form)
    #   direct path:     (bytes / pool_tokens) * tp         → b_tp1
    # (vLLM's per-worker pool holds its head-shard of ALL ctx*seqs tokens.)
    measured_tp1: Optional[float] = None
    measured_card: Optional[float] = None
    solve_path: Optional[str] = None
    if available_kv_gib is not None:
        if kv_pool_tokens:
            if tp is None:
                missing.append("tp (log 'tensor_parallel_size' or registry)")
            else:
                measured_card = (available_kv_gib * 1024**3) / kv_pool_tokens
                measured_tp1 = measured_card * tp
                solve_path = "direct-bytes-over-tokens"
        else:
            if max_ctx is None:
                missing.append("max_ctx (log 'max_model_len' or registry)")
            if max_num_seqs is None:
                missing.append("max_num_seqs (log or registry)")
            if tp is None:
                missing.append("tp (log 'tensor_parallel_size' or registry)")
            if not missing and max_ctx and max_num_seqs and tp:
                denom = max_ctx * max_num_seqs / tp
                if denom > 0:
                    measured_tp1 = (available_kv_gib * 1024**3) / denom
                    measured_card = measured_tp1 / tp
                    solve_path = "back-solve-ctx-seqs-tp"

    out: dict[str, Any] = {
        "verdict": VERDICT_INSUFFICIENT,
        "suspected_cause": None,
        "measured_per_token_bytes": round(measured_card, 2) if measured_card is not None else None,
        "measured_per_token_bytes_tp1": round(measured_tp1, 2) if measured_tp1 is not None else None,
        "predicted_per_token_bytes": round(float(predicted_tp1) / tp, 2)
        if (predicted_tp1 is not None and tp)
        else None,
        "predicted_per_token_bytes_tp1": round(float(predicted_tp1), 2)
        if predicted_tp1 is not None
        else None,
        "ratio": round(measured_tp1 / float(predicted_tp1), 4)
        if (measured_tp1 is not None and predicted_tp1)
        else None,
        "solve_path": solve_path,
        "available_kv_gib": available_kv_gib,
        "kv_pool_tokens": kv_pool_tokens,
        "max_ctx": max_ctx,
        "max_num_seqs": max_num_seqs,
        "tp": tp,
        "field_sources": field_sources,
        "missing_fields": missing,
        "evidence": parsed.get("evidence", {}),
    }
    if measured_tp1 is None or predicted_tp1 is None:
        if measured_tp1 is not None:
            # Measured but nothing to compare against (non-catalog slug etc.).
            out["missing_fields"].append("predicted-per-token (no calibrated kv-calc spec)")
        out["verdict"] = VERDICT_INSUFFICIENT
        out["suspected_cause"] = (
            "could not assemble both sides of the comparison — fields missing: "
            + "; ".join(out["missing_fields"])
            if out["missing_fields"]
            else "comparison unavailable"
        )
        return out

    verdict, cause = classify(measured_tp1, float(predicted_tp1))
    out["verdict"] = verdict
    out["suspected_cause"] = cause
    return out



# =============================================================================
# Registry + kv-calc integration (LAZY heavy imports)
# =============================================================================

_KVCALC_MOD: Any = None


def _load_kvcalc() -> Any:
    """Load tools/kv-calc.py by file location (dash in the filename) and cache
    it.  Raises on failure — callers convert to honest errors, never guesses."""
    global _KVCALC_MOD
    if _KVCALC_MOD is not None:
        return _KVCALC_MOD
    import importlib.util

    root = REPO_ROOT
    for p in (str(root), str(root / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("kv_calc", root / "tools" / "kv-calc.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kv_calc"] = mod
    spec.loader.exec_module(mod)
    _KVCALC_MOD = mod
    return mod


def slug_facts(slug: str) -> dict[str, Any]:
    """Resolve a registry slug to the facts the comparison needs:
    {model_id, kv_format, registry_cfg{max_ctx,max_num_seqs,tp}, predicted_tp1}.
    On failure returns {'error': str} — honest, no fabricated numbers."""
    try:
        kc = _load_kvcalc()
    except Exception as exc:
        return {"error": f"could not load tools/kv-calc.py: {exc}"}
    try:
        resolved, model_id, err = kc._resolve_fit_slug(slug)
    except Exception as exc:
        return {"error": f"could not resolve slug {slug!r}: {exc}"}
    if err is not None:
        return {"error": err}
    entry = kc.COMPOSE_REGISTRY[resolved]
    if entry.get("kvcalc_key") in (None, "SKIP"):
        return {"error": f"{slug!r} is not kv-calc priceable (SKIP engine)"}
    cfg = kc._compose_cfg_from_registry(kc.PROFILES, model_id, "__bootlog__", resolved)
    spec = kc.MODEL_SPECS[model_id]
    # ctx=1/seqs=1/TP=1 isolates the per-token growing term — exactly the
    # KV_MATH predicted_per_token_bytes formula (fixed terms returned separately).
    predicted_tp1 = float(kc.kv_pool_per_card_bytes(spec, cfg["kv_format"], 1, 1, 1)[0])
    return {
        "slug": resolved,
        "model_id": model_id,
        "kv_format": cfg["kv_format"],
        "registry_cfg": {
            "max_ctx": cfg.get("max_ctx"),
            "max_num_seqs": cfg.get("max_num_seqs"),
            "tp": cfg.get("tp"),
        },
        "predicted_per_token_bytes_tp1": predicted_tp1,
    }


def solve_log(slug: str, log_text: str) -> dict[str, Any]:
    """Full path: parse the log, resolve the slug's facts, classify.  The
    returned dict IS the ``--json`` contract."""
    facts = slug_facts(slug)
    if facts.get("error"):
        return {
            "verdict": VERDICT_UNSUPPORTED,
            "suspected_cause": facts["error"],
            "error": facts["error"],
            "slug": slug,
        }
    parsed = parse_bootlog(log_text)
    out = back_solve(
        parsed,
        predicted_tp1=facts["predicted_per_token_bytes_tp1"],
        registry_cfg=facts["registry_cfg"],
    )
    out.update({"slug": facts["slug"], "model": facts["model_id"], "kv_format": facts["kv_format"]})
    return out


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="bootlog_solve.py",
        description=(
            "Back-solve measured per-token KV bytes from a vLLM boot log and "
            "classify vs the kv-calc prediction (ADDING_MODELS.md Step 5)."
        ),
    )
    p.add_argument("--slug", required=True, metavar="ENGINE/VARIANT",
                   help="registry slug (e.g. vllm/gemma-dual) for the prediction side")
    p.add_argument("--log-file", default="-",
                   help="boot-log file to parse ('-' = stdin, default)")
    p.add_argument("--json", action="store_true", help="emit the verdict as JSON (always on; reserved)")
    args = p.parse_args(argv)

    if args.log_file == "-":
        log_text = sys.stdin.read()
    else:
        try:
            log_text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(json.dumps({
                "verdict": VERDICT_INSUFFICIENT,
                "suspected_cause": f"log unavailable: {exc}",
                "error": str(exc),
                "slug": args.slug,
            }))
            return 1
    print(json.dumps(solve_log(args.slug, log_text), indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
