"""Measurement-record producer — turn a completed bench run into ONE structured
record of measured runtime facts.

WHAT THIS IS (and is NOT)
=========================
This is a *pure producer*. It reads the **output** of ``scripts/bench.sh`` (its
stdout text, plus the ``nvidia-smi`` GPU line it prints) together with a config
identity (a ``compose_registry`` tag) and emits a single measurement-record
JSON. It writes that record to a per-rig, gitignored corpus directory.

It is deliberately decoupled from live execution:

  * it does NOT require a GPU, a running model, or a live bench;
  * it parses *saved* bench output, so a maintainer can retro-capture a run
    that already happened;
  * it makes NO decisions — there is no lookup, no consumer, no timeout-sizing,
    no optimizer accretion here. Those are gated behind a separate design
    unlock (the v0.9.0 "compose optimizer" track). Adding any consumer logic to
    this file would cross that boundary.

SCHEMA
======
The frozen field names come from the optimizer design's measurement-record
schema (the "stable measurement-record schema defined now" / "frozen here so
tests/docs/M2 have a fixed target" paragraphs). They are used VERBATIM:

    model_slug, arch, arch_class, engine_id, engine_pin, hardware, topology,
    kv_dtype, max_model_len, max_num_seqs, mem_util, objective,
    confidence_tier, margin_applied, result_class, smoke_status, soak_status,
    kv_calc_version, provenance

    provenance = {source, n_obs, cohort, last_confirmed, kv_calc_version}

Fields a *bench run cannot legitimately know* (``objective``,
``confidence_tier``, ``margin_applied`` — these are optimizer/deriver concerns)
are emitted as ``None``/sentinel rather than fabricated. ``smoke_status`` and
``soak_status`` default to ``"not-run"`` for a pure bench unless that data is
supplied.

PRODUCER-PROPOSED EXTENSIONS (not in the frozen contract)
=========================================================
Two additions the current frozen schema lacks live under the clearly-named
``measured_extensions`` nested key, so a future reader knows they are
producer-proposed, NOT yet in the frozen contract. They are candidates for the
optimizer's Lock-criteria #6 schema-typing pass:

  * ``decode_tps_by_ctx`` — TPS as a *context-depth ladder* (a dict keyed by
    context-token depth), not a scalar. ``bench.sh`` today measures only the
    short-context canonical prompts, so the ladder will have ONE point for now;
    structuring it as a ladder lets depth points be added later without a
    schema change. Alongside it: ``prefill_tps``, ``ttft_s``, ``wall_tps``,
    and per-card peak VRAM.
  * ``power_cap_w`` — the GPU power cap, captured as part of the conditions
    fingerprint. The optimizer's cohort key does NOT include power cap, but a
    230W-vs-370W throughput artifact has burned this stack before, so the cap
    is captured to keep a record's TPS interpretable.

See ``compose-optimizer-design.md`` Lock-criteria #6 (typed measurement-record
schema) before promoting either extension into the frozen contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # package-relative when imported as scripts.lib.profiles.measurement_record
    from .compose_registry import COMPOSE_REGISTRY
except ImportError:  # pragma: no cover - direct-script fallback
    from compose_registry import COMPOSE_REGISTRY  # type: ignore


# Repo root = three parents up from this file (scripts/lib/profiles/ -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Default corpus directory, repo-relative. ``results/`` is gitignored
# (``results/*`` with explicit ``!`` negations for committed seed subsets), so
# records written here are per-rig local data and are NOT committed by default.
# A TPS from a 3090@370W is wrong for a 4090, so cross-rig records must never be
# committed blindly; a curated committed seed subset (if ever wanted) is a
# separate, later, deliberate act — not this producer's job.
CORPUS_SUBDIR = "results/measurement-records"

# Sentinel for fields a bench cannot know but the schema requires present.
NOT_RUN = "not-run"


class MeasuredRecordError(ValueError):
    """A measured record is missing data it MUST carry to be trustworthy.

    Raised (fail-loud) when ``result_class`` says a record is *measured* but the
    parse produced no decode TPS (or no parseable bench summary block at all).
    A measured record with null TPS is worse than no record — it looks like real
    calibration data while carrying none — so the producer refuses to write it.
    This mirrors the module's existing fail-loud posture for an unknown registry
    tag (``KeyError``). Soft gaps (e.g. a malformed GPU-state line) are surfaced
    as ``parse_warnings`` instead, not raised.
    """


def _is_measured_result_class(result_class: Optional[str]) -> bool:
    """True if ``result_class`` denotes a MEASURED record.

    The frozen 3-state runtime contract uses ``boot-fit predicted|measured`` and
    the producer's default is ``boot-fit-measured``. We treat any class whose
    final ``-``-delimited token is ``measured`` (e.g. ``boot-fit-measured``) as
    measured. ``predicted`` / ``unknown`` / ``*-derived`` classes are NOT
    measured and impose no decode-TPS requirement.
    """
    if not result_class:
        return False
    return result_class.strip().lower().split("-")[-1] == "measured"


# --------------------------------------------------------------------------- #
# arch / arch_class derivation
# --------------------------------------------------------------------------- #
# ``arch`` is the model's family slug (the verbatim ``family:`` value from
# scripts/lib/profiles/models/<id>.yml, consistent with how the [F] loop's
# loop_input uses ``arch_family`` VERBATIM).
#
# ``arch_class`` is the optimizer's attention-family granularity
# (MHA / GQA / MLA / deltanet-hybrid / sliding-window) that drives KV legality
# and cliff-class behavior (design Lock-criteria #1). A bench does not
# authoritatively measure attention family, so we map it only where it is
# unambiguous from the known family slug, and leave it ``None`` otherwise
# (fail-soft, never guess). Callers may override.
_FAMILY_TO_ARCH_CLASS = {
    "qwen3-next-hybrid": "deltanet-hybrid",
    "qwen3-next-moe": "deltanet-hybrid",
    "gemma4-swa-dense": "sliding-window",
    "gemma4-swa-moe": "sliding-window",
}


def _read_model_family(model_slug: str) -> Optional[str]:
    """Best-effort read of ``family:`` from models/<model_slug>.yml.

    Uses a tiny line scan rather than importing a YAML lib or the full profile
    loader — this keeps the producer dependency-light and GPU/IO-free. Returns
    ``None`` if the file or key is absent (honest degrade, never crash).
    """
    path = _REPO_ROOT / "scripts" / "lib" / "profiles" / "models" / f"{model_slug}.yml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^family:\s*(\S+)", text, re.MULTILINE)
    return m.group(1) if m else None


def _topology_from_tp(tp: int) -> str:
    """Map a tensor-parallel degree to the repo's topology slug.

    single (TP=1) / dual (TP=2) have no count ambiguity; TP>2 -> ``multi<N>``
    (matches the compose layout's topology rule).
    """
    if tp == 1:
        return "single"
    if tp == 2:
        return "dual"
    return f"multi{tp}"


# --------------------------------------------------------------------------- #
# bench.sh stdout parsing
# --------------------------------------------------------------------------- #
@dataclass
class BenchMetrics:
    """Parsed numbers from one ``scripts/bench.sh`` stdout capture.

    All fields optional — a partial bench (e.g. ``ONLY=code``, or no GPU line)
    still yields a usable :class:`BenchMetrics`. The producer never fabricates
    absent numbers. Whether an absent number is *tolerable* is decided later in
    :func:`build_record`: for a MEASURED ``result_class`` a missing decode TPS
    is fatal (fail-loud), while a missing GPU line is a soft warning. The parse
    provenance fields below let ``build_record`` tell those cases apart.
    """

    decode_tps: Optional[float] = None      # mean decode_TPS (model decode rate)
    wall_tps: Optional[float] = None        # mean wall_TPS (user-perceived)
    ttft_s: Optional[float] = None          # mean TTFT, seconds
    prefill_tps: Optional[float] = None     # mean PP tok/s (prompt-processing)
    # Per-card peak VRAM in MiB, indexed by GPU index, from the nvidia-smi line.
    vram_used_mib: dict[int, int] = field(default_factory=dict)
    # Power cap (limit) in watts, from nvidia-smi power.limit if present.
    power_cap_w: Optional[float] = None
    # Power draw in watts, per GPU index, if present.
    power_draw_w: dict[int, float] = field(default_factory=dict)

    # --- Parse provenance (drift / hollow-record detection) ----------------- #
    # How many ``=== summary [...] ===`` blocks carrying a ``decode_TPS mean=``
    # line we matched. Zero on a measured record means bench-output drift (the
    # summary section the producer keys off is absent or unparseable) -> the
    # record must NOT be written with null TPS. See ``build_record``.
    decode_summary_blocks: int = 0
    # ``=== GPU state ===`` marker seen at all (regardless of parseability).
    gpu_state_section_present: bool = False
    # A ``=== GPU state ===`` section was present but yielded no usable
    # per-GPU VRAM row -> malformed GPU-state line (soft gap -> warn, not raise).
    gpu_state_unparsed: bool = False


def _f(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_bench_output(text: str) -> BenchMetrics:
    """Parse ``scripts/bench.sh`` stdout into a :class:`BenchMetrics`.

    Robust to:
      * narrative-only / code-only / both runs (takes the last summary block's
        ``decode_TPS``/``wall_TPS``/``TTFT`` means — the canonical short-context
        point);
      * llama.cpp fallback PP block vs vLLM log-scraped ``PP tok/s``;
      * presence/absence of the ``=== GPU state ===`` nvidia-smi line.
    """
    m = BenchMetrics()

    # Summary lines look like:
    #   wall_TPS       mean= 238.10   std=  0.00 ...
    #   decode_TPS     mean= 245.10   std=  0.00 ...
    #   PP tok/s       mean=2843.21   std=  0.00 ...
    # Take the LAST occurrence of each (the final summary block).
    for label, attr in (
        (r"decode_TPS", "decode_tps"),
        (r"wall_TPS", "wall_tps"),
        (r"PP tok/s", "prefill_tps"),
    ):
        matches = re.findall(rf"^\s*{label}\s+mean=\s*([0-9.]+)", text, re.MULTILINE)
        if matches:
            setattr(m, attr, _f(matches[-1]))
        if label == r"decode_TPS":
            # ``decode_TPS mean=`` lines appear ONLY inside a summary block, so
            # the count is the count of parseable decode-summary blocks. Zero on
            # a measured record => the summary section the producer keys off is
            # absent / unparseable (bench-output drift). build_record fails loud.
            m.decode_summary_blocks = len(matches)

    # TTFT summary line:  TTFT          mean=   120ms  std= ...
    ttft = re.findall(r"^\s*TTFT\s+mean=\s*([0-9.]+)ms", text, re.MULTILINE)
    if ttft:
        val = _f(ttft[-1])
        m.ttft_s = (val / 1000.0) if val is not None else None

    # nvidia-smi line (CSV, noheader):
    #   index, utilization.gpu, memory.used, memory.total, power.draw,
    #   temperature.gpu  — and power.limit only if the caller queried it.
    # bench.sh queries: index,utilization.gpu,memory.used,memory.total,
    #                   power.draw,temperature.gpu
    # We parse defensively: locate the "memory.used" MiB token and a power
    # "W" token positionally per row.
    gpu_block = text.split("=== GPU state ===", 1)
    if len(gpu_block) == 2:
        m.gpu_state_section_present = True
        for row in gpu_block[1].splitlines():
            row = row.strip()
            if not row or "," not in row:
                continue
            cells = [c.strip() for c in row.split(",")]
            # First cell must be a GPU index (int).
            try:
                idx = int(cells[0])
            except (ValueError, IndexError):
                continue
            for cell in cells[1:]:
                mib = re.match(r"^([0-9]+)\s*MiB$", cell)
                if mib and idx not in m.vram_used_mib:
                    # First MiB cell in CSV order is memory.used.
                    m.vram_used_mib[idx] = int(mib.group(1))
                    continue
                watt = re.match(r"^([0-9.]+)\s*W$", cell)
                if watt and idx not in m.power_draw_w:
                    m.power_draw_w[idx] = float(watt.group(1))
        # Section present but no usable per-GPU VRAM row => malformed GPU-state
        # line. Soft gap (VRAM is a fingerprint extension, not the core
        # measured TPS): warn in build_record, do NOT raise.
        if not m.vram_used_mib:
            m.gpu_state_unparsed = True

    # Optional explicit power.limit line, if a caller appended one, e.g.:
    #   power.limit: 370.00 W   (per-rig fingerprint)
    plim = re.search(r"power\.limit[^0-9]*([0-9.]+)\s*W", text)
    if plim:
        m.power_cap_w = _f(plim.group(1))

    return m


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #
def build_record(
    *,
    tag: str,
    bench_metrics: BenchMetrics,
    # Conditions fingerprint inputs the registry cannot know. These are
    # provenance the caller supplies (or leaves None); the producer never
    # invents them.
    hardware: Optional[str] = None,
    engine_pin: Optional[str] = None,
    power_cap_w: Optional[float] = None,
    # Bench outcome.
    result_class: str = "boot-fit-measured",
    # Pure-bench defaults; override if smoke/soak data is actually present.
    smoke_status: str = NOT_RUN,
    soak_status: str = NOT_RUN,
    kv_calc_version: Optional[str] = None,
    # Optimizer-only fields a bench cannot know — sentinel, never fabricated.
    objective: Optional[str] = None,
    confidence_tier: Optional[str] = None,
    margin_applied: Optional[float] = None,
    # arch_class override (attention-family granularity); else best-effort map.
    arch_class: Optional[str] = None,
    n_obs: int = 1,
    last_confirmed: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble ONE measurement record from a registry tag + parsed bench output.

    Fail-loud, never write hollow calibration data:

      * ``KeyError`` for an unknown tag (a typo'd tag should not silently
        produce a garbage record);
      * :class:`MeasuredRecordError` when ``result_class`` says the record is
        *measured* (e.g. ``boot-fit-measured``) but the parse produced **no
        decode TPS** — either because the bench summary block is absent /
        unparseable (output drift) or no ``decode_TPS mean=`` was found. A
        measured record with null TPS looks like real data to the optimizer's
        calibration backbone but carries none, so the producer refuses it.

    Softer gaps that do NOT invalidate the measured TPS (e.g. a malformed /
    absent ``=== GPU state ===`` line, so per-GPU VRAM is unknown) are surfaced
    as a top-level ``parse_warnings`` list on the record rather than raised, so
    the record is still written but the gap is explicit (never a silent null).
    Genuinely-optional/unknowable optimizer fields (``objective``,
    ``confidence_tier``, ...) stay ``None`` — that is not a gap.
    """
    entry = COMPOSE_REGISTRY[tag]
    parse_warnings: list[str] = []

    model_slug = entry["model"]
    arch = _read_model_family(model_slug)  # family slug, used verbatim
    if arch_class is None and arch is not None:
        arch_class = _FAMILY_TO_ARCH_CLASS.get(arch)  # None if unknown — no guess

    topology = _topology_from_tp(int(entry["tp"]))

    # power_cap_w precedence: explicit arg > parsed from bench output > None.
    eff_power_cap = power_cap_w if power_cap_w is not None else bench_metrics.power_cap_w

    # Conditions fingerprint = (hardware, engine_pin, kv_dtype, topology,
    # power_cap_w). power_cap_w is a producer extension (see module docstring):
    # the optimizer cohort key omits it, but it is load-bearing for TPS
    # interpretability on this stack.
    fingerprint = {
        "hardware": hardware,
        "engine_pin": engine_pin,
        "kv_dtype": entry["kv_format"],
        "topology": topology,
        "power_cap_w": eff_power_cap,
    }

    # provenance — frozen sub-schema {source, n_obs, cohort, last_confirmed,
    # kv_calc_version}. A direct bench run is `source: measured`. The cohort key
    # mirrors the optimizer's (hardware, engine_pin, arch, objective); objective
    # is None for a pure bench (it is an optimizer input, not a bench fact).
    provenance = {
        "source": "measured",
        "n_obs": n_obs,
        "cohort": {
            "hardware": hardware,
            "engine_pin": engine_pin,
            "arch": arch,
            "objective": objective,
        },
        "last_confirmed": last_confirmed,
        "kv_calc_version": kv_calc_version,
    }

    # --- Producer-proposed extensions (NOT in the frozen contract) ---------- #
    # Candidates for the optimizer's Lock-criteria #6 schema-typing pass.
    # decode_tps_by_ctx is a LADDER keyed by context-token depth. bench.sh today
    # only measures the short canonical prompts, so there is ONE point now,
    # keyed by the config's max_model_len-context... no: keyed by the actual
    # measured prompt depth. We do not know the exact canonical prompt token
    # depth from the summary alone, so we key the single point by the
    # short-context canonical bucket label "canonical-short". Future depth
    # points (e.g. "30000", "120000") slot in without a schema change.
    decode_tps_by_ctx: dict[str, Optional[float]] = {}
    if bench_metrics.decode_tps is not None:
        decode_tps_by_ctx["canonical-short"] = bench_metrics.decode_tps

    # --- Fail-loud validation for MEASURED records -------------------------- #
    # The producer's whole value is the measured decode TPS. For a measured
    # result_class, refuse to emit a record with null TPS — distinguish the two
    # ways it goes null so the error names the actual cause:
    #   1. No parseable bench summary block at all => bench-output drift.
    #   2. A summary block parsed but no decode number => incomplete metrics.
    if _is_measured_result_class(result_class):
        if bench_metrics.decode_summary_blocks == 0:
            raise MeasuredRecordError(
                f"result_class={result_class!r} is a MEASURED record but no "
                "parseable bench summary block was found (expected a "
                "'=== summary [...] ===' section with a 'decode_TPS mean=' "
                "line). This is scripts/bench.sh output drift -- refusing to "
                "write a hollow record with null TPS. Check the bench capture "
                "format, or pass a non-measured --result-class if this run "
                "genuinely produced no decode metrics."
            )
        if not decode_tps_by_ctx:
            raise MeasuredRecordError(
                f"result_class={result_class!r} is a MEASURED record but the "
                "parse produced no decode TPS (empty decode_tps_by_ctx). A "
                "measured record with null TPS is worse than no record for "
                "optimizer calibration -- refusing to write it."
            )

    # Soft gap: GPU-state line absent/malformed -> per-GPU VRAM unknown. Warn,
    # do NOT raise (VRAM is a fingerprint extension, not the core measured TPS).
    if bench_metrics.gpu_state_unparsed:
        parse_warnings.append(
            "GPU-state section present but unparseable: no per-GPU VRAM row "
            "matched (expected 'index, ... , <N> MiB, ...' from nvidia-smi); "
            "peak_vram_mib_by_gpu is empty."
        )
    elif not bench_metrics.gpu_state_section_present:
        parse_warnings.append(
            "no '=== GPU state ===' section in bench output; per-GPU VRAM and "
            "power-draw are unknown (peak_vram_mib_by_gpu is empty)."
        )

    measured_extensions = {
        # Flagged producer-proposed; see module docstring + design Lock #6.
        "_note": (
            "producer-proposed; NOT in frozen measurement-record schema. "
            "Candidates for optimizer Lock-criteria #6 schema-typing pass."
        ),
        "decode_tps_by_ctx": decode_tps_by_ctx,
        "prefill_tps": bench_metrics.prefill_tps,
        "ttft_s": bench_metrics.ttft_s,
        "wall_tps": bench_metrics.wall_tps,
        "peak_vram_mib_by_gpu": {str(k): v for k, v in sorted(bench_metrics.vram_used_mib.items())},
        "power_draw_w_by_gpu": {str(k): v for k, v in sorted(bench_metrics.power_draw_w.items())},
        # power_cap_w is part of the conditions fingerprint AND an extension.
        "power_cap_w": eff_power_cap,
        "conditions_fingerprint": fingerprint,
    }

    record = {
        # --- Frozen schema fields (verbatim names) --------------------------- #
        "model_slug": model_slug,
        "arch": arch,
        "arch_class": arch_class,
        "engine_id": entry["engine"],
        "engine_pin": engine_pin,
        "hardware": hardware,
        "topology": topology,
        "kv_dtype": entry["kv_format"],
        "max_model_len": entry["max_ctx"],
        "max_num_seqs": entry["max_num_seqs"],
        "mem_util": entry["mem_util"],
        "objective": objective,            # optimizer concern -> None for bench
        "confidence_tier": confidence_tier,  # deriver concern -> None for bench
        "margin_applied": margin_applied,    # optimizer concern -> None for bench
        "result_class": result_class,
        "smoke_status": smoke_status,
        "soak_status": soak_status,
        "kv_calc_version": kv_calc_version,
        "provenance": provenance,
        # --- Producer extensions (clearly namespaced) ------------------------ #
        "measured_extensions": measured_extensions,
        # Soft-gap diagnostics (never a silent null): a list naming each
        # expected-but-absent/malformed datum that did NOT rise to a fail-loud
        # raise. Empty list on a clean, well-formed bench output.
        "parse_warnings": parse_warnings,
        # Carry the registry tag for traceability (not a frozen field; aids
        # the maintainer joining a record back to its config identity).
        "_tag": tag,
    }
    return record


# --------------------------------------------------------------------------- #
# Corpus location + filename
# --------------------------------------------------------------------------- #
def short_fingerprint(record: dict[str, Any]) -> str:
    """Stable 8-hex short hash of the conditions fingerprint.

    Keyed by (hardware, engine_pin, kv_dtype, topology, power_cap_w) so records
    from the same config under the same conditions append to one file, and a
    condition change (e.g. 230W -> 370W) lands in a different file. ``None``
    fields are rendered as the literal string "null" so a partially-known
    fingerprint is still deterministic.
    """
    import hashlib

    fp = record["measured_extensions"]["conditions_fingerprint"]
    key = "|".join(
        f"{k}={fp.get(k) if fp.get(k) is not None else 'null'}"
        for k in ("hardware", "engine_pin", "kv_dtype", "topology", "power_cap_w")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def corpus_path_for(record: dict[str, Any], *, corpus_dir: Optional[Path] = None) -> Path:
    """Compute the JSONL corpus file path for a record.

    ``<corpus_dir>/<tag-slug>__<short-fingerprint>.jsonl`` (appended to). The
    tag's ``/`` is slugified to ``-`` so it is a single path component.
    """
    base = corpus_dir if corpus_dir is not None else (_REPO_ROOT / CORPUS_SUBDIR)
    tag_slug = record["_tag"].replace("/", "-")
    return base / f"{tag_slug}__{short_fingerprint(record)}.jsonl"


def write_record(record: dict[str, Any], *, corpus_dir: Optional[Path] = None) -> Path:
    """Append one record as a JSON line to its corpus file. Returns the path.

    Creates the corpus directory if needed. The corpus lives under a gitignored
    ``results/`` subtree — per-rig local data, never committed by default.
    """
    path = corpus_path_for(record, corpus_dir=corpus_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=False) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI — standalone, runs on SAVED bench output (no GPU, no live model)
# --------------------------------------------------------------------------- #
def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="measurement_record",
        description=(
            "Emit ONE measurement-record JSON from a saved scripts/bench.sh "
            "stdout capture + a compose_registry tag. Pure producer: reads "
            "bench OUTPUT, writes a per-rig gitignored record. No GPU required."
        ),
    )
    p.add_argument("--tag", required=True, help="compose_registry tag, e.g. ik-llama/iq4ks-mtp")
    p.add_argument(
        "--bench-output",
        type=Path,
        default=None,
        help="Path to a saved bench.sh stdout capture. Default: read stdin.",
    )
    p.add_argument("--hardware", default=None, help="Hardware id, e.g. rtx-3090 (fingerprint).")
    p.add_argument("--engine-pin", default=None, help="Engine pin/SHA (fingerprint + cohort).")
    p.add_argument(
        "--power-cap-w",
        type=float,
        default=None,
        help="GPU power cap in watts (fingerprint). Overrides any parsed value.",
    )
    p.add_argument(
        "--result-class",
        default="boot-fit-measured",
        help="Bench outcome class. Default: boot-fit-measured.",
    )
    p.add_argument("--smoke-status", default=NOT_RUN, help='Default: "not-run".')
    p.add_argument("--soak-status", default=NOT_RUN, help='Default: "not-run".')
    p.add_argument("--kv-calc-version", default=None, help="kv_calc_version content hash, if known.")
    p.add_argument("--arch-class", default=None, help="Override attention-family arch_class.")
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Override corpus directory (default: <repo>/results/measurement-records).",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print the record JSON to stdout and do NOT write the corpus file.",
    )
    return p


def main(argv=None) -> int:
    import sys

    args = _build_arg_parser().parse_args(argv)

    if args.bench_output is not None:
        text = args.bench_output.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    metrics = parse_bench_output(text)
    try:
        record = build_record(
            tag=args.tag,
            bench_metrics=metrics,
            hardware=args.hardware,
            engine_pin=args.engine_pin,
            power_cap_w=args.power_cap_w,
            result_class=args.result_class,
            smoke_status=args.smoke_status,
            soak_status=args.soak_status,
            kv_calc_version=args.kv_calc_version,
            arch_class=args.arch_class,
        )
    except MeasuredRecordError as exc:
        # Fail loud, but with a clean operator message (not a raw traceback).
        print(f"[measurement_record] ERROR: {exc}", file=sys.stderr)
        return 2

    # Surface soft-gap warnings so a partial record is never silently accepted.
    for w in record.get("parse_warnings", []):
        print(f"[measurement_record] WARNING: {w}", file=sys.stderr)

    if args.print_only:
        print(json.dumps(record, indent=2, sort_keys=False))
        return 0

    path = write_record(record, corpus_dir=args.corpus_dir)
    print(json.dumps(record, indent=2, sort_keys=False))
    print(f"\n[measurement_record] appended -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
