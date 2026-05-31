#!/usr/bin/env bash
set -euo pipefail

# test-measurement-record.sh — producer-only measurement-record emitter.
#
# Validates scripts/lib/profiles/measurement_record.py WITHOUT a GPU or a
# running model: it feeds a captured sample of `scripts/bench.sh` stdout (we
# reuse bench.sh's own BENCH_MOCK=1 canonical output so the sample never
# drifts from the real format) into the emitter and asserts the emitted JSON.
#
# Asserts:
#   (a) every FROZEN schema field is present with the right value/source;
#   (b) optimizer-only fields (objective/confidence_tier/margin_applied) are
#       null — never fabricated by a bench;
#   (c) smoke_status/soak_status default to "not-run" for a pure bench;
#   (d) provenance carries {source,n_obs,cohort,last_confirmed,kv_calc_version};
#   (e) the producer-proposed `measured_extensions` ladder + power_cap_w + VRAM
#       parse correctly and are clearly namespaced;
#   (f) topology derives from tp; arch/arch_class from the model family;
#   (g) corpus path is under the gitignored results/ subtree and filename
#       carries the conditions-fingerprint short hash; write+append works;
#   (h) an unknown tag fails loud (KeyError), never a silent garbage record.
#   (i) a MEASURED record built from bench output MISSING the decode summary
#       fails loud (MeasuredRecordError), never a hollow null-TPS record;
#   (j) a MEASURED record with a malformed/absent GPU-state line surfaces a
#       parse_warnings entry (not a silent null), without raising;
#   (k) regression: the complete well-formed sample yields EMPTY parse_warnings.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Capture REAL bench.sh canonical output via its built-in mock (no GPU/model).
# We append a synthetic GPU-state block + power.limit line because BENCH_MOCK
# short-circuits before the nvidia-smi section; the format mirrors what
# bench.sh emits from `nvidia-smi --query-gpu=...,memory.used,...,power.draw`.
BENCH_SAMPLE="$(BENCH_MOCK=1 bash scripts/bench.sh 2>/dev/null)"
BENCH_SAMPLE+="
=== GPU state ===
0, 99 %, 22310 MiB, 24576 MiB, 351.20 W, 71
power.limit: 370.00 W
"

export BENCH_SAMPLE

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1])

from scripts.lib.profiles import measurement_record as mr

sample = os.environ["BENCH_SAMPLE"]

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# --- parse ----------------------------------------------------------------- #
metrics = mr.parse_bench_output(sample)
check(metrics.decode_tps == 245.10, f"(e) decode_tps parse: {metrics.decode_tps}")
check(metrics.wall_tps == 238.10, f"(e) wall_tps parse: {metrics.wall_tps}")
check(abs((metrics.ttft_s or 0) - 0.120) < 1e-9, f"(e) ttft_s parse: {metrics.ttft_s}")
check(metrics.vram_used_mib.get(0) == 22310, f"(g/e) vram parse: {metrics.vram_used_mib}")
check(abs((metrics.power_draw_w.get(0) or 0) - 351.20) < 1e-9, f"(e) power_draw parse: {metrics.power_draw_w}")
check(metrics.power_cap_w == 370.00, f"(e) power.limit parse: {metrics.power_cap_w}")

# --- build (realistic ik-llama iq4ks Qwen3.6-27B single-card @370W) -------- #
rec = mr.build_record(
    tag="ik-llama/iq4ks-mtp",
    bench_metrics=metrics,
    hardware="rtx-3090",
    engine_pin="ik-llama-cpp@b1234",
    # power_cap_w left None on purpose -> must fall back to the parsed 370W.
    result_class="boot-fit-measured",
)

# (a) frozen schema fields present + correct value/source
FROZEN = [
    "model_slug", "arch", "arch_class", "engine_id", "engine_pin", "hardware",
    "topology", "kv_dtype", "max_model_len", "max_num_seqs", "mem_util",
    "objective", "confidence_tier", "margin_applied", "result_class",
    "smoke_status", "soak_status", "kv_calc_version", "provenance",
]
for fld in FROZEN:
    check(fld in rec, f"(a) frozen field missing: {fld}")

check(rec["model_slug"] == "qwen3.6-27b", f"(a) model_slug: {rec['model_slug']}")
check(rec["engine_id"] == "llama-cpp-local", f"(a) engine_id: {rec['engine_id']}")
check(rec["kv_dtype"] == "q4_0", f"(a) kv_dtype: {rec['kv_dtype']}")
check(rec["max_model_len"] == 200000, f"(a) max_model_len: {rec['max_model_len']}")
check(rec["max_num_seqs"] == 1, f"(a) max_num_seqs: {rec['max_num_seqs']}")
check(rec["hardware"] == "rtx-3090", f"(a) hardware: {rec['hardware']}")
check(rec["engine_pin"] == "ik-llama-cpp@b1234", f"(a) engine_pin: {rec['engine_pin']}")
check(rec["result_class"] == "boot-fit-measured", f"(a) result_class: {rec['result_class']}")

# (f) topology from tp=1 -> single ; arch/arch_class from family
check(rec["topology"] == "single", f"(f) topology: {rec['topology']}")
check(rec["arch"] == "qwen3-next-hybrid", f"(f) arch (family verbatim): {rec['arch']}")
check(rec["arch_class"] == "deltanet-hybrid", f"(f) arch_class mapped: {rec['arch_class']}")

# (b) optimizer-only fields must be null, never fabricated
check(rec["objective"] is None, f"(b) objective not null: {rec['objective']}")
check(rec["confidence_tier"] is None, f"(b) confidence_tier not null: {rec['confidence_tier']}")
check(rec["margin_applied"] is None, f"(b) margin_applied not null: {rec['margin_applied']}")

# (c) smoke/soak default to not-run for a pure bench
check(rec["smoke_status"] == "not-run", f"(c) smoke_status: {rec['smoke_status']}")
check(rec["soak_status"] == "not-run", f"(c) soak_status: {rec['soak_status']}")

# (d) provenance sub-schema {source,n_obs,cohort,last_confirmed,kv_calc_version}
prov = rec["provenance"]
for k in ("source", "n_obs", "cohort", "last_confirmed", "kv_calc_version"):
    check(k in prov, f"(d) provenance missing: {k}")
check(prov["source"] == "measured", f"(d) provenance.source: {prov['source']}")
check(prov["n_obs"] == 1, f"(d) provenance.n_obs: {prov['n_obs']}")
check(prov["cohort"]["hardware"] == "rtx-3090", "(d) cohort.hardware")
check(prov["cohort"]["arch"] == "qwen3-next-hybrid", "(d) cohort.arch")
check(prov["cohort"]["objective"] is None, "(d) cohort.objective null")

# (e) producer extensions clearly namespaced + populated
ext = rec["measured_extensions"]
check("_note" in ext and "NOT in frozen" in ext["_note"], "(e) extensions _note flag missing")
check(ext["decode_tps_by_ctx"] == {"canonical-short": 245.10}, f"(e) ladder: {ext['decode_tps_by_ctx']}")
check(ext["wall_tps"] == 238.10, f"(e) ext wall_tps: {ext['wall_tps']}")
check(ext["prefill_tps"] is None or isinstance(ext["prefill_tps"], float), "(e) prefill_tps type")
check(ext["peak_vram_mib_by_gpu"] == {"0": 22310}, f"(e) peak vram: {ext['peak_vram_mib_by_gpu']}")
# power_cap_w fell back to the parsed 370W (no explicit override)
check(ext["power_cap_w"] == 370.00, f"(e) ext power_cap_w fallback: {ext['power_cap_w']}")
fp = ext["conditions_fingerprint"]
check(fp["power_cap_w"] == 370.00, f"(e) fingerprint power_cap_w: {fp['power_cap_w']}")
check(fp["kv_dtype"] == "q4_0" and fp["topology"] == "single", "(e) fingerprint kv/topology")

# (k) regression: complete, well-formed sample => clean record, NO warnings
check("parse_warnings" in rec, "(k) parse_warnings field missing")
check(rec["parse_warnings"] == [], f"(k) happy path has warnings: {rec['parse_warnings']}")

# explicit power_cap_w arg must win over parsed value
rec2 = mr.build_record(tag="ik-llama/iq4ks-mtp", bench_metrics=metrics,
                       hardware="rtx-3090", power_cap_w=230.0)
check(rec2["measured_extensions"]["power_cap_w"] == 230.0,
      f"(e) explicit power_cap_w override: {rec2['measured_extensions']['power_cap_w']}")

# (g) corpus path under gitignored results/ + fingerprint hash in name + write
with tempfile.TemporaryDirectory() as td:
    corpus = Path(td) / "mr"
    path = mr.write_record(rec, corpus_dir=corpus)
    check(path.parent == corpus, f"(g) corpus dir: {path.parent}")
    check(path.name.startswith("ik-llama-iq4ks-mtp__"), f"(g) filename slug: {path.name}")
    check(path.name.endswith(".jsonl"), f"(g) jsonl ext: {path.name}")
    # append a second record -> two lines, same file (same fingerprint)
    mr.write_record(rec, corpus_dir=corpus)
    lines = path.read_text().strip().splitlines()
    check(len(lines) == 2, f"(g) append: {len(lines)} lines")
    json.loads(lines[0])  # valid JSON line

# default corpus location is under results/ (gitignored subtree)
default_path = mr.corpus_path_for(rec)
check("results/measurement-records" in default_path.as_posix(),
      f"(g) default corpus under results/: {default_path}")

# (h) unknown tag fails loud
try:
    mr.build_record(tag="vllm/does-not-exist", bench_metrics=metrics)
    failures.append("(h) unknown tag did NOT raise")
except KeyError:
    pass

# (i) MEASURED record + bench output missing the decode summary => fail loud.
# Strip the summary block from the sample (keep the GPU-state line) so only the
# decode metric is gone; the parser must report zero decode-summary blocks and
# build_record must refuse rather than write null TPS.
no_summary = "\n".join(
    ln for ln in sample.splitlines()
    if "decode_TPS" not in ln and "wall_TPS" not in ln and "=== summary" not in ln
)
m_nosum = mr.parse_bench_output(no_summary)
check(m_nosum.decode_tps is None, f"(i) precondition decode_tps gone: {m_nosum.decode_tps}")
check(m_nosum.decode_summary_blocks == 0, f"(i) precondition summary blocks: {m_nosum.decode_summary_blocks}")
try:
    mr.build_record(tag="ik-llama/iq4ks-mtp", bench_metrics=m_nosum,
                    hardware="rtx-3090", result_class="boot-fit-measured")
    failures.append("(i) measured record w/ no decode summary did NOT raise")
except mr.MeasuredRecordError:
    pass

# A NON-measured result_class with the same missing-decode input must NOT raise
# (the requirement is scoped to measured records only).
try:
    rec_pred = mr.build_record(tag="ik-llama/iq4ks-mtp", bench_metrics=m_nosum,
                               hardware="rtx-3090", result_class="boot-fit-predicted")
    check(rec_pred["measured_extensions"]["decode_tps_by_ctx"] == {},
          "(i) predicted record ladder should be empty, not fabricated")
except mr.MeasuredRecordError:
    failures.append("(i) non-measured result_class wrongly raised on missing decode")

# (j) MEASURED record + malformed GPU-state line => parse_warnings, NOT raise.
# Keep the decode summary; corrupt the GPU-state row so no VRAM MiB matches.
bad_gpu = """\
=== summary [narrative] (n=1) ===
  wall_TPS       mean= 238.10   std=  0.00
  decode_TPS     mean= 245.10   std=  0.00
  TTFT          mean=   120ms  std=    0ms
=== GPU state ===
<nvidia-smi unavailable: query failed>
"""
m_badgpu = mr.parse_bench_output(bad_gpu)
check(m_badgpu.decode_tps == 245.10, f"(j) precondition decode parsed: {m_badgpu.decode_tps}")
check(m_badgpu.gpu_state_section_present, "(j) precondition GPU section present")
check(m_badgpu.gpu_state_unparsed, "(j) precondition GPU section flagged unparsed")
check(m_badgpu.vram_used_mib == {}, f"(j) precondition no VRAM rows: {m_badgpu.vram_used_mib}")
rec_badgpu = mr.build_record(tag="ik-llama/iq4ks-mtp", bench_metrics=m_badgpu,
                             hardware="rtx-3090", result_class="boot-fit-measured")
check(len(rec_badgpu["parse_warnings"]) >= 1, "(j) no parse_warnings surfaced for bad GPU line")
check(any("GPU-state" in w for w in rec_badgpu["parse_warnings"]),
      f"(j) GPU-state warning not named: {rec_badgpu['parse_warnings']}")
# record is still WRITTEN (soft gap), and the measured TPS is intact
check(rec_badgpu["measured_extensions"]["decode_tps_by_ctx"] == {"canonical-short": 245.10},
      "(j) decode TPS lost despite only GPU line being bad")
check(rec_badgpu["measured_extensions"]["peak_vram_mib_by_gpu"] == {},
      "(j) VRAM should be empty, not fabricated")

# An absent GPU-state section also warns (distinct message), still no raise.
no_gpu = "\n".join(ln for ln in bad_gpu.splitlines() if "GPU state" not in ln and "nvidia-smi" not in ln)
m_nogpu = mr.parse_bench_output(no_gpu)
check(not m_nogpu.gpu_state_section_present, "(j) precondition no GPU section")
rec_nogpu = mr.build_record(tag="ik-llama/iq4ks-mtp", bench_metrics=m_nogpu,
                            hardware="rtx-3090", result_class="boot-fit-measured")
check(any("GPU state" in w for w in rec_nogpu["parse_warnings"]),
      f"(j) absent-GPU-section warning not surfaced: {rec_nogpu['parse_warnings']}")

if failures:
    print("FAIL test-measurement-record:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("PASS test-measurement-record (all assertions)")
PY
