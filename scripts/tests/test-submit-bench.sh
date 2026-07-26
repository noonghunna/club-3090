#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

# test-submit-bench.sh — BENCHMARKS.md row formatter + submit-bench.sh plumbing.
#
# SELF-CONTAINED BY CONSTRUCTION (#776). This test used to require six specific
# results/rebench/<tag>/ dirs of real 2026-05 benchmark output. Those were never
# committed — results/* is gitignored — so they existed only as untracked files
# on the maintainer rig. The test therefore PASSED here and could not pass on
# any fresh clone, which made the suite permanently red for every external
# contributor and meant this path had never actually been exercised in CI.
#
# It now synthesizes its own fixtures under results/rebench/_selftest-*/ (the
# gitignored subtree, same convention as test-measurement-record.sh) and asserts
# against those, so the verdict is identical on a fresh clone and on this rig.
# Any real local rebench dirs are additionally validated read-only as a bonus —
# never depended on, and never written to.
#
# Asserts:
#   (a) bench_row_fixtures() discovers every synthetic fixture, and SKIPS a dir
#       missing a required artifact rather than handing it downstream to fail;
#   (b) section_name() routes all FIVE branches — dual, quad, single vLLM,
#       single llama.cpp, Gemma — where the old real fixtures covered only
#       dual + Gemma, so three branches were never tested at all;
#   (c) the Gemma section emits 11 columns and every other section 9;
#   (d) every row is a markdown table row citing results/rebench/<tag>/REPORT.md;
#   (e) formatter fallbacks work: docker-inspect list AND bare-dict shapes,
#       compose path from label AND inferred from container name, date from the
#       tag AND from REPORT.md, peak VRAM present AND absent ("TBD"), all three
#       soak verdicts, and pp from narrative+code AND from prompt_processing;
#   (f) bench.sh's BENCH_MOCK output shape, incl. thinking + PP modes;
#   (g) submit-bench.sh writes BENCHMARKS-row.md and prints all three routes;
#   (h) a missing tag fails loud;
#   (i) --auto-submit and --auto-submit --as-pr generate ISSUE/PR bodies and
#       invoke gh with the right title (via GH_MOCK);
#   (j) an unauthenticated gh aborts --auto-submit instead of half-submitting.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=../lib/bench-row-formatter.sh
source "$ROOT_DIR/scripts/lib/bench-row-formatter.sh"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_columns() {
  local row="$1"
  local expected="$2"
  local cols
  cols="$(awk -F'|' '{print NF - 2}' <<< "$row")"
  if [[ "$cols" != "$expected" ]]; then
    echo "ASSERTION FAILED: expected $expected columns, got $cols" >&2
    echo "$row" >&2
    exit 1
  fi
}

REBENCH_DIR="${ROOT_DIR}/results/rebench"

# Clear leftovers from a hard-killed previous run BEFORE generating, so the test
# is idempotent even if a SIGKILL skipped the EXIT trap. Only ever _selftest-*:
# real benchmark history is never touched, read or written.
rm -rf "${REBENCH_DIR}"/_selftest-*

TMP_BIN="$(mktemp -d)"
cleanup() {
  rm -rf "${REBENCH_DIR}"/_selftest-* "$TMP_BIN"
}
trap cleanup EXIT

# --- synthesize fixtures ----------------------------------------------------
# Each fixture carries the four artifacts the formatter requires (_internal.json,
# REPORT.md, container-config.json, rig.txt), shaped to drive one section branch
# plus a distinct set of formatter fallbacks.
python3 - "$REBENCH_DIR" "$ROOT_DIR" <<'PY'
import json
import sys
from pathlib import Path

REBENCH = Path(sys.argv[1])
ROOT = Path(sys.argv[2])

QWEN_DUAL = str(ROOT / "models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml")
QWEN_SINGLE = str(ROOT / "models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml")
GEMMA_DUAL = str(ROOT / "models/gemma-4-31b/vllm/compose/dual/qat-w4a16/int8.yml")


def cmd(served, tp, kv, ctx, spec=None):
    out = [
        "--served-model-name", served,
        "--tensor-parallel-size", str(tp),
        "--kv-cache-dtype", kv,
        "--max-model-len", str(ctx),
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.92",
    ]
    if spec is not None:
        out += ["--speculative-config", json.dumps(spec)]
    return out


def rig(n_gpus, power="300"):
    lines = ["hostname: selftest-rig"]
    for i in range(n_gpus):
        lines.append(f"GPU {i}: NVIDIA GeForce RTX 3090 (UUID: GPU-{i:04d})")
    if power:
        lines.append(f"power_cap_w: {power}")
    return "\n".join(lines) + "\n"


def internal(tps=(70.0, 85.0), pp=(1400.0, 1500.0), vram=("22345 MiB", "22990 MiB"),
             mtp=None, quality=(110, 150), aider=(38, 225), soak=None):
    bench = {}
    if tps:
        bench["narrative"] = {"wall_tps_mean": tps[0]}
        bench["code"] = {"wall_tps_mean": tps[1]}
        if pp:
            bench["narrative"]["pp_tps_mean"] = pp[0]
            bench["code"]["pp_tps_mean"] = pp[1]
    if vram:
        bench["gpu_state"] = [{"mem_used": v} for v in vram]
    if mtp:
        bench["mtp"] = mtp
    out = {"bench": bench}
    if quality:
        out["quality"] = {"total_passed": quality[0], "total_total": quality[1]}
    if aider:
        out["aider"] = {"passed_count": aider[0], "total_count": aider[1]}
    if soak:
        out["soak"] = soak
    return out


# tag -> (container-config blob, _internal.json, REPORT.md body, rig.txt)
FIXTURES = {
    # Dual-card TP=2. Compose path from the docker-compose LABEL, MTP notes,
    # clean soak, date parsed from the TAG, docker-inspect LIST shape.
    "_selftest-qwen-dual-mtp-2026-01-02": (
        [{
            "Name": "/vllm-qwen-dual",
            "Config": {
                "Image": "vllm/vllm-openai:v0.25.1",
                "Labels": {"com.docker.compose.project.config_files": QWEN_DUAL},
                "Cmd": cmd("qwen3.6-27b-autoround", 2, "fp8_e5m2", 262144,
                           {"method": "mtp", "num_speculative_tokens": 3}),
                "Env": ["VLLM_USE_V1=1"],
            },
        }],
        internal(mtp={"mean_accept_length": 2.41, "avg_accept_rate": 62.8},
                 soak={"verdict": "PASS", "silent_empty": "0 / 40"}),
        "**Date:** 2026-01-02\n\nVerify-stress: **12/12**\n",
        rig(2),
    ),
    # Dual-card with GENESIS_* env -> "Genesis on" note. Compose path INFERRED
    # from the container name (no label), TQ3 kv display, borderline soak.
    "_selftest-qwen-dual-genesis-2026-01-02": (
        [{
            "Name": "/vllm-qwen-dual-tq3-mtp-genesis",
            "Config": {
                "Image": "vllm/vllm-openai:v0.25.1",
                "Labels": {},
                "Cmd": cmd("qwen3.6-27b-autoround", 2, "turboquant_3bit_nc", 262144,
                           {"method": "mtp", "num_speculative_tokens": 2}),
                "Env": ["GENESIS_ENABLE=1"],
            },
        }],
        internal(mtp={"mean_accept_length": 1.98, "avg_accept_rate": 51.0},
                 soak={"verdict": "PASS", "silent_empty": "2 / 40", "max_growth_mib": "148"}),
        "**Date:** 2026-01-02\n\nVerify-stress: **11/12**\n",
        rig(2, power="275"),
    ),
    # Single-card vLLM TP=1. BARE DICT container-config (not a list), NO date in
    # the tag so it must come from REPORT.md, no MTP, no aider.
    "_selftest-qwen-single-vllm": (
        {
            "Name": "/vllm-qwen-minimal",
            "Config": {
                "Image": "vllm/vllm-openai:v0.25.1",
                "Labels": {"com.docker.compose.project.config_files": QWEN_SINGLE},
                "Cmd": cmd("qwen3.6-27b-autoround", 1, "auto", 48000),
                "Env": [],
            },
        },
        internal(vram=("21008 MiB",), aider=None,
                 soak={"verdict": "PASS", "silent_empty": "0"}),
        "**Date:** 2026-01-03\n\n**Overall:** PASS\n",
        rig(1),
    ),
    # Single-card llama.cpp, routed by container NAME rather than compose path.
    # No gpu_state -> peak VRAM must degrade to "TBD" instead of crashing.
    "_selftest-qwen-llamacpp-2026-01-04": (
        [{
            "Name": "/llama-cpp-qwen-single",
            "Config": {
                "Image": "ghcr.io/ggml-org/llama.cpp:server-cuda-b9967",
                "Labels": {},
                "Cmd": cmd("qwen3.6-27b", 1, "q8_0", 131072),
                "Env": [],
            },
        }],
        internal(vram=None, quality=None, aider=None,
                 soak={"verdict": "FAIL", "silent_empty": "5 / 40"}),
        "Verify-stress: **9/12**\n",
        rig(1, power=""),
    ),
    # Quad-card: TP=4 -> multi4 section, compose path inferred from tp alone.
    "_selftest-qwen-multi4-2026-01-05": (
        [{
            "Name": "/vllm-qwen-multi4",
            "Config": {
                "Image": "vllm/vllm-openai:v0.25.1",
                "Labels": {},
                "Cmd": cmd("qwen3.6-27b-autoround", 4, "fp8_e4m3", 262144,
                           {"method": "mtp", "num_speculative_tokens": 3}),
                "Env": [],
            },
        }],
        internal(tps=(96.0, 118.0), vram=("21500 MiB",) * 4,
                 mtp={"mean_accept_length": 2.55, "avg_accept_rate": 66.1},
                 soak={"verdict": "PASS", "silent_empty": "0 / 40"}),
        "**Date:** 2026-01-05\n\nVerify-stress: **12/12**\n",
        rig(4),
    ),
    # Gemma -> the ONLY 11-column section. Exercises mtp.per_position and
    # pp_display's single-key "prompt_processing" fallback.
    "_selftest-gemma-dual-2026-01-06": (
        [{
            "Name": "/vllm-gemma-dual",
            "Config": {
                "Image": "vllm/vllm-openai:v0.25.1",
                "Labels": {"com.docker.compose.project.config_files": GEMMA_DUAL},
                "Cmd": cmd("gemma-4-31b", 2, "int8_per_token_head", 224000,
                           {"method": "mtp", "num_speculative_tokens": 4}),
                "Env": [],
            },
        }],
        {
            "bench": {
                "narrative": {"wall_tps_mean": 53.2},
                "code": {"wall_tps_mean": 57.9},
                "prompt_processing": {"pp_tps_mean": 980.0},
                "gpu_state": [{"mem_used": "23100 MiB"}, {"mem_used": "23050 MiB"}],
                "mtp": {"mean_accept_length": 1.74, "avg_accept_rate": 44.2,
                        "per_position": "0.62/0.31/0.14/0.05"},
            },
            "quality": {"total_passed": 104, "total_total": 150},
            "soak": {"verdict": "PASS", "silent_empty": "0 / 40"},
        },
        "**Date:** 2026-01-06\n\nVerify-stress: **12/12**\n",
        rig(2),
    ),
}

for tag, (config_blob, internal_blob, report, rig_txt) in FIXTURES.items():
    d = REBENCH / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "container-config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")
    (d / "_internal.json").write_text(json.dumps(internal_blob, indent=2), encoding="utf-8")
    (d / "REPORT.md").write_text(f"# {tag}\n\n{report}", encoding="utf-8")
    (d / "rig.txt").write_text(rig_txt, encoding="utf-8")

# Negative fixture: a dir left by an INTERRUPTED rebench, missing _internal.json.
# bench_row_fixtures() must skip it, not emit a path that then fails to format.
partial = REBENCH / "_selftest-partial-interrupted"
partial.mkdir(parents=True, exist_ok=True)
(partial / "REPORT.md").write_text("# interrupted\n", encoding="utf-8")
(partial / "rig.txt").write_text(rig(2), encoding="utf-8")
PY

# --- expected section per synthetic fixture ---------------------------------
declare -A EXPECT_SECTION=(
  [_selftest-qwen-dual-mtp-2026-01-02]="Dual-card (2× RTX 3090, TP=2)"
  [_selftest-qwen-dual-genesis-2026-01-02]="Dual-card (2× RTX 3090, TP=2)"
  [_selftest-qwen-single-vllm]="Single-card (1× RTX 3090) — vLLM"
  [_selftest-qwen-llamacpp-2026-01-04]="Single-card (1× RTX 3090) — llama.cpp"
  [_selftest-qwen-multi4-2026-01-05]="Quad-card (4× RTX 3090, TP=4)"
  [_selftest-gemma-dual-2026-01-06]="Gemma 4 31B (community-experimental)"
)

# (a) discovery finds every synthetic fixture and skips the partial one.
fixtures=()
while IFS= read -r fixture; do
  fixtures+=("$fixture")
done < <(bench_row_fixtures)

for tag in "${!EXPECT_SECTION[@]}"; do
  found=0
  for dir in "${fixtures[@]}"; do
    if [[ "${dir##*/}" == "$tag" ]]; then
      found=1
      break
    fi
  done
  if [[ "$found" != 1 ]]; then
    echo "ASSERTION FAILED: bench_row_fixtures() did not discover synthetic fixture: $tag" >&2
    printf '  discovered: %s\n' "${fixtures[@]##*/}" >&2
    exit 1
  fi
done

for dir in "${fixtures[@]}"; do
  if [[ "${dir##*/}" == "_selftest-partial-interrupted" ]]; then
    echo "ASSERTION FAILED: bench_row_fixtures() emitted a dir missing _internal.json" >&2
    exit 1
  fi
done

# (b)(c)(d) each synthetic fixture routes to the right section with the right
# column count and cites its own REPORT.md.
for tag in "${!EXPECT_SECTION[@]}"; do
  dir="${REBENCH_DIR}/${tag}"
  section="$(bench_row_section "$dir")"
  row="$(bench_row_format "$dir")"
  if [[ "$section" != "${EXPECT_SECTION[$tag]}" ]]; then
    echo "ASSERTION FAILED: $tag -> section '$section', expected '${EXPECT_SECTION[$tag]}'" >&2
    exit 1
  fi
  [[ "$row" == \|*\| ]] || {
    echo "ASSERTION FAILED: row is not a markdown table row for $dir" >&2
    echo "$row" >&2
    exit 1
  }
  assert_contains "$row" "Report: \`results/rebench/${tag}/REPORT.md\`"
  if [[ "$section" == "Gemma 4 31B (community-experimental)" ]]; then
    assert_columns "$row" 11
  else
    assert_columns "$row" 9
  fi
done

# (e) the fallbacks the previous real fixtures never exercised.
row="$(bench_row_format "${REBENCH_DIR}/_selftest-qwen-llamacpp-2026-01-04")"
assert_contains "$row" "TBD"                      # no gpu_state -> peak VRAM degrades
assert_contains "$row" "Soak: ✗ FAIL"
row="$(bench_row_format "${REBENCH_DIR}/_selftest-qwen-single-vllm")"
assert_contains "$row" "2026-01-03"               # date from REPORT.md, not the tag
assert_contains "$row" "verify-stress PASS"       # **Overall:** PASS form
assert_contains "$row" "Soak: ✓ PASS"
row="$(bench_row_format "${REBENCH_DIR}/_selftest-qwen-dual-genesis-2026-01-02")"
assert_contains "$row" "Genesis on"
assert_contains "$row" "TQ3"
assert_contains "$row" "Soak: ⚠ borderline"
row="$(bench_row_format "${REBENCH_DIR}/_selftest-gemma-dual-2026-01-06")"
assert_contains "$row" "0.62/0.31/0.14/0.05"      # mtp.per_position (Gemma-only column)
assert_contains "$row" "980"                      # prompt_processing pp fallback
row="$(bench_row_format "${REBENCH_DIR}/_selftest-qwen-multi4-2026-01-05")"
assert_contains "$row" "/card"                    # multi-GPU peak VRAM suffix

# Bonus: validate any REAL local rebench output read-only. Never depended on —
# on a fresh clone there is none, and the assertions above already stand alone.
real_count=0
for dir in "${fixtures[@]}"; do
  case "${dir##*/}" in
    _selftest-*) continue ;;
  esac
  section="$(bench_row_section "$dir")"
  row="$(bench_row_format "$dir")"
  [[ "$row" == \|*\| ]] || {
    echo "ASSERTION FAILED: row is not a markdown table row for $dir" >&2
    echo "$row" >&2
    exit 1
  }
  assert_contains "$row" "Report: \`results/rebench/${dir##*/}/REPORT.md\`"
  if [[ "$section" == "Gemma 4 31B (community-experimental)" ]]; then
    assert_columns "$row" 11
  else
    assert_columns "$row" 9
  fi
  real_count=$((real_count + 1))
done

# (f) bench.sh mock output shape.
#
# PREFLIGHT_NO_AUTODETECT=1 + CONTAINER=none + ENGINE_KIND pinned, per the #478
# precedent in test-measurement-record.sh: unpinned, bench.sh adopts whatever
# container happens to be running on the rig, and ENGINE_KIND=llamacpp flips
# BENCH_MOCK from the PP_MODE=log branch to PP_MODE=fallback (bench.sh:126-128).
# Both branches emit "PP tok/s", so this was not failing — but WHICH branch got
# tested depended on rig state, so on a rig with a llama.cpp container up the
# log branch was never exercised at all. Pinning makes the coverage of both
# branches deliberate, and stops the autodetect naming an unrelated container.
MOCK_ENV=(PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none BENCH_MOCK=1 RUNS=1 WARMUPS=0)

out="$(env "${MOCK_ENV[@]}" ENGINE_KIND=vllm bash scripts/bench.sh)"
assert_contains "$out" "PP tok/s"
out="$(env "${MOCK_ENV[@]}" ENGINE_KIND=vllm ENABLE_THINKING=1 bash scripts/bench.sh 2>&1)"
assert_contains "$out" "[bench] thinking: enabled"
assert_contains "$out" "PP tok/s"
out="$(env "${MOCK_ENV[@]}" ENGINE_KIND=vllm PP=1 bash scripts/bench.sh)"
assert_contains "$out" "summary [prompt-processing]"
assert_contains "$out" "PP tok/s"
# The llamacpp engine forces PP_MODE=fallback independently of PP=1 — cover it
# explicitly rather than inheriting it by accident from a running container.
out="$(env "${MOCK_ENV[@]}" ENGINE_KIND=llamacpp bash scripts/bench.sh)"
assert_contains "$out" "PP tok/s"

# (g) submit-bench.sh end-to-end against a synthetic tag.
tag="_selftest-qwen-dual-mtp-2026-01-02"

out="$(bash scripts/submit-bench.sh --tag "$tag")"
assert_contains "$out" "Generated BENCHMARKS row for section: Dual-card (2× RTX 3090, TP=2)"
assert_contains "$out" "Wrote: results/rebench/${tag}/BENCHMARKS-row.md"
assert_contains "$out" "1. Issue + maintainer integrates"
assert_contains "$out" "2. Direct PR"
assert_contains "$out" "3. Manual edit"
test -s "results/rebench/${tag}/BENCHMARKS-row.md"

# (h) missing tag fails loud.
if out="$(bash scripts/submit-bench.sh --tag does-not-exist 2>&1)"; then
  echo "ASSERTION FAILED: missing tag unexpectedly succeeded" >&2
  exit 1
fi
assert_contains "$out" "tag dir not found: results/rebench/does-not-exist"

# (i) --auto-submit (issue) and --as-pr (PR) via GH_MOCK.
out="$(GH_MOCK=1 GH_MOCK_USER=octocat bash scripts/submit-bench.sh --tag "$tag" --auto-submit)"
assert_contains "$out" "Issue title: [bench] @octocat ${tag}"
test -s "results/rebench/${tag}/auto-submit-mock.log"
assert_contains "$(cat "results/rebench/${tag}/auto-submit-mock.log")" "gh issue create --title [bench] @octocat ${tag}"
test -s "results/rebench/${tag}/ISSUE-body.md"
assert_contains "$(cat "results/rebench/${tag}/ISSUE-body.md")" "results/rebench/${tag}/REPORT.md"
assert_contains "$(cat "results/rebench/${tag}/ISSUE-body.md")" "Proposed BENCHMARKS.md row"

out="$(GH_MOCK=1 GH_MOCK_USER=octocat bash scripts/submit-bench.sh --tag "$tag" --auto-submit --as-pr)"
assert_contains "$out" "PR title: bench(matrix): @octocat ${tag}"
test -s "results/rebench/${tag}/PR-body.md"
assert_contains "$(cat "results/rebench/${tag}/auto-submit-mock.log")" "gh pr create --title bench(matrix): @octocat ${tag}"
assert_contains "$(cat "results/rebench/${tag}/PR-body.md")" "results/rebench/${tag}/REPORT.md"

# (j) unauthenticated gh must abort, not half-submit.
cat > "${TMP_BIN}/gh" <<'MOCK_GH'
#!/usr/bin/env bash
if [[ "$1" == "auth" && "$2" == "status" ]]; then
  exit 1
fi
echo "unexpected gh call: $*" >&2
exit 2
MOCK_GH
chmod +x "${TMP_BIN}/gh"

if out="$(PATH="${TMP_BIN}:${PATH}" bash scripts/submit-bench.sh --tag "$tag" --auto-submit 2>&1)"; then
  echo "ASSERTION FAILED: unauthenticated gh path unexpectedly succeeded" >&2
  exit 1
fi
assert_contains "$out" "not authed with gh. Run: gh auth login"

# (k) a non-UTF-8 locale, end to end. Two independent failure mechanisms lived
# here, both invisible on the maintainer rig (#776) because the path never ran
# off it:
#   1. every section name carries "×", and a piped stdout defaults to ASCII, so
#      the formatter died with UnicodeEncodeError (#599 class, fixed in #778);
#   2. Python decodes sys.argv with ASCII+surrogateescape under such a locale, so
#      the row reached submit-bench.sh's python heredocs as LONE SURROGATES —
#      which strict utf-8 refuses, so no encoding= pin helps. Fixed with
#      os.fsencode()-based recovery (#777).
# PYTHONUTF8=0 + PYTHONCOERCECLOCALE=0 are load-bearing: modern Python coerces a
# bare LC_ALL=C to C.UTF-8, so without them these assertions are vacuous.
locale_env=(PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 LC_ALL=C LANG=C)
row="$(env "${locale_env[@]}" bash -c "source scripts/lib/bench-row-formatter.sh; bench_row_format '${REBENCH_DIR}/${tag}'")"
[[ "$row" == \|*\| ]] || {
  echo "ASSERTION FAILED: formatter did not emit a row under a non-UTF-8 locale" >&2
  echo "$row" >&2
  exit 1
}
assert_contains "$row" "Report: \`results/rebench/${tag}/REPORT.md\`"
assert_contains "$row" "2× 3090"   # simplify_gpu() drops the "NVIDIA GeForce RTX " prefix
section="$(env "${locale_env[@]}" bash -c "source scripts/lib/bench-row-formatter.sh; bench_row_section '${REBENCH_DIR}/${tag}'")"
if [[ "$section" != "Dual-card (2× RTX 3090, TP=2)" ]]; then
  echo "ASSERTION FAILED: section under non-UTF-8 locale was '$section'" >&2
  exit 1
fi

# ...and submit-bench.sh itself, on the --as-pr path, which is where the argv
# surrogates bit: it reads the unicode-bearing PR template and writes the body.
#
# Checked with an explicit `if !` rather than a bare assignment: under `set -e` a
# failing command substitution aborts the test with NO message, so a regression
# here would surface as a silent rc=1 and cost someone an afternoon.
run_submit_locale() {
  local what="$1"; shift
  if ! out="$(env "${locale_env[@]}" GH_MOCK=1 GH_MOCK_USER=octocat bash scripts/submit-bench.sh "$@" 2>&1)"; then
    echo "ASSERTION FAILED: submit-bench.sh ${what} failed under a non-UTF-8 locale" >&2
    echo "--- output ---" >&2
    echo "$out" >&2
    exit 1
  fi
}

run_submit_locale "--auto-submit --as-pr" --tag "$tag" --auto-submit --as-pr
assert_contains "$out" "PR title: bench(matrix): @octocat ${tag}"
assert_contains "$(cat "results/rebench/${tag}/PR-body.md")" "2× 3090"
run_submit_locale "--auto-submit" --tag "$tag" --auto-submit
assert_contains "$out" "Issue title: [bench] @octocat ${tag}"

# (l) insert_row() — the only function here that mutates a tracked file, and
# until now the only one with NO coverage: GH_MOCK exits before the real-gh path
# that calls it. Exercised against a throwaway copy via BENCHMARKS_FILE, under
# both locales, because its write is where a half-done encoding fix destroys
# data: Path.write_text truncates on open, so the failed encode left the target
# at 0 bytes (measured 46 -> 0). It now writes a temp sibling + os.replace.
bench_copy="${TMP_BIN}/BENCHMARKS-copy.md"
for loc in utf8 c; do
  cp BENCHMARKS.md "$bench_copy"
  before_bytes="$(wc -c < "$bench_copy")"
  if [[ "$loc" == "c" ]]; then
    insert_out="$(env "${locale_env[@]}" GH_MOCK=1 GH_MOCK_USER=octocat BENCHMARKS_FILE="$bench_copy" \
      bash scripts/submit-bench.sh --tag "$tag" --auto-submit --as-pr 2>&1)" || insert_rc=$?
  else
    insert_out="$(GH_MOCK=1 GH_MOCK_USER=octocat BENCHMARKS_FILE="$bench_copy" \
      bash scripts/submit-bench.sh --tag "$tag" --auto-submit --as-pr 2>&1)" || insert_rc=$?
  fi
  if [[ "${insert_rc:-0}" != 0 ]]; then
    echo "ASSERTION FAILED: insert_row run failed ($loc locale, rc=${insert_rc})" >&2
    echo "$insert_out" >&2
    exit 1
  fi
  assert_contains "$insert_out" "inserted row under 'Dual-card (2× RTX 3090, TP=2)'"
  after_bytes="$(wc -c < "$bench_copy")"
  if (( after_bytes <= before_bytes )); then
    echo "ASSERTION FAILED: insert_row ($loc locale) did not grow the file: ${before_bytes} -> ${after_bytes} bytes" >&2
    exit 1
  fi
  # The row must land inside the Dual-card table, and the rest of the file must
  # survive intact — a truncating write would take the unicode headings with it.
  assert_contains "$(cat "$bench_copy")" "results/rebench/${tag}/REPORT.md"
  assert_contains "$(cat "$bench_copy")" "Dual-card (2× RTX 3090, TP=2)"
  inserted_under="$(awk '/^#+ Dual-card \(2× RTX 3090, TP=2\)/{f=1;next} f&&/^#/{exit} f&&/^\|/{print;exit}' "$bench_copy")"
  [[ -n "$inserted_under" ]] || {
    echo "ASSERTION FAILED: no table row under the Dual-card heading after insert ($loc)" >&2
    exit 1
  }
  # No temp file left behind by the atomic write.
  if compgen -G "${bench_copy}.submit-bench.tmp" >/dev/null; then
    echo "ASSERTION FAILED: atomic-write temp file leaked ($loc)" >&2
    exit 1
  fi
done

# A FAILED insert must leave the target byte-identical. The reachable failure is
# an unknown section (SystemExit before any write); the unreachable-by-default one
# is an encode error mid-write, which is why the write goes through a temp sibling
# + os.replace rather than write_text — the latter truncates on open, and with an
# ASCII-named section (the Gemma row is the only one) that turned a #777 encode
# error into 180693 -> 0 bytes. That A/B is in the PR; asserted here is the
# invariant it protects: a failing insert never damages the file.
# A target file WITHOUT the derived section reaches insert_row's own not-found
# branch. (`--section` can't be used for this: it is validated against the file
# up front, so it exits before insert_row ever runs.)
bench_stub="${TMP_BIN}/BENCHMARKS-stub.md"
printf '# Bench\n\n## Some Other Section\n\n| a | b |\n|---|---|\n| 1 | 2 |\n' > "$bench_stub"
before_sum="$(sha256sum < "$bench_stub")"
if out="$(GH_MOCK=1 GH_MOCK_USER=octocat BENCHMARKS_FILE="$bench_stub" \
          bash scripts/submit-bench.sh --tag "$tag" --auto-submit --as-pr 2>&1)"; then
  echo "ASSERTION FAILED: insert into a file lacking the section unexpectedly succeeded" >&2
  exit 1
fi
assert_contains "$out" "section not found in BENCHMARKS.md"
bench_copy="$bench_stub"
if [[ "$(sha256sum < "$bench_copy")" != "$before_sum" ]]; then
  echo "ASSERTION FAILED: a failed insert modified the target file" >&2
  exit 1
fi
if compgen -G "${bench_copy}.submit-bench.tmp" >/dev/null; then
  echo "ASSERTION FAILED: atomic-write temp file leaked after a failed insert" >&2
  exit 1
fi

# The repo's own BENCHMARKS.md must be refused outright, so a bare GH_MOCK run
# can never mutate tracked history.
if out="$(GH_MOCK=1 GH_MOCK_USER=octocat BENCHMARKS_FILE="${ROOT_DIR}/BENCHMARKS.md" \
          bash scripts/submit-bench.sh --tag "$tag" --auto-submit --as-pr 2>&1)"; then
  echo "ASSERTION FAILED: insert into the repo's own BENCHMARKS.md was allowed" >&2
  exit 1
fi
assert_contains "$out" "refusing to insert into the repo's own BENCHMARKS.md"
if ! git diff --quiet -- BENCHMARKS.md; then
  echo "ASSERTION FAILED: BENCHMARKS.md was modified by the test" >&2
  exit 1
fi

echo "test-submit-bench: ok (${#EXPECT_SECTION[@]} synthetic fixtures, ${real_count} local rebench dir(s) also validated)"
