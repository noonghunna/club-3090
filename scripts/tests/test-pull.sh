#!/usr/bin/env bash
# v0.8.0 Pull-Gate P4 — the EXHAUSTIVE state-machine truth table.
#
# Per the LOCKED stop-condition (brief v6): prose iteration ended; THIS test
# is now the authoritative spec for the Pull-Gate state machine. It covers,
# with NO live network / NO GPU / NO HF-API / NO real [D]-emit:
#
#   * the §4.1 9 cells of [C1] (3 confidences × 3 raw-verdicts) with the
#     correct terminal + every flag interaction, asserted TOTAL + design-
#     locked (terminal set EXACTLY the locked 4);
#   * every stratum's abort, in order, with ordering assertions
#     (stratum-4 [C2a] AFTER stratum-3 [C0]; stratum-5 after [C2a];
#     stratum-6 Path-A-only);
#   * --experimental-arch bypasses ONLY no-arch-row (NOT
#     runtime-incompatible, NOT stratum-5 no-fit-model);
#   * golden cases g0..g15 from the brief;
#   * no --trust-remote-code ever in an in-scope Path-A emitted body
#     (reusing the patch_attribution.service_body()==0 check pattern).
#
# Hermetic: hardware_sm injected; HF fetcher is a recorded FixtureFetcher;
# disk free is an injected fake statvfs; [D] emit goes through the REAL
# generate_compose.generate (pure, no container) for the integration
# goldens and an injected d_runner for the dry-run-refusal golden.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.pop("HF_TOKEN", None)  # deterministic gated path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import pull as P  # noqa: E402  (P4 — under test)
from scripts.lib.profiles import deriver as D  # noqa: E402 (P2, frozen)
from scripts.lib.profiles import gates as G  # noqa: E402  (P3, frozen)
from scripts.lib.profiles import patch_attribution as PA  # noqa: E402
from scripts.lib.profiles.compat import load_profiles  # noqa: E402
from scripts.lib import generate_compose as GC  # noqa: E402  ([D], frozen)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


profiles = load_profiles()
SM_86 = 8.6   # RTX 3090 (Ampere)
SM_90 = 9.0   # Hopper-class

CFG = f"{D._HF_RESOLVE}/{{slug}}/resolve/main/config.json"
API = f"{D._HF_API}/{{slug}}?blobs=true"


class FixtureFetcher:
    """Recorded-response fetcher (the P2/P3 pattern). Raises if a Path-A
    curated run touches it (curated hits must be network-free)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list = []

    def get(self, url, headers=None, range_=None):
        self.calls.append((url, range_))
        if url not in self.routes:
            return D.FetchResponse(status=404, body=b"")
        spec = self.routes[url]
        if isinstance(spec, D.FetchResponse):
            return spec
        if isinstance(spec, dict):
            return D.FetchResponse(
                status=200, body=json.dumps(spec).encode("utf-8")
            )
        raise AssertionError(f"bad fixture for {url}")


class NoNet:
    def get(self, *a, **k):
        raise AssertionError("curated Path-A run must not hit the network")


def fake_statvfs(free_gb: float):
    def _sv(_p):
        class S:
            f_frsize = 4096
            f_bavail = int(free_gb * (1024 ** 3) / 4096)
        return S()
    return _sv


BIG_DISK = fake_statvfs(500.0)
TINY_DISK = fake_statvfs(2.0)


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


def dense_api(weight_gb=8.0):
    return {
        "siblings": [
            {"rfilename": "config.json", "size": 700},
            {"rfilename": "tokenizer.json", "size": 1_000_000},
            {"rfilename": "model.safetensors",
             "size": int(weight_gb * (1024 ** 3))},
        ]
    }


def ff_derived(slug, cfg, weight_gb=8.0):
    return FixtureFetcher({
        API.format(slug=slug): dense_api(weight_gb),
        CFG.format(slug=slug): cfg,
    })


CURATED_SLUG = "Lorbus/Qwen3.6-27B-int4-AutoRound"
CURATED_MODEL_ID = "qwen3.6-27b"           # Tier-1 curated model CURATED_SLUG resolves to
CURATED_VARIANT = "autoround-int4"          # variant CURATED_SLUG resolves to

# ---------------------------------------------------------------------------
# NON-MOCKED kv-calc parameterization.
#
# The pre-fix bug: `_curated_spec` priced Tier-1 curated hits through P1's
# conservative generic-dense LOWER-BOUND instead of the model's authoritative
# curated-exact family branch. The mocked truth-table never ran the real
# `_curated_spec` -> real `kv.raw_verdict` for a curated model, so a curated
# config that genuinely FITS got a false hard-block and every test stayed
# green.
#
# This helper derives the EXPECTED Pull-Gate `[B]` raw_verdict + `[C1]`
# terminal for a curated `(model_id, variant, profile-like)` straight from
# kv-calc's OWN authoritative curated-exact spec (`kv.MODEL_SPECS[...]`, the
# same specs `tools/kv-calc.py --calibration` validates at 22/22) and a LIVE
# `kv.predict()` call. Expectations are parameterized off kv-calc itself, so
# the Pull-Gate `[B]` verdict can NEVER silently diverge from kv-calc again
# (a divergence becomes a hard FAIL here, not a frozen-buggy green).
# ---------------------------------------------------------------------------
import importlib.util as _ilu  # noqa: E402

_kv_path = root / "tools" / "kv-calc.py"
_kv_spec = _ilu.spec_from_file_location("kv_calc", _kv_path)
_KVC = _ilu.module_from_spec(_kv_spec)
sys.modules["kv_calc"] = _KVC          # MUST precede exec_module (@dataclass)
_kv_spec.loader.exec_module(_KVC)

sys.path.insert(0, str(root / "scripts" / "lib" / "profiles"))
from compose_registry import COMPOSE_REGISTRY as _COMPOSE_REGISTRY  # noqa: E402

# kv-calc raw_verdict -> Pull-Gate raw_verdict; then §4.1 exact-row terminal.
_RAWMAP = {"PASS": "fits-clean", "TIGHT": "fits-constrained",
           "FAIL": "wont-fit"}
_EXACT_TERMINAL = {  # confidence == "exact" rows of the §4.1 table
    "fits-clean": "proceed",
    "fits-constrained": "confirm→proceed",
    "wont-fit": "hard-block",
}


def curated_exact_expectation(profile_like: str,
                              model_id: str = CURATED_MODEL_ID,
                              variant: str = CURATED_VARIANT):
    """Return (expected_raw_verdict, expected_exact_terminal) computed LIVE
    from kv-calc's authoritative curated-exact spec — never hardcoded."""
    mm = _COMPOSE_REGISTRY[profile_like]
    spec = dict(_KVC.MODEL_SPECS[model_id])  # authoritative; copy
    assert spec["model_family"] != "generic-dense", (
        f"{model_id} curated-exact spec must NOT be generic-dense")
    size_gb = profiles.models[model_id].weights[variant].get("size_gb")
    if isinstance(size_gb, (int, float)) and float(size_gb) > 0:
        spec["weights_total_gb"] = float(size_gb)  # qwen3-next-hybrid weight field
    p = _KVC.predict(
        spec=spec,
        kv_format=mm["kv_format"],
        max_ctx=int(mm["max_ctx"]),
        max_num_seqs=int(mm["max_num_seqs"] or 1),
        tp=int(mm["tp"]),
        mem_util=float(mm["mem_util"] or 0.95),
        vram_gb=24,
    )
    rv = _RAWMAP[p.verdict]
    return rv, _EXACT_TERMINAL[rv]


# ===========================================================================
# SECTION 1 — [C1] §4.1 TOTAL FUNCTION: all 9 cells × flag interactions.
#   This is a PURE-function unit. It is the single authoritative mapping;
#   the rest of the orchestrator only consumes it.
# ===========================================================================
print("\n--- [C1] §4.1 3×3 total function (pure) ---")

# design-lock: terminal set is EXACTLY the locked 4.
check(
    P.LOCKED_TERMINALS == {"proceed", "confirm→proceed", "hard-block",
                           "override-accepted"},
    f"design-lock: Terminal set == locked 4 (got {sorted(P.LOCKED_TERMINALS)})",
)
check(
    {t.value for t in P.Terminal} == P.LOCKED_TERMINALS,
    "design-lock: Terminal enum == LOCKED_TERMINALS (no extra states)",
)

# TOTAL: every (confidence, raw_verdict) in the domain maps to a cell.
unmapped = [
    (c, v)
    for c in P.C1_CONFIDENCE_DOMAIN
    for v in P.C1_RAW_VERDICT_DOMAIN
    if (c, v) not in P._C1_TABLE
]
check(not unmapped, f"[C1] is TOTAL over 3×3 domain (unmapped: {unmapped})")
check(
    len(P._C1_TABLE) == 9,
    f"[C1] table is exactly 9 cells (got {len(P._C1_TABLE)})",
)
# every cell's terminal is in the locked set
bad_term = [
    k for k, c in P._C1_TABLE.items()
    if c.base_terminal.value not in P.LOCKED_TERMINALS
]
check(not bad_term, f"[C1] every cell terminal ∈ locked 4 (bad: {bad_term})")

# Expected §4.1 outcomes (terminal, satisfied) per cell + flags.
# Source: v0.8.x-design.md §4.1 table (lines 62-66) + footnote line 68.
EXACT = D.Confidence.EXACT.value
DERV = D.Confidence.DERIVED.value
ELB = D.Confidence.ESTIMATED_LOWER_BOUND.value

# exact × fits-clean  -> proceed, SILENT (the ONLY silent gate-pass).
o = P.c1_terminal(EXACT, "fits-clean", {})
check(o.terminal is P.Terminal.PROCEED and o.satisfied and o.needs == "",
      "[C1] exact×fits-clean -> proceed (silent, no flag)")

# exact × fits-constrained -> confirm→proceed; needs --yes.
o = P.c1_terminal(EXACT, "fits-constrained", {})
check(o.terminal is P.Terminal.CONFIRM_PROCEED and not o.satisfied
      and o.needs == "--yes",
      "[C1] exact×fits-constrained -> confirm→proceed, needs --yes")
o = P.c1_terminal(EXACT, "fits-constrained", {"yes": True})
check(o.terminal is P.Terminal.CONFIRM_PROCEED and o.satisfied,
      "[C1] exact×fits-constrained + --yes -> satisfied")

# exact × wont-fit -> hard-block (no flag clears it).
o = P.c1_terminal(EXACT, "wont-fit", {"yes": True, "force_download": True})
check(o.terminal is P.Terminal.HARD_BLOCK and not o.satisfied,
      "[C1] exact×wont-fit -> hard-block (NO flag clears it)")

# derived × fits-clean -> confirm→proceed, needs --yes.
o = P.c1_terminal(DERV, "fits-clean", {})
check(o.terminal is P.Terminal.CONFIRM_PROCEED and not o.satisfied
      and o.needs == "--yes",
      "[C1] derived×fits-clean -> confirm→proceed, needs --yes")
o = P.c1_terminal(DERV, "fits-clean", {"yes": True})
check(o.satisfied, "[C1] derived×fits-clean + --yes -> satisfied")

# derived × fits-constrained -> confirm→proceed, needs --yes.
o = P.c1_terminal(DERV, "fits-constrained", {})
check(o.terminal is P.Terminal.CONFIRM_PROCEED and not o.satisfied,
      "[C1] derived×fits-constrained -> confirm→proceed, needs --yes")

# derived × wont-fit -> advisory → --force-download → override-accepted.
o = P.c1_terminal(DERV, "wont-fit", {})
check(o.terminal is P.Terminal.OVERRIDE_ACCEPTED and not o.satisfied
      and o.needs == "--force-download",
      "[C1] derived×wont-fit -> override-accepted, needs --force-download")
o = P.c1_terminal(DERV, "wont-fit", {"force_download": True})
check(o.terminal is P.Terminal.OVERRIDE_ACCEPTED and o.satisfied,
      "[C1] derived×wont-fit + --force-download -> override-accepted")
# --yes alone does NOT clear a wont-fit advisory.
o = P.c1_terminal(DERV, "wont-fit", {"yes": True})
check(not o.satisfied,
      "[C1] derived×wont-fit + --yes (no --force-download) -> NOT satisfied")

# estimated-lower-bound × fits-clean -> confirm→proceed, needs --yes.
o = P.c1_terminal(ELB, "fits-clean", {})
check(o.terminal is P.Terminal.CONFIRM_PROCEED and not o.satisfied
      and o.needs == "--yes",
      "[C1] estimated-lower-bound×fits-clean -> confirm→proceed, needs --yes")
o = P.c1_terminal(ELB, "fits-clean", {"yes": True})
check(o.satisfied,
      "[C1] estimated-lower-bound×fits-clean + --yes -> satisfied")

# estimated-lower-bound × fits-constrained -> confirm→proceed, needs --yes.
o = P.c1_terminal(ELB, "fits-constrained", {})
check(o.terminal is P.Terminal.CONFIRM_PROCEED and not o.satisfied,
      "[C1] estimated-lower-bound×fits-constrained -> confirm→proceed")

# estimated-lower-bound × wont-fit -> advisory → --force-download.
o = P.c1_terminal(ELB, "wont-fit", {})
check(o.terminal is P.Terminal.OVERRIDE_ACCEPTED and not o.satisfied
      and o.needs == "--force-download",
      "[C1] estimated-lower-bound×wont-fit -> override-accepted, "
      "needs --force-download")
o = P.c1_terminal(ELB, "wont-fit", {"force_download": True})
check(o.terminal is P.Terminal.OVERRIDE_ACCEPTED and o.satisfied,
      "[C1] estimated-lower-bound×wont-fit + --force-download -> "
      "override-accepted")

# §4.1 footnote line 68: ONLY exact×fits-clean reaches proceed without --yes.
silent = [
    (c, v)
    for c in P.C1_CONFIDENCE_DOMAIN
    for v in P.C1_RAW_VERDICT_DOMAIN
    if P.c1_terminal(c, v, {}).satisfied
    and P.c1_terminal(c, v, {}).terminal is P.Terminal.PROCEED
]
check(silent == [(EXACT, "fits-clean")],
      f"[C1] ONLY exact×fits-clean silent-passes (got {silent})")

# ===========================================================================
# SECTION 2 — STRATUM 1: deriver structured errors (g12 trio + stratum-1).
#   Aborts BEFORE any gate.
# ===========================================================================
print("\n--- stratum-1: deriver structured errors (g12) ---")

# g12a: no *.safetensors -> unsupported-format.
s = "fixtures/gguf-only"
ff = FixtureFetcher({
    API.format(slug=s): {"siblings": [{"rfilename": "model.gguf",
                                       "size": 9}]},
    CFG.format(slug=s): dense_cfg(),
})
r = P.run_pull(s, "vllm/minimal", hardware_sm=SM_86, fetcher=ff,
               profiles=profiles, statvfs=BIG_DISK)
check(r.stratum is P.Stratum.DERIVER
      and r.abort_reason == "unsupported-format" and not r.ok,
      f"g12: no safetensors -> stratum-1 unsupported-format "
      f"(got {r.stratum.name}/{r.abort_reason})")

# g12b: multiple top-level safetensors, no index -> ambiguous-weight-set.
s = "fixtures/ambiguous"
ff = FixtureFetcher({
    API.format(slug=s): {"siblings": [
        {"rfilename": "modelA.safetensors", "size": 9000000000},
        {"rfilename": "modelB.safetensors", "size": 9000000000},
        {"rfilename": "config.json", "size": 700},
    ]},
    CFG.format(slug=s): dense_cfg(),
})
r = P.run_pull(s, "vllm/minimal", hardware_sm=SM_86, fetcher=ff,
               profiles=profiles, statvfs=BIG_DISK)
check(r.stratum is P.Stratum.DERIVER
      and r.abort_reason == "ambiguous-weight-set",
      f"g12: multiple sets -> stratum-1 ambiguous-weight-set "
      f"(got {r.stratum.name}/{r.abort_reason})")

# g12c: repo-not-found (HF model API 404).
s = "fixtures/nope"
ff = FixtureFetcher({API.format(slug=s): D.FetchResponse(404, b"")})
r = P.run_pull(s, "vllm/minimal", hardware_sm=SM_86, fetcher=ff,
               profiles=profiles, statvfs=BIG_DISK)
check(r.stratum is P.Stratum.DERIVER
      and r.abort_reason == "repo-not-found",
      f"g12: 404 -> stratum-1 repo-not-found "
      f"(got {r.stratum.name}/{r.abort_reason})")
# stratum-1 aborts BEFORE [C0] / [B] (no fabricated verdict, §1).
check(r.raw_verdict is None and r.terminal is None,
      "g12: stratum-1 abort emits NO fit verdict (§1 honesty)")

# ===========================================================================
# SECTION 3 — STRATUM 2: --profile-like precondition (g0, g3b, g13).
#   Both paths, BEFORE [C0].
# ===========================================================================
print("\n--- stratum-2: --profile-like precondition (g0/g3b/g13) ---")

# g13: non-vLLM --profile-like (llamacpp/default) -> unsupported-runtime-engine
# (both paths), BEFORE [C0]/[B].
r = P.run_pull(CURATED_SLUG, "llamacpp/default", path="B",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is P.Stratum.PROFILE_LIKE
      and r.abort_reason == "unsupported-runtime-engine",
      f"g13: llamacpp/default -> stratum-2 unsupported-runtime-engine "
      f"(got {r.stratum.name}/{r.abort_reason})")

# g0: Genesis profile-not-emittable coverage moved to test-pullgate-gates,
# where the gate accepts synthetic registry/runtime fixtures. No
# Genesis-equipped compose remains in COMPOSE_REGISTRY post-#254, and
# run_pull() intentionally takes only real registry profile-like slugs.

# g3b: Path-A model/variant mismatch -> stratum-2 profile-mismatch.
# vllm/gemma-bf16-mtp is vLLM + emittable but model=gemma-4-31b != curated qwen.
r = P.run_pull(CURATED_SLUG, "vllm/gemma-bf16-mtp", path="A",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is P.Stratum.PROFILE_LIKE
      and r.abort_reason == "profile-mismatch",
      f"g3b: model/variant mismatch -> stratum-2 profile-mismatch "
      f"(got {r.stratum.name}/{r.abort_reason})")

# ===========================================================================
# SECTION 4 — STRATUM 3: [C0] engine-support / runtime / SM
#   (g4, g5, g6a, g6b, g10, g14) + flag-bypass scoping.
# ===========================================================================
print("\n--- stratum-3: [C0] (g4/g5/g6a/g6b/g10/g14) ---")

# g4: Llama (head_dim derived), trc:unverified -> needs-trust-remote-code-ack;
# --trust-remote-code -> engine-supported -> eligible -> estimated-lower-bound
# verdict, NO [D] (Path B).
s = "fixtures/llama-trc"
ffL = lambda: ff_derived(s, dense_cfg("LlamaForCausalLM"), weight_gb=8.0)
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ffL(), profiles=profiles, statvfs=BIG_DISK)
check(r.stratum is P.Stratum.C0
      and r.abort_reason == "needs-trust-remote-code-ack",
      f"g4: Llama trc:unverified -> stratum-3 needs-trust-remote-code-ack "
      f"(got {r.stratum.name}/{r.abort_reason})")
# + --trust-remote-code clears stratum-3; an estimated-lower-bound model
# then reaches [C1] = confirm→proceed which itself needs --yes (§4.1 — a
# low-confidence model is NEVER a silent pass). Without --yes it is an
# honest non-pass (NOT a fabricated fit).
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ffL(), profiles=profiles, statvfs=BIG_DISK,
                trust_remote_code=True)
check(not r.ok and r.stratum is P.Stratum.DECIDED
      and r.confidence == "estimated-lower-bound"
      and r.raw_verdict is not None and not r.emitted
      and r.abort_reason.startswith("confirm→proceed"),
      f"g4: + --trust-remote-code -> engine-supported -> [C1] "
      f"confirm→proceed (needs --yes; honest non-pass, NO [D]) "
      f"(ok={r.ok}, conf={r.confidence}, reason={r.abort_reason})")
# + --yes -> the Path-B verdict is download-eligible, still NO [D].
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ffL(), profiles=profiles, statvfs=BIG_DISK,
                trust_remote_code=True, yes=True)
check(r.ok and r.stratum is P.Stratum.DECIDED
      and r.confidence == "estimated-lower-bound"
      and r.raw_verdict is not None and not r.emitted,
      f"g4: + --trust-remote-code --yes -> Path-B verdict, NO [D] "
      f"(ok={r.ok}, conf={r.confidence}, emitted={r.emitted})")
check(any("soak-continuous" in n for n in r.notices),
      "g4: Path-B verdict carries the §7 boot-fit≠runtime caveat")

# g5: no-arch-row; --experimental-arch -> eligible: verdict;
#     ineligible -> stratum-5 no-fit-model (NOT bypassed) [also g11].
s = "fixtures/exotic-dense"
exo = dense_cfg("TotallyExoticForCausalLM")
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, exo), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is P.Stratum.C0
      and r.abort_reason == "engine-support-unknown/no-arch-row",
      f"g5: exotic arch -> stratum-3 no-arch-row "
      f"(got {r.stratum.name}/{r.abort_reason})")
# + --experimental-arch clears no-arch-row; eligible derived model reaches
# [C1] confirm→proceed (estimated-lower-bound) — needs --yes to be
# download-eligible.
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, exo), profiles=profiles,
                statvfs=BIG_DISK, experimental_arch=True, yes=True)
check(r.ok and r.stratum is P.Stratum.DECIDED and r.raw_verdict is not None,
      f"g5: + --experimental-arch --yes (eligible) -> Path-B verdict "
      f"(ok={r.ok}, stratum={r.stratum.name})")
check(any("bypassed" in n for n in r.notices),
      "g5: bypass surfaces a notice (Path B only, capture deferred)")

# g6a: known arch + auto_map -> --trust-remote-code only.
s = "fixtures/automap-llama"
am = dense_cfg("LlamaForCausalLM", auto_map={"x": "y"})
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, am), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is P.Stratum.C0
      and r.abort_reason == "needs-trust-remote-code-ack"
      and r.diagnostics.get("c0_bypassable_by") == ["--trust-remote-code"],
      f"g6a: known arch + auto_map -> needs-trc-ack, --trust-remote-code "
      f"ONLY (got {r.diagnostics.get('c0_bypassable_by')})")
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, am), profiles=profiles,
                statvfs=BIG_DISK, trust_remote_code=True, yes=True)
check(r.ok and r.stratum is P.Stratum.DECIDED,
      "g6a: + --trust-remote-code --yes -> proceeds (Path-B verdict)")
# g6a: --experimental-arch alone does NOT clear an auto_map+known-arch trc.
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, am), profiles=profiles,
                statvfs=BIG_DISK, experimental_arch=True)
check(r.stratum is P.Stratum.C0,
      "g6a: --experimental-arch alone does NOT clear auto_map trc")

# g6b: no-arch-row + auto_map -> BOTH flags required.
s = "fixtures/automap-exotic"
amx = dense_cfg("ExoticAutoMapForCausalLM", auto_map={"x": "y"})
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, amx), profiles=profiles,
                statvfs=BIG_DISK)
check(set(r.diagnostics.get("c0_bypassable_by") or [])
      == {"--trust-remote-code", "--experimental-arch"},
      f"g6b: no-arch-row+auto_map -> needs BOTH flags "
      f"(got {r.diagnostics.get('c0_bypassable_by')})")
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, amx), profiles=profiles,
                statvfs=BIG_DISK, trust_remote_code=True)
check(r.stratum is P.Stratum.C0,
      "g6b: only --trust-remote-code (missing --experimental-arch) -> "
      "still stratum-3 (subset rule)")
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, amx), profiles=profiles,
                statvfs=BIG_DISK, trust_remote_code=True,
                experimental_arch=True, yes=True)
check(r.ok and r.stratum is P.Stratum.DECIDED,
      "g6b: BOTH flags (+ --yes) -> clears stratum-3 -> Path-B verdict")

# g10: runtime-incompatible is NON-bypassable; --experimental-arch does
# NOT bypass it. Curated MoE arch with no loads:true row for the selected
# engine pin (Path B, arch from config) -> runtime-incompatible.
s = "fixtures/qwen35-moe"
moe = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "num_local_experts": 128, "torch_dtype": "bfloat16",
}
r = P.run_pull(s, "vllm/gemma-int8-mtp", path="B", hardware_sm=SM_90,
                fetcher=ff_derived(s, moe), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is P.Stratum.C0
      and r.abort_reason == "engine-support-unknown/runtime-incompatible",
      f"g10: missing loads:true pin -> stratum-3 runtime-incompatible "
      f"(got {r.stratum.name}/{r.abort_reason})")
r = P.run_pull(s, "vllm/gemma-int8-mtp", path="B", hardware_sm=SM_90,
                fetcher=ff_derived(s, moe), profiles=profiles,
                statvfs=BIG_DISK, experimental_arch=True)
check(r.stratum is P.Stratum.C0
      and r.abort_reason == "engine-support-unknown/runtime-incompatible",
      "g10: --experimental-arch does NOT bypass runtime-incompatible "
      "(non-bypassable)")

# g14: hardware-SM mismatch. Gemma fp8_e4m3 / required_sm 9.0 profile on a
# detected RTX 3090 (sm_86) -> runtime-incompatible (no false 'fits'),
# non-bypassable. (Curated Gemma 31B + vllm/gemma-mtp-tp1.)
GEMMA_SLUG = "Intel/gemma-4-31B-it-int4-AutoRound"
r = P.run_pull(GEMMA_SLUG, "vllm/gemma-mtp-tp1", path="A",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is P.Stratum.C0
      and r.abort_reason == "engine-support-unknown/runtime-incompatible",
      f"g14: fp8_e4m3/required_sm 9.0 on sm_86 -> stratum-3 "
      f"runtime-incompatible (got {r.stratum.name}/{r.abort_reason})")
r = P.run_pull(GEMMA_SLUG, "vllm/gemma-mtp-tp1", path="A",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK, experimental_arch=True)
check(r.stratum is P.Stratum.C0,
      "g14: --experimental-arch does NOT bypass the SM mismatch "
      "(non-bypassable)")
# Same profile on sm_90 clears the SM gate (proves it IS the SM gate).
r = P.run_pull(GEMMA_SLUG, "vllm/gemma-mtp-tp1", path="A",
                hardware_sm=SM_90, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK)
check(r.stratum is not P.Stratum.C0
      or r.abort_reason != "engine-support-unknown/runtime-incompatible",
      "g14: same profile on sm_90 passes [C0] SM gate (proves SM gate)")

# ===========================================================================
# SECTION 5 — STRATUM 4: [C2a] disk pre-gate (g7) + ordering AFTER [C0].
# ===========================================================================
print("\n--- stratum-4: [C2a] disk (g7) + order vs [C0] ---")

# g7: disk-short -> stratum-4 hard-abort AFTER [C0], BEFORE [B].
# (--trust-remote-code clears the trc:unverified [C0] so the run REACHES
#  stratum-4 — proving [C2a] is evaluated AFTER a clean [C0].)
s = "fixtures/big-llama"
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=200.0),
                profiles=profiles, statvfs=TINY_DISK,
                trust_remote_code=True)
check(r.stratum is P.Stratum.C2A_DISK and r.abort_reason == "disk-short",
      f"g7: 200 GiB model, 2 GiB free, [C0] clean -> stratum-4 "
      f"disk-short (got {r.stratum.name}/{r.abort_reason})")
# Ordering: a CLEAN [C0] still hard-aborts on disk-short, AND [B] never ran
# (no fit verdict).
check(r.raw_verdict is None,
      "g7 ordering: [C2a] is AFTER [C0] and BEFORE [B] "
      "(no fit verdict produced before the disk abort)")
# A no-arch-row model with disk-short stops at stratum-3 FIRST (proves
# strict 3-before-4 ordering: [C0] precedes [C2a]).
s = "fixtures/exotic-bigdisk-short"
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, dense_cfg("ZQZForCausalLM"),
                                   weight_gb=200.0),
                profiles=profiles, statvfs=TINY_DISK)
check(r.stratum is P.Stratum.C0,
      "ordering: no-arch-row + disk-short stops at stratum-3 FIRST "
      "([C0] strictly precedes [C2a])")

# ===========================================================================
# SECTION 6 — STRATUM 5: pre-[B] no-fit-model (g11) — monotonic,
#   non-bypassable; [C0] stays engine-supported.
# ===========================================================================
print("\n--- stratum-5: pre-[B] no-fit-model (g11) ---")

# g11: Gemma2-SWA-only style ineligible derived model with a KNOWN arch row
# (so [C0]=engine-supported, stays) -> stratum-5 no-fit-model.
# Use a SWA-only config on a no-arch-row arch + --experimental-arch so [C0]
# resolves (engine-support bypassed) then stratum-5 STILL fires (proves
# --experimental-arch does NOT bypass stratum-5).
s = "fixtures/swa-only"
swa = dense_cfg("SwaOnlyForCausalLM", sliding_window=4096)
swa.pop("max_window_layers", None)
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, swa), profiles=profiles,
                statvfs=BIG_DISK, experimental_arch=True)
check(r.stratum is P.Stratum.ELIGIBILITY
      and r.abort_reason == "no-fit-model",
      f"g11: SWA-only ineligible -> stratum-5 no-fit-model "
      f"(got {r.stratum.name}/{r.abort_reason})")
check("non-bypassable" in r.detail,
      "g11: no-fit-model is non-bypassable (detail states it)")
# MoE-on-known-arch path: [C0]=engine-supported (stays) -> stratum-5.
# Mixtral has no arch row; use a config that is MoE AND has the curated
# Qwen MoE arch row (engine-supported on a clean engine) to prove [C0] is
# NOT rewritten by stratum-5.
s = "fixtures/moe-known"
moe2 = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "num_local_experts": 128, "torch_dtype": "bfloat16",
}
c0 = G.c0_engine_support(
    "vllm/qwen-35b-a3b-dual", P.D.DeriveResult(
        slug=s, profile={"arch": "Qwen3_5MoeForConditionalGeneration",
                         "auto_map": False}),
    path="B", hardware_sm=SM_86, root=root)
r = P.run_pull(s, "vllm/qwen-35b-a3b-dual", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, moe2), profiles=profiles,
                statvfs=BIG_DISK, experimental_arch=True)
check(c0.state is G.C0State.ENGINE_SUPPORTED
      and r.stratum is P.Stratum.ELIGIBILITY
      and r.abort_reason == "no-fit-model",
      f"g11 monotonic: [C0]=engine-supported (stays) -> stratum-5 "
      f"no-fit-model ([C0]={c0.state.value}, "
      f"got {r.stratum.name}/{r.abort_reason})")
# --experimental-arch did NOT bypass stratum-5 (it ran WITH the flag set).
check(r.abort_reason == "no-fit-model",
      "g11: --experimental-arch does NOT bypass stratum-5 no-fit-model")

# ===========================================================================
# SECTION 7 — STRATUM 6: Path-A [D] dry-run (g1, g2, g8, g15) + Path-B
#   never touches [D].
# ===========================================================================
print("\n--- stratum-6: Path-A [D] dry-run + Path-B isolation "
      "(g1/g2/g8/g15) ---")

# g8: curated + --dry-run -> Path-B verdict, NO [D] (forced Path B even
# though the slug is curated and an --out is given).
r = P.run_pull(CURATED_SLUG, "vllm/dual", dry_run=True, out="/tmp/_no.yml",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK, yes=True)
check(r.path == "B" and not r.emitted and r.compose_text is None,
      f"g8: curated + --dry-run -> Path B, NO [D] emit "
      f"(path={r.path}, emitted={r.emitted})")
check(not os.path.exists("/tmp/_no.yml"),
      "g8: Path B wrote NO compose file even with --out (never emits)")

# g2: Path-A curated `vllm/dual` -> [C1] terminal is whatever the
# AUTHORITATIVE curated-exact kv-calc spec prices it (parameterized — NOT
# hardcoded; can never silently diverge from kv-calc). When the curated-exact
# verdict is download-eligible, --yes (when the §4.1 row needs it) -> [D]
# dry-run -> emit; the non-silent rows still require --yes.
OUT = "/tmp/_pull_g2.yml"
if os.path.exists(OUT):
    os.unlink(OUT)
g2_rv, g2_term = curated_exact_expectation("vllm/dual")
g2_silent = (g2_term == "proceed")          # only exact×fits-clean is silent
r = P.run_pull(CURATED_SLUG, "vllm/dual", path="A", out=OUT,
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK, yes=True)
check(r.ok and r.path == "A" and r.raw_verdict == g2_rv
      and r.terminal == g2_term and r.emitted,
      f"g2: Path-A curated vllm/dual + --yes -> kv-calc curated-exact "
      f"verdict={g2_rv} terminal={g2_term} -> [D] emit "
      f"(ok={r.ok}, verdict={r.raw_verdict}, terminal={r.terminal}, "
      f"emitted={r.emitted})")
if g2_rv == "fits-constrained":
    # fits-constrained invariant: surfaces the known effective-cap warning
    # (NOT 'applied constraint'; no rewrite). Only assertable when kv-calc
    # actually prices this curated config constrained.
    cap_notice = [n for n in r.notices if "effective-cap warning" in n]
    check(cap_notice and "no compose config rewritten" in cap_notice[0]
          and "applied constraint" not in cap_notice[0],
          "g2: fits-constrained prints 'known effective-cap warning' "
          "(NOT 'applied constraint'; no rewrite)")
# without --yes: a non-silent terminal is NOT satisfied (needs --yes); a
# silent exact×fits-clean `proceed` IS satisfied even without --yes.
r2 = P.run_pull(CURATED_SLUG, "vllm/dual", path="A", out=OUT + ".x",
                 hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                 statvfs=BIG_DISK)
if g2_silent:
    check(r2.ok and r2.emitted and r2.terminal == "proceed",
          f"g2: Path-A curated vllm/dual silent {g2_term} -> eligible "
          f"WITHOUT --yes (ok={r2.ok}, emitted={r2.emitted})")
    check(os.path.exists(OUT + ".x"),
          "g2: a satisfied silent proceed wrote the compose file")
    os.unlink(OUT + ".x")
else:
    check(not r2.ok and not r2.emitted
          and r2.abort_reason.startswith(g2_term),
          f"g2: Path-A curated vllm/dual non-silent {g2_term} WITHOUT "
          f"--yes -> NOT eligible, NO [D] (reason={r2.abort_reason})")
    check(not os.path.exists(OUT + ".x"),
          "g2: a non-satisfied non-silent terminal emits NO compose file")

# g1: Path-A curated, gate-passing, [D] dry-run ok -> proceed → [D] invoked.
# (We use the satisfied confirm→proceed above as the integration analogue
# of g1; here also assert the emitted body + that --project-directory note
# is surfaced, and the trc-leak invariant.)
check(r.compose_text is not None and "services:" in r.compose_text,
      "g1/g2: [D] produced a compose body on a clean dry-run")
check(os.path.exists(OUT),
      "g1/g2: Path-A satisfied terminal wrote the compose to --out")
check(any("--project-directory" in n for n in r.notices),
      "g1/g2: emitted compose carries the COMPOSE_GENERATOR.md "
      "--project-directory consumption note")

# trc-leak invariant: no --trust-remote-code in the in-scope Path-A emitted
# body (reuse the patch_attribution.service_body()==0 check pattern).
body = PA.service_body(r.compose_text)
check(body.count("--trust-remote-code") == 0,
      "g1/g2: in-scope Path-A emitted body contains ZERO "
      "--trust-remote-code (governed-slot invariant, service_body()==0)")

# g15: Path-A passes stratum-2 + [C1] download-eligible but the [D] dry-run
# REFUSES at a later point (foundational/degraded patch drift) -> Path-A
# abort, NOT reported download-eligible. Inject a d_runner that refuses.
def refusing_runner(_root, _profile, _ad):
    raise GC.Refuse("PN12: foundational drift-guard failed (compile-safe) "
                    "-> hard-refuse")

r = P.run_pull(CURATED_SLUG, "vllm/dual", path="A", out=OUT + ".g15",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK, yes=True, d_runner=refusing_runner)
check(r.stratum is P.Stratum.D_DRY_RUN
      and r.abort_reason == "d-refused:foundational-drift"
      and not r.ok and not r.emitted,
      f"g15: [D] dry-run refuses -> stratum-6 Path-A abort, NOT "
      f"download-eligible (got {r.stratum.name}/{r.abort_reason}, "
      f"ok={r.ok})")
check(not os.path.exists(OUT + ".g15"),
      "g15: a [D]-refused dry-run wrote NO compose file")

# g3: Path-A curated `vllm/minimal` -> [C1] terminal is whatever the
# AUTHORITATIVE curated-exact kv-calc spec prices it (parameterized — NOT
# hardcoded; can never silently freeze a misprice again). The structural
# invariant under test: exact×wont-fit is an UNCLEARABLE hard-block (no flag
# clears it, no [D]); a download-eligible curated-exact verdict emits.
g3_rv, g3_term = curated_exact_expectation("vllm/minimal")
r = P.run_pull(CURATED_SLUG, "vllm/minimal", path="A", out=OUT + ".g3",
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK, yes=True, force_download=True)
check(r.terminal == g3_term,
      f"g3: Path-A curated vllm/minimal terminal == kv-calc curated-exact "
      f"({g3_rv} -> {g3_term}); got {r.terminal}")
if g3_rv == "wont-fit":
    check(not r.ok and not r.emitted and r.abort_reason == "hard-block",
          f"g3: exact×wont-fit -> hard-block UNCLEARABLE even with --yes "
          f"--force-download, NO [D] (ok={r.ok}, "
          f"reason={r.abort_reason})")
    check(not os.path.exists(OUT + ".g3"),
          "g3: hard-block wrote NO compose file")
else:
    check(r.ok and r.emitted,
          f"g3: curated-exact download-eligible ({g3_term}) -> [D] emit "
          f"(ok={r.ok}, emitted={r.emitted})")
    check(os.path.exists(OUT + ".g3"),
          "g3: a download-eligible curated terminal wrote the compose")

# ---------------------------------------------------------------------------
# P4-fix REGRESSION (NON-MOCKED): Tier-1 curated hit must be priced through
# the model's authoritative curated-exact kv-calc family branch, NOT P1's
# conservative generic-dense lower-bound.
#
# This drives the REAL `run_pull` -> REAL `_curated_spec` -> REAL
# `kv.raw_verdict` (real kv-calc) for a Tier-1 curated `(slug, profile-like)`
# pair (no FixtureFetcher for the verdict; NoNet proves it stays network-free)
# and asserts the Pull-Gate `[B]` raw_verdict + `[C1]` terminal EQUAL kv-calc's
# OWN curated-exact `predict()` mapped through FAIL/TIGHT/PASS ->
# wont-fit/fits-constrained/fits-clean. Pre-fix: `_curated_spec` emitted
# model_family="generic-dense" -> wont-fit -> false `hard-block` (no compose).
# Expectation is parameterized off `kv.predict()` + `kv.MODEL_SPECS[...]`, so
# the gate can NEVER silently diverge from kv-calc again. Both curated vLLM
# profiles for this model FIT under curated-exact pricing -> NOT a false
# hard-block (the proven symptom this fix closes).
for _pl in ("vllm/minimal", "vllm/dual"):
    exp_rv, exp_term = curated_exact_expectation(_pl)
    _o = f"/tmp/_pull_p4fix_{_pl.replace('/', '_')}.yml"
    if os.path.exists(_o):
        os.unlink(_o)
    rr = P.run_pull(CURATED_SLUG, _pl, path="A", out=_o,
                     hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                     statvfs=BIG_DISK, yes=True)
    check(rr.raw_verdict == exp_rv,
          f"P4-fix[{_pl}]: Pull-Gate [B] raw_verdict == kv-calc "
          f"curated-exact ({exp_rv}); got {rr.raw_verdict} "
          f"(curated model NOT priced as generic-dense)")
    check(rr.terminal == exp_term,
          f"P4-fix[{_pl}]: [C1] terminal == kv-calc-derived exact-row "
          f"terminal ({exp_term}); got {rr.terminal}")
    if exp_rv != "wont-fit":
        check(rr.terminal != "hard-block" and rr.ok and rr.emitted
              and os.path.exists(_o),
              f"P4-fix[{_pl}]: kv-calc PASS/TIGHT -> NOT a false "
              f"hard-block; curated download-eligible + compose emitted "
              f"(terminal={rr.terminal}, ok={rr.ok}, emitted={rr.emitted})")
    if os.path.exists(_o):
        os.unlink(_o)

# Path-B structural isolation: NO Path-B run can ever set emitted/
# compose_text. Sweep a representative matrix.
pathb_emit = False
for pl in ("vllm/minimal", "vllm/dual", "vllm/qwen-35b-a3b-dual"):
    rb = P.run_pull(CURATED_SLUG, pl, dry_run=True, hardware_sm=SM_86,
                    fetcher=NoNet(), profiles=profiles, statvfs=BIG_DISK,
                    yes=True, force_download=True)
    if rb.emitted or rb.compose_text is not None:
        pathb_emit = True
check(not pathb_emit,
      "Path B is structurally incapable of [D] emit / download "
      "(--force-download is a no-op + notice this phase)")

# g9: generic-dense lower-bound is honored end-to-end + --calibration 22/22
# is asserted by the harness below (kv-calc unchanged).
s = "fixtures/llama-dense-elb"
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=4.0),
                profiles=profiles, statvfs=BIG_DISK, yes=True,
                trust_remote_code=True)
check(r.confidence == "estimated-lower-bound" and r.raw_verdict is not None,
      f"g9: derived dense -> estimated-lower-bound confidence, real "
      f"verdict (conf={r.confidence}, verdict={r.raw_verdict})")

# ===========================================================================
# SECTION 8 — STRATUM ORDERING PROOF (1→2→3→4→5→6 strict & monotonic).
#   Stack multiple failing conditions; assert the EARLIEST stratum wins.
# ===========================================================================
print("\n--- stratum ordering proof (earliest-wins, monotonic) ---")

# stratum-1 BEFORE stratum-2: a deriver error + a bad profile-like ->
# stratum-1 wins (deriver runs first).
s = "fixtures/order-s1"
ff = FixtureFetcher({API.format(slug=s): D.FetchResponse(404, b"")})
r = P.run_pull(s, "llamacpp/default", hardware_sm=SM_86, fetcher=ff,
                profiles=profiles, statvfs=TINY_DISK)
check(r.stratum is P.Stratum.DERIVER,
      "order: stratum-1 (deriver) precedes stratum-2 (bad profile-like) "
      "+ stratum-4 (tiny disk)")

# stratum-2 BEFORE stratum-3: bad profile-like + would-be-bad [C0] ->
# stratum-2 wins.
s = "fixtures/order-s2"
r = P.run_pull(s, "llamacpp/default", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, dense_cfg("ExoticXForCausalLM")),
                profiles=profiles, statvfs=TINY_DISK)
check(r.stratum is P.Stratum.PROFILE_LIKE,
      "order: stratum-2 (profile-like) precedes stratum-3 ([C0]) + "
      "stratum-4 (disk)")

# stratum-3 BEFORE stratum-4: no-arch-row + disk-short -> stratum-3 wins.
s = "fixtures/order-s3"
r = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, dense_cfg("NoRowForCausalLM"),
                                   weight_gb=200.0),
                profiles=profiles, statvfs=TINY_DISK)
check(r.stratum is P.Stratum.C0,
      "order: stratum-3 ([C0]) precedes stratum-4 ([C2a] disk)")

# stratum-4 BEFORE stratum-5: disk-short + ineligible (MoE) on a clean
# [C0] -> stratum-4 wins.
s = "fixtures/order-s4"
moec = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "num_local_experts": 128, "torch_dtype": "bfloat16",
}
r = P.run_pull(s, "vllm/qwen-35b-a3b-dual", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, moec, weight_gb=300.0),
                profiles=profiles, statvfs=TINY_DISK,
                experimental_arch=True)
check(r.stratum is P.Stratum.C2A_DISK,
      "order: stratum-4 ([C2a] disk-short) precedes stratum-5 "
      "(no-fit-model)")

# stratum-5 BEFORE [B]/[C1]: ineligible model never reaches a verdict.
s = "fixtures/order-s5"
r = P.run_pull(s, "vllm/qwen-35b-a3b-dual", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(s, moec, weight_gb=4.0),
                profiles=profiles, statvfs=BIG_DISK,
                experimental_arch=True)
check(r.stratum is P.Stratum.ELIGIBILITY and r.raw_verdict is None,
      "order: stratum-5 (no-fit-model) precedes [B] (no fit verdict "
      "produced)")

# stratum-6 is Path-A-only: the same gate-passing model on Path B NEVER
# reaches stratum-6 (no [D] dry-run on Path B).
r = P.run_pull(CURATED_SLUG, "vllm/dual", dry_run=True, hardware_sm=SM_86,
                fetcher=NoNet(), profiles=profiles, statvfs=BIG_DISK,
                yes=True)
check(r.stratum is not P.Stratum.D_DRY_RUN and not r.emitted,
      "order: stratum-6 ([D] dry-run) is Path-A ONLY (Path B never "
      "reaches it)")

# --experimental-arch bypass scoping — consolidated assertion:
#   bypasses no-arch-row; does NOT bypass runtime-incompatible; does NOT
#   bypass stratum-5 no-fit-model. (Cited individually in g5/g10/g14/g11.)
s = "fixtures/scope-norow"
r_norow = P.run_pull(s, "vllm/minimal", path="B", hardware_sm=SM_86,
                      fetcher=ff_derived(s, dense_cfg("ScopeXForCausalLM")),
                      profiles=profiles, statvfs=BIG_DISK,
                      experimental_arch=True)
check(r_norow.ok or r_norow.stratum is not P.Stratum.C0,
      "scope: --experimental-arch BYPASSES no-arch-row")
check(
    P.run_pull("fixtures/qwen35-moe2", "vllm/gemma-int8-mtp", path="B",
               hardware_sm=SM_90,
               fetcher=ff_derived("fixtures/qwen35-moe2", {
                   "model_type": "qwen3_5_moe",
                   "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                   "hidden_size": 4096, "num_hidden_layers": 32,
                   "num_attention_heads": 32, "num_key_value_heads": 8,
                   "num_local_experts": 128, "torch_dtype": "bfloat16"}),
               profiles=profiles, statvfs=BIG_DISK,
               experimental_arch=True).abort_reason
    == "engine-support-unknown/runtime-incompatible",
    "scope: --experimental-arch does NOT bypass runtime-incompatible",
)

# ===========================================================================
# SECTION 9 — v0.8.0 [E] E4: post-[C1] derived-[E] orchestration + trigger
#   semantics + override force-capture (pt5). All [E]-stage funcs MOCKED
#   (NO real Docker / GPU / network — real on-rig is E5). These are ADDED
#   cases (g16..g22); every assertion above (§4.1 9-cell, 6-stratum, g0..g15,
#   trc-leak, Path-B isolation) is unchanged and stays green.
# ===========================================================================
print("\n--- [E] E4: post-[C1] derived orchestration + triggers "
      "(g16..g22, mocked) ---")

import shutil as _sh  # noqa: E402

from scripts.lib.profiles import capture as _CAP  # noqa: E402

# E4 [E]-stage capture artifacts land under <root>/.pull-captures (the
# real runtime dir); these are mocked-run artifacts — purge the whole tree
# at the end so the test leaves NO repo residue (and it is .gitignore'd).
_CAP_ROOT = root / ".pull-captures"


def _purge_captures():
    _sh.rmtree(_CAP_ROOT, ignore_errors=True)


# Injected GPU topology (no real nvidia-smi dependence): 2× 24 GiB.
TOPO_2 = (2, [24576, 24576], ["NVIDIA GeForce RTX 3090",
                              "NVIDIA GeForce RTX 3090"])
TOPO_1 = (1, [24576], ["NVIDIA GeForce RTX 3090"])


class _Calls:
    def __init__(self):
        self.emit = self.dl = self.boot = self.smoke = 0
        self.cap = self.p5 = 0
        # E3/E4-fix ordering ledger: every lifecycle event appended in
        # invocation order so a test can PROVE the new contract —
        #   boot-up -> (server ALIVE) smoke -> capture [-> pt5] -> teardown
        # i.e. teardown index > capture index > smoke index > boot index,
        # and smoke/capture see `server_alive == True`.
        self.order: list[str] = []
        self.server_alive = False
        self.smoke_saw_alive = None
        self.capture_saw_alive = None


def mk_emocks(calls, *, emittable=True, boot_ok=True):
    """Mocked E1/E2/E3 + E4-pt5 funcs. NO Docker/GPU/network."""
    from scripts.lib.profiles.downloader import DownloadResult
    from scripts.lib.profiles.booter import BootResult

    def emit_fn(root, ei):
        calls.emit += 1
        if not emittable:
            raise GC.Refuse("derived-runtime-unsupported:kv")
        return ("services:\n  vllm-derived-x:\n    image: img:pin\n",
                {"resolved_image": "img:pin", "max_model_len": 32768,
                 "kv_format": ei.runtime.get("kv_format"),
                 "engine": ei.runtime.get("engine")})

    def download_fn(ei, fetcher=None):
        calls.dl += 1
        return DownloadResult(ok=True, files=["model.safetensors"],
                              bytes=8_000_000_000, sha_verified=True,
                              failure=None,
                              local_dir=str(ei.hf_home))

    import contextlib as _cl

    @_cl.contextmanager
    def boot_cm(ei, compose_text):
        # E3/E4-fix: the boot lifecycle is a CONTEXT MANAGER. up() FIRST,
        # then `yield` the live handle WHILE the (fixture) server is "up";
        # teardown (down) ALWAYS in the finally, on __exit__, AFTER the
        # with-body (smoke + capture + pt5). This fixture records the exact
        # event order so the test can prove up -> smoke -> capture -> down
        # and that teardown still runs if the with-body raises.
        calls.boot += 1
        calls.order.append("up")
        calls.server_alive = True
        try:
            if boot_ok:
                yield BootResult(ok=True, seconds=1.0, failure=None,
                                 endpoint="http://127.0.0.1:8020/v1")
            else:
                yield BootResult(ok=False, seconds=0.5,
                                 failure="CUDA OOM; worker exited",
                                 endpoint=None)
        finally:
            # ALWAYS — even if the with-body raised (no-orphan guarantee at
            # the CORRECT scope: AFTER smoke + capture, never before).
            calls.server_alive = False
            calls.order.append("down")

    def smoke_fn(ei, endpoint):
        calls.smoke += 1
        calls.order.append("smoke")
        # PROOF the server is ALIVE when smoke runs (the whole point of the
        # E3/E4-fix — the prior teardown-in-finally-before-return killed it).
        calls.smoke_saw_alive = calls.server_alive
        return _CAP.SmokeResult(
            smoke_capability_set=["plain-chat", "streaming"],
            results={"plain-chat": "green", "streaming": "green",
                     "tool-call": "unsmoked", "reasoning-streaming":
                     "unsmoked", "structured-output": "unsmoked",
                     "vision": "unsmoked", "long-context": "unsmoked"},
            partial=True)

    def capture_fn(ei, **kw):
        calls.cap += 1
        calls.order.append("capture")
        # Capture is also emitted while the server is still UP (before the
        # CM __exit__ teardown).
        calls.capture_saw_alive = calls.server_alive
        return _CAP.emit_capture(ei, **kw)

    def override_capture_fn(ei, **kw):
        calls.p5 += 1
        calls.order.append("pt5")
        return _CAP.emit_override_capture(ei, **kw)

    return dict(emit_fn=emit_fn, download_fn=download_fn, boot_cm=boot_cm,
                smoke_fn=smoke_fn, capture_fn=capture_fn,
                override_capture_fn=override_capture_fn)


import tempfile as _tf  # noqa: E402

# A non-curated generic-dense Llama (Qwen2 arch -> known arch row, no trc).
# torch_dtype bfloat16 -> weight_format bfloat16 (pure-dtype CONTRACT-2 row;
# CONTRACT-5 dispatch passes). vllm/minimal: clean engine, fp8_e5m2 KV,
# drafter None, no required features, tp=1 -> derived-emittable.
DSLUG = "fixtures/e4-dense"


def derived_run(profile_like, *, calls, emittable=True, boot_ok=True,
                topo=TOPO_2, **kw):
    with _tf.TemporaryDirectory() as _td:
        # repo_root for capture goes to a tmp dir via root override is not
        # possible (root selects the registry); capture writes under
        # <root>/.pull-captures — use the real root but a unique slug+ts so
        # artifacts are isolated. We assert on res.capture_paths.
        return P.run_pull(
            DSLUG, profile_like, path="B", hardware_sm=SM_86,
            fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"),
                               weight_gb=4.0),
            profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
            gpu_topology=topo, **mk_emocks(calls, emittable=emittable,
                                           boot_ok=boot_ok), **kw)


# --- g16: non-curated proceed (download-eligible), NO --dry-run -> [E] ----
c = _Calls()
r = derived_run("vllm/minimal", calls=c, yes=True)
check(r.ok and not r.emitted and r.confidence == "estimated-lower-bound",
      f"g16: non-curated download-eligible verdict stands "
      f"(ok={r.ok}, emitted={r.emitted}, conf={r.confidence})")
check(c.emit == 1 and c.dl == 1 and c.boot == 1 and c.cap == 1,
      f"g16: [E] ran emit+download+boot+capture "
      f"(emit={c.emit} dl={c.dl} boot={c.boot} cap={c.cap})")
check(r.download_ok is True and r.boot_ok is True
      and isinstance(r.smoke, dict)
      and r.smoke["smoke_capability_set"] == ["plain-chat", "streaming"],
      f"g16: PullResult [E] additive fields populated "
      f"(dl_ok={r.download_ok} boot_ok={r.boot_ok} smoke={r.smoke})")
_g16caps = [x for x in r.capture_paths if x.endswith(".json")]
check(any("pt1-gate" in x for x in _g16caps)
      and any("pt2-download" in x for x in _g16caps)
      and any("pt3-boot" in x for x in _g16caps)
      and any("pt4-smoke" in x for x in _g16caps)
      and any("manifest" in x for x in _g16caps)
      and not any("pt5" in x or "override" in x for x in _g16caps),
      f"g16: pt1-4 + manifest emitted, NO pt5 (non-override) "
      f"(paths={[os.path.basename(x) for x in _g16caps]})")
check(c.p5 == 0, "g16: override-capture (pt5) NOT invoked (not override)")

# --- g16b: E3/E4-fix — boot lifecycle is a context manager; the server
#     stays UP for smoke+capture; teardown happens on CM __exit__ AFTER
#     the with-body. This is the assertion that PROVES the on-rig E5
#     teardown-in-finally-before-smoke defect is fixed: prior code tore the
#     container down BEFORE smoke -> ConnectionRefused, by construction. ---
_oi = {ev: i for i, ev in enumerate(c.order)}
check(c.order == ["up", "smoke", "capture", "down"],
      f"g16b: lifecycle order is up -> smoke -> capture -> down EXACTLY "
      f"(got {c.order})")
check(_oi["up"] < _oi["smoke"] < _oi["capture"] < _oi["down"],
      f"g16b: teardown idx > capture idx > smoke idx > boot/up idx "
      f"(order={c.order})")
check(c.smoke_saw_alive is True,
      "g16b: smoke ran while the server was ALIVE (NOT torn down first — "
      "the on-rig E5 ConnectionRefused root-cause is fixed)")
check(c.capture_saw_alive is True,
      "g16b: capture (pt1-4 + manifest) emitted while the server was still "
      "UP, BEFORE the CM __exit__ teardown")
check(c.server_alive is False,
      "g16b: after run_pull returned the CM __exit__ ran -> server torn "
      "down (no-orphan guarantee preserved at the CORRECT scope)")
for _x in r.capture_paths:
    try:
        os.unlink(_x)
    except OSError:
        pass

# --- g16c: E3/E4-fix — teardown STILL runs (no-orphan) even if capture
#     RAISES inside the `with` body. The CM's finally must fire on the
#     exceptional __exit__; run_pull must NOT raise (PullResult intact). --
c = _Calls()


def _boom_capture(ei, **kw):
    c.cap += 1
    c.order.append("capture")
    c.capture_saw_alive = c.server_alive
    raise RuntimeError("capture blew up mid-with-body")


_em = mk_emocks(c, emittable=True, boot_ok=True)
_em["capture_fn"] = _boom_capture
r = P.run_pull(
    DSLUG, "vllm/minimal", path="B", hardware_sm=SM_86,
    fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"), weight_gb=4.0),
    profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
    yes=True, gpu_topology=TOPO_2, **_em)
check(isinstance(r, P.PullResult) and r.ok,
      f"g16c: capture raising inside the `with` does NOT raise out of "
      f"run_pull; PullResult intact (ok={r.ok})")
check("down" in c.order and c.order.index("down") > c.order.index("capture")
      and c.server_alive is False,
      f"g16c: CM teardown STILL ran on the exceptional __exit__, AFTER the "
      f"capture that raised (no-orphan preserved) (order={c.order})")
check(c.capture_saw_alive is True,
      "g16c: the raising capture still saw the server ALIVE (smoke+capture "
      "scope is correct even on the failure path)")
check(any("capture emit failed" in n for n in r.notices),
      "g16c: the capture exception is recorded structurally as a notice")
_purge_captures()

# --- g17: non-curated + --dry-run -> verdict-only, NO [E] ----------------
c = _Calls()
r = P.run_pull(DSLUG, "vllm/minimal", dry_run=True, hardware_sm=SM_86,
                fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=4.0),
                profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
                yes=True, gpu_topology=TOPO_2, **mk_emocks(c))
check(r.path == "B" and (c.emit + c.dl + c.boot + c.cap + c.p5) == 0,
      f"g17: --dry-run stays verdict-only — NO [E] stage at all "
      f"(emit={c.emit} dl={c.dl} boot={c.boot} cap={c.cap} p5={c.p5})")
check(r.download_ok is None and r.boot_ok is None and r.smoke is None
      and r.capture_paths == [],
      f"g17: --dry-run leaves all [E] additive fields None/empty "
      f"(dl_ok={r.download_ok} boot_ok={r.boot_ok})")
check(any("soak-continuous" in n for n in r.notices),
      "g17: --dry-run verdict still carries the §7 caveat (unchanged)")

# --- g18: override-accepted + --force-download -> [E] + pt5 emitted ------
# estimated-lower-bound × wont-fit + --force-download -> override-accepted.
# A huge-weight derived model forces wont-fit on this hardware.
c = _Calls()
r = P.run_pull(DSLUG, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=400.0),
                profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
                force_download=True, gpu_topology=TOPO_2,
                **mk_emocks(c, boot_ok=False))
check(r.ok and r.terminal == "override-accepted",
      f"g18: derived wont-fit + --force-download -> override-accepted "
      f"(ok={r.ok}, terminal={r.terminal})")
check(c.emit == 1 and c.dl == 1 and c.boot == 1 and c.cap == 1
      and c.p5 == 1,
      f"g18: --force-download ACTIVATES [E] for override-accepted incl "
      f"pt5 (emit={c.emit} dl={c.dl} boot={c.boot} cap={c.cap} p5={c.p5})")
_p5 = [x for x in r.capture_paths if "pt5" in x or "override" in x]
check(len(_p5) == 1, f"g18: exactly one pt5 artifact (got {_p5})")
_p5doc = json.loads(Path(_p5[0]).read_text())
check(_p5doc["point"] == "override_capture"
      and _p5doc["calibration_signal_not_validated"] is True,
      f"g18: pt5 carries point=override_capture + literal "
      f"calibration_signal_not_validated:true (got "
      f"{_p5doc.get('calibration_signal_not_validated')!r})")
check(_p5doc["actual"] is None and _p5doc["exit_error_summary"]
      and _p5doc["predicted_vs_actual_delta_mib"] is None,
      f"g18: boot never reached allocation -> actual null, "
      f"exit_error_summary set, delta null (got actual="
      f"{_p5doc['actual']!r}, exit={_p5doc['exit_error_summary']!r})")
check("predicted_b_breakdown" in _p5doc,
      "g18: pt5 carries the full [B] kv-calc predicted breakdown")
for _x in r.capture_paths:
    try:
        os.unlink(_x)
    except OSError:
        pass

# --- g19: override-accepted WITHOUT --force-download -> NO [E] -----------
# Without --force-download the wont-fit advisory is NOT satisfied (the [C1]
# row needs --force-download); honest non-pass, no [E].
c = _Calls()
r = P.run_pull(DSLUG, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=400.0),
                profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
                gpu_topology=TOPO_2, **mk_emocks(c))
check(not r.ok and r.abort_reason.startswith("override-accepted")
      and (c.emit + c.dl + c.boot + c.cap + c.p5) == 0,
      f"g19: override-accepted WITHOUT --force-download -> NOT satisfied, "
      f"NO [E] (ok={r.ok}, reason={r.abort_reason}, emit={c.emit})")

# --- g20: confirm→proceed WITHOUT --yes -> NOT eligible, NO [E] ----------
c = _Calls()
r = P.run_pull(DSLUG, "vllm/minimal", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=4.0),
                profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
                gpu_topology=TOPO_2, **mk_emocks(c))
check(not r.ok and r.abort_reason.startswith("confirm→proceed")
      and (c.emit + c.dl + c.boot + c.cap + c.p5) == 0,
      f"g20: confirm→proceed WITHOUT --yes -> NOT download-eligible, "
      f"NO [E] (ok={r.ok}, reason={r.abort_reason}, emit={c.emit})")

# --- g21: curated Path-A -> NO download stage (unchanged) ----------------
c = _Calls()
_o21 = "/tmp/_pull_g21.yml"
if os.path.exists(_o21):
    os.unlink(_o21)
r = P.run_pull(CURATED_SLUG, "vllm/dual", path="A", out=_o21,
                hardware_sm=SM_86, fetcher=NoNet(), profiles=profiles,
                statvfs=BIG_DISK, yes=True, gpu_topology=TOPO_2,
                **mk_emocks(c))
check(r.ok and r.emitted and r.path == "A"
      and (c.emit + c.dl + c.boot + c.cap + c.p5) == 0,
      f"g21: curated Path-A emits via [D], NO derived [E] download stage "
      f"(emitted={r.emitted}, emit={c.emit} dl={c.dl})")
check(r.download_ok is None and r.boot_ok is None,
      "g21: curated Path-A leaves [E] additive fields untouched (unchanged)")
if os.path.exists(_o21):
    os.unlink(_o21)

# --- g22: derived but CONTRACT-5 reject -> structured refuse, NO dl/boot -
# Point --profile-like at the single-card fp8 Gemma shape so derived_emittable
# refuses BEFORE any download/boot on the Ampere hardware gate.
c = _Calls()
r = P.run_pull(DSLUG, "vllm/gemma-mtp-tp1", path="B", hardware_sm=SM_86,
                fetcher=ff_derived(DSLUG, dense_cfg("Qwen2ForCausalLM"),
                                   weight_gb=4.0),
                profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True,
                yes=True, gpu_topology=TOPO_2, **mk_emocks(c))
_n22 = " ".join(r.notices)
check("derived-runtime-unsupported:" in _n22
      and (c.emit + c.dl + c.boot + c.cap + c.p5) == 0,
      f"g22: CONTRACT-5 reject -> structured derived-runtime-unsupported "
      f"refuse, NO emit/download/boot (notices={_n22!r}, emit={c.emit} "
      f"dl={c.dl})")
check(r.download_ok is None and r.boot_ok is None,
      "g22: CONTRACT-5 reject leaves [E] additive fields None (no stage ran)")

_purge_captures()

# ===========================================================================
# v0.8.2 CONTRACT-1.1 — capture-on-hard-block pass-through (V1).
#
# The pt1-gate emitter is wired on the terminal hard-block `return res`
# paths as a PURE PASS-THROUGH: the decision (ok/stratum/abort_reason) is
# UNCHANGED; a pt1-gate.json + schema:2 manifest.json bundle is emitted
# BEFORE the existing return. We inject a mock `gate_capture_fn` to keep
# this hermetic (real emit_gate_capture is unit-tested in
# test-pullemit-capture.sh + on-rig V6); here we assert (a) the decision is
# byte-unchanged vs the SAME run without the hook, and (b) the hook is
# invoked with the EXACT shipped abort_reason at every enumerated stratum.
# ===========================================================================
print("\n--- v0.8.2 CONTRACT-1.1: capture-on-hard-block pass-through ---")

_GATE_CALLS: list = []


def _mock_gate_capture(**kw):
    _GATE_CALLS.append(kw)
    return {"paths": {"gate": "/tmp/x/pt1-gate.json",
                      "manifest": "/tmp/x/manifest.json"},
            "dir": "/tmp/x", "manifest": {"schema": 2}}


def _gate_case(name, run_kwargs, *, expect_reason, expect_stratum):
    # Baseline: the SAME run with NO gate hook (decision reference).
    base = P.run_pull(**run_kwargs)
    # With the pass-through hook injected.
    _GATE_CALLS.clear()
    hooked = P.run_pull(**run_kwargs, gate_capture_fn=_mock_gate_capture)
    # (a) the decision is byte-unchanged (pass-through, NOT decision logic).
    check(base.ok == hooked.ok
          and base.stratum == hooked.stratum
          and base.abort_reason == hooked.abort_reason
          and base.detail == hooked.detail,
          f"{name}: pass-through is decision-NEUTRAL (ok/stratum/"
          f"abort_reason/detail byte-identical with vs without the hook)")
    check(hooked.abort_reason == expect_reason
          and hooked.stratum is expect_stratum,
          f"{name}: terminal decision unchanged "
          f"(stratum={hooked.stratum.name} reason={hooked.abort_reason!r})")
    # (b) the gate emitter was invoked once with the EXACT shipped
    #     abort_reason (NOT a semantic alias) at this stratum.
    check(len(_GATE_CALLS) == 1
          and _GATE_CALLS[0].get("abort_reason") == expect_reason
          and _GATE_CALLS[0].get("slug") == run_kwargs["slug"],
          f"{name}: emit_gate_capture invoked once with the exact shipped "
          f"abort_reason {expect_reason!r} "
          f"(got {[c.get('abort_reason') for c in _GATE_CALLS]})")
    return hooked


# C0 — engine-support-unknown/no-arch-row (the §10-R9 solicited lever).
_s = "fixtures/exotic-dense-gate"
_exo = dense_cfg("TotallyExoticForCausalLM")
_gate_case(
    "gate-C0-no-arch-row",
    dict(slug=_s, profile_like="vllm/minimal", path="B", hardware_sm=SM_86,
         fetcher=ff_derived(_s, _exo), profiles=profiles, statvfs=BIG_DISK),
    expect_reason="engine-support-unknown/no-arch-row",
    expect_stratum=P.Stratum.C0)

# C2a — disk-short (a user-environment correct-refusal).
_sd = "fixtures/big-llama-gate"
_gate_case(
    "gate-C2a-disk-short",
    dict(slug=_sd, profile_like="vllm/minimal", path="B", hardware_sm=SM_86,
         fetcher=ff_derived(_sd, dense_cfg("Qwen2ForCausalLM"),
                            weight_gb=200.0),
         profiles=profiles, statvfs=TINY_DISK, trust_remote_code=True),
    expect_reason="disk-short", expect_stratum=P.Stratum.C2A_DISK)

# Deriver stratum — a pre-deriver terminal (no der.profile -> model/arch
# /quant null in the bundle). Uses the same fixture shape as g1 (unknown
# weight format -> unsupported-format).
_sdr = "fixtures/derr-gate"
_ffdr = FixtureFetcher({
    API.format(slug=_sdr): {"siblings": [
        {"rfilename": "config.json", "size": 700},
        {"rfilename": "model.bin", "size": 8 * 1024 ** 3}]},
    CFG.format(slug=_sdr): dense_cfg("LlamaForCausalLM"),
})
_dr = P.run_pull(slug=_sdr, profile_like="vllm/minimal", hardware_sm=SM_86,
                 fetcher=_ffdr, profiles=profiles, statvfs=BIG_DISK,
                 gate_capture_fn=_mock_gate_capture)
check(_dr.stratum is P.Stratum.DERIVER and not _dr.ok,
      f"gate-deriver: terminal decision unchanged "
      f"(stratum={_dr.stratum.name} reason={_dr.abort_reason!r})")

# REAL end-to-end (no mock): a C0 hard-block writes a genuine schema:2
# bundle on disk; assert pt1-gate.json + manifest.json exist, schema==2,
# carry the EXACT abort_reason, outcome=='hard-block', failure_class null,
# and NO absolute path leaked into either artifact (the V1 RED-LINE).
_purge_captures()
_real = P.run_pull(slug=_s, profile_like="vllm/minimal", path="B",
                   hardware_sm=SM_86, fetcher=ff_derived(_s, _exo),
                   profiles=profiles, statvfs=BIG_DISK)
check(_real.abort_reason == "engine-support-unknown/no-arch-row",
      "gate-real: C0 hard-block decision unchanged (real emitter path)")
_gdir = _real.diagnostics.get("gate_capture", {}).get("dir")
check(_gdir and Path(_gdir).is_dir(),
      f"gate-real: a real .pull-captures/ bundle dir was written "
      f"(dir={_gdir!r})")
if _gdir:
    _gp = Path(_gdir)
    _g1 = _gp / "pt1-gate.json"
    _gm = _gp / "manifest.json"
    check(_g1.is_file() and _gm.is_file(),
          "gate-real: pt1-gate.json + manifest.json both written "
          "(NO pt2/3/4/5 — gate-only terminated pre-download)")
    _names = sorted(x.name for x in _gp.iterdir())
    check(_names == ["manifest.json", "pt1-gate.json"],
          f"gate-real: ONLY pt1-gate.json + manifest.json (no pt2-5) "
          f"(got {_names})")
    _mj = json.loads(_gm.read_text())
    _pj = json.loads(_g1.read_text())
    check(_mj.get("schema") == 2 and _mj.get("outcome") == "hard-block"
          and _mj.get("failure_class") is None
          and _mj.get("abort_reason") == "engine-support-unknown/no-arch-row",
          f"gate-real: manifest schema:2 / outcome:hard-block / "
          f"failure_class:null / exact abort_reason (got "
          f"schema={_mj.get('schema')} outcome={_mj.get('outcome')!r})")
    check(_pj.get("schema") == 2 and _pj.get("point") == "gate"
          and _pj.get("abort_reason")
          == "engine-support-unknown/no-arch-row"
          and _pj.get("is_gate_only") is True,
          "gate-real: pt1-gate.json schema:2/point:gate carries the raw "
          "abort_reason (H2 maintainer-distinguishability)")
    for _art in (_g1, _gm):
        _blob = _art.read_text()
        check("/opt/ai" not in _blob and "/home/" not in _blob
              and "/mnt/" not in _blob,
              f"gate-real: NO absolute path leaked in {_art.name} "
              f"(redacted; the V1 RED-LINE)")
    # `.last` marker written by the SHARED helper (centralization mandate).
    _last = root / ".pull-captures" / ".last"
    check(_last.is_file()
          and _last.read_text().strip() == str(
              _gp.relative_to(root / ".pull-captures")),
          "gate-real: the SHARED .last marker points at the gate bundle "
          "(centralized: gate-only updates .last too — --submit-last works)")
_purge_captures()

# ===========================================================================
# v0.8.2 CONTRACT-4 — the `recommend` UX (STEP V5).
#
# RED-LINE / outcome-not-addition: the recommendation MUST reflect the REAL
# shipped verdict and CHANGE WITH IT. We drive THREE genuinely-different
# real `run_pull` outcomes through `_render_recommendation` and assert the
# rendered block differs and matches the underlying `res` each time:
#   (1) FITS + emitted   — a curated Path-A proceed (r.ok, r.emitted)
#   (2) confirm→proceed  — an estimated-lower-bound non-pass (not r.ok)
#   (3) hard-block       — a C0 no-arch-row terminal (not r.ok, no [B])
# A static/templated summary that did not consume the real verdict would
# render identically for all three (or carry a §7 caveat on the blocked
# leg that never reached [B]); that is the explicit FAIL the brief names.
# Pure presentation: the SAME `res` objects the frozen gate produced are
# fed in; no decision field is read, mutated, or re-derived.
# ===========================================================================
print("\n--- v0.8.2 CONTRACT-4: recommend UX tracks the REAL verdict ---")

# (1) FITS + emitted — curated Path-A vllm/dual + --yes (the g2 shape).
_rec_fit = P.run_pull(CURATED_SLUG, "vllm/dual", path="A",
                       out="/tmp/_v5_rec.yml", hardware_sm=SM_86,
                       fetcher=NoNet(), profiles=profiles,
                       statvfs=BIG_DISK, yes=True)
if os.path.exists("/tmp/_v5_rec.yml"):
    os.unlink("/tmp/_v5_rec.yml")
_blk_fit = "\n".join(P._render_recommendation(_rec_fit))
check(_rec_fit.ok and _rec_fit.emitted,
      "rec(1): the FITS leg is a REAL ok+emitted curated verdict "
      f"(ok={_rec_fit.ok}, emitted={_rec_fit.emitted})")
check("verdict: FITS" in _blk_fit
      and "DOES NOT FIT" not in _blk_fit
      and "a ready compose was emitted" in _blk_fit
      and f"stratum={_rec_fit.stratum.name}" in _blk_fit,
      "rec(1): FITS+emitted -> 'FITS' + the emitted-compose line + the "
      "REAL deciding stratum (tracks res.ok/res.emitted/res.stratum)")

# (2a) confirm→proceed, NOT satisfied — estimated-lower-bound non-pass
# (the g4 shape: Llama + --trust-remote-code, NO --yes -> [C1]
# confirm→proceed:needs --yes). This is `not ok` and — per the shipped
# gate — the §7 caveat is appended ONLY on a download-eligible/accepted
# verdict, NOT on a not-yet-satisfied confirm→proceed; so this leg carries
# NO CAVEAT_S7 (verified against the live shipped run, not assumed). The
# recommendation MUST therefore omit the soak pointer here too.
_s_cp = "fixtures/v5-llama-cp"
_cp_kw = dict(slug=_s_cp, profile_like="vllm/minimal", path="B",
              hardware_sm=SM_86,
              fetcher=ff_derived(_s_cp, dense_cfg("LlamaForCausalLM"),
                                 weight_gb=8.0),
              profiles=profiles, statvfs=BIG_DISK, trust_remote_code=True)
_rec_cp = P.run_pull(**_cp_kw)
_blk_cp = "\n".join(P._render_recommendation(_rec_cp))
check(not _rec_cp.ok and _rec_cp.stratum is P.Stratum.DECIDED
      and _rec_cp.confidence == "estimated-lower-bound"
      and _rec_cp.abort_reason
      and _rec_cp.abort_reason.startswith("confirm→proceed"),
      "rec(2a): the confirm→proceed leg is a REAL estimated-lower-bound "
      f"non-pass (ok={_rec_cp.ok}, reason={_rec_cp.abort_reason!r})")
check("FITS (estimated) — NOT YET ACCEPTED" in _blk_cp
      and "DOES NOT FIT" not in _blk_cp
      and _rec_cp.abort_reason in _blk_cp
      and "ACCEPTANCE gate, not a fit failure" in _blk_cp
      and "--submit-last" not in _blk_cp
      and f"stratum={_rec_cp.stratum.name}" in _blk_cp,
      "rec(2a): a fits-clean confirm→proceed (not ok, NOT --yes) is HONESTLY "
      "rendered as FITS-but-NOT-YET-ACCEPTED (never 'DOES NOT FIT'), carries "
      "the REAL abort_reason + the acceptance-gate guidance + the REAL "
      "stratum, and does NOT misroute to the failure on-ramp")
check(P.CAVEAT_S7 not in _rec_cp.notices
      and "SOAK_MODE=continuous" not in _blk_cp,
      "rec(2a): a not-yet-satisfied confirm→proceed carries NO §7 caveat "
      "(the shipped gate appends it only on an accepted verdict) -> the "
      "recommendation correctly omits the soak pointer (tracks the REAL "
      "res.notices, NOT a static template)")

# (2b) the SAME model + --yes -> a download-eligible estimated-lower-bound
# verdict (r.ok, Path B so NOT emitted). The shipped gate DOES append the
# §7 caveat here; the recommendation MUST carry it verbatim AND honestly
# state "no compose emitted" (Path B). This proves the §7-caveat + the
# no-fabricated-artifact behavior both track the REAL res.
_rec_cy = P.run_pull(**_cp_kw, yes=True)
_blk_cy = "\n".join(P._render_recommendation(_rec_cy))
check(_rec_cy.ok and not _rec_cy.emitted
      and _rec_cy.confidence == "estimated-lower-bound"
      and P.CAVEAT_S7 in _rec_cy.notices,
      "rec(2b): --yes -> a REAL download-eligible estimated-lower-bound "
      f"verdict carrying CAVEAT_S7 (ok={_rec_cy.ok}, "
      f"emitted={_rec_cy.emitted})")
check("verdict: FITS" in _blk_cy
      and "ESTIMATED LOWER BOUND" in _blk_cy
      and "no compose was emitted this run" in _blk_cy
      and P.CAVEAT_S7 in _blk_cy
      and "SOAK_MODE=continuous" in _blk_cy,
      "rec(2b): FITS(estimated-lower-bound, Path B) -> floor stated, NO "
      "fabricated compose artifact, §7 caveat + soak pointer carried "
      "verbatim BECAUSE the real res carries CAVEAT_S7 (echoed from "
      "res.notices, not re-derived)")

# (3) hard-block — C0 no-arch-row (the gate-C0 shape; never reaches [B]).
_s_hb = "fixtures/v5-exotic"
_rec_hb = P.run_pull(_s_hb, "vllm/minimal", path="B", hardware_sm=SM_86,
                     fetcher=ff_derived(_s_hb,
                                        dense_cfg("V5ExoticForCausalLM")),
                     profiles=profiles, statvfs=BIG_DISK)
_blk_hb = "\n".join(P._render_recommendation(_rec_hb))
check(not _rec_hb.ok
      and _rec_hb.abort_reason == "engine-support-unknown/no-arch-row"
      and _rec_hb.stratum is P.Stratum.C0,
      "rec(3): the hard-block leg is a REAL C0 no-arch-row terminal "
      f"(reason={_rec_hb.abort_reason!r}, stratum={_rec_hb.stratum.name})")
check("DOES NOT FIT / BLOCKED" in _blk_hb
      and "engine-support-unknown/no-arch-row" in _blk_hb
      and f"stratum={_rec_hb.stratum.name}" in _blk_hb,
      "rec(3): hard-block -> the REAL abort_reason + the REAL deciding "
      "stratum are rendered (tracks res.abort_reason/res.stratum)")
# This leg NEVER reached [B], so the gate produced NO §7 caveat — the
# recommendation MUST NOT fabricate a soak pointer (the static-template
# FAIL the brief calls out: a templated block would carry it anyway).
check(P.CAVEAT_S7 not in _rec_hb.notices
      and "SOAK_MODE=continuous" not in _blk_hb
      and "§7 caveat" not in _blk_hb,
      "rec(3): a pre-[B] hard-block carries NO §7 caveat -> the "
      "recommendation correctly omits the soak pointer (tracks the REAL "
      "res.notices; a static template would falsely include it)")

# Outcome-not-addition: the rendered blocks are PAIRWISE DIFFERENT (a
# static/templated summary would be identical) AND each matches its own
# real res — the recommendation provably TRACKS the verdict, not merely
# "exists". Four genuinely-different real outcomes (fit+emitted /
# confirm→proceed-blocked / estimated-lower-bound-fit / hard-block).
_blocks = (_blk_fit, _blk_cp, _blk_cy, _blk_hb)
check(len(set(_blocks)) == len(_blocks),
      "rec: the four recommendation blocks are pairwise DIFFERENT "
      "(it tracks the real verdict; NOT a static template)")

# Leak-clean (rig-independent assertion convention — assert the absolute
# repo root is absent, NOT a /opt|/home substring allowlist): no recommend
# block leaks an absolute filesystem path.
for _name, _blk in (("fit", _blk_fit), ("cp", _blk_cp),
                    ("cy", _blk_cy), ("hb", _blk_hb)):
    check(str(root) not in _blk,
          f"rec({_name}): the recommendation block contains NO absolute "
          f"repo path (rig-independent leak assertion: str(root) absent)")

# vLLM-only by construction + honest-confidence echo (CONTRACT-4 invariants,
# read straight off the real res — not re-asserted policy).
check("engine=vLLM" in _blk_fit and "engine=vLLM" in _blk_hb,
      "rec: vLLM-only is stated on every leg (the gate is vLLM-only by "
      "construction; recommend only echoes it)")
check(_rec_cy.confidence == "estimated-lower-bound"
      and "ESTIMATED LOWER BOUND" in _blk_cy,
      "rec: an estimated-lower-bound FIT states the floor honestly "
      "(echoes res.confidence; never hides the lower-bound)")

_purge_captures()

# Purge ALL mocked-[E] capture residue (leave NO repo artifact).
_purge_captures()

# ===========================================================================
# Done.
# ===========================================================================
if failures:
    print(f"\nSUMMARY: {len(failures)} assertion(s) FAILED.",
          file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)

print("\nSUMMARY: all Pull-Gate P4 truth-table assertions passed "
      "(§4.1 9 cells + 6 strata + ordering + g0..g15 + trc-leak + "
      "Path-B isolation) + v0.8.0 [E] E4 orchestration (g16..g22: "
      "derived-[E] continuation, E3/E4-fix boot-lifecycle CM ordering "
      "[g16b up->smoke->capture->down, smoke+capture see server ALIVE] + "
      "[g16c teardown-on-exception], --dry-run verdict-only, "
      "--force-download override + pt5 force-capture, confirm-without-yes/"
      "override-without-force gating, curated-Path-A unchanged, "
      "CONTRACT-5 reject).")
PY

# ---------------------------------------------------------------------------
# CLI-contract: exit-code boundary (the pure truth-table above can't cover
# argv parsing / process exit). #370 regression lock: argparse usage errors
# MUST exit 64 (not 2 — argparse default), so a typo is distinguishable
# from an honest gate hard-stop (2); --help stays 0; hard-stop stays 2.
_clifail=0
_ec(){ bash scripts/pull.sh "$@" >/dev/null 2>&1; echo $?; }
[ "$(_ec)" = 64 ]                                              || { echo "FAIL: no-args -> 64 (#370)" >&2; _clifail=1; }
[ "$(_ec Qwen/Qwen2.5-0.5B-Instruct)" = 64 ]                   || { echo "FAIL: missing required --profile-like -> 64 (#370)" >&2; _clifail=1; }
[ "$(_ec --nope x)" = 64 ]                                     || { echo "FAIL: unknown flag -> 64 (#370)" >&2; _clifail=1; }
[ "$(_ec --help)" = 0 ]                                        || { echo "FAIL: --help -> 0 (#370 must not regress help)" >&2; _clifail=1; }
[ "$(_ec definitely/nonexistent-xyz123 --profile-like vllm/minimal --dry-run)" = 2 ] || { echo "FAIL: honest hard-stop -> 2 (must stay 2, not 64) (#370)" >&2; _clifail=1; }
[ "$_clifail" = 0 ] && echo "PASS: CLI exit-code contract (#370): usage=64, --help=0, hard-stop=2" || { echo "1+ CLI-contract assertion(s) failed." >&2; exit 1; }

# v0.8.2 CONTRACT-1.1: the real-CLI hard-stop above exercises the genuine
# pt1-gate capture-on-hard-block pass-through end-to-end (pull.sh ->
# emit_gate_capture). That writes a real gitignored .pull-captures/ bundle;
# purge it so the test leaves NO repo residue and the CI condition
# (gitignored runtime state ABSENT) is restored (same discipline as the
# in-heredoc _purge_captures for the mocked-[E] residue).
rm -rf "$ROOT_DIR/.pull-captures"

echo "test-pull.sh OK"
