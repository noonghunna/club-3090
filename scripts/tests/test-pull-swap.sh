#!/usr/bin/env bash
# Pull-Gate — `pull.sh --profile-like ... --dry-run --json` SHAPE contract.
#
# Exercises the ADDITIVE `--json` emit on the evaluate-only gate and asserts:
#
#   1. SHAPE: stdout is exactly ONE valid JSON object carrying the contracted
#      top-level keys {fit_verdict, arch, eligible, note, swap_path} and the
#      swap_path sub-keys {route, sibling_slug, quant_match, drop_spec_config}
#      with the right value types. (Driven against a curated Tier-1 slug, so
#      the run is hermetic — NO live network, NO GPU.)
#   2. swap_path is STRUCTURED FIELDS, not a {"message": "..."} sentence
#      blob — the load-bearing TUI requirement. A curated hit carries
#      route=null (its own compose serves it).
#   3. The BRING_YOUR_OWN route-C swap path is derived from the SAME
#      `arch_model_xref` registry the gate's human `_hint` consults — asserted
#      structurally (route C / sibling / quant_match / drop_spec_config) for a
#      wrapper-arch (Qwen3_5MoeForConditionalGeneration) that maps to a
#      curated MoE we serve, and route B for an arch with no curated sibling.
#   4. STRICT ADDITIVITY: without `--json` the wrapper's stdout/stderr/exit
#      are byte-identical to a direct `python3 pull.py` invocation (modulo a
#      wall-clock capture timestamp).
#
# Hermetic: HF_TOKEN unset (deterministic gated path); the curated slug is
# resolved network-free by the deriver's Tier-1 lookup; the route-C/route-B
# swap derivation reuses the shipped read-only registry helpers directly.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset HF_TOKEN || true

CURATED_SLUG="Lorbus/Qwen3.6-27B-int4-AutoRound"   # Tier-1 curated (network-free)

failures=0

# ---------------------------------------------------------------------------
# 1+2: --json SHAPE on the curated (hermetic) slug. stdout is JSON-only (the
#      [compat] banner goes to stderr); pipe stdout into the validator.
# ---------------------------------------------------------------------------
# Capture the JSON to a file (the [compat] banner is on stderr -> discarded);
# the validator reads it as argv[1] so the heredoc owns python's stdin.
_json_out="$(mktemp)"
bash scripts/pull.sh "$CURATED_SLUG" --profile-like vllm/minimal --dry-run --json \
    >"$_json_out" 2>/dev/null || true
python3 - "$_json_out" <<'PY' || failures=$((failures+1))
import json
import sys

with open(sys.argv[1]) as fh:
    raw = fh.read().strip()
try:
    obj = json.loads(raw)
except Exception as exc:
    print(f"FAIL: --json stdout is not a single JSON object: {exc!r}\n{raw!r}",
          file=sys.stderr)
    sys.exit(1)

ok = True


def check(cond, msg):
    global ok
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        ok = False


check(isinstance(obj, dict), "--json emits a JSON object")
check(set(obj) == {"fit_verdict", "arch", "eligible", "note", "swap_path"},
      f"top-level keys exactly {{fit_verdict, arch, eligible, note, swap_path}} "
      f"(got {sorted(obj)})")
check(isinstance(obj.get("eligible"), bool), "eligible is a bool")
check("note" in obj and isinstance(obj["note"], str), "note is a string")

sp = obj.get("swap_path")
check(isinstance(sp, dict), "swap_path is an object (NOT a sentence/string)")
if isinstance(sp, dict):
    check(set(sp) == {"route", "sibling_slug", "quant_match", "drop_spec_config"},
          f"swap_path sub-keys exactly {{route, sibling_slug, quant_match, "
          f"drop_spec_config}} (got {sorted(sp)})")
    check("message" not in sp,
          "swap_path is STRUCTURED FIELDS, not a {message: ...} blob")
    check(sp.get("route") in ("B", "C", None), "swap_path.route in {B,C,null}")
    check(isinstance(sp.get("drop_spec_config"), bool),
          "swap_path.drop_spec_config is a bool")
    # A curated Tier-1 hit needs no swap path (its own compose serves it).
    check(sp.get("route") is None,
          "curated slug -> swap_path.route is null (no swap needed)")

sys.exit(0 if ok else 1)
PY
rm -f "$_json_out"

# ---------------------------------------------------------------------------
# 3: route-C / route-B swap-path DERIVATION reuses the shipped registry.
#    Asserts the structured mapping the --json heredoc produces for a
#    wrapper-arch that maps to a curated MoE (route C) and an arch with no
#    curated sibling (route B) — the BRING_YOUR_OWN swap path as FIELDS.
# ---------------------------------------------------------------------------
python3 - "$ROOT_DIR" <<'PY' || failures=$((failures+1))
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import pull as P            # READ-ONLY
from scripts.lib import generate_compose as gc        # READ-ONLY
from scripts.lib.profiles.compat import load_profiles

ok = True


def check(cond, msg):
    global ok
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        ok = False


rt = gc._load_yaml(root, "scripts/lib/profiles/profile_runtime.yml")
arches = gc.load_arches(root)
profiles = load_profiles()


def resolve(arch):
    """Replays the EXACT registry lookup the --json swap_path derivation runs
    (gc.resolve_arch_from_config + arch_model_xref). Returns
    (sibling_slug, family)."""
    canon, row = gc.resolve_arch_from_config(rt, arches, arch)
    family = row.get("family") if row else None
    slugs = (((rt.get("arch_model_xref") or {}).get(canon) or {})
             .get("model_slugs") or [])
    return (slugs[0] if slugs else None), family


# Route C: wrapper arch -> curated MoE sibling we price via the curated path.
sib, fam = resolve("Qwen3_5MoeForConditionalGeneration")
check(sib == "qwen3.6-35b-a3b" and fam in P._FAMILY_WEIGHT_FIELDS,
      f"route-C: Qwen3_5MoeForConditionalGeneration -> curated sibling "
      f"'qwen3.6-35b-a3b' (got {sib!r}, family={fam!r})")
m = profiles.models.get(sib)
first = next(iter(m.weights))
quant_match = (m.weights[first] or {}).get("format") or first
check(isinstance(quant_match, str) and quant_match,
      f"route-C: quant_match resolved from the curated model's weights "
      f"(got {quant_match!r})")
# drop_spec_config is now DETECTED, not blanket-True: _swap_path keeps
# --speculative-config when the brought checkpoint actually carries an MTP head
# (deriver.detect_mtp_head: config DECLARES the MTP layers + a dedicated mtp
# weights file). Regression guard for the bug where head-preserving fine-tunes
# (e.g. ThinkingCap AutoRound) were silently served MTP-off.
from scripts.lib.profiles.deriver import detect_mtp_head
_mtp_api = {"siblings": [{"rfilename": "model-00001-of-00007.safetensors"},
                         {"rfilename": "model_mtp_bf16.safetensors"}]}
_plain_api = {"siblings": [{"rfilename": "model.safetensors"}]}
check(detect_mtp_head({"mtp_num_hidden_layers": 1}, _mtp_api) is True,
      "route-C: MTP head PRESENT (config declares + mtp weights file) -> keep "
      "--speculative-config (drop_spec_config=False)")
check(detect_mtp_head({"num_hidden_layers": 48}, _plain_api) is False,
      "route-C: plain re-quant (no MTP declared, no mtp file) -> drop "
      "--speculative-config (drop_spec_config=True)")
check(detect_mtp_head({"text_config": {"num_nextn_predict_layers": 1}}, _mtp_api) is True,
      "route-C: MTP declared in nested text_config (VLM) + file present -> keep")
check(detect_mtp_head({"mtp_num_hidden_layers": 1}, _plain_api) is False,
      "route-C: declares MTP but no dedicated mtp weights file -> drop "
      "(embedded head not detectable without the index; conservative)")

# Route B: an arch with NO curated sibling -> the self-contained GGUF
# fallback (route B, no sibling/quant_match).
sib2, fam2 = resolve("LlamaForCausalLM")
check(sib2 is None,
      f"route-B: LlamaForCausalLM has no curated sibling -> route B / "
      f"sibling null (got sibling={sib2!r})")

sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------------------------
# 4: STRICT ADDITIVITY — without --json, pull.sh stdout/stderr/exit are
#    byte-identical to a direct pull.py invocation (modulo the wall-clock
#    capture-dir timestamp, which is non-deterministic by construction).
# ---------------------------------------------------------------------------
_sh_out="$(mktemp)"; _sh_err="$(mktemp)"
_py_out="$(mktemp)"; _py_err="$(mktemp)"
# Guard the exit-code capture against `set -e` (the curated+minimal combo is
# an honest hard-stop -> exit 2; we WANT both invocations to match on that).
_sh_ec=0
bash scripts/pull.sh "$CURATED_SLUG" --profile-like vllm/minimal --dry-run \
    >"$_sh_out" 2>"$_sh_err" || _sh_ec=$?
_py_ec=0
python3 scripts/lib/profiles/pull.py "$CURATED_SLUG" --profile-like vllm/minimal --dry-run \
    >"$_py_out" 2>"$_py_err" || _py_ec=$?

# Normalize the only non-deterministic byte run: the capture-dir timestamp.
_strip() { sed -E 's#[0-9]{8}T[0-9]{6}Z#<TS>#g' "$1"; }

if [ "$_sh_ec" = "$_py_ec" ] \
   && diff <(_strip "$_sh_out") <(_strip "$_py_out") >/dev/null \
   && diff <(_strip "$_sh_err") <(_strip "$_py_err") >/dev/null; then
    echo "PASS: no-\"--json\" passthrough byte-identical to direct pull.py "\
"(exit=$_sh_ec, stdout+stderr match modulo capture timestamp)"
else
    echo "FAIL: passthrough drift (sh_ec=$_sh_ec py_ec=$_py_ec)" >&2
    diff <(_strip "$_sh_out") <(_strip "$_py_out") >&2 || true
    diff <(_strip "$_sh_err") <(_strip "$_py_err") >&2 || true
    failures=$((failures+1))
fi
rm -f "$_sh_out" "$_sh_err" "$_py_out" "$_py_err"

# The hard-stop passthrough above emits a real gitignored .pull-captures/
# bundle (the genuine pt1-gate capture); purge it so the test leaves NO repo
# residue and the CI condition (gitignored runtime state ABSENT) is restored
# (same discipline as test-pull.sh).
rm -rf "$ROOT_DIR/.pull-captures"

# ---------------------------------------------------------------------------
# 4: apply-swap EMIT — swap_apply.emit_swap_compose clones the sibling's REAL
#    compose (curated chat-template / parsers preserved) with --model re-pointed
#    at the brought weights, keeping --speculative-config iff the checkpoint has
#    an MTP head. No network (reads the on-disk vllm/dual compose). The download
#    + Route-C derive is exercised live by the c3 dogfood, not here.
# ---------------------------------------------------------------------------
python3 - "$ROOT_DIR" <<'PY' || failures=$((failures+1))
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

import yaml
from scripts.lib.profiles import swap_apply as SA

ok = True


def check(cond, msg):
    global ok
    print(("PASS: " if cond else "FAIL: ") + msg, file=sys.stdout if cond else sys.stderr)
    if not cond:
        ok = False


def _cmd(compose_path):
    doc = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))
    svc = next(iter(doc["services"].values()))
    return svc, svc["command"], Path(compose_path).read_text(encoding="utf-8")


def _val(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# pure command-list surgery (no I/O)
_in = ["--model", "/old", "--served-model-name", "a", "b",
       "--quantization", "auto_round", "--speculative-config", '{"method":"mtp"}']
_keep = SA.apply_command_overrides(_in, model_path="/brought-model",
                                   served_name="mine", drop_spec=False)
check(_val(_keep, "--model") == "/brought-model", "override: --model repointed")
check(_val(_keep, "--served-model-name") == "mine" and "a" not in _keep and "b" not in _keep,
      "override: --served-model-name replaces ALL sibling served values")
check(_val(_keep, "--quantization") == "auto_round", "override: --quantization untouched")
check("--speculative-config" in _keep, "override: spec-config KEPT when drop_spec=False")
_drop = SA.apply_command_overrides(_in, model_path="/b", served_name="m", drop_spec=True)
check("--speculative-config" not in _drop, "override: spec-config + value DROPPED when drop_spec=True")

# full emit against the real vllm/dual sibling compose — MTP-head present
p_mtp = SA.emit_swap_compose(root, "vllm/dual", Path("/tmp/brought-weights"),
                             served_name="ThinkingCap", has_mtp_head=True,
                             brought_san="test-mtp")
svc, cmd, txt = _cmd(p_mtp)
check(_val(cmd, "--model") == "/brought-model", "emit(MTP): --model at the mounted brought weights")
check("--speculative-config" in cmd, "emit(MTP): --speculative-config KEPT (head present)")
check('"method":"mtp"' in _val(cmd, "--speculative-config"),
      "emit(MTP): spec-config JSON survives the YAML round-trip")
check("qwen-froggeric-chat-template" in txt, "emit(MTP): curated chat template PRESERVED")
check(_val(cmd, "--reasoning-parser") == "qwen3" and _val(cmd, "--tool-call-parser") == "qwen3_coder",
      "emit(MTP): curated parsers PRESERVED")
check(any("/brought-model:ro" in str(v) for v in svc["volumes"]),
      "emit(MTP): brought-weights volume mount added")
check(not any(str(v).split(":/", 1)[0].startswith(("./", "../"))
              or "/../" in str(v).split(":/", 1)[0] for v in svc["volumes"]),
      "emit(MTP): sibling relative ../ mounts absolutized (compose is relocatable)")
check(str(svc.get("container_name", "")).startswith("vllm-brought-"),
      "emit(MTP): container_name distinct (no collision with the sibling)")
p_mtp.unlink()

# weights under a pulls/ dir → compose lands in the RUNTIME composes dir, not repo
import tempfile
_pd = Path(tempfile.mkdtemp()) / "club3090" / "pulls" / "repo-z"
_pd.mkdir(parents=True)
p_loc = SA.emit_swap_compose(root, "vllm/dual", _pd, served_name="loc",
                             has_mtp_head=False, brought_san="loc-z")
check("club3090/composes" in str(p_loc) and str(root) not in str(p_loc),
      "emit: compose written to RUNTIME composes dir, NOT the project tree")
p_loc.unlink()

# no MTP head → spec-config dropped, curated flags still intact
p_plain = SA.emit_swap_compose(root, "vllm/dual", Path("/tmp/brought-weights"),
                               served_name="plain", has_mtp_head=False,
                               brought_san="test-plain")
_, cmd2, txt2 = _cmd(p_plain)
check("--speculative-config" not in cmd2, "emit(no-head): --speculative-config DROPPED")
check("--reasoning-parser" in cmd2 and "qwen-froggeric" in txt2,
      "emit(no-head): curated flags still preserved")
p_plain.unlink()

sys.exit(0 if ok else 1)
PY

if [ "$failures" != 0 ]; then
    echo "$failures assertion group(s) failed." >&2
    exit 1
fi
echo "test-pull-swap.sh OK"
