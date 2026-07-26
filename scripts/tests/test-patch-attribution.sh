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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402
from scripts.lib.profiles import patch_attribution as pa  # noqa: E402

patches_path = root / "scripts/lib/profiles/patches.yml"
arches_path = root / "scripts/lib/profiles/arch_patches.yml"
seed_path = root / "scripts/lib/profiles/calibration_seed.yml"

errors: list[str] = []
known_gaps: list[str] = []


def load(path: Path) -> dict:
    return pa.load(path, errors=errors, root=root)


patch_doc = load(patches_path)
arch_doc = load(arches_path)
seed_doc = load(seed_path)

patches = patch_doc.get("patches", [])
arches = arch_doc.get("arches", [])
seeds = seed_doc.get("anchors", [])

patch_ids: set[str] = set()
covered_files: list[Path] = []
genesis_envs: set[str] = set()

required_patch_keys = pa.REQUIRED_PATCH_KEYS
valid_patch_status = pa.VALID_PATCH_STATUS

for patch in patches:
    missing = required_patch_keys - set(patch)
    if missing:
        errors.append(f"patch {patch.get('id', '<missing>')} missing keys: {sorted(missing)}")
    pid = patch.get("id")
    if not pid:
        errors.append("patch entry missing id")
        continue
    if pid in patch_ids:
        errors.append(f"duplicate patch id: {pid}")
    patch_ids.add(pid)
    if patch.get("status") not in valid_patch_status:
        errors.append(f"{pid} has invalid status {patch.get('status')!r}")
    delivery = patch.get("delivery") or {}
    for key in ("dockerfile_bake", "entrypoint_invoke", "genesis"):
        if key not in delivery or not isinstance(delivery.get(key), bool):
            errors.append(f"{pid}.delivery.{key} must be boolean")
    upstream = patch.get("upstream") or {}
    for key in ("ref", "status", "drop_when"):
        if not upstream.get(key):
            errors.append(f"{pid}.upstream.{key} missing")
    for rel in patch.get("files") or []:
        target = root / rel
        if not target.exists():
            errors.append(f"{pid} references missing file/dir: {rel}")
        else:
            covered_files.append(target)
    if patch.get("genesis_env"):
        genesis_envs.add(patch["genesis_env"])


for artifact in sorted(p for p in (root / "models").rglob("*") if p.is_file() and pa.is_artifact(p)):
    if not pa.covered(artifact, covered_files):
        errors.append(f"orphan patch artifact lacks patches.yml entry: {artifact.relative_to(root)}")

compose_files = sorted((root / "models").glob("**/compose/**/*.yml"))
found_genesis = set()
for compose in compose_files:
    text = compose.read_text(encoding="utf-8")
    found_genesis.update(re.findall(r"GENESIS_ENABLE_[A-Z0-9_]+", text))
missing_genesis = found_genesis - genesis_envs
if missing_genesis:
    errors.append(f"Genesis env flags missing patches.yml entries: {sorted(missing_genesis)}")


def compose_text(compose_name: str, seen: set[Path] | None = None) -> str:
    return pa.compose_text(root, compose_name, seen)


def gap_declared(patch: dict, compose_name: str) -> bool:
    return pa.gap_declared(patch, compose_name)


def reaches(patch: dict, compose_name: str) -> bool:
    return pa.reaches(root, patch, compose_name)


for patch in patches:
    for lb in patch.get("load_bearing_when") or []:
        for compose_name in lb.get("composes") or []:
            if compose_name not in COMPOSE_REGISTRY:
                errors.append(f"{patch['id']} load_bearing_when references unknown compose {compose_name}")
                continue
            if reaches(patch, compose_name):
                continue
            msg = f"{patch['id']} does not reach {compose_name}"
            if gap_declared(patch, compose_name):
                known_gaps.append(msg)
            else:
                errors.append(msg)

# ---------------------------------------------------------------------------
# CONTRACT-2b-i — chat_template delivery class + REAL extends merge.
# ---------------------------------------------------------------------------
# (1) Every patch's delivery_mechanism is in the v0.8.2 valid set (the
#     vocabulary now includes `chat_template`).
for patch in patches:
    dm = patch.get("delivery_mechanism")
    if dm not in pa.VALID_DELIVERY_MECHANISM:
        errors.append(
            f"{patch['id']} delivery_mechanism {dm!r} not in "
            f"{sorted(pa.VALID_DELIVERY_MECHANISM)}"
        )

# (2) chat_template patches: spec shape + a behavioral drift_guard whose
#     check encodes the SELF-CONTAINED symmetric restart+settle protocol
#     (the #150 lesson — a non-symmetric guard flaps and is ignored).
chat_template_patches = [
    p for p in patches if p.get("delivery_mechanism") == "chat_template"
]
for patch in chat_template_patches:
    spec = patch.get("delivery_spec") or {}
    for k in ("jinja", "mounted_at", "wired_at"):
        if not spec.get(k):
            errors.append(f"{patch['id']} chat_template delivery_spec missing {k}")
    jinja_rel = spec.get("jinja")
    if jinja_rel and not (root / jinja_rel).exists():
        errors.append(f"{patch['id']} chat_template jinja missing on disk: {jinja_rel}")
    if jinja_rel and Path(jinja_rel).suffix not in pa.CHAT_TEMPLATE_ARTIFACT_SUFFIXES:
        errors.append(f"{patch['id']} chat_template jinja is not a .jinja artifact: {jinja_rel}")
    dg = patch.get("drift_guard") or {}
    if dg.get("kind") != "behavioral":
        errors.append(f"{patch['id']} chat_template drift_guard must be kind: behavioral")
    chk = (dg.get("check") or "").lower()
    for token in ("symmetric", "docker restart", "settle", ">=3", "grand mean"):
        if token not in chk:
            errors.append(
                f"{patch['id']} chat_template drift_guard.check must encode the "
                f"self-contained symmetric protocol (missing {token!r})"
            )

# (3) Orphan-artifact discovery for vendored `.jinja` templates: every
#     chat-template `.jinja` under a model patches/ tree MUST be owned by a
#     delivery_mechanism: chat_template patch (so a bad/regressed/re-vendored
#     template can never ship with ZERO attribution coverage).
ct_covered_files = []
for patch in chat_template_patches:
    for rel in patch.get("files") or []:
        ct_covered_files.append(root / rel)
for jinja in sorted(
    p for p in (root / "models").rglob("*") if p.is_file() and pa.is_chat_template_artifact(p)
):
    if not pa.covered(jinja, ct_covered_files):
        errors.append(
            f"orphan chat-template artifact lacks a chat_template patches.yml entry: "
            f"{jinja.relative_to(root)}"
        )

# (4) Effective coverage MUST use REAL Docker Compose merge semantics, NOT
#     declared lines. The dangerous direction is the FALSE NEGATIVE: a child
#     that !reset/overrides/REMOVES the mount must be caught as a coverage
#     loss. Build a synthetic base+child fixture where the child re-declares
#     `volumes`/`command` WITHOUT the template and assert reaches() == False
#     (the legacy single-base text-concat would still "see" the base's mount
#     line and wrongly return True — that is the #377 failure mode).
import tempfile  # noqa: E402

if chat_template_patches:
    ctp = chat_template_patches[0]
    with tempfile.TemporaryDirectory() as _td:
        _tdp = Path(_td)
        mounted = (ctp.get("delivery_spec") or {}).get("mounted_at")
        base = _tdp / "base.yml"
        keep_child = _tdp / "keep.yml"
        reset_child = _tdp / "reset.yml"
        noextend = _tdp / "noextend.yml"
        base.write_text(
            "services:\n"
            "  base-svc:\n"
            "    image: scratch\n"
            "    command: [--model, m, --chat-template, %s]\n"
            "    volumes:\n"
            "      - ../../patches/froggeric-chat-template/chat_template.jinja:%s:ro\n"
            % (mounted, mounted),
            encoding="utf-8",
        )
        # (a) Child that KEEPS inheritance (only overrides an env) -> still
        #     covered. Proves extends: IS merged (not ignored).
        keep_child.write_text(
            "services:\n"
            "  keep-svc:\n"
            "    extends:\n"
            "      file: base.yml\n"
            "      service: base-svc\n"
            "    environment:\n"
            "      - X=1\n",
            encoding="utf-8",
        )
        # (b) Child that REMOVES the mount + --chat-template via the Compose
        #     `!reset` tag — the ONLY in-Compose removal mechanism across
        #     extends: (a plain re-declared `[]` does NOT drop a base
        #     sequence; Compose merges extends: sequences additively). The
        #     coverage loss MUST be caught (reaches == False). A text-concat
        #     would still "see" the base's mount line -> false-negative.
        reset_child.write_text(
            "services:\n"
            "  reset-svc:\n"
            "    extends:\n"
            "      file: base.yml\n"
            "      service: base-svc\n"
            "    volumes: !reset []\n"
            "    command: !reset [--model, m]\n",
            encoding="utf-8",
        )
        # (c) The real #377 mode: a compose that simply STOPPED extending
        #     its template-bearing base — no extends: at all. Must NOT be
        #     reported covered. (Deterministic everywhere; no docker / tags.)
        noextend.write_text(
            "services:\n"
            "  noextend-svc:\n"
            "    image: scratch\n"
            "    command: [--model, m]\n",
            encoding="utf-8",
        )
        if not pa.reaches(root, ctp, str(keep_child)):
            errors.append(
                "chat_template real-merge: a child that inherits the base "
                "(extends:, no override) lost coverage — extends not merged"
            )
        if pa.reaches(root, ctp, str(reset_child)):
            errors.append(
                "chat_template real-merge FALSE-NEGATIVE: a child that "
                "!reset-removed the chat-template mount/--chat-template "
                "wiring was still reported covered (the #377 dangerous "
                "direction — extends resolved by text concat, not real "
                "Docker Compose merge)"
            )
        if pa.reaches(root, ctp, str(noextend)):
            errors.append(
                "chat_template real-merge FALSE-NEGATIVE: a compose that "
                "STOPPED extending its template-bearing base was still "
                "reported covered (the literal #377 silent-drift mode)"
            )

# (5) Rig-independent leak-assertion convention (mandatory — the V2 on-rig
#     lesson). The chat_template effective-coverage path resolves the
#     merged compose via `docker compose config`, which renders the mount
#     SOURCE as an ABSOLUTE host path (e.g. /opt/ai/.../patches/...jinja).
#     That absolute path MUST NOT leak into any committed/shared artifact:
#     the patches.yml `jinja`/`mounted_at` must be repo-relative/container
#     paths, NEVER absolute. Assert with `str(abs_dir) not in shared` AND
#     that only the repo-relative form appears — NEVER a bare `/opt|/home`
#     substring allowlist (a sandbox path structurally defeats that; that
#     exact miss shipped a real leak in V2).
abs_root = str(root.resolve())
for patch in chat_template_patches:
    spec = patch.get("delivery_spec") or {}
    jinja_rel = spec.get("jinja") or ""
    mounted = spec.get("mounted_at") or ""
    shared = f"{jinja_rel}\n{mounted}\n{patch.get('id','')}"
    if abs_root in shared:
        errors.append(
            f"{patch['id']} chat_template delivery_spec LEAKS the absolute "
            f"repo path ({abs_root!r}) — must be repo-relative/container only"
        )
    if jinja_rel.startswith("/") or jinja_rel.startswith(abs_root):
        errors.append(
            f"{patch['id']} chat_template jinja must be repo-relative, "
            f"got absolute: {jinja_rel}"
        )
    # The repo-relative form (the ONLY acceptable shape) must be present.
    if jinja_rel and not jinja_rel.startswith("models/"):
        errors.append(
            f"{patch['id']} chat_template jinja must be a repo-relative "
            f"models/... path, got: {jinja_rel}"
        )

# Every load-bearing chat_template compose's SHIPPED mount line must use
# the repo-relative `../../patches/...` form, never the absolute path the
# `docker compose config` merge renders (the merge output is internal to
# reaches() and is NEVER emitted/shared — assert that invariant on the
# committed composes directly, rig-independently).
for patch in chat_template_patches:
    for lb in patch.get("load_bearing_when") or []:
        for compose_name in lb.get("composes") or []:
            if compose_name not in COMPOSE_REGISTRY:
                continue
            cpath = root / COMPOSE_REGISTRY[compose_name]["compose_path"]
            ctext = cpath.read_text(encoding="utf-8")
            if abs_root in ctext:
                errors.append(
                    f"committed compose {compose_name} LEAKS the absolute "
                    f"repo path {abs_root!r} (must use the repo-relative "
                    f"../../patches/... mount form)"
                )

arch_allowed_keys = pa.ARCH_ALLOWED_KEYS
arch_required = pa.ARCH_REQUIRED_KEYS
valid_trc = pa.VALID_TRC
valid_arch_status = pa.VALID_ARCH_STATUS
valid_confidence = pa.VALID_CONFIDENCE


def c0_state(row: dict, tp: int, trust_ack: bool = False) -> str:
    return pa.c0_state(row, tp, trust_ack)


for row in arches:
    arch = row.get("arch", "<missing>")
    unknown = set(row) - arch_allowed_keys
    missing = arch_required - set(row)
    if unknown:
        errors.append(f"arch {arch} has unknown keys: {sorted(unknown)}")
    if missing:
        errors.append(f"arch {arch} missing keys: {sorted(missing)}")
        continue
    if row["status"] not in valid_arch_status:
        errors.append(f"arch {arch} invalid status {row['status']!r}")
    if row["confidence"] not in valid_confidence:
        errors.append(f"arch {arch} invalid confidence {row['confidence']!r}")
    trc = row["requires_trust_remote_code"]
    if trc not in valid_trc:
        errors.append(f"arch {arch} invalid requires_trust_remote_code={trc!r}")
    evidence = row.get("requires_trust_remote_code_evidence")
    if not evidence:
        errors.append(f"arch {arch} missing requires_trust_remote_code_evidence")
    if trc == "unverified" and evidence != "none":
        errors.append(f"arch {arch} has unverified trust_remote_code but evidence is not none")
    if trc in {"true", "false"} and evidence == "none":
        errors.append(f"arch {arch} has {trc} trust_remote_code without real evidence")
    for pid in row.get("required_patches") or []:
        if pid not in patch_ids:
            errors.append(f"arch {arch} references unknown patch id {pid}")
    valid_tp = row.get("valid_tp") or {}
    divisors = valid_tp.get("tp_divisors")
    if not isinstance(divisors, list) or not divisors or not all(isinstance(tp, int) for tp in divisors):
        errors.append(f"arch {arch} valid_tp.tp_divisors must be a non-empty integer list")
        continue
    if not isinstance(valid_tp.get("marlin_alignment_required"), bool):
        errors.append(f"arch {arch} valid_tp.marlin_alignment_required must be boolean")
    if valid_tp.get("moe_layout") not in {"dense", "moe"}:
        errors.append(f"arch {arch} valid_tp.moe_layout must be dense|moe")
    states = {c0_state(row, tp) for tp in divisors}
    if len(states) != 1:
        errors.append(f"arch {arch} C0 declared TP states not singular: {sorted(states)}")
    negative_tp = max(divisors) + 1
    if c0_state({**row, "requires_trust_remote_code": "false"}, negative_tp) != "engine-support-unknown":
        errors.append(f"arch {arch} negative TP did not resolve engine-support-unknown")
    if trc == "unverified" and c0_state(row, divisors[0]) != "needs-trust-remote-code-ack":
        errors.append(f"arch {arch} unverified trust_remote_code did not fail closed")

seed_required = pa.SEED_REQUIRED_KEYS
for seed in seeds:
    label = f"{seed.get('model', '<missing>')}:{seed.get('kv_format', '<missing>')}:{seed.get('selected_ctx', '<missing>')}"
    missing = seed_required - set(seed)
    if missing:
        errors.append(f"seed {label} missing keys: {sorted(missing)}")
        continue
    if seed["provenance"] != "seed-from-measured-corpus":
        errors.append(f"seed {label} has invalid provenance {seed['provenance']!r}")
    if seed["confidence"] == "exact" and not seed["source"].startswith("BENCHMARKS.md#"):
        errors.append(f"seed {label} exact confidence lacks BENCHMARKS source")
    source_file = seed["source"].split("#", 1)[0]
    if not (root / source_file).exists():
        errors.append(f"seed {label} source file missing: {source_file}")
    measured = seed["measured"] or {}
    for key in ("vram_mib_per_card", "tps_short", "tps_loaded_ctx", "soak_continuous"):
        if key not in measured:
            errors.append(f"seed {label} measured.{key} missing")
    if measured.get("soak_continuous") not in {"pass", "fail", "not-run"}:
        errors.append(f"seed {label} invalid soak_continuous {measured.get('soak_continuous')!r}")
    smoked = set(seed.get("smoked_capabilities") or [])
    unsmoked = set(seed.get("unsmoked_capabilities") or [])
    if smoked & unsmoked:
        errors.append(f"seed {label} capabilities appear in both smoked and unsmoked: {sorted(smoked & unsmoked)}")
    if "tool-call-stream" in smoked:
        errors.append(f"seed {label} claims tool-call-stream smoked; #145 guard forbids that without explicit working-source evidence")

if known_gaps:
    print("[patch-attribution] known delivery gaps:")
    for gap in sorted(known_gaps):
        print(f"  - {gap}")

if errors:
    print("[patch-attribution] FAIL")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)

print(f"[patch-attribution] PASS: {len(patches)} patch entries, {len(arches)} arch rows, {len(seeds)} calibration seeds")
PY
