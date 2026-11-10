#!/usr/bin/env python3
"""Render the per-topology slug tables in docs/{SINGLE,DUAL,MULTI}_CARD.md from the registry.

    python3 tools/docs/slug_tables.py          # rewrite the generated blocks in place
    python3 tools/docs/slug_tables.py --check  # exit 1 (with a diff) if any block is stale
    python3 tools/docs/slug_tables.py --pages  # the pages it manages, one repo-relative path per line

Each card page carries one or more marked blocks:

    <!-- BEGIN GENERATED: slug-table dual -->
    ...
    <!-- END GENERATED: slug-table dual -->

The table lists every non-deprecated CORE slug for that topology (the gitignored
local layer is never included), grouped by model: status, max context, a link to
the compose file and a link to the announcement thread that introduced it.

Why generated: the hand-written tables fell a month behind the catalog (disc #498)
while the registry stayed current. Rows come from the registry; the only hand-kept
data here is ANNOUNCEMENTS, which maps slugs to the discussion that announced them.
Stdlib only, like the rest of the launcher path.
"""
from __future__ import annotations

import difflib
import fnmatch
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib" / "profiles"))
import compose_registry as reg  # noqa: E402

DISCUSSIONS = "https://github.com/noonghunna/club-3090/discussions"

# slug glob -> announcement discussion numbers (first = the one linked). First match
# wins, so specific patterns go before general ones. [] = no announcement exists.
# Derived 2026-09-23 from which announcement bodies name each slug (or, for families
# announced together, the family's distinctive tokens).
ANNOUNCEMENTS: list[tuple[str, list[int]]] = [
    # Qwen3.8-27B series: part 1 (#993) max tier + llama.cpp, part 2 (#1024) fast +
    # NVFP4, part 3 (#1076) DFlash2 super/ultra tiers, part 4 (#1245) SGLang.
    ("sgl/qwen38-27b-*", [1245]),
    ("vllm/qwen38-27b-*-super*", [1076]),
    ("vllm/qwen38-27b-*-ultra*", [1076]),
    ("vllm/qwen38-27b-*-fast", [1024]),
    ("vllm/qwen38-27b-*-nvfp4", [1024]),
    ("vllm/qwen38-27b-*-max", [993]),
    ("llamacpp/qwen38-27b-*", [993]),
    # ThinkingCap-Qwen3.8-27B (bottlecapai fine-tune): every Qwen3.8 tier, both engines (#1418).
    ("sgl/thinkingcap38-27b-*", [1418]),
    ("vllm/thinkingcap38-27b-*", [1418]),
    # Six engines announced together (#1700): Strata, Project Maya, bucko-vllm, HyperQwen,
    # ExLlamaV3 and PrismML's llama.cpp.
    ("strata/*", [1700]),
    ("strata-maya/*", [1700]),
    ("bucko-vllm/*", [1700]),
    ("vllm-hyperqwen/*", [1700]),
    ("exllamav3/*", [1700]),
    ("llama-cpp-prism/*", [1700]),
    # CPU expert offload / the moe-cache engine
    ("llamacpp-club3090/glm53-flash-*", [1117]),
    ("llamacpp-club3090/qwen38-flash-next-*", [1117]),
    ("llamacpp-club3090/inkling-small-*", [959]),
    ("llamacpp-club3090/deepseek-flash-vision-*", []),
    ("llamacpp-club3090/deepseek-flash-*", [951]),
    ("llamacpp/deepseek-flash-*", [909]),
    # Other models
    ("vllm/agents-a1-*", [547]),
    ("ik-llama/ornith35b-*", [480]),
    ("ik-llama/ornith9b-*", [478]),
    ("vllm/thinkingcap-*", [749]),
    ("llamacpp/tess-*", [662]),
    ("vllm/tess-*", [662]),
    ("llamacpp/deckard40B-*", [350]),
    ("vllm/gemma-12b-*", [412]),
    ("llamacpp/gemma-12b-*", [412]),
    ("vllm/gemma-31b-dual", [67]),
    ("vllm/nemotron-75b-*", [706]),
    ("llamacpp/hauhaucs-*", [411]),
    # Qwen3.6
    ("vllm/*-nvfp4*", [608]),
    ("vllm/qwen-27b-dual-lmcache", [423]),
    ("vllm/dual", [733]),
    ("vllm/minimal", [733]),
    ("vllm/qwen-27b-dual-fast", [733]),
    ("vllm/qwen-27b-dual-max", [733]),
    ("vllm/qwen-27b-multi-fast", [733]),
]

STATUS = {
    "production": "✅ production",
    "caveats": "⚠️ caveats",
    "experimental": "🧪 experimental",
    "incubating": "🐣 incubating",
    "preview": "👁️ preview",
    "upstream-gated": "⏸️ upstream-gated",
}
STATUS_ORDER = ["production", "caveats", "preview", "experimental", "incubating", "upstream-gated"]

PAGES = {
    "single": ROOT / "docs" / "SINGLE_CARD.md",
    "dual": ROOT / "docs" / "DUAL_CARD.md",
    "multi3": ROOT / "docs" / "MULTI_CARD.md",
    "multi4": ROOT / "docs" / "MULTI_CARD.md",
    "multi8": ROOT / "docs" / "MULTI_CARD.md",
}


def topology_of(entry: dict) -> str:
    return entry["compose_path"].split("/compose/", 1)[1].split("/", 1)[0]


def display_names() -> dict[str, str]:
    names = {}
    for path in (ROOT / "scripts" / "lib" / "profiles" / "models").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        mid = re.search(r"^id:\s*['\"]?([^'\"\n]+)", text, re.M)
        name = re.search(r"^display_name:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", text, re.M)
        if mid and name:
            names[mid.group(1).strip()] = name.group(1).strip()
    return names


def announcement(slug: str) -> list[int]:
    for pattern, numbers in ANNOUNCEMENTS:
        if fnmatch.fnmatchcase(slug, pattern):
            return numbers
    return []


def notes(entry: dict) -> str:
    out = []
    sm = entry.get("required_sm")
    if sm and float(sm) >= 9.0:
        out.append(f"needs sm {float(sm):.1f}+ (not a 3090)")
    if entry.get("host_ram_gb"):
        out.append(f"host RAM ≥ {entry['host_ram_gb']} GB")
    if entry.get("requires_nvlink"):
        out.append("needs NVLink")
    return "; ".join(out)


def curated_defaults(topology: str, models: set[str]) -> set[str]:
    picks = set()
    for model in models:
        try:
            slug = reg.curated_default_target(model, topology)
        except Exception:  # noqa: BLE001 - a resolver hiccup must not break the docs build
            slug = None
        if slug:
            picks.add(slug)
    return picks


def render(topology: str) -> str:
    rows = [
        (slug, e) for slug, e in reg.COMPOSE_REGISTRY.items()
        if e.get("origin", "core") == "core" and e["status"] != "deprecated" and topology_of(e) == topology
    ]
    names = display_names()
    stars = curated_defaults(topology, {e["model"] for _, e in rows})
    rows.sort(key=lambda r: (names.get(r[1]["model"], r[1]["model"]).lower(),
                             STATUS_ORDER.index(r[1]["status"]) if r[1]["status"] in STATUS_ORDER else 99,
                             r[0]))
    lines = [
        "| Model | Slug | Status | Max ctx | Compose | Announced in | Notes |",
        "|---|---|---|--:|---|---|---|",
    ]
    last_model = None
    for slug, e in rows:
        model = names.get(e["model"], e["model"])
        shown = f"**{model}**" if model != last_model else ""
        last_model = model
        star = " ⭐" if slug in stars else ""
        compose = e["compose_path"]
        short = compose.split("/compose/", 1)[1]
        nums = announcement(slug)
        ann = " · ".join(f"[#{n}]({DISCUSSIONS}/{n})" for n in nums) if nums else "—"
        lines.append(
            f"| {shown} | `{slug}`{star} | {STATUS.get(e['status'], e['status'])} | {e['max_ctx']} "
            f"| [{short}](../{compose}) | {ann} | {notes(e)} |"
        )
    lines.append("")
    legend = (" ⭐ = the model's default for this topology (`bash scripts/switch.sh <model>/default`)."
              if stars else "")
    lines.append(f"{len(rows)} slugs.{legend} Generated from the registry by "
                 f"`tools/docs/slug_tables.py`; don't edit by hand.")
    return "\n".join(lines)


BLOCK = re.compile(r"(<!-- BEGIN GENERATED: slug-table (\S+) -->\n)(.*?)(<!-- END GENERATED: slug-table \2 -->)", re.S)


def rebuild(text: str) -> str:
    return BLOCK.sub(lambda m: m.group(1) + render(m.group(2)) + "\n" + m.group(4), text)


def main(argv: list[str]) -> int:
    if "--pages" in argv:
        for path in sorted(set(PAGES.values())):
            print(path.relative_to(ROOT).as_posix())
        return 0
    check = "--check" in argv
    stale = 0
    for path in sorted(set(PAGES.values())):
        old = path.read_text(encoding="utf-8")
        if not BLOCK.search(old):
            print(f"slug_tables: no generated block in {path.relative_to(ROOT)}", file=sys.stderr)
            return 2
        new = rebuild(old)
        if new == old:
            continue
        if check:
            stale += 1
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(True), new.splitlines(True),
                str(path.relative_to(ROOT)), str(path.relative_to(ROOT)) + " (regenerated)"))
        else:
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text(new, encoding="utf-8")
            os.replace(tmp, path)
            print(f"slug_tables: updated {path.relative_to(ROOT)}")
    if check and stale:
        print(f"slug_tables: {stale} page(s) out of date — run python3 tools/docs/slug_tables.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
