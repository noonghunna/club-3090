#!/usr/bin/env python3
"""The weights folders a slug mounts — show them, and remove them safely (#1814).

Run through ``scripts/weights.sh`` (it loads MODEL_DIR from the settings).

    weights.sh show   <slug> [--json]
    weights.sh remove <slug> [--dry-run] [--yes --digest D] [--allow-shared] [--json]

Nothing here re-implements a rule another part of the repo owns. Three resolvers
answer the three questions, the same ones a launch asks:

  * which slugs exist, and their composes  → compose_registry.get_registry()
                                             (core + local layer, EVERY status)
  * which paths a compose mounts           → preflight_compose_weight_paths
                                             (scripts/preflight.sh)
  * which recipe a mounted path belongs to → weights.lookup_many()
                                             (scripts/lib/profiles/weights.py)

Removal is safe by construction. A removal is refused, and nothing at all is
deleted, when any of these holds:

  shared      another slug (deprecated and non-functional ones count) needs a
              file that would go. Overridable with --allow-shared, never silently.
  container   a running container mounts the folder (or mounts a parent of it and
              is a slug that uses it, or can't be identified). Not overridable.
  escape      a path leaves the model directories, a recipe path is absolute or
              has `..`, or a symlink sits below the folder's own link. Not
              overridable.
  symlink     a recipe file is itself a symlink (removing it frees nothing and
              its target may serve something else). Not overridable.
  no-files    the recipe declares no `files:` — it IS the folder, and this tool
              deletes a recipe's files, never a whole folder. Not overridable.
  docker      the running-container check could not be made. Not overridable.

What goes is ONLY the recipe's `files:` (plus hf's own bookkeeping for exactly
those files), compared by RESOLVED path — variant names lie, paths don't. A
folder linked from MODEL_DIR onto another disk loses the files at the target,
and the link goes only once nothing is left behind it.

The model directories are MODEL_DIR plus the target of each symlink that sits
directly in MODEL_DIR (one hop: "this folder lives on another disk"). A target
that is "/" or contains MODEL_DIR is not a model directory.

Exit codes: 0 ok · 1 error · 3 refused, another slug needs it (--allow-shared)
· 4 refused, not overridable · 5 not confirmed (--yes, or answer the prompt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_LIB = Path(__file__).resolve().parent
for _p in (_LIB / "profiles", _LIB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import compose_registry  # noqa: E402
import weights as weights_reader  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_SHARED, EXIT_REFUSED, EXIT_UNCONFIRMED = 0, 1, 3, 4, 5
FILE_KNOBS = ("GGUF_FILE", "DRAFT_FILE", "MMPROJ_FILE")
# hf --local-dir bookkeeping for one file: <subdir>/.cache/huggingface/download/<f>…
HF_STAGE = Path(".cache") / "huggingface" / "download"
# Bookkeeping of a whole recipe folder, not of one file: setup.sh's hash marker and
# hf --local-dir's cache tags. They go only when NOTHING else would be left in the
# recipe's own local_subdir — otherwise the emptied folder lingers, and c3 reads a
# folder with no weights in it as an interrupted (PARTIAL) download, not an absent one.
FOLDER_BOOKKEEPING = (".setup-verified.tsv", ".cache/huggingface/CACHEDIR.TAG", ".cache/huggingface/.gitignore")
# hf --local-dir also keeps the repo's file listing per revision here (newer huggingface_hub:
# .cache/huggingface/trees/<revision>.json). Missing from the list above, it alone kept a
# removed recipe's folder on disk, so c3 went on showing its slugs as a partial download.
HF_TREES = Path(".cache") / "huggingface" / "trees"
GIB = 1024 ** 3


def _folder_bookkeeping(subdir: str, f: str) -> bool:
    """Is ``f`` (a path under the recipe folder ``subdir``) bookkeeping of the folder itself?"""
    rel = os.path.relpath(f, subdir)
    if rel in FOLDER_BOOKKEEPING:
        return True
    p = Path(rel)
    return p.parent == HF_TREES and p.suffix == ".json"


class WeightsError(Exception):
    """A failure that means 'can't answer', reported with exit 1."""


# ── the index: every slug's mounts, mapped to recipes ────────────────────────────


@dataclass
class Mount:
    slug: str
    cls: str                    # gguf | draft | mmproj | hf | volume | command
    rel: str                    # mounted path, relative to MODEL_DIR
    recipe: Optional[dict]      # weights.py recipe (WEIGHT_* keys), None = no recipe
    meta: dict                  # the raw weights: entry
    override: str = ""          # "GGUF_FILE=…" when found through a saved override

    @property
    def key(self) -> str:
        return self.recipe["WEIGHT_KEY"] if self.recipe else ""

    @property
    def subdir(self) -> str:
        return (self.recipe or {}).get("WEIGHT_SUBDIR", "").strip("/")

    @property
    def files(self) -> list[str]:
        raw = self.meta.get("files") or []
        return [raw] if isinstance(raw, str) else [str(f) for f in raw]

    @property
    def folder_rel(self) -> str:
        """The folder shown for this mount: the recipe's own dir (`path`, which is
        where its files live, e.g. bucket/UD-IQ4_XS), else what was mounted."""
        if self.recipe:
            return str(self.meta.get("path") or self.subdir).strip("/")
        if self.cls in ("gguf", "draft", "mmproj"):
            return os.path.dirname(self.rel)
        return self.rel.strip("/")

    @property
    def role(self) -> str:
        kind = str(self.meta.get("kind") or "")
        if kind in ("draft", "mmproj") or self.cls in ("draft", "mmproj"):
            return "mmproj" if "mmproj" in (kind, self.cls) else "draft"
        return "main"


@dataclass
class Index:
    root: Path
    model_dir: Path
    registry: dict
    mounts: list[Mount] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def of(self, slug: str) -> list[Mount]:
        return [m for m in self.mounts if m.slug == slug]


def _emit_mounts(root: Path, jobs: list[tuple[str, str, dict[str, str]]]) -> list[tuple[int, str, str]]:
    """Run preflight_compose_weight_paths over every job in ONE bash process.

    jobs: [(job id, compose path, {knob: value})]. The file knobs are CLEARED for
    every job and set only to the job's own values — an exported GGUF_FILE would
    otherwise be applied to every compose in the catalog.
    """
    script = r'''
source "$1/scripts/preflight.sh" >/dev/null 2>&1 || exit 97
while IFS=$'\t' read -r id compose g d m; do
  [[ -f "$compose" ]] || { printf '%s\t!missing\t%s\n' "$id" "$compose"; continue; }
  while IFS=$'\t' read -r cls rel; do
    printf '%s\t%s\t%s\n' "$id" "$cls" "$rel"
  done < <(GGUF_FILE="$g" DRAFT_FILE="$d" MMPROJ_FILE="$m" preflight_compose_weight_paths "$compose" 2>/dev/null)
done
'''
    env = {k: v for k, v in os.environ.items() if k not in FILE_KNOBS}
    env["PYTHONUTF8"] = env.get("PYTHONUTF8", "1")
    feed = "".join(
        f"{jid}\t{compose}\t{knobs.get('GGUF_FILE', '')}\t{knobs.get('DRAFT_FILE', '')}"
        f"\t{knobs.get('MMPROJ_FILE', '')}\n"
        for jid, compose, knobs in jobs
    )
    try:
        res = subprocess.run(
            ["bash", "-c", script, "_", str(root)], input=feed, capture_output=True,
            encoding="utf-8", errors="replace", env=env, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WeightsError(f"reading the composes failed: {exc}") from None
    if res.returncode == 97:
        raise WeightsError(f"could not source {root}/scripts/preflight.sh")
    out = []
    for line in res.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            out.append((int(parts[0]), parts[1], parts[2]))
    return out


def build_index(root: Path, model_dir: Path, *, global_knobs: Optional[dict[str, str]] = None) -> Index:
    try:
        registry = compose_registry.get_registry()
    except Exception as exc:  # LocalRegistryError / RegistryDataError
        raise WeightsError(f"cannot load the registry: {exc}") from None
    idx = Index(root=root, model_dir=model_dir, registry=registry)

    saved: dict[str, dict[str, str]] = {}
    try:
        import slug_settings  # noqa: PLC0415
        saved = {s: dict(v) for s, v in slug_settings.read().slugs.items()}
    except Exception as exc:  # a broken store must not hide a user
        idx.warnings.append(f"per-slug settings unreadable ({exc}); their file overrides are not counted")

    # Job 0..n: every slug at its compose defaults. Then one extra job per slug whose
    # saved (or exported/global) file knobs point somewhere else — counted as users
    # TOO, never instead: the default files stay "needed" by that slug.
    jobs: list[tuple[str, str, dict[str, str]]] = []
    job_meta: list[tuple[str, str]] = []
    for slug, entry in sorted(registry.items()):
        cp = str(entry.get("compose_path") or "")
        if not cp:
            idx.warnings.append(f"{slug}: no compose_path in the registry")
            continue
        compose = cp if os.path.isabs(cp) else str(root / cp)
        jobs.append((str(len(jobs)), compose, {}))
        job_meta.append((slug, ""))
        knobs = {k: v for k, v in (global_knobs or {}).items() if v}
        knobs.update({k: v for k, v in saved.get(slug, {}).items() if k in FILE_KNOBS and v})
        if knobs:
            jobs.append((str(len(jobs)), compose, knobs))
            job_meta.append((slug, " ".join(f"{k}={v}" for k, v in sorted(knobs.items()))))

    raw = _emit_mounts(root, jobs)
    seen: set[tuple[str, str, str]] = set()
    pending: list[tuple[str, str, str, str]] = []
    for jid, cls, rel in raw:
        slug, override = job_meta[jid]
        if cls == "!missing":
            idx.warnings.append(f"{slug}: compose not found: {rel}")
            continue
        rel = rel.strip()
        if rel.startswith("/models/"):
            rel = rel[len("/models/"):]
        if not rel or (slug, cls, rel) in seen:
            continue
        seen.add((slug, cls, rel))
        pending.append((slug, cls, rel, override))

    lookups = weights_reader.lookup_many(
        [(rel, str(registry[slug].get("model") or "") or None) for slug, _c, rel, _o in pending]
    )
    for (slug, cls, rel, override), hit in zip(pending, lookups):
        recipe, meta = hit if hit else (None, {})
        idx.mounts.append(Mount(slug=slug, cls=cls, rel=rel, recipe=recipe, meta=meta, override=override))
    return idx


# ── paths ────────────────────────────────────────────────────────────────────────


def _real(p: os.PathLike | str) -> str:
    return os.path.realpath(p)


def _under(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def model_roots(model_dir: Path) -> list[str]:
    """MODEL_DIR, plus each directory a symlink directly IN it points to (one hop)."""
    md_real = _real(model_dir)
    roots = [md_real]
    try:
        entries = list(os.scandir(model_dir))
    except OSError:
        return roots
    for e in entries:
        if not e.is_symlink():
            continue
        tgt = _real(e.path)
        if not os.path.isdir(tgt) or tgt == "/" or _under(md_real, tgt):
            continue
        roots.append(tgt)
    return roots


def _claims(m: Mount, model_dir: Path) -> list[str]:
    """Resolved paths this mount needs; each covers itself and everything below it."""
    out = [_real(model_dir / m.rel)]
    if m.recipe:
        base = model_dir / m.subdir
        files = m.files
        out += [_real(base / f) for f in files] if files else [_real(base)]
    return out


def _covers(claim: str, target: str) -> bool:
    return _under(target, claim)


def _intersects(a: str, b: str) -> bool:
    return _under(a, b) or _under(b, a)


def _du(path: Path) -> tuple[int, int]:
    """(on-disk bytes, file count) under ``path``, hardlinks once, no symlink follow."""
    total, count, seen = 0, 0, set()
    if path.is_file():
        st = path.stat()
        return st.st_blocks * 512, 1
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_blocks * 512
            count += 1
    return total, count


def _shard_scan_missing(root: Path, folder: Path) -> Optional[list[str]]:
    """preflight's #1042 shard scan on one folder → the absent weight files (None = couldn't run)."""
    try:
        res = subprocess.run(
            ["bash", "-c", 'source "$1/scripts/preflight.sh" >/dev/null 2>&1 && _preflight_shard_scan "$2"',
             "_", str(root), str(folder)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    return [ln.split(":", 1)[-1] for ln in res.stdout.splitlines() if ln.strip()]


# ── show ─────────────────────────────────────────────────────────────────────────


def _status(idx: Index, slug: str) -> str:
    return str((idx.registry.get(slug) or {}).get("status") or "")


def _folder_users(idx: Index, slug: str, folder_real: str, needed: list[str]) -> list[dict]:
    """Other slugs with weights in this folder; needs_files = one of `needed` is theirs."""
    users: dict[str, dict] = {}
    for m in idx.mounts:
        if m.slug == slug:
            continue
        claims = _claims(m, idx.model_dir)
        if not any(_intersects(c, folder_real) for c in claims):
            continue
        u = users.setdefault(m.slug, {"slug": m.slug, "status": _status(idx, m.slug), "needs_files": False})
        if any(_covers(c, t) for c in claims for t in needed):
            u["needs_files"] = True
    return sorted(users.values(), key=lambda u: (not u["needs_files"], u["slug"]))


def show(idx: Index, slug: str) -> dict:
    if slug not in idx.registry:
        raise WeightsError(f"unknown slug: {slug}")
    folders: dict[str, dict] = {}
    for m in idx.of(slug):
        path = idx.model_dir / m.folder_rel
        real = _real(path)
        f = folders.get(real)
        if f is None:
            f = folders[real] = {
                "path": str(path), "resolved": real,
                "linked": real != os.path.normpath(str(path)),
                "role": m.role, "recipes": [], "mounts": [], "_needed": [],
            }
        if m.role == "main":
            f["role"] = "main"
        if m.key and m.key not in f["recipes"]:
            f["recipes"].append(m.key)
        f["mounts"].append({"class": m.cls, "rel": m.rel, **({"override": m.override} if m.override else {})})
        if m.recipe and m.files:
            base = idx.model_dir / m.subdir
            f["_needed"] += [_real(base / x) for x in m.files]
        f.setdefault("_m", []).append(m)
    out = []
    for real, f in folders.items():
        ms: list[Mount] = f.pop("_m")
        needed = f.pop("_needed")
        state, note = _folder_state(idx, ms, Path(real))
        size, count = _du(Path(real)) if os.path.exists(real) else (0, 0)
        f.update({"state": state, "size_bytes": size, "file_count": count,
                  "users": _folder_users(idx, slug, real, needed or [real])})
        # A recipe's own files can be a sliver of the folder they sit in (a vision
        # projector in a bucket of quants): say both, or "75 GiB" reads as its size.
        if needed:
            f["recipe_bytes"] = sum(_du(Path(n))[0] for n in set(needed) if os.path.isfile(n))
        if note:
            f["note"] = note
        out.append(f)
    out.sort(key=lambda f: ({"main": 0, "draft": 1, "mmproj": 2}.get(f["role"], 3), f["path"]))
    return {"slug": slug, "status": _status(idx, slug), "model_dir": str(idx.model_dir),
            "folders": out, "warnings": idx.warnings}


def _folder_state(idx: Index, ms: list[Mount], real: Path) -> tuple[str, str]:
    """complete / incomplete / missing / present (present = no recipe to check against)."""
    with_files = [m for m in ms if m.recipe and m.files]
    if with_files:
        paths = {idx.model_dir / m.subdir / x for m in with_files for x in m.files}
        have = sum(1 for p in paths if p.is_file())
        if have == len(paths):
            return "complete", ""
        if have == 0:
            return "missing", ""
        return "incomplete", f"{len(paths) - have} of {len(paths)} recipe files absent"
    if not real.exists():
        return "missing", ""
    if any(m.recipe for m in ms):
        m = next(m for m in ms if m.recipe)
        glob = m.recipe.get("WEIGHT_VERIFY_GLOB") or "*.safetensors"
        if not any((idx.model_dir / m.subdir).glob(glob)):
            return "incomplete", f"no {glob} under the folder"
        absent = _shard_scan_missing(idx.root, real) if real.is_dir() else []
        if absent:
            return "incomplete", f"{len(absent)} weight file(s) named by the index are absent"
        return "complete", ""
    return "present", "no recipe — completeness unknown"


# ── remove: the plan ─────────────────────────────────────────────────────────────


@dataclass
class Plan:
    slug: str
    targets: list[dict] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)       # emptied dirs that go too (rmdir)
    links: list[dict] = field(default_factory=list)
    refusals: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)

    def refuse(self, code: str, message: str, *, folder: str = "", slugs=None) -> None:
        r = {"code": code, "message": message, "overridable": code == "shared"}
        if folder:
            r["folder"] = folder
        if slugs:
            r["slugs"] = slugs
        self.refusals.append(r)

    def digest(self) -> str:
        body = json.dumps(
            {"t": sorted((t["path"], t["resolved"], t["bytes"], t["ino"]) for t in self.targets),
             "d": sorted(self.dirs), "l": sorted(lk["path"] for lk in self.links)}, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    def exit_code(self, allow_shared: bool) -> int:
        hard = [r for r in self.refusals if not r["overridable"]]
        if hard:
            return EXIT_REFUSED
        if self.refusals and not allow_shared:
            return EXIT_SHARED
        return EXIT_OK


def _rel_ok(rel: str) -> bool:
    p = Path(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts


def _link_component(model_dir: Path, rel: str) -> tuple[Optional[Path], Optional[Path]]:
    """(the folder's link directly in MODEL_DIR or None, the first symlink BELOW it or None)."""
    top: Optional[Path] = None
    cur = model_dir
    for i, part in enumerate(Path(rel).parts[:-1]):
        cur = cur / part
        if cur.is_symlink():
            if i == 0:
                top = cur
            else:
                return top, cur
    return top, None


def _hf_bookkeeping(base: Path, f: str) -> list[Path]:
    stage = base / HF_STAGE
    out = [stage / f"{f}.metadata", stage / f"{f}.lock", stage / f"{f}.incomplete", base / f"{f}.incomplete"]
    parent = (stage / f).parent
    if parent.is_dir():
        out += sorted(parent.glob(Path(f).name + ".*.incomplete"))
    return out


def plan_remove(idx: Index, slug: str, *, docker: str = "docker") -> Plan:
    if slug not in idx.registry:
        raise WeightsError(f"unknown slug: {slug}")
    plan = Plan(slug=slug)
    model_dir = idx.model_dir
    roots = model_roots(model_dir)
    entry = idx.registry[slug]
    others = [m for m in idx.mounts if m.slug != slug]
    other_claims = [(m, _claims(m, model_dir)) for m in others]

    mine = idx.of(slug)
    if not mine:
        plan.notes.append("this slug's compose mounts no weights folder under MODEL_DIR")
    done_keys: set[str] = set()
    touched: dict[str, str] = {}          # resolved folder → as configured
    top_links: dict[str, Path] = {}
    shared: dict[str, dict] = {}          # folder → {users, files}: ONE refusal per folder
    subdirs: dict[str, tuple[str, str]] = {}   # resolved recipe subdir → (folder shown, recipe)
    for m in mine:
        folder = model_dir / m.folder_rel
        if not m.recipe:
            plan.notes.append(f"{m.rel}: no weights recipe — left alone (only a recipe's files are removed)")
            continue
        if m.key in done_keys:
            continue
        done_keys.add(m.key)
        touched[_real(folder)] = str(folder)
        # Variant names lie: the path's recipe must be one the slug's registry row names.
        model, variant = m.key.split(":", 1)
        expected = {f"{entry.get('model')}:{entry.get('weights_variant')}"}
        for c in entry.get("weights_companions") or []:
            expected.add(c if ":" in str(c) else f"{entry.get('model')}:{c}")
        if m.role == "main" and m.key not in expected:
            plan.refuse("mismatch",
                        f"{m.rel} resolves to recipe {m.key}, but the registry row says "
                        f"{entry.get('model')}:{entry.get('weights_variant')} — refusing to guess which is right",
                        folder=str(folder))
            continue
        if not m.files:
            plan.refuse("no-files",
                        f"recipe {m.key} declares no files: — it is the whole folder {folder}, and this "
                        "tool removes a recipe's files, never a folder. Remove it by hand if you mean to.",
                        folder=str(folder))
            continue
        if not _rel_ok(m.subdir):
            plan.refuse("escape", f"recipe {m.key}: local_subdir {m.subdir!r} is not a plain relative path",
                        folder=str(folder))
            continue
        base = model_dir / m.subdir
        subdirs.setdefault(_real(base), (str(folder), m.key))
        for f in m.files:
            if not _rel_ok(f):
                plan.refuse("escape", f"recipe {m.key}: file {f!r} is not a plain relative path", folder=str(folder))
                continue
            raw = base / f
            for extra_i, p in enumerate([raw] + _hf_bookkeeping(base, f)):
                bookkeeping = extra_i > 0
                rel = os.path.relpath(p, model_dir)
                top, nested = _link_component(model_dir, rel)
                if nested is not None:
                    plan.refuse("escape", f"{p}: goes through a symlink below the folder ({nested}) — not followed",
                                folder=str(folder))
                    continue
                if not os.path.lexists(p):
                    continue
                st = os.lstat(p)
                if stat.S_ISLNK(st.st_mode):
                    plan.refuse("symlink", f"{p} is a symlink (→ {os.readlink(p)}); removing it frees nothing "
                                "and its target may serve something else", folder=str(folder))
                    continue
                if not stat.S_ISREG(st.st_mode):
                    plan.refuse("escape", f"{p} is not a regular file", folder=str(folder))
                    continue
                real = _real(p)
                if not any(_under(real, r) for r in roots):
                    plan.refuse("escape", f"{p} resolves to {real}, outside the model directories "
                                f"({', '.join(roots)})", folder=str(folder))
                    continue
                if top is not None:
                    top_links[str(top)] = top
                users = sorted({om.slug for om, cl in other_claims if any(_covers(c, real) for c in cl)})
                if users and not bookkeeping:
                    sb = shared.setdefault(str(folder), {"users": set(), "files": 0})
                    sb["users"].update(users)
                    sb["files"] += 1
                plan.targets.append({
                    "path": str(p), "resolved": real, "folder": str(folder), "recipe": m.key,
                    "bytes": st.st_blocks * 512, "freed_bytes": st.st_blocks * 512 if st.st_nlink == 1 else 0,
                    "ino": st.st_ino, "bookkeeping": bookkeeping,
                    **({"note": f"{st.st_nlink} hardlinks — frees nothing until the others go"}
                       if st.st_nlink > 1 else {}),
                })
    for folder, sb in shared.items():
        plan.refuse("shared", f"{sb['files']} file(s) in {folder} are needed by "
                    f"{len(sb['users'])} other slug(s)", folder=folder,
                    slugs=[{"slug": u, "status": _status(idx, u)} for u in sorted(sb["users"])])
    plan.folders = sorted(set(touched.values()))

    # A recipe folder left with nothing but bookkeeping once the planned files go is
    # cleared whole: its bookkeeping files and its emptied dirs. This also runs when the
    # weights are ALREADY gone (an earlier remove, or a hand delete) — such a folder
    # still exists, so c3 shows its slugs as a partial download and offers this very
    # remove, which must then clear it rather than find nothing to do.
    going = {t["resolved"] for t in plan.targets}
    cleared: set[str] = set()
    for sd, (folder, key) in sorted(subdirs.items()):
        if not any(_under(sd, r) for r in roots) or sd in roots:
            continue        # never MODEL_DIR / a model root itself, never outside one
        if not os.path.isdir(sd) or os.path.islink(sd):
            continue
        left = [f for f in _regular_and_links_under(sd) if f not in going]
        if not all(_folder_bookkeeping(sd, f) for f in left):
            continue
        cleared.add(sd)
        for f in sorted(left):
            st = os.lstat(f)
            if not stat.S_ISREG(st.st_mode):
                continue
            plan.targets.append({
                "path": f, "resolved": f, "folder": folder, "recipe": key,
                "bytes": st.st_blocks * 512, "freed_bytes": st.st_blocks * 512 if st.st_nlink == 1 else 0,
                "ino": st.st_ino, "bookkeeping": True,
            })
    going = {t["resolved"] for t in plan.targets}
    plan.dirs = _emptied_dirs(going, set(subdirs), set(roots), cleared)
    if not plan.refusals and not any(not t["bookkeeping"] for t in plan.targets):
        if plan.targets or plan.dirs:
            plan.notes.append("the weights are already gone; only the download's bookkeeping and empty "
                              "folders are left, which keep the folder on disk (c3 shows such a folder "
                              "as a partial download)")
        else:
            plan.notes.append("nothing of this slug's recipes is on disk")
    # A folder link goes only when nothing is left behind it (hf bookkeeping aside).
    for top in top_links.values():
        tgt = _real(top)
        left = _files_left(tgt, going)
        if left:
            plan.notes.append(f"{top} → {tgt} stays: {len(left)} other file(s) live behind it")
        else:
            plan.links.append({"path": str(top), "target": tgt})

    _check_containers(idx, plan, touched, docker=docker)
    return plan


def _emptied_dirs(going: set[str], subdirs: set[str], roots: set[str],
                  cleared: set[str] = frozenset()) -> list[str]:
    """Dirs that hold nothing once ``going`` is gone, deepest first — inside a recipe's
    own subdir only (never the subdir's parent, MODEL_DIR or a model root).

    Left in place, an emptied quant dir reads to c3 as a PARTIAL download ("subdir
    exists but no verify_glob match") instead of absent. rmdir only ever removes an
    empty dir, so this can't take a file with it.

    ``cleared`` are recipe folders being emptied whole: every dir under one is a
    candidate too, so dirs that were already empty (hf's ``download/`` once its records
    went, a quant dir emptied by hand) go with it, not only the ones a planned file sits in.
    """
    cands: set[str] = set()
    for t in going:
        d = os.path.dirname(t)
        while any(_under(d, s) for s in subdirs) and d not in roots:
            cands.add(d)
            d = os.path.dirname(d)
    for sd in cleared:
        for dirpath, dirnames, _files in os.walk(sd, followlinks=False):
            cands.add(dirpath)
            for n in dirnames:
                p = os.path.join(dirpath, n)
                if os.path.islink(p):      # a link is an entry, never a dir to remove
                    continue
                cands.add(p)
    cands = {d for d in cands if d not in roots and not os.path.islink(d)}
    empty: list[str] = []
    for d in sorted(cands, key=lambda x: -x.count("/")):
        try:
            entries = [os.path.join(d, n) for n in os.listdir(d)]
        except OSError:
            continue
        if all((e in going) or (e in empty) for e in entries):
            empty.append(d)
    return empty


def _regular_and_links_under(d: str) -> list[str]:
    """Every non-directory entry under ``d`` (files AND symlinks), no link followed."""
    out = []
    for dirpath, _dirs, files in os.walk(d, followlinks=False):
        out += [os.path.join(dirpath, n) for n in files]
    return out


def _files_left(target: str, going: set[str]) -> list[str]:
    """Files behind a folder link that are NOT about to go (hf's .cache/ aside)."""
    left = []
    for dirpath, _dirs, files in os.walk(target, followlinks=False):
        rel_dir = os.path.relpath(dirpath, target)
        if rel_dir == ".cache" or rel_dir.startswith(".cache" + os.sep):
            continue
        left += [os.path.join(dirpath, n) for n in files if _real(os.path.join(dirpath, n)) not in going]
    return left


def _docker_json(docker: str, args: list[str]) -> Any:
    try:
        res = subprocess.run([docker, *args], capture_output=True, encoding="utf-8",
                             errors="replace", timeout=30)
    except FileNotFoundError:
        raise WeightsError(f"`{docker}` not found") from None
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WeightsError(f"`{docker} {args[0]}` failed: {exc}") from None
    if res.returncode != 0:
        raise WeightsError(f"`{docker} {args[0]}` exited {res.returncode}: {res.stderr.strip()[:200]}")
    return res.stdout


def _check_containers(idx: Index, plan: Plan, touched: dict[str, str], *, docker: str) -> None:
    """Refuse while a running container mounts a folder that would lose files."""
    if not touched or not (plan.targets or plan.links):
        return
    try:
        ids = _docker_json(docker, ["ps", "-q", "--no-trunc"]).split()
        info = json.loads(_docker_json(docker, ["inspect", *ids])) if ids else []
    except (WeightsError, ValueError) as exc:
        plan.refuse("docker", f"can't tell whether a running container uses these weights: {exc}")
        return
    folders = set(touched) | set(touched.values())     # resolved AND as configured
    users = {u for t in plan.targets for u in _target_users(idx, plan.slug, t["resolved"])} | {plan.slug}
    for c in info if isinstance(info, list) else []:
        name = str(c.get("Name") or c.get("Id") or "?").lstrip("/")
        cslug = _container_slug(idx, c)
        for mnt in c.get("Mounts") or []:
            if mnt.get("Type") != "bind":
                continue
            src = str(mnt.get("Source") or "")
            srcs = {os.path.normpath(src), _real(src)} if src else set()
            for folder in folders:
                if any(_under(s, folder) for s in srcs):
                    plan.refuse("container", f"{name} is running and mounts {src} (inside {folder}) — stop it first",
                                folder=folder)
                elif any(_under(folder, s) for s in srcs):
                    if cslug is None:
                        plan.refuse("container", f"{name} is running with {src} mounted, which holds {folder}, "
                                    "and isn't a slug this tool can identify — stop it first", folder=folder)
                    elif cslug in users:
                        plan.refuse("container", f"{name} is running {cslug}, which uses {folder} — stop it first",
                                    folder=folder)


def _target_users(idx: Index, slug: str, real: str) -> set[str]:
    return {m.slug for m in idx.mounts if m.slug != slug and any(_covers(c, real) for c in _claims(m, idx.model_dir))}


def _container_slug(idx: Index, c: dict) -> Optional[str]:
    labels = ((c.get("Config") or {}).get("Labels") or {})
    files = str(labels.get("com.docker.compose.project.config_files") or "")
    for f in [x for x in files.split(",") if x]:
        for slug, entry in idx.registry.items():
            cp = str(entry.get("compose_path") or "")
            if cp and (f == cp or f.endswith("/" + cp.lstrip("/"))):
                return slug
    return None


# ── remove: doing it ─────────────────────────────────────────────────────────────


def execute(plan: Plan) -> tuple[int, int, list[str]]:
    """Delete exactly the planned targets, re-checking each one first."""
    removed, freed, errors = 0, 0, []
    for t in plan.targets:
        p = t["resolved"]
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_ino != t["ino"]:
            errors.append(f"{p} changed since the plan — left alone")
            continue
        try:
            os.unlink(p)
        except OSError as exc:
            errors.append(f"{p}: {exc.strerror or exc}")
            continue
        removed += 1
        freed += t["freed_bytes"]
    for d in plan.dirs:                      # deepest first; rmdir refuses a non-empty dir
        try:
            os.rmdir(d)
        except OSError:
            pass
    for lk in plan.links:
        p = Path(lk["path"])
        if p.is_symlink() and _real(p) == lk["target"]:
            if _files_left(lk["target"], set()):
                errors.append(f"{p}: files appeared behind the link — kept")
                continue
            try:
                os.unlink(p)
            except OSError as exc:
                errors.append(f"{p}: {exc.strerror or exc}")
    return removed, freed, errors


# ── output ───────────────────────────────────────────────────────────────────────


def _gib(n: int) -> str:
    return f"{n / GIB:,.2f} GiB"


def print_show(d: dict) -> None:
    print(f"{d['slug']}  ({d['status'] or '—'})  —  MODEL_DIR={d['model_dir']}")
    if not d["folders"]:
        print("  no weights folders under MODEL_DIR")
    for f in d["folders"]:
        print(f"\n  [{f['role']}] {f['path']}")
        if f["linked"]:
            print(f"    resolved  {f['resolved']}")
        print(f"    state     {f['state']}" + (f"  ({f['note']})" if f.get("note") else ""))
        own = (f"  (this recipe's files: {_gib(f['recipe_bytes'])})"
               if "recipe_bytes" in f and f["recipe_bytes"] < 0.99 * f["size_bytes"] else "")
        print(f"    size      {_gib(f['size_bytes'])} · {f['file_count']} file(s){own}")
        if f["recipes"]:
            print(f"    recipe    {', '.join(f['recipes'])}")
        need = [u for u in f["users"] if u["needs_files"]]
        beside = [u for u in f["users"] if not u["needs_files"]]
        if need:
            print(f"    also used by {len(need)} slug(s): " + ", ".join(
                f"{u['slug']}{' (' + u['status'] + ')' if u['status'] not in ('production', '') else ''}" for u in need))
        if beside:
            print(f"    other weights in this folder: {len(beside)} slug(s): " + ", ".join(u["slug"] for u in beside))
        if not f["users"]:
            print("    used by no other slug")
    for w in d.get("warnings") or []:
        print(f"  note: {w}", file=sys.stderr)


def plan_json(plan: Plan, allow_shared: bool, dry_run: bool) -> dict:
    return {
        "slug": plan.slug, "dry_run": dry_run, "allow_shared": allow_shared,
        "folders": plan.folders, "targets": plan.targets, "dirs": plan.dirs, "links": plan.links,
        "total_bytes": sum(t["bytes"] for t in plan.targets),
        "freed_bytes": sum(t["freed_bytes"] for t in plan.targets),
        "refusals": plan.refusals, "notes": plan.notes, "digest": plan.digest(),
        "exit": plan.exit_code(allow_shared),
    }


def print_plan(d: dict) -> None:
    print(f"weights remove {d['slug']}{'  (dry run — nothing is deleted)' if d['dry_run'] else ''}")
    by_folder: dict[str, list[dict]] = {}
    for t in d["targets"]:
        by_folder.setdefault(t["folder"], []).append(t)
    for folder, ts in by_folder.items():
        real = _real(folder)
        print(f"\n  {folder}" + (f"\n    → {real}" if real != os.path.normpath(folder) else ""))
        for t in ts:
            if t["bookkeeping"]:
                continue
            shown = os.path.relpath(t["path"], folder) if _under(os.path.normpath(t["path"]), folder) else t["path"]
            print(f"    delete  {shown}  {_gib(t['bytes'])}" + (f"  ({t['note']})" if t.get("note") else ""))
        bk = [t for t in ts if t["bookkeeping"]]
        if bk:
            print(f"    delete  {len(bk)} bookkeeping file(s) (hf download records, setup.sh's verify marker)")
    for dd in d.get("dirs") or []:
        print(f"  rmdir   {dd}  (empty once the files go)")
    for lk in d["links"]:
        print(f"  unlink  {lk['path']}  (→ {lk['target']}, empty once the files go)")
    if not d["targets"] and not d["links"] and not d.get("dirs"):
        print("  nothing to delete")
    n = sum(1 for t in d["targets"] if not t["bookkeeping"])
    print(f"\n  total   {n} weight file(s), {_gib(d['total_bytes'])} — frees {_gib(d['freed_bytes'])}")
    for note in d["notes"]:
        print(f"  note    {note}")
    for r in d["refusals"]:
        tag = "REFUSED" if not r["overridable"] else ("override" if d["allow_shared"] else "REFUSED")
        print(f"  {tag:7} [{r['code']}] {r['message']}")
        for s in r.get("slugs") or []:
            print(f"            - {s['slug']}" + (f" ({s['status']})" if s["status"] else ""))
    if d["exit"] == EXIT_SHARED:
        print("  Fix: pass --allow-shared to delete them anyway (those slugs stop working until re-downloaded).")
    print(f"  digest  {d['digest']}")


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="weights.sh", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="repo checkout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="the weights folders a slug mounts")
    s.add_argument("slug")
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("remove", help="remove a slug's recipe files (dry-run first)")
    r.add_argument("slug")
    for p in (s, r):
        # The model dir to look in; default MODEL_DIR (settings), as switch.sh uses.
        p.add_argument("--model-dir", default="")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--yes", action="store_true",
                   help="don't ask — needs --digest, so you've read the dry run first")
    r.add_argument("--allow-shared", action="store_true",
                   help="delete files other slugs also need (they stop working until re-downloaded)")
    r.add_argument("--digest", default="", help="refuse unless the plan matches this dry-run digest")
    r.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = Path(args.root).resolve()
    md = (args.model_dir or os.environ.get("MODEL_DIR") or "").strip() or str(root / "models-cache")
    model_dir = Path(os.path.abspath(md))
    knobs = {k: os.environ.get(k, "") for k in FILE_KNOBS}
    try:
        idx = build_index(root, model_dir, global_knobs=knobs)
        if args.cmd == "show":
            d = show(idx, args.slug)
            print(json.dumps(d, indent=1)) if args.json else print_show(d)
            return EXIT_OK
        plan = plan_remove(idx, args.slug)
    except WeightsError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc)}))
        print(f"[weights] ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    d = plan_json(plan, args.allow_shared, args.dry_run)
    rc = d["exit"]
    if args.digest and args.digest != d["digest"] and rc == EXIT_OK:
        d["refusals"].append({"code": "changed", "overridable": False,
                              "message": f"the plan changed since the dry run (digest {args.digest} → {d['digest']})"})
        d["exit"] = rc = EXIT_REFUSED
    if args.dry_run or rc != EXIT_OK or not (plan.targets or plan.links or plan.dirs):
        print(json.dumps(d, indent=1)) if args.json else print_plan(d)
        return rc
    if args.yes and not args.digest:
        # "Dry-run first" by construction: an unattended delete must name the dry run
        # it was read from. On a terminal (no --yes) the plan is printed above the
        # prompt instead, which is the same dry run, read before answering.
        msg = (f"--yes needs --digest from the dry run you read: bash scripts/weights.sh remove "
               f"{args.slug} --dry-run  (its last line is the digest) — nothing deleted")
        print(json.dumps({**d, "exit": EXIT_UNCONFIRMED, "error": msg}) if args.json else f"[weights] {msg}")
        return EXIT_UNCONFIRMED
    if not args.json:
        print_plan(d)
    if not args.yes:
        if not sys.stdin.isatty():
            msg = "not confirmed: pass --yes (after reading the dry run), or run it in a terminal"
            print(json.dumps({**d, "exit": EXIT_UNCONFIRMED, "error": msg}) if args.json else f"[weights] {msg}")
            return EXIT_UNCONFIRMED
        what = f"{len(plan.targets)} file(s), {_gib(d['total_bytes'])}" + (
            f", and {len(plan.dirs)} folder(s) once empty" if plan.dirs else "")
        ans = input(f"Delete these {what}? Type 'yes': ")
        if ans.strip() != "yes":
            print("[weights] not confirmed — nothing deleted")
            return EXIT_UNCONFIRMED
    removed, freed, errors = execute(plan)
    result = {**d, "dry_run": False, "removed": removed, "freed_bytes_actual": freed, "errors": errors}
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        gone_dirs = sum(1 for dd in plan.dirs if not os.path.lexists(dd))
        print(f"[weights] removed {removed} file(s), freed {_gib(freed)}"
              + (f", {gone_dirs} empty folder(s)" if gone_dirs else "")
              + (f", {len(plan.links)} link(s)" if plan.links else ""))
        for e in errors:
            print(f"[weights] WARN: {e}", file=sys.stderr)
    return EXIT_ERROR if errors else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
