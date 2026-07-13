#!/usr/bin/env bash
set -euo pipefail

# test-pullgate-download.sh — v0.8.0 [E] STEP E2 (club-3090 #141 / #147).
#
# Contract test for CONTRACT-3: the HF download stage. The test IS the spec;
# the code is fixed to it. NO live network / GPU / Docker — a recorded-
# fixture fetcher replays snapshot + HF HEAD x-linked-etag responses and a
# real tmp staging tree is exercised so atomic-move + .incomplete cleanup
# are asserted on disk.
#
# Coverage:
#   * download_set == the CONTRACT-3 reconciled union (weights + index +
#     REQUIRED_METADATA incl. vocab.json/merges.txt AND *.jinja).
#   * fetched-set == sized-set: allow_patterns passed to snapshot_download
#     IS literally download_set() (nothing outside the set is fetched);
#     deriver.sized_download_gb sizes exactly that set; gates.c2a_disk
#     consumes the SHARED download_set() (no drift).
#   * SHA pass: every *.safetensors verified vs the HF API
#     siblings[].lfs.sha256 (the deriver-already-fetched _hf_api payload) ==
#     sha256(file) -> ok, atomic move into the CONTRACT-2 host --model dir,
#     .incomplete gone. (E2-fix-2: the verify SOURCE moved off the
#     Xet-redirect-fragile HEAD x-linked-etag onto the API lfs.sha256.)
#   * E2-fix-2 regression: a Xet-backed repo whose redirect-following HEAD
#     returns NO x-linked-etag (head_etag -> None) but whose API publishes
#     siblings[].lfs.sha256 matching the file -> ok=True/sha_verified=True
#     (this exact on-rig scenario, Qwen/Qwen2.5-0.5B-Instruct, FALSE-failed
#     "no-etag" before the fix).
#   * no-etag -> HARD fail (DownloadResult.failure=="no-etag") when a
#     *.safetensors has NO retrievable API lfs.sha256 (genuinely
#     unverifiable), .incomplete tree deleted, NOT skipped (the deliberate
#     opposite of setup.sh:437) — CONTRACT-3 hard-fail-if-unverifiable
#     preserved; only the hash source moved (token kept "no-etag").
#   * sha-mismatch -> failure + .incomplete cleanup, no final dir.
#   * gated-401 mid-fetch -> failure="gated-401" + cleanup, no partials.
#   * E1 dtype header-probe now LIVE: a quantized repo lacking torch_dtype
#     resolves --dtype bfloat16 via the standardized probe-fetcher wiring
#     (was fail-closed in E1).
#   * DEFAULT real fetcher shells out to the `hf` CLI (NOT a
#     huggingface_hub lib-import): resolves hf->huggingface-cli, builds
#     argv with one --include per download_set entry (set == download_set,
#     no extra), --local-dir == staging; subprocess.run MOCKED. The on-rig
#     E5 ModuleNotFoundError regression fix.
#   * NEITHER hf NOR huggingface-cli -> STRUCTURED DownloadResult(failure==
#     "hf-cli-missing") with the canonical actionable message — NOT a
#     raised/​swallowed ModuleNotFoundError.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import deriver as D          # noqa: E402
from scripts.lib.profiles import downloader as DL       # noqa: E402
from scripts.lib import generate_compose as gc          # noqa: E402
from scripts.lib.profiles.einput import EInput          # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# A sharded quantized repo (awq) with the full reconciled metadata spread:
#   weights + index + config/generation_config/tokenizer*/special_tokens_map
#   + vocab.json + merges.txt + a *.jinja chat template + an adapter (must
#   be EXCLUDED) + an unrelated README (must NOT be fetched).
# ---------------------------------------------------------------------------
API = {
    "siblings": [
        {"rfilename": "config.json", "size": 900},
        {"rfilename": "generation_config.json", "size": 120},
        {"rfilename": "tokenizer.json", "size": 1_800_000},
        {"rfilename": "tokenizer_config.json", "size": 4_000},
        {"rfilename": "tokenizer.model", "size": 500_000},
        {"rfilename": "special_tokens_map.json", "size": 300},
        {"rfilename": "vocab.json", "size": 800_000},
        {"rfilename": "merges.txt", "size": 400_000},
        {"rfilename": "chat_template.jinja", "size": 2_000},
        {"rfilename": "model.safetensors.index.json", "size": 50_000},
        {"rfilename": "model-00001-of-00002.safetensors", "size": 6_000_000_000},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 5_000_000_000},
        {"rfilename": "adapter_model.safetensors", "size": 90_000_000},
        {"rfilename": "README.md", "size": 12_000},
    ]
}

# ---------------------------------------------------------------------------
# 1. download_set == the CONTRACT-3 reconciled union.
# ---------------------------------------------------------------------------
ds = D.download_set(API)
expected = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]
check(ds == expected,
      f"download_set == reconciled union (weights+index+REQUIRED_METADATA"
      f"+jinja, adapter+README excluded) got={ds}")
check("adapter_model.safetensors" not in ds,
      "download_set excludes the LoRA/adapter safetensors")
check("README.md" not in ds, "download_set excludes unrelated repo files")
check("chat_template.jinja" in ds and "vocab.json" in ds
      and "merges.txt" in ds,
      "download_set reconciles BOTH v2's *.jinja AND legacy vocab/merges")

# 1b. A dedicated MTP/nextn head (e.g. `mtp_grafted.safetensors`) is a real
# weight the model needs with MTP enabled, but it's neither a `model-*` nor
# `-of-` shard — the shard filter used to DROP it, silently omitting
# Tess-4-27B-FP8's MTP head → broken MTP serving (club-3090 #617). It must now
# be unioned into the weight set.
MTP_API = {"siblings": [{"rfilename": n} for n in (
    "model-00001-of-00007.safetensors", "model-00007-of-00007.safetensors",
    "model.safetensors.index.json", "mtp_grafted.safetensors",
    "config.json", "tokenizer.json",
)]}
mtp_ds = D.download_set(MTP_API)
check("mtp_grafted.safetensors" in mtp_ds,
      f"download_set unions a dedicated MTP head (mtp_grafted.safetensors); got={sorted(mtp_ds)}")
check("model-00001-of-00007.safetensors" in mtp_ds,
      "download_set still includes the main shards alongside the MTP head")

# ---------------------------------------------------------------------------
# 2. sized_download_gb sizes EXACTLY download_set; gates.c2a_disk consumes
#    the SHARED set (no drift between [C2a] / E2 / E3).
# ---------------------------------------------------------------------------
sized = D.sized_download_gb(API)
manual = round(
    sum(
        s["size"]
        for s in API["siblings"]
        if s["rfilename"] in set(ds)
    ) / (1024 ** 3),
    4,
)
check(abs(sized - manual) < 1e-6,
      f"sized_download_gb sums exactly download_set ({sized} == {manual})")
# adapter NOT counted (would add 90 MB if drift existed)
no_adapter = round(
    sum(s["size"] for s in API["siblings"]
        if s["rfilename"] in set(ds)) / (1024 ** 3), 4)
check(abs(sized - no_adapter) < 1e-6,
      "sized footprint excludes the adapter (download_set is authority)")

from scripts.lib.profiles import gates as G  # noqa: E402

der_for_c2a = SimpleNamespace(profile={"_hf_api": API})


class FakeStat:
    def __init__(self, free_gb: float):
        self.f_frsize = 4096
        self.f_bavail = int(free_gb * (1024 ** 3) / 4096)


c2a = G.c2a_disk(der_for_c2a, statvfs=lambda p: FakeStat(500.0))
check(abs(c2a.required_gb - round(sized * 1.2, 4)) < 1e-3,
      f"gates.c2a_disk sizes the SHARED download_set() "
      f"(required={c2a.required_gb} == sized*1.2={round(sized*1.2,4)})")

# ---------------------------------------------------------------------------
# Fixture fetcher: replays snapshot (writes EXACTLY allow_patterns) + HEAD
# x-linked-etag. Records the allow_patterns it was asked to fetch so we can
# assert fetched-set == sized-set (nothing extra).
# ---------------------------------------------------------------------------
class FixtureFetcher:
    def __init__(self, *, etags=None, gated_on_head=None,
                 gated_on_snapshot=False, disk_on_snapshot=False,
                 bad_bytes=None):
        # etags: {filename: etag_or_None}; gated_on_head: filename set
        self.etags = etags or {}
        self.gated_on_head = set(gated_on_head or ())
        self.gated_on_snapshot = gated_on_snapshot
        self.disk_on_snapshot = disk_on_snapshot
        self.bad_bytes = bad_bytes or {}  # filename -> bytes (sha won't match)
        self.allow_seen = None
        self.head_calls: list[str] = []

    def snapshot(self, repo_id, local_dir, allow_patterns):
        self.allow_seen = list(allow_patterns)
        if self.gated_on_snapshot:
            raise DL.GatedError("401")
        if self.disk_on_snapshot:
            raise DL.DiskError("ENOSPC")
        written = []
        rootp = Path(local_dir)
        for name in allow_patterns:
            p = rootp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            if name in self.bad_bytes:
                p.write_bytes(self.bad_bytes[name])
            else:
                # deterministic content so the etag fixture == sha256(file)
                p.write_bytes(f"CONTENT::{name}".encode())
            written.append(name)
        return written

    def head_etag(self, repo_id, filename):
        self.head_calls.append(filename)
        if filename in self.gated_on_head:
            return "__gated-401__"
        return self.etags.get(filename, None)


def content_sha(name: str) -> str:
    return hashlib.sha256(f"CONTENT::{name}".encode()).hexdigest()


import copy  # noqa: E402


def api_with_lfs(sha_for=None, drop_lfs=()):
    """A deep copy of API where every *.safetensors sibling carries an
    `lfs.sha256` == sha256("CONTENT::<name>") (the deterministic content the
    FixtureFetcher writes) — i.e. the canonical per-file LFS SHA256 the
    deriver stashes in `_hf_api`. `sha_for` overrides specific filenames'
    lfs.sha256; `drop_lfs` removes lfs entirely for a filename (a genuinely
    unverifiable *.safetensors -> CONTRACT-3 hard `no-etag`)."""
    sha_for = sha_for or {}
    a = copy.deepcopy(API)
    for s in a["siblings"]:
        n = s["rfilename"]
        if not n.endswith(".safetensors"):
            continue
        if n in drop_lfs:
            s.pop("lfs", None)
            continue
        sha = sha_for.get(n, content_sha(n))
        s["lfs"] = {"sha256": sha, "size": s.get("size", 0)}
    return a


def mk_einput(slug, hf_home, api=None):
    der = SimpleNamespace(
        slug=slug,
        profile={
            "_hf_api": api if api is not None else api_with_lfs(),
            "selected_weight_files": [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            "weight_format": "awq",
            "torch_dtype": None,
        },
    )
    return EInput(
        slug=slug, terminal="proceed", is_override_accepted=False,
        der=der, runtime={}, selected_files=[], hf_home=Path(hf_home),
        c2a=None, hardware_sm=8.6, visible_gpu_count=1,
        per_gpu_vram_mib=[24576], selected_gpu_indices=[0],
        selected_gpu_vram_mib=[24576], topology_summary="(RTX 3090, 24576)",
        club3090_commit="deadbeef", diagnostics={},
    )


# ---------------------------------------------------------------------------
# 3. SHA pass -> ok, atomic move, .incomplete gone, fetched-set == sized-set.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/Awq-Model", td)
    good_etags = {
        n: content_sha(n) for n in ds if n.endswith(".safetensors")
    }
    f = FixtureFetcher(etags=good_etags)
    r = DL.download_model(ei, fetcher=f)
    check(r.ok and r.failure is None,
          f"SHA-pass: DownloadResult ok (got ok={r.ok} fail={r.failure})")
    check(r.sha_verified is True, "SHA-pass: sha_verified True")
    check(sorted(r.files) == sorted(ds),
          f"SHA-pass: fetched-set == download_set (got {sorted(r.files)})")
    # fetched-set == sized-set: allow_patterns IS literally download_set
    check(f.allow_seen == ds,
          f"allow_patterns passed to snapshot IS literally download_set "
          f"(got {f.allow_seen})")
    final = DL.pull_dir(Path(td), "Org/Awq-Model")
    check(final.is_dir(), f"SHA-pass: final --model dir created at {final}")
    check(not (final / ".incomplete").exists(),
          "SHA-pass: .incomplete tree removed after atomic move")
    check(all((final / n).exists() for n in ds),
          "SHA-pass: every download_set file present in the final dir")
    check(r.local_dir == str(final),
          "SHA-pass: DownloadResult.local_dir is the host --model dir")
    check(r.bytes > 0, "SHA-pass: bytes accounted")

# ---------------------------------------------------------------------------
# 3b. E2-fix-2 REGRESSION (the exact on-rig E5 scenario):
#     a Xet-backed repo. The redirect-following HEAD into cas-bridge.
#     xethub.hf.co carries a CAS-blob ETag but NO x-linked-etag, so
#     head_etag() -> None for every *.safetensors. BEFORE the fix this
#     false-failed failure=="no-etag" even though the canonical LFS SHA256
#     IS published in the HF model API siblings[].lfs.sha256 (which the
#     deriver already fetched into _hf_api). AFTER the fix: verification
#     uses the API lfs.sha256 -> ok=True, sha_verified=True. (On-rig:
#     `pull.sh Qwen/Qwen2.5-0.5B-Instruct` reached [E], hf-download ran,
#     then false `no-etag`.)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    # API publishes the canonical lfs.sha256 (== sha256 of the fixture
    # content) for every shard — exactly what /api/models?blobs=true does.
    ei = mk_einput("Qwen/Qwen2.5-0.5B-Instruct", td)
    # Xet redirect: HEAD sees NO x-linked-etag for ANY *.safetensors.
    xet_no_etag = {
        n: None for n in ds if n.endswith(".safetensors")
    }
    f = FixtureFetcher(etags=xet_no_etag)
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is True and r.failure is None,
          f"E2-fix-2: Xet-redirect HEAD has NO x-linked-etag yet verify "
          f"PASSES off the API lfs.sha256 (got ok={r.ok} fail={r.failure})"
          f" — this exact case FALSE-failed 'no-etag' before the fix")
    check(r.sha_verified is True,
          "E2-fix-2: sha_verified True via API lfs.sha256 (redirect-immune)")
    # the HEAD seam WAS still consulted (gated-401 probe) — proves we did
    # NOT just delete the HEAD call, only stopped trusting its hash value.
    check(set(f.head_calls) == {
              n for n in ds if n.endswith(".safetensors")},
          "E2-fix-2: HEAD still consulted as the gated-401 probe seam "
          f"(head_calls={sorted(f.head_calls)})")
    final = DL.pull_dir(Path(td), "Qwen/Qwen2.5-0.5B-Instruct")
    check(final.is_dir() and not (final / ".incomplete").exists(),
          "E2-fix-2: atomic move + .incomplete cleanup unchanged")
    check(all((final / n).exists() for n in ds),
          "E2-fix-2: every download_set file promoted to the final dir")

# ---------------------------------------------------------------------------
# 3c. E2-fix-2: SHA verification is sourced from the API lfs.sha256, NOT
#     the HEAD value — a WRONG/STALE HEAD x-linked-etag must NOT cause a
#     false sha-mismatch when the file matches the API lfs.sha256.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/StaleHeadEtag", td)
    bogus = {n: "deadbeef" * 8 for n in ds if n.endswith(".safetensors")}
    f = FixtureFetcher(etags=bogus)  # HEAD lies; API lfs.sha256 is truth
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is True and r.failure is None and r.sha_verified is True,
          f"E2-fix-2: a wrong HEAD x-linked-etag is IGNORED; verify uses "
          f"API lfs.sha256 (got ok={r.ok} fail={r.failure})")

# ---------------------------------------------------------------------------
# 4. no-etag -> HARD fail (NOT skip), .incomplete deleted.
#    CONTRACT-3 hard-fail-if-unverifiable PRESERVED through E2-fix-2: the
#    trusted hash now comes from the API siblings[].lfs.sha256, so the
#    genuinely-unverifiable case is "a *.safetensors with NO lfs.sha256 in
#    the API payload" (NOT a blank HEAD x-linked-etag — that is now the
#    EXPECTED Xet-redirect state, see regression case 3b). HEAD etags are
#    fully present here to PROVE the hard fail comes from the missing API
#    hash, not the HEAD seam. Failure token UNCHANGED ("no-etag").
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    # shard 2 has NO lfs.sha256 in the API -> genuinely unverifiable.
    api_no_lfs = api_with_lfs(drop_lfs={"model-00002-of-00002.safetensors"})
    ei = mk_einput("Org/NoEtag", td, api=api_no_lfs)
    # every HEAD x-linked-etag present & correct: proves the hard fail is
    # from the MISSING API lfs.sha256, not the (now-ignored) HEAD value.
    etags = {n: content_sha(n) for n in ds if n.endswith(".safetensors")}
    f = FixtureFetcher(etags=etags)
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is False and r.failure == "no-etag",
          f"no-etag: HARD fail failure=='no-etag' when a *.safetensors has "
          f"NO API lfs.sha256 (got ok={r.ok} fail={r.failure}) — NOT "
          f"skipped like setup.sh:437; token kept across the source move")
    final = DL.pull_dir(Path(td), "Org/NoEtag")
    check(not (final / ".incomplete").exists(),
          "no-etag: .incomplete tree deleted (no corrupt residue)")
    check(not any(p.is_file() for p in final.rglob("*")) if final.exists()
          else True,
          "no-etag: no partial/corrupt files left behind")

# ---------------------------------------------------------------------------
# 5. sha-mismatch -> fail + cleanup.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    # API lfs.sha256 (mk_einput default) says one thing; written bytes are
    # different -> mismatch. HEAD etags present & correct to prove the
    # mismatch is judged against the API hash, not the (ignored) HEAD value.
    ei = mk_einput("Org/ShaBad", td)
    etags = {n: content_sha(n) for n in ds if n.endswith(".safetensors")}
    f = FixtureFetcher(
        etags=etags,
        bad_bytes={"model-00001-of-00002.safetensors": b"CORRUPT"},
    )
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is False and r.failure == "sha-mismatch",
          f"sha-mismatch: fail failure=='sha-mismatch' (got {r.failure})")
    final = DL.pull_dir(Path(td), "Org/ShaBad")
    check(not (final / ".incomplete").exists(),
          "sha-mismatch: .incomplete deleted")
    check(not final.exists() or not any(
              p.name.endswith(".safetensors") for p in final.rglob("*")),
          "sha-mismatch: no staged weights promoted to final")

# ---------------------------------------------------------------------------
# 6. gated-401 (on snapshot AND on HEAD) -> fail + cleanup, no partials.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/Gated", td)
    f = FixtureFetcher(gated_on_snapshot=True)
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is False and r.failure == "gated-401",
          f"gated-401 (snapshot): failure=='gated-401' (got {r.failure})")
    final = DL.pull_dir(Path(td), "Org/Gated")
    check(not (final / ".incomplete").exists(),
          "gated-401: .incomplete deleted, no partials")

with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/GatedHead", td)
    etags = {n: content_sha(n) for n in ds if n.endswith(".safetensors")}
    f = FixtureFetcher(
        etags=etags,
        gated_on_head={"model-00001-of-00002.safetensors"},
    )
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is False and r.failure == "gated-401",
          f"gated-401 (HEAD mid-verify): failure=='gated-401' "
          f"(got {r.failure})")
    final = DL.pull_dir(Path(td), "Org/GatedHead")
    check(not (final / ".incomplete").exists(),
          "gated-401 (HEAD): .incomplete deleted")

# ---------------------------------------------------------------------------
# 7. disk failure -> failure="disk" + cleanup.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/Disk", td)
    f = FixtureFetcher(disk_on_snapshot=True)
    r = DL.download_model(ei, fetcher=f)
    check(r.ok is False and r.failure == "disk",
          f"disk: failure=='disk' (got {r.failure})")
    final = DL.pull_dir(Path(td), "Org/Disk")
    check(not (final / ".incomplete").exists(),
          "disk: .incomplete deleted")

# ---------------------------------------------------------------------------
# 8. E1 dtype header-probe NOW LIVE via standardized probe-fetcher wiring.
#    A quantized (awq) repo with NO torch_dtype: in E1 step (2) was dead
#    (diagnostics["fetcher"] never set) -> fail-closed missing-torch-dtype.
#    E2's downloader.set_probe_fetcher() standardizes the wiring; with a
#    fixture fetcher exposing a bf16 safetensors header, _resolve_compute_
#    dtype now resolves --dtype bfloat16 (NOT fail-closed).
# ---------------------------------------------------------------------------
def st_header(obj):
    blob = json.dumps(obj).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob


class HeaderProbeFetcher:
    """Replays the deriver's bounded header probe: range-GET 0-7 then the
    header bytes. Exposes a bf16 storage dtype."""
    def __init__(self, slug, weight):
        self._url = f"{D._HF_RESOLVE}/{slug}/resolve/main/{weight}"
        self._blob = st_header({"__metadata__": {},
                                "w": {"dtype": "BF16"}})

    def get(self, url, headers=None, range_=None):
        if url != self._url or range_ is None:
            return D.FetchResponse(status=404, body=b"")
        lo, hi = range_
        return D.FetchResponse(status=206, body=self._blob[lo:hi + 1])


SLUG = "Org/Awq-NoTorchDtype"
WEIGHT = "model-00001-of-00002.safetensors"
der = SimpleNamespace(
    slug=SLUG,
    profile={
        "weight_format": "awq",
        "torch_dtype": None,                       # NO config torch_dtype
        "selected_weight_files": [WEIGHT, "model-00002-of-00002.safetensors"],
    },
)
ei = EInput(
    slug=SLUG, terminal="proceed", is_override_accepted=False, der=der,
    runtime={}, selected_files=[], hf_home=Path("/data/hf"), c2a=None,
    hardware_sm=8.6, visible_gpu_count=1, per_gpu_vram_mib=[24576],
    selected_gpu_indices=[0], selected_gpu_vram_mib=[24576],
    topology_summary="(RTX 3090, 24576)", club3090_commit="deadbeef",
    diagnostics={},
)

# Before wiring: E1 step (2) is dead -> fail-closed missing-torch-dtype.
dt0, rej0 = gc._resolve_compute_dtype(ei)
check(dt0 == "" and rej0 == "missing-torch-dtype",
      f"E1 baseline: no probe fetcher -> fail-closed missing-torch-dtype "
      f"(got dt={dt0!r} rej={rej0!r})")

# E2 standardizer wires diagnostics['fetcher']; probe is now LIVE.
DL.set_probe_fetcher(ei, HeaderProbeFetcher(SLUG, WEIGHT))
check(ei.diagnostics.get("fetcher") is not None,
      "set_probe_fetcher populated einput.diagnostics['fetcher']")
dt1, rej1 = gc._resolve_compute_dtype(ei)
check(dt1 == "bfloat16" and rej1 is None,
      f"E1 dtype probe NOW LIVE: awg-no-torch_dtype resolves --dtype "
      f"bfloat16 via the standardized probe-fetcher (was fail-closed in "
      f"E1); got dt={dt1!r} rej={rej1!r}")

# default_probe_fetcher() yields the canonical real bounded fetcher.
check(isinstance(D.default_probe_fetcher(), D.HttpFetcher),
      "deriver.default_probe_fetcher() == canonical HttpFetcher")

# ---------------------------------------------------------------------------
# 9. DEFAULT REAL FETCHER (no injected fixture) shells out to the `hf` CLI:
#    resolves `hf` -> `huggingface-cli`, builds argv with EXACTLY one
#    `--include` per download_set entry (set == download_set, no extra
#    patterns), --local-dir == staging. subprocess.run is MOCKED (no live
#    network/subprocess); the mock materializes the include files so the
#    rest of download_model (SHA verify / atomic stage) runs unchanged.
#    This is the on-rig E5 regression fix: the prior default did
#    `from huggingface_hub import snapshot_download` -> ModuleNotFoundError
#    under the bare python3 pull.sh runs.
# ---------------------------------------------------------------------------
import subprocess as _subprocess  # noqa: E402

_orig_which = DL.shutil.which
_orig_run = DL.subprocess.run

captured = {}


def fake_which_hf(name):
    # `hf` present, `huggingface-cli` also present (hf must WIN).
    if name == "hf":
        return "/usr/bin/hf"
    if name == "huggingface-cli":
        return "/usr/bin/huggingface-cli"
    return None


def fake_run_ok(argv, **kw):
    captured["argv"] = list(argv)
    captured["env"] = kw.get("env")
    # Materialize EXACTLY the --include set into --local-dir so the
    # downstream SHA/stage logic (unchanged) operates on real files.
    local_dir = None
    includes = []
    i = 0
    while i < len(argv):
        if argv[i] == "--local-dir":
            local_dir = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--include":
            includes.append(argv[i + 1])
            i += 2
            continue
        i += 1
    captured["includes"] = includes
    captured["local_dir"] = local_dir
    rootp = Path(local_dir)
    for name in includes:
        p = rootp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"CONTENT::{name}".encode())
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/RealCli", td)
    DL.shutil.which = fake_which_hf
    DL.subprocess.run = fake_run_ok
    try:
        # NO fetcher= -> the DEFAULT real HubFetcher (CLI subprocess) runs.
        # head_etag still hits urllib (network) — short-circuit it so the
        # SHA stage uses our deterministic content fixture.
        good_etags = {
            n: content_sha(n) for n in ds if n.endswith(".safetensors")
        }
        _orig_head = DL.HubFetcher.head_etag
        DL.HubFetcher.head_etag = (
            lambda self, repo_id, filename: good_etags.get(filename)
        )
        try:
            r = DL.download_model(ei)
        finally:
            DL.HubFetcher.head_etag = _orig_head
    finally:
        DL.shutil.which = _orig_which
        DL.subprocess.run = _orig_run

    argv = captured.get("argv", [])
    check(argv[:1] == ["/usr/bin/hf"],
          f"real fetcher resolves `hf` (wins over huggingface-cli); "
          f"argv[0]={argv[:1]}")
    check(argv[1:4] == ["download", "Org/RealCli", "--local-dir"],
          f"real fetcher argv == `hf download <slug> --local-dir ...` "
          f"(got {argv[1:4]})")
    staging = str(DL.incomplete_dir(Path(td), "Org/RealCli"))
    check(captured.get("local_dir") == staging,
          f"--local-dir == the .incomplete staging dir (got "
          f"{captured.get('local_dir')!r} want {staging!r})")
    inc = captured.get("includes", [])
    check(inc == ds,
          f"ONE --include per download_set entry, in order, set == "
          f"download_set, NOTHING extra (got {inc})")
    check(set(inc) == set(ds) and len(inc) == len(ds),
          "real fetcher --include set is EXACTLY download_set (Codex-r7 "
          "High-1 exact-set guarantee holds via CLI transport)")
    check(r.ok and r.failure is None and r.sha_verified is True,
          f"real CLI fetcher: SHA verify / atomic stage logic UNCHANGED "
          f"(ok={r.ok} fail={r.failure} sha={r.sha_verified})")
    final = DL.pull_dir(Path(td), "Org/RealCli")
    check(final.is_dir() and not (final / ".incomplete").exists(),
          "real CLI fetcher: atomic move + .incomplete cleanup unchanged")

# ---------------------------------------------------------------------------
# 10. NEITHER `hf` NOR `huggingface-cli` resolvable -> STRUCTURED
#     DownloadResult(ok=False, failure="hf-cli-missing") with the canonical
#     actionable message — NOT a raised/​swallowed ModuleNotFoundError (the
#     exact on-rig E5 failure mode this fix closes).
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    ei = mk_einput("Org/NoCli", td)
    DL.shutil.which = lambda name: None  # nothing on PATH
    try:
        raised = None
        try:
            r = DL.download_model(ei)
        except BaseException as exc:  # MUST NOT happen — structured only
            raised = exc
            r = None
    finally:
        DL.shutil.which = _orig_which

    check(raised is None,
          f"missing-CLI is NOT a raised exception (the E5 "
          f"ModuleNotFoundError mode is closed); raised={raised!r}")
    check(r is not None and r.ok is False
          and r.failure == "hf-cli-missing",
          f"missing-CLI -> structured DownloadResult(failure="
          f"'hf-cli-missing') (got ok={getattr(r,'ok',None)} "
          f"fail={getattr(r,'failure',None)})")
    check(r is not None and "hf download" not in (r.detail or "")
          and "pipx" in (r.detail or "")
          and "uv tool install" in (r.detail or "")
          and "break-system-packages" in (r.detail or ""),
          f"missing-CLI carries the canonical PEP-668-aware install hint "
          f"(pipx / uv / --break-system-packages; in sync with setup.sh "
          f"ensure_hf_cli); detail={getattr(r,'detail',None)!r}")
    final = DL.pull_dir(Path(td), "Org/NoCli")
    check(not (final / ".incomplete").exists(),
          "missing-CLI: .incomplete tree deleted (no residue)")

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll E2 download-stage (CONTRACT-3) assertions passed.")
PY

echo "test-pullgate-download.sh OK"
