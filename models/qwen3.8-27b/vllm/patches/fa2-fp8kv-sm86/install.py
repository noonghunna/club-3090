"""Build pinned FA2 sources and attach the version-checked vLLM sidecar."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import urllib.request

import torch

FA2_REV = "ef08ffb9a23f9493435fdd81e3addba6ad4f5fff"
CUTLASS_REV = "62750a2b75c802660e4894434dc55e839f322277"
SOURCES = (
    ("fa2", "AntonProkopyev/fa2-fp8kv-sm86", FA2_REV,
     "d4bd7906c9878c2a81666d0ec99eef55687e3674f333fa419dbd64e75d9892aa"),
    ("cutlass", "NVIDIA/cutlass", CUTLASS_REV,
     "78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2"),
)
STOCK_SHA = "95f9e66762860f94a428c0f7f9ff6564f375ed95c46fc8b9019b3d91c99a57b0"
HOOK = "\n# club3090-fa2-sm86\nfrom fa2_sm86_adapter import Backend as FlashInferBackend\n"


def main():
    version = importlib.metadata.version("vllm")
    if version != "0.29.0":
        raise SystemExit(f"FA2 requires vLLM 0.29.0; found {version}")
    backend = Path(importlib.metadata.distribution("vllm").locate_file(
        "vllm/v1/attention/backends/flashinfer.py"))
    original = backend.read_text(encoding="utf-8").removesuffix(HOOK)
    if hashlib.sha256(original.encode("utf-8")).hexdigest() != STOCK_SHA:
        raise SystemExit("FA2 refused: stock FlashInfer backend source has drifted")
    identity = {"vllm": version, "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flashinfer": importlib.metadata.version("flashinfer-python"),
                "source": FA2_REV, "cutlass": CUTLASS_REV, "sm": "86"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path("/opt/club3090/fa2-cache") / key
    cache.mkdir(parents=True, exist_ok=True)
    for name, repo, revision, digest in SOURCES:
        archive = cache / f"{name}.tar.gz"
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            with urllib.request.urlopen(
                f"https://codeload.github.com/{repo}/tar.gz/{revision}", timeout=120
            ) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            os.replace(temporary, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"FA2 refused: {name} archive SHA256 mismatch")
        directory = cache / name
        if not directory.exists():
            staging = cache / f"{name}-extract"
            staging.mkdir(exist_ok=True)
            with tarfile.open(archive) as source:
                source.extractall(staging, filter="data")
            roots = list(staging.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise SystemExit(f"FA2 refused: unexpected {name} archive layout")
            os.replace(roots[0], directory)
            staging.rmdir()
    source = cache / "fa2"
    libraries = (source / "build-pipeline/fa2_fp8kv.so",
                 source / "build-prefill/fa2_fp8kv_prefill.so")
    manifest = cache / "build.json"
    if not manifest.exists():
        subprocess.run(
            ["python3", "build.py", "--pipeline", "--prefill", "--cutlass-include",
             str(cache / "cutlass/include")], cwd=source, check=True,
            env={**os.environ, "TORCH_CUDA_ARCH_LIST": "8.6",
                 "MAX_JOBS": os.environ.get("MAX_JOBS", "2")},
        )
        checksums = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (*libraries, source / "fa2_prefill.py", source / "paged_prefill.py")}
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": identity, "files": checksums}), encoding="utf-8")
        os.replace(temporary, manifest)
    built = json.loads(manifest.read_text(encoding="utf-8"))
    if built["identity"] != identity or any(
        hashlib.sha256((source / p).read_bytes()).hexdigest() != digest
        for p, digest in built["files"].items()
    ):
        raise SystemExit("FA2 refused: cached build checksum mismatch")
    temporary = backend.with_suffix(".fa2-tmp")
    temporary.write_text(original + HOOK, encoding="utf-8")
    os.replace(temporary, backend)
    runtime = {"FA2_FP8KV_LIBRARY": str(libraries[0]),
               "FA2_FP8KV_PREFILL_LIBRARY": str(libraries[1]),
               "PYTHONPATH": ":".join((str(Path(__file__).parent), str(source),
                                       os.environ.get("PYTHONPATH", "")))}
    Path("/etc/club3090/fa2-runtime.env").write_text(
        "".join(f"export {key}={shlex.quote(value)}\n" for key, value in runtime.items()),
        encoding="utf-8",
    )
    print(f"[fa2] source={FA2_REV} vllm={version} cache={cache}", flush=True)


if __name__ == "__main__":
    main()
