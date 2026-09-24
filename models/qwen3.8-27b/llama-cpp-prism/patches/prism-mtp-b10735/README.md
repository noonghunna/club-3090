# prism b10735 + 1 upstream PR — build recipe for `llama-cpp-prism-mtp`

⚠️ This patch is **baked into the engine image at build time**, not mounted at
runtime. The engine profile marks it `image_baked: true` for that reason. It is
vendored here so the image is **reproducible**, and so its upstream status can be
tracked to its drop trigger.

| patch | upstream | diff | why |
|---|---|---|---|
| `pr189.diff` | [#189](https://github.com/PrismML-Eng/llama.cpp/pull/189) OPEN | +6/-0 | Guards a CUDA illegal memory access in short-query MMA FA with quantized V (Q 3-4). Inert at the shipped `SPEC_N=4` (Q=5); carried for `SPEC_N=2`/`3`, reachable via env. |

**Dropped 2026-09-24** (drop trigger fired — merged upstream AND shipped in `prism-b10735-842b188`):

| patch | merged | why it was carried |
|---|---|---|
| [#205](https://github.com/PrismML-Eng/llama.cpp/pull/205) | 2026-09-21 | `--spec-type draft-mtp` could not START on a Hadamard-folded target. The reason this engine was created. |
| [#216](https://github.com/PrismML-Eng/llama.cpp/pull/216) | 2026-09-23 | GDN `cols_per_warp=4` was gated to GB10; widening to Ampere+ measured **+8.6% prefill** on sm_86. |

⛔ **Drop trigger for #189:** when it merges AND ships in a prism release tarball, drop
this patch, delete the `llama-cpp-prism-mtp` engine profile and fold the slug back onto
`llama-cpp-prism` (whose stock b10735 image already carries #205 — see that profile's
`supported_drafters` note: an MTP boot on the stock image is still unmeasured).

## Rebuild

```bash
git clone --depth 1 --branch prism-b10735-842b188 \
  https://github.com/PrismML-Eng/llama.cpp.git
cd llama.cpp && git apply ../pr189.diff
```

Then build in a CUDA **12.8** devel image and copy `build/bin/` over `/app/` in the
stock prism image (its own recipe: `../../image/Dockerfile`).

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential libcurl4-openssl-dev ca-certificates
COPY llama.cpp /src
WORKDIR /src
# ⚠️ libcuda.so.1 is the DRIVER library, absent in any build container, so the link
# of llama-server fails on cuMemMap/cuMemCreate/cuDeviceGet. Use the devel image's
# stub. The linker wants the SONAME, hence the symlink.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN cmake -B build -G Ninja \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
      -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
 && cmake --build build --target llama-server -j 32
FROM ghcr.io/noonghunna/llamacpp-prism@sha256:20c43cb73ad3075cb689770682a7cde3c9dc0df2d16ba585986de9d69ddaec62
COPY --from=build /src/build/bin/ /app/
```

## Three build traps, each of which cost a build

1. **`-DCMAKE_CUDA_ARCHITECTURES` must be explicit.** Bare `-DGGML_CUDA=ON` fails with
   `CUDA_ARCHITECTURES is empty for target "llama-common"` (upstream
   [#182](https://github.com/PrismML-Eng/llama.cpp/issues/182)). `86;89;120` **matches
   the stock release matrix** — verify with `cuobjdump --list-elf`, do not read it off
   release notes. A first build at `86` alone was *narrower than stock* and would not
   have run on a 4090 or 5090.
2. **`-Wl,-rpath-link`, not `-L`.** `-L` resolves explicit `-lfoo`; `libcuda.so.1` is a
   transitive `DT_NEEDED` of `libggml-cuda.so` and is searched via `rpath-link`. `ld`
   says so verbatim and it still took two builds to act on.
3. ⛔ **Never `-rpath`** — that bakes the 66 KB stub into the runtime search path, so the
   container loads the stub instead of the real driver. Verify with
   `readelf -d /app/llama-server` that **no stub path** (`…/lib64/stubs`) appears in
   `RPATH`/`RUNPATH`. ⚠️ An empty RUNPATH is NOT what you will see: CMake's build-tree
   `RUNPATH [/src/build/bin:]` is present on both the b10709 and b10735 images. That is
   harmless — the directory does not exist in the runtime image and `LD_LIBRARY_PATH=/app`
   resolves first — so check for the stub path, not for an empty field. (This line
   used to say "no RPATH/RUNPATH is present", which neither image ever satisfied.)
   `docker cp` of `/app/libggml-cuda.so` copies the SYMLINK; inspect
   `libggml-cuda.so.0.21.0`.

⚠️ The built binary reports `build 1` because the shallow clone has no git history to
count from. The **commit hash** (`842b188`) is the reliable identifier.
