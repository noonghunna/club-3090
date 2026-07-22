# Pods — running multiple models on one host

A **pod** is one model pinned to a chosen **GPU set + port** — an *estate instance*. On a multi-GPU box you compose several pods (e.g. a chat model on GPU 0, a coder on GPUs 1+2), each its own endpoint, side by side. This page is the home for creating and managing them; for the raw GPU-pinning mechanics (UUIDs, CDI/NixOS runtimes) see [HARDWARE.md → "Pinning specific GPUs"](HARDWARE.md#pinning-specific-gpus-on-multi-gpu-rigs-and-cdi--nixos-runtimes).

> **Pod vs TP=N.** A single model split *across* cards (tensor-parallel, `vllm/dual` = TP=2) is one endpoint on N GPUs — see [DUAL_CARD.md](DUAL_CARD.md) / [MULTI_CARD.md](MULTI_CARD.md). A *pod* is the other axis: **many** models, each on its own subset of the cards. A pod's GPU count still equals its compose's TP (a TP=2 slug claims 2 GPUs).

**Non-goal:** pods are **single-host**. Routing across multiple boxes is the LiteLLM gateway's job — these tools don't do multi-node.

---

## Defaults — before you make any pod

**There is no default pod, and no pod claims "all GPUs" for you.** A fresh setup has an **empty estate** — `pod.sh list` shows no pods and every GPU free. Pods are entirely opt-in; you only get them when you `pod.sh create` (or use the c3 `[N]` wizard).

What happens *without* pods: a plain `bash scripts/launch.sh` (or `switch.sh <slug>`) boots **one model**, and how many cards it uses is the **compose's tensor-parallel size**, not "all your GPUs":

- **1 GPU** → that card.
- **2 GPUs** → it *asks* — `Use GPU 0, GPU 1, or both? [both]`. The **both** default runs one model across both cards (TP=2), e.g. `vllm/dual` — a single endpoint, not two pods.
- **3+ GPUs** → it *asks* — `Which GPU(s)? [all]`. "all" runs one model across all of them (TP=N).

So the default is *one* model that spans the cards you pick — which you can loosely think of as "one pod on all GPUs," but it's a single TP=N endpoint, not the pod/estate machinery. The moment you want **several** models at once — one on GPU 0, another on GPUs 1+2 — that's when you create pods. Pin the single-model case to specific cards without any pod setup via `bash scripts/launch.sh --gpus <a,b>`.

---

## Quickstart (CLI)

```bash
# create — fit-checked against the SELECTED GPUs, written to the estate file
bash scripts/pod.sh create chat  --gpus 0   --slug beellama/dflash   # TP=1 on GPU 0
bash scripts/pod.sh create coder --gpus 1,2 --slug vllm/dual         # TP=2 on GPUs 1+2

bash scripts/pod.sh up coder      # boot just this pod (UUID-pinned + placement-asserted)
bash scripts/pod.sh list          # pods + free GPUs      (--json for tooling)
bash scripts/pod.sh status        # live serving state + placement per pod
bash scripts/pod.sh down coder    # stop just this pod
bash scripts/pod.sh rm coder      # remove from the estate file (refuses while running)
```

`up` / `down` / `rm` act on **one** pod; to boot the whole file at once use `bash scripts/launch.sh --estate-file <path>` (add `--parallel` to boot instances concurrently).

## Quickstart (c3 cockpit)

The **Operate · Orchestration** tab shows a live **pod view** — each pod with its GPUs stacked beneath its header and a placement health badge — and:

- **`[N]`** — open the **New-pod** modal: name · slug (catalog picker) · GPU set (prefilled with the free GPUs). It runs the same fit-checked, gated `pod.sh create`.
- **`[o]`** — stop the whole estate (gated).

CLI and cockpit share **one estate file and one validation path** — a pod you make in either shows up in the other.

---

## Two or more pods — a worked example

Say you have a 4-GPU box and want a fast chat model and a bigger coder running at once:

```bash
# define two pods (writes the estate file; nothing boots yet)
bash scripts/pod.sh create chat  --gpus 0   --slug beellama/dflash   # TP=1 on GPU 0
bash scripts/pod.sh create coder --gpus 1,2 --slug vllm/dual         # TP=2 on GPUs 1+2

bash scripts/pod.sh list         # → 2 pods; GPU 3 free
bash scripts/pod.sh up chat      # boot each (or `launch.sh --estate-file <path> --parallel` for all)
bash scripts/pod.sh up coder
bash scripts/pod.sh status       # both serving, ✓ placed, on the GPUs you picked

# later: free up the coder's cards without touching chat
bash scripts/pod.sh down coder
bash scripts/pod.sh rm coder     # (or leave it defined and `up` it again later)
```

Each pod is its own endpoint (`chat` on `:8080`, `coder` on `:8010`) — point different clients at different ports, or front them with the LiteLLM gateway.

### Heterogeneous rigs — one pod per card family

Local rigs increasingly mix families (2× 3090 + a GB10 + an RTX 6000 Pro). The clean pattern is **one homogeneous pod per family**, each sized to what that hardware does well:

```bash
bash scripts/pod.sh create qwen  --gpus 0,1 --slug vllm/dual                  # the two 3090s (TP=2)
bash scripts/pod.sh create big   --gpus 2   --slug vllm/qwen-27b-single-nvfp4  # the GB10 (sm ≥ 9)
bash scripts/pod.sh create vision --gpus 3  --slug <a-6000-pro-slug>          # the 6000 Pro
```

Keep card families in **separate pods**, not mixed inside one TP group: tensor-parallel across mismatched cards makes NCCL wait on the slowest card and wastes the bigger card's VRAM. `create` *lets* you build a mixed-family pod (it estimates fit against the smallest card in the set), but a same-family pod is faster and its fit estimate is exact. `pod.sh create` also enforces per-slug `required_sm` — an NVFP4 slug won't land on your 3090s.

---

## How `create` decides fit (D1)

`create` prices the slug against the **GPU set you selected** (via `kv-calc --card`) before writing anything:

- **GPU count must equal the compose's tensor-parallel size** — a TP=2 slug needs exactly 2 GPUs. A mismatch is a **hard reject** with a clear message (no silent drop).
- **Homogeneous set** (all same card) → priced against that card.
- **Heterogeneous set** (e.g. a 3090 + a 5090) → estimated against the **smallest card** in the set (a conservative floor; a note says which card was used). True per-card heterogeneous modelling is a deferred `kv-calc` enhancement.
- The whole estate is then re-validated (`validate_estate`) for **GPU collisions** (two pods claiming the same card) and **port collisions** before the new pod is appended.

`create` only *writes the plan* — no GPU is claimed until `up`. In c3 the create routes through the standard confirm gate; a bad set is refused by the same checks.

## Placement verification

Pinning specific GPUs is only trustworthy if you can confirm it worked. After any pod boots, a **placement assertion** compares where the model *actually* ran (`nvidia-smi --query-compute-apps=gpu_uuid` — runtime-agnostic ground truth) against the GPUs you requested:

- CLI `up` prints `✓ placement verified` or a loud `⚠ PLACEMENT MISMATCH`.
- `pod.sh status` / the c3 pod view carry the verdict per pod (`✓ placed` / `⚠ PLACEMENT MISMATCH`).

A mismatch on a CDI/NixOS rig usually means `CUDA_VISIBLE_DEVICES` didn't reach the container — see [HARDWARE.md](HARDWARE.md#pinning-specific-gpus-on-multi-gpu-rigs-and-cdi--nixos-runtimes) for the CDI deploy-block recipe. The view **never** shows a requested-but-not-actual placement.

### ⚠️ Known incompatibility — `vllm-gemma-stable` slugs × UUID pinning

Pods boot GPUs **UUID-pinned** (#610). The `vllm-gemma-stable` engine is pinned at **vLLM v0.22.0**, whose `Platform.device_id_to_physical_device_id` casts the `CUDA_VISIBLE_DEVICES` entry with a bare `int()` — so a `GPU-<uuid>` mask raises before the model loads:

```
ValueError: invalid literal for int() with base 10: 'GPU-1c6f148d-...'
  → pydantic ValidationError: Model architectures [...] failed to be inspected
  → container restart loop
```

**Affected slugs** (all six on that engine): `vllm/gemma-mtp-tp1` · `vllm/gemma-bf16-mtp` · `vllm/gemma-int8-mtp` · `vllm/gemma-31b-dual` · `vllm/gemma-31b-qat-w4a16-dual` · `vllm/gemma-26ba4b-single`.

This is **not Gemma-specific and not a pod-tooling bug** — it's the engine pin. Upstream fixed the parse between v0.22.0 and v0.24.0, so `vllm-stable` slugs (v0.25.1) pin by UUID cleanly; only the un-bumped gemma engine trips it. The combination was never gate-tested because those slugs were last gated before #610 landed.

**Until the pin bump lands** ([UPSTREAM.md](UPSTREAM.md#vllm-vllm-projectvllm) → `device_id_to_physical_device_id`, [#750](https://github.com/noonghunna/club-3090/issues/750)), run these slugs outside the pod path — e.g. `launch.sh` with plain Docker GPU selection, or `docker compose` directly with `--gpus device=N` and no `CUDA_VISIBLE_DEVICES` override.

> On a **classic** nvidia runtime you can instead keep `NVIDIA_VISIBLE_DEVICES` on UUIDs (it performs the card selection; the container renumbers the exposed set to `0..N-1`) and pass `CUDA_VISIBLE_DEVICES` in index form. **Don't do this on CDI** — there `NVIDIA_VISIBLE_DEVICES` is ignored and the UUID mask is the only thing pinning cards.

## The estate file

Pods live in a YAML estate file (default `scripts/lib/profiles/estate.yml`; override with `--file`). GPU indices are stored **index-based**; they're resolved to UUIDs at boot so pods land on the right cards on both container runtimes. Hand-written files pass the same validation as `pod.sh create`.

```yaml
schema_version: 1
estate:
  - name: chat
    compose: beellama/dflash    # a registry slug (see `switch.sh --list`)
    gpus: [0]                    # host GPU indices — count must equal the slug's TP
    port: 8080
  - name: coder
    compose: vllm/dual
    gpus: [1, 2]
    port: 8010
```

---

## Reference

| Command | What |
|---|---|
| `pod.sh create <name> --gpus <a,b> --slug <slug> [--port N] [--file P]` | fit-check + validate + append a pod |
| `pod.sh list [--json]` | pods + membership + free GPUs |
| `pod.sh status [--json]` | live serving state + placement verdict per pod |
| `pod.sh up <name>` / `down <name>` | boot / stop one pod |
| `pod.sh rm <name>` | remove from the estate file (refuses while running) |
| `launch.sh --estate-file <P> [--parallel]` | boot the whole estate file |

Related: [HARDWARE.md](HARDWARE.md#pinning-specific-gpus-on-multi-gpu-rigs-and-cdi--nixos-runtimes) (GPU pinning / CDI runtimes) · [MULTI_CARD.md](MULTI_CARD.md) (TP scaling for a single model across cards) · [ARCHITECTURE.md](ARCHITECTURE.md) (services, ports).
