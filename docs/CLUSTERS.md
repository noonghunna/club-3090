# Clusters — running multiple models on one host

A **cluster** is one model pinned to a chosen **GPU set + port** — an *estate instance*. On a multi-GPU box you compose several clusters (e.g. a chat model on GPU 0, a coder on GPUs 1+2), each its own endpoint, side by side. This page is the home for creating and managing them; for the raw GPU-pinning mechanics (UUIDs, CDI/NixOS runtimes) see [HARDWARE.md → "Pinning specific GPUs"](HARDWARE.md#pinning-specific-gpus-on-multi-gpu-rigs-and-cdi--nixos-runtimes).

> **Cluster vs TP=N.** A single model split *across* cards (tensor-parallel, `vllm/dual` = TP=2) is one endpoint on N GPUs — see [DUAL_CARD.md](DUAL_CARD.md) / [MULTI_CARD.md](MULTI_CARD.md). A *cluster* is the other axis: **many** models, each on its own subset of the cards. A cluster's GPU count still equals its compose's TP (a TP=2 slug claims 2 GPUs).

**Non-goal:** clusters are **single-host**. Routing across multiple boxes is the LiteLLM gateway's job — these tools don't do multi-node.

---

## Defaults — before you make any cluster

**There is no default cluster, and no cluster claims "all GPUs" for you.** A fresh setup has an **empty estate** — `cluster.sh list` shows no clusters and every GPU free. Clusters are entirely opt-in; you only get them when you `cluster.sh create` (or use the c3 `[n]` wizard).

What happens *without* clusters: a plain `bash scripts/launch.sh` (or `switch.sh <slug>`) boots **one model**, and how many cards it uses is the **compose's tensor-parallel size**, not "all your GPUs":

- **1 GPU** → that card.
- **2 GPUs** → it *asks* — `Use GPU 0, GPU 1, or both? [both]`. The **both** default runs one model across both cards (TP=2), e.g. `vllm/dual` — a single endpoint, not two clusters.
- **3+ GPUs** → it *asks* — `Which GPU(s)? [all]`. "all" runs one model across all of them (TP=N).

So the default is *one* model that spans the cards you pick — which you can loosely think of as "one cluster on all GPUs," but it's a single TP=N endpoint, not the cluster/estate machinery. The moment you want **several** models at once — one on GPU 0, another on GPUs 1+2 — that's when you create clusters. Pin the single-model case to specific cards without any cluster setup via `bash scripts/launch.sh --gpus <a,b>`.

---

## Quickstart (CLI)

```bash
# create — fit-checked against the SELECTED GPUs, written to the estate file
bash scripts/cluster.sh create chat  --gpus 0   --slug beellama/dflash   # TP=1 on GPU 0
bash scripts/cluster.sh create coder --gpus 1,2 --slug vllm/dual         # TP=2 on GPUs 1+2

bash scripts/cluster.sh up coder      # boot just this cluster (UUID-pinned + placement-asserted)
bash scripts/cluster.sh list          # clusters + free GPUs      (--json for tooling)
bash scripts/cluster.sh status        # live serving state + placement per cluster
bash scripts/cluster.sh down coder    # stop just this cluster
bash scripts/cluster.sh rm coder      # remove from the estate file (refuses while running)
```

`up` / `down` / `rm` act on **one** cluster; to boot the whole file at once use `bash scripts/launch.sh --estate-file <path>` (add `--parallel` to boot instances concurrently).

## Quickstart (c3 cockpit)

The **Operate · Orchestration** tab shows a live **cluster view** — each cluster with its GPUs stacked beneath its header and a placement health badge — and:

- **`[n]`** — open the **New-cluster** modal: name · slug (catalog picker) · GPU set (prefilled with the free GPUs). It runs the same fit-checked, gated `cluster.sh create`.
- **`[o]`** — stop the whole estate (gated).

CLI and cockpit share **one estate file and one validation path** — a cluster you make in either shows up in the other.

---

## How `create` decides fit (D1)

`create` prices the slug against the **GPU set you selected** (via `kv-calc --card`) before writing anything:

- **GPU count must equal the compose's tensor-parallel size** — a TP=2 slug needs exactly 2 GPUs. A mismatch is a **hard reject** with a clear message (no silent drop).
- **Homogeneous set** (all same card) → priced against that card.
- **Heterogeneous set** (e.g. a 3090 + a 5090) → estimated against the **smallest card** in the set (a conservative floor; a note says which card was used). True per-card heterogeneous modelling is a deferred `kv-calc` enhancement.
- The whole estate is then re-validated (`validate_estate`) for **GPU collisions** (two clusters claiming the same card) and **port collisions** before the new cluster is appended.

`create` only *writes the plan* — no GPU is claimed until `up`. In c3 the create routes through the standard confirm gate; a bad set is refused by the same checks.

## Placement verification

Pinning specific GPUs is only trustworthy if you can confirm it worked. After any cluster boots, a **placement assertion** compares where the model *actually* ran (`nvidia-smi --query-compute-apps=gpu_uuid` — runtime-agnostic ground truth) against the GPUs you requested:

- CLI `up` prints `✓ placement verified` or a loud `⚠ PLACEMENT MISMATCH`.
- `cluster.sh status` / the c3 cluster view carry the verdict per cluster (`✓ placed` / `⚠ PLACEMENT MISMATCH`).

A mismatch on a CDI/NixOS rig usually means `CUDA_VISIBLE_DEVICES` didn't reach the container — see [HARDWARE.md](HARDWARE.md#pinning-specific-gpus-on-multi-gpu-rigs-and-cdi--nixos-runtimes) for the CDI deploy-block recipe. The view **never** shows a requested-but-not-actual placement.

## The estate file

Clusters live in a YAML estate file (default `scripts/lib/profiles/estate.yml`; override with `--file`). GPU indices are stored **index-based**; they're resolved to UUIDs at boot so clusters land on the right cards on both container runtimes. Hand-written files pass the same validation as `cluster.sh create`.

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
| `cluster.sh create <name> --gpus <a,b> --slug <slug> [--port N] [--file P]` | fit-check + validate + append a cluster |
| `cluster.sh list [--json]` | clusters + membership + free GPUs |
| `cluster.sh status [--json]` | live serving state + placement verdict per cluster |
| `cluster.sh up <name>` / `down <name>` | boot / stop one cluster |
| `cluster.sh rm <name>` | remove from the estate file (refuses while running) |
| `launch.sh --estate-file <P> [--parallel]` | boot the whole estate file |

Related: [HARDWARE.md](HARDWARE.md#pinning-specific-gpus-on-multi-gpu-rigs-and-cdi--nixos-runtimes) (GPU pinning / CDI runtimes) · [MULTI_CARD.md](MULTI_CARD.md) (TP scaling for a single model across cards) · [ARCHITECTURE.md](ARCHITECTURE.md) (services, ports).
