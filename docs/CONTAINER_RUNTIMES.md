# Container runtimes — non-Docker / non-bare-metal notes

The shipped composes assume **Docker (or Docker-compatible runtime) on bare-metal Linux**. Most reporters run that exact stack and the `verify-*` scripts target it. This page captures what we know about non-default runtime/host combinations — not as recipes that are tested end-to-end, but as breadcrumbs for users hitting environmental issues that aren't club-3090 bugs.

> **Tested environment (the "known good" baseline):** bare-metal Ubuntu 22.04 / 24.04, kernel 6.8.x, Docker Engine 27+ with NVIDIA Container Toolkit, no `default-runtime: nvidia` in `daemon.json` (opt-in via `--gpus '"device=N,M"'` per-command). Variants that materially diverge from this can surface separate bugs that aren't pin-fixable.

---

## Soft-warn: the docker preflight in `scripts/setup.sh` is not load-bearing

`scripts/setup.sh` only fetches Genesis + models — no docker invocations until you actually run `docker compose up` later. As of [`2f8ed19`](https://github.com/noonghunna/club-3090/commit/2f8ed19), the docker preflight in setup.sh is a soft warning rather than a hard fail, so users on non-Docker container runtimes can run setup without working around the gate.

`scripts/launch.sh` and `scripts/switch.sh` keep the hard check because they actually invoke docker commands directly.

---

## Podman / Podman Compose

Already supported via env override — `scripts/switch.sh` reads `COMPOSE_BIN` and defaults to `docker compose`:

```bash
COMPOSE_BIN="podman compose" bash scripts/switch.sh vllm/dual
```

Caveats not validated end-to-end:

- `docker ps` / `docker inspect` calls inside `switch.sh` (lines 119, 128-133) assume the binary is named `docker` and emit Docker-shaped JSON. Aliasing `docker → podman` mostly works but won't cover every case (e.g., `docker compose project labels` may differ).
- Per-command flag conventions differ: `--gpus '"device=0,1"'` is Docker-specific syntax; podman uses `--device nvidia.com/gpu=all` or similar. Adjust your compose / override accordingly.

If you've shipped a Podman pipeline that works end-to-end through `setup.sh` → `launch.sh` → `verify-full.sh`, please file a docs PR with the diff so the next person inherits a working recipe.

---

## microk8s

@apnar reported success running club-3090 under microk8s ([disc #48](https://github.com/noonghunna/club-3090/discussions/48), [disc #51](https://github.com/noonghunna/club-3090/discussions/51) for 5090 single-card data). Not pre-baked — the user translates the compose file to a k8s manifest manually. The `setup.sh` soft-warn ([`2f8ed19`](https://github.com/noonghunna/club-3090/commit/2f8ed19)) is what unblocks setup; `verify-full.sh` works because it talks HTTP to the engine.

The path forward for microk8s:

- **Models cache** mounted into the k8s pod's volume (analogous to `~/.cache/huggingface` mount in compose)
- **GPUs assigned via** the NVIDIA k8s device plugin, not `--gpus`
- **vLLM arguments** translated 1:1 from the compose `command:` block into the pod spec's `args:`
- **Genesis / patch mounts** (RO bind-mounts of the `models/qwen3.6-27b/vllm/patches/*` files in the compose) → k8s ConfigMap or initContainer that places the same files

If you have a working microk8s pipeline and are willing to PR a manifest example, the docs should grow a `docs/microk8s/` subfolder with one or two sample YAMLs. Open invitation.

---

## Proxmox VE — known footgun on kernel 6.17.x

> **Proxmox VE 8.x / kernel 6.17.x users**: `vllm/vllm-openai:nightly-7a1eb8ac2…` (and likely current nightlies) crashes with `RuntimeError: this event loop is already running` at `vllm.entrypoints.cli.serve.cmd → uvloop.run(run_server(args))` regardless of Genesis pin, TP, runtime config, or `--init`. Same image boots clean on bare-metal Ubuntu 6.8.x. **If you're on Proxmox + 6.17.x kernel and hit this, the bug is environmental, not in club-3090.**

### The data trail (cite for upstream filing)

[@lexhoefsloot's club-3090 #49](https://github.com/noonghunna/club-3090/issues/49) ran a thorough elimination sequence on the bug. What's been ruled out:

| Suspect | Status | Probe |
|---|---|---|
| Genesis patches | ❌ ruled out | Bare `docker run` (no Genesis) crashes |
| `torch.compile` × uvloop | ❌ ruled out | `--enforce-eager` crashes |
| Multiproc spawn (TP > 1) | ❌ ruled out | TP=1 also crashes |
| Module import / lib-stack | ❌ ruled out | `import vllm` works |
| CLI dispatch / argparse | ❌ ruled out | `vllm serve --help` works |
| vLLM upstream regression | ❌ ruled out | Same image boots clean on bare-metal Ubuntu 2× 3090 PCIe (cross-rig, [Probe B reproduction](https://github.com/noonghunna/club-3090/issues/49#issuecomment-4373220693)) |
| `default-runtime: nvidia` in daemon.json | ❌ ruled out | Removed + per-command `--runtime=nvidia` still crashes |
| PID 1 / signal forwarding | ❌ ruled out | `--init` (tini PID 1) still crashes |
| **Proxmox VE kernel 6.17.2-pve specific** | ⚠️ leading candidate | (kernel × Python 3.12 × this image's lib stack interaction) |
| **Proxmox VE containerd / namespace setup** | ⚠️ adjacent candidate | (LXC / cgroup / seccomp policy interaction at process startup) |

### What this means for you

If you're on bare-metal anything (Ubuntu, Debian, RHEL family) on a kernel 6.8-6.16-ish, none of this affects you — that's the tested baseline.

If you're on Proxmox VE LXC/VM and hit the same `uvloop` trace at boot:

1. Confirm it's the same crash: container exits within ~5 seconds of `docker run` (or compose up), the trace mentions `uvloop.loop.Loop.run_forever` → `RuntimeError: this event loop is already running`.
2. If yes, no club-3090-side fix exists today. The image works on different host stacks; your environment is the differentiator.
3. **Cheapest workaround attempt** (low signal but free to try): boot with `--privileged`. If that boots clean, the issue is namespace-policy-related and the workaround for your daily driver is `--privileged` (not ideal, but unblocks).
4. **Better long-term**: file with Proxmox forum (PVE + Docker + GPU passthrough thread) or NVIDIA Container Toolkit — they have the audience that can debug containerd × kernel × Python interactions.

If you find a workaround that gets dual.yml booting on your Proxmox rig, **please file a docs PR back here** so the next Proxmox user inherits the answer faster.

### Re-check triggers

This footgun page should be revisited when:

- Someone independently reproduces the asyncio crash on a non-Proxmox bare-metal Linux 6.17.x rig — would distinguish "kernel 6.17.x bug" from "Proxmox-specific" and re-route the diagnosis
- vLLM nightly bumps past `nightly-7a1eb8ac2…` and a different SHA gets tested on the same Proxmox setup — would tell us whether it's nightly-7a1eb8ac2 specific or persistent across the nightly stream
- Proxmox ships PVE kernel 6.18+ — kernel revisions historically resolve namespace × cgroup × user-space-runtime interactions

---

## What this page is NOT

- **Not a recipe for Proxmox / microk8s / podman setups.** Those are user-driven; we don't have CI runners to validate them.
- **Not a list of all possible non-Docker setups.** Just the ones that have surfaced via reporters with concrete data trails.
- **Not a substitute for [docs/HARDWARE.md](HARDWARE.md)** which covers GPU/driver requirements regardless of runtime.

---

## See also

- [HARDWARE.md](HARDWARE.md) — GPU / driver / power / NVLink requirements
- [CLIFFS.md](CLIFFS.md) — known failure modes (the engine-side bugs, not environmental ones)
- [MULTI_CARD.md](MULTI_CARD.md) — TP topology and the `nvidia-smi topo -m` ranking
- [club-3090 #49](https://github.com/noonghunna/club-3090/issues/49) — full Proxmox VE asyncio investigation trail (parked)
- [club-3090 disc #48](https://github.com/noonghunna/club-3090/discussions/48) — original microk8s docker-soft-warn request
