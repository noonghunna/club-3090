# Tailscale Integration — Design & Behavior

## Overview

The RunPod entrypoint optionally connects pods to a Tailscale tailnet so vLLM is accessible via a stable, private WireGuard IP address — no RunPod proxy URLs, no port forwarding, no changing IPs.

Two modes:

| Mode | Env vars | Behavior |
|---|---|---|
| **Stable IP** (default) | `TAILSCALE_AUTH_KEY` + `TAILSCALE_STATE_REPO`, suffix OFF | Same node identity every boot → stable IP and MagicDNS across pod restarts |
| **Multi-pod** | `TAILSCALE_AUTH_KEY` + `TAILSCALE_STATE_REPO` + `ENABLE_TAILSCALE_SUFFIX=1` | Each pod gets a unique identity → `runpod-llm`, `runpod-llm-1`, `runpod-llm-2`... |

Without `TAILSCALE_AUTH_KEY`, Tailscale is skipped entirely.

---

## Environment Variables

### Required to enable Tailscale

| Variable | Purpose |
|---|---|
| `TAILSCALE_AUTH_KEY` | Auth key from [Tailscale admin console → Keys](https://login.tailscale.com/admin/settings/keys). Use reusable + ephemeral + tagged `tag:runpod`. |

### State persistence (optional but recommended)

| Variable | Default | Purpose |
|---|---|---|
| `TAILSCALE_STATE_REPO` | (unset) | HF dataset repo (`user/repo`) to persist node identity across pods |
| `TAILSCALE_HOSTNAME` | `runpod-llm` | Node name in your tailnet |
| `ENABLE_TAILSCALE_SUFFIX` | `0` | Sequential numbering mode for multiple simultaneous pods |

### One-time setup

```bash
# Create a private HF dataset repo to store node identity
hf repos create runpod-llms-tailscale --type=dataset --private

# Generate a Tailscale auth key at:
# https://login.tailscale.com/admin/settings/keys
#   - Reusable: yes
#   - Ephemeral: yes
#   - Tag: tag:runpod
```

Template env vars:
```
TAILSCALE_AUTH_KEY=tskey-auth-xxxxx
TAILSCALE_STATE_REPO=fierysurf/runpod-llms-tailscale
ENABLE_TAILSCALE_SUFFIX=1   (only if running multiple pods)
```

---

## Boot Flow

### No Tailscale (`TAILSCALE_AUTH_KEY` unset)

```
→ Skip Tailscale entirely
→ Proceed to vLLM boot
```

### Stable IP Mode (`TAILSCALE_AUTH_KEY` set, `ENABLE_TAILSCALE_SUFFIX=0` or unset)

```
1. Install Tailscale from official apt repo
2. Try downloading tailscale.state from HF (TAILSCALE_STATE_REPO)
   ├─ Exists → "Restored state from previous pod" (same node identity)
   └─ Missing → "No prior state — fresh identity"
3. Start tailscaled in userspace-networking mode (required for containers)
4. tailscale up --hostname=$TAILSCALE_HOSTNAME
5. Print Tailscale IP (e.g., 100.112.244.67)
6. Upload tailscale.state to HF (immediately, before vLLM boots)
7. Proceed to vLLM boot

On shutdown:
   trap → tailscale down → upload final state → exit
```

### Sequential Mode (`TAILSCALE_AUTH_KEY` + `ENABLE_TAILSCALE_SUFFIX=1`)

```
1. Install Tailscale from official apt repo
2. Find first free slot:
   N=0
   ├─ tailscale.state EXISTS on HF → "Slot 0 taken — trying next"
   ├─ tailscale-1.state EXISTS on HF → "Slot 1 taken — trying next"
   └─ tailscale-2.state MISSING → "Slot 2 free — claiming"
3. Hostname becomes: runpod-llm-2
4. Fresh identity (no state restore in sequential mode — new identity each time)
5. Start tailscaled in userspace-networking mode
6. tailscale up --hostname=runpod-llm-2
7. Print Tailscale IP
8. Upload tailscale-2.state to HF
9. Proceed to vLLM boot

On shutdown:
   trap → tailscale down → upload final state → exit
```

---

## State Persistence Mechanics

### What is `tailscale.state`?

A binary file produced by `tailscaled --state=<path>`. Contains:
- Machine key (hardware identity)
- Node key (network identity)  
- WireGuard private key
- Login session token
- Hostname preference

Uploaded to HF as a regular file in the dataset repo.

### Upload timing

State is uploaded at two points:
1. **Immediately after `tailscale up` succeeds** — ensures even if the pod is SIGKILL'd, state is saved (main path for Community Cloud)
2. **On graceful shutdown (trap EXIT/SIGTERM/SIGINT)** — best-effort backup

### Download timing

State is downloaded BEFORE `tailscale up`:
1. Download from HF
2. If download succeeds → restore identity (pod becomes the same node)
3. If download fails → fresh identity (new node key generated)

### Storage

Uses `hf download` and `hf upload` (modern HuggingFace CLI). The `HF_TOKEN` env var must be set if the dataset repo is private.

---

## Slot Discovery (Sequential Mode) — Deep Dive

### Problem

Two pods sharing the same `tailscale.state` = same WireGuard identity = only one can be connected at a time. The second pod kicks the first one off. Need a way to assign unique identities to simultaneous pods.

### Solution

Use sequential state file naming. Each pod gets a unique numbered slot:

| Pod | State file | Hostname |
|---|---|---|
| 1st | `tailscale.state` | `runpod-llm` |
| 2nd | `tailscale-1.state` | `runpod-llm-1` |
| 3rd | `tailscale-2.state` | `runpod-llm-2` |

### Algorithm

```
for N in 0, 1, 2, ...:
    sfx = (N==0) ? "" : "-{N}"
    try download tailscale{sfx}.state from HF
    if download succeeds:
        # Slot is taken (another pod created it)
        continue to next N
    else:
        # Slot is free
        hostname = base_hostname + sfx
        create fresh identity
        upload tailscale{sfx}.state to HF
        break
```

### Why download-based discovery works

- HF file existence is atomic — either the file exists or it doesn't
- No race condition: if two pods check simultaneously and both see the slot as free, the first one to upload "claims" it. The second one's upload will overwrite but by then both have already connected with different generated node keys. Since each pod generates its own node key on `tailscaled` start (fresh identity), there's no identity collision — just duplicate hostname which Tailscale handles via auto-suffixing.
- No timestamps, no heartbeats, no background processes needed

### Cleanup

Dead pod slots **accumulate** on HF. To free slots:
1. Stop all pods
2. Delete state files from HF dataset repo
3. Start pods fresh — sequential numbering restarts from slot 0

---

## Userspace Networking

### Why

RunPod containers (and most cloud GPU containers) lack `/dev/net/tun` — there's no kernel TUN device to create a virtual network interface.

### What it means

`tailscaled --tun=userspace-networking` runs the WireGuard tunnel entirely in userspace. Tailscale acts as a SOCKS5/HTTP proxy rather than creating a kernel network interface.

### Limitations (all acceptable for our use case)

| Feature | Works? | Notes |
|---|---|---|
| TCP connections (HTTP, vLLM API) | ✅ Yes | This is all we need |
| `tailscale ping` | ❌ No | Uses ICMP, not supported in userspace |
| Subnet routing | ❌ No | Needs kernel TUN |
| Exit node (routing all traffic) | ❌ No | Needs kernel TUN |
| Tailscale SSH | ❌ No | Needs kernel TUN |
| MagicDNS | ✅ Yes | Resolves `runpod-llm.yak-ya.ts.net` to Tailscale IP |

### Buffer size warning

The log warning about failed UDP buffer size is harmless — throughput impact for API traffic (small JSON payloads) is negligible.

---

## Accessing vLLM via Tailscale

### From any device on your tailnet

```bash
# Direct IP
curl http://100.112.244.67:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Hi"}],"max_tokens":30}'

# MagicDNS (if enabled on your tailnet)
curl http://runpod-llm.yak-ya.ts.net:8000/v1/chat/completions ...
```

### With API_KEY set

If `API_KEY` is set on the RunPod template, vLLM requires the `Authorization: Bearer` header. Without `API_KEY`, no auth required (open access within the tailnet).

### Connectivity model

Tailscale tries direct peer-to-peer first (WireGuard UDP). If NAT traversal fails (common on cloud GPU networks), it falls back to DERP relays. The `netstack` log messages about connection refused are the userspace networking stack probing for local services — expected and harmless.

---

## Known Limitations & Edge Cases

### SIGKILL on Community Cloud

RunPod Community Cloud pods may receive SIGKILL instead of SIGTERM. SIGKILL cannot be trapped — cleanup code does NOT run. Mitigation: state is uploaded immediately after `tailscale up` (before vLLM boots), so node identity is already persisted even if killed abruptly.

### Stale state files

In sequential mode, state files from dead pods accumulate on HF. No automatic cleanup. Manual cleanup: delete state files from the HF dataset repo.

### Clock skew

N/A — no timestamp-based logic. Discovery is purely file-existence-based.

### Auth key expiration

Tailscale auth keys expire after 90 days (default). Generate a new one and update the template env var before expiry. Node keys (device identity) auto-renew.

### Duplicate node key warning

The Tailscale admin console may show "Duplicate node key" when the same identity connects from a different IP (new pod). This is expected and harmless — Tailscale just notes the IP change. No action needed.

### Multiple pods, shared state (non-suffix mode)

If two pods share the same `tailscale.state` (both using the non-suffix stable-IP mode simultaneously), they fight over the same node identity. The last one to connect wins; the other gets silently disconnected. Use sequential mode for multi-pod setups.

### State file corruption

If `tailscale.state` is corrupted (partial upload, network error during upload), `tailscaled` will reject it and create a fresh identity. The pod will boot normally but lose its previous IP. No error handling needed — Tailscale handles this gracefully.

### HF rate limiting

During boot, `hf download` and `hf upload` are called a few times (download state, download heartbeat/check slot, upload state). Rate limits are unlikely to be hit with this volume. HF's free tier allows generous API usage.

### Tailscale free tier limits

- 100 devices per tailnet — fine for personal use
- 3 users — fine for solo/small team
- DERP relay bandwidth is adequate for API traffic (not streaming video)
```

## Quick Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Slot N taken" for first pod | Old state files on HF from previous tests | Delete all `tailscale*.state` files from HF repo |
| Two pods share same hostname | `ENABLE_TAILSCALE_SUFFIX=0` with both pods running | Set `ENABLE_TAILSCALE_SUFFIX=1` |
| `cuInit 999` error | RunPod GPU passthrough broken (Community Cloud specific) | Switch to Secure Cloud or retry on a different host |
| State not uploading to HF | `HF_TOKEN` not set or `hf` CLI not installed | `HF_TOKEN` must be set as RunPod secret; `hf` is pre-installed in the image |
| MagicDNS doesn't resolve | `systemd-resolved` not properly wired with NetworkManager | Use Tailscale IP directly, or fix DNS on your local machine |
```

## File Structure

```
HF dataset repo (e.g., fierysurf/runpod-llms-tailscale):
├── tailscale.state          # Slot 0: runpod-llm
├── tailscale-1.state        # Slot 1: runpod-llm-1
├── tailscale-2.state        # Slot 2: runpod-llm-2
└── ...

/workspace/ (pod-local, ephemeral):
├── tailscale.state          # Current pod's state file (downloaded or fresh)
└── tailscale-1.state        # (sequential mode, if this pod got slot 1)
```
