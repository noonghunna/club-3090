# Troubleshooting

Something isn't working, or you want to share what your rig does. Start with the report; if the
launcher itself is the problem, boot the compose directly.

For symptom-by-symptom help ("OOM after a few turns", "TPS is low", "tool calls come back empty"),
see the [FAQ](FAQ.md), starting with its
[troubleshooting ladder](FAQ.md#troubleshooting-ladder--boot-the-simplest-stack-first).

## Generate a report

`report.sh` writes a paste-ready markdown report: hardware, OS, GPUs, power limits, container
runtime, stack version, your saved settings and the running container's state. **Home paths,
hostnames, usernames, Hugging Face tokens and every secret in your settings are redacted by
default**, so it's safe to paste into a public issue.

```bash
bash scripts/report.sh                     # ~2 s: hardware + stack + boot-log highlights
bash scripts/report.sh --full > my-rig.md  # ~43 min: the full cross-rig pass (see below)
```

Add live test output for what the thread needs:

| Flag | Adds | Time |
|---|---|--:|
| `--verify` | `verify-full.sh` | 1–2 min |
| `--stress` | `verify-stress.sh` (long-context ladder, tool-prefill OOM) | 5–10 min |
| `--soak` | continuous soak: the only test that catches the multi-turn Cliff 2b | ~25 min |
| `--bench` | `bench.sh` throughput | ~3 min |
| `--agentic` | `bench-agentic.sh` curve | ~8 min |
| `--full` | all five | ~43 min |
| `--studio` | AI Studio container logs (image / video / audio bugs) | ~2 s |
| `--engine-args` | the engine's full startup dump | ~2 s |
| `--no-redact` | turns redaction off (internal sharing only) | |

A config can pass verify, stress and bench and still fail the soak: see [CLIFFS.md](CLIFFS.md).
To contribute numbers, follow [Run the evals yourself](RUN_EVALS.md).

## If `launch.sh` / `switch.sh` won't boot: load the compose directly

The launchers wrap the boot in a preflight (hardware and free-VRAM checks), your saved settings and
the slug → compose registry. If one of those misfires (a false preflight failure, a saved setting
you can't track down, missing PyYAML), go around them. This assumes the weights are already
downloaded. (`bash scripts/settings.sh show` lists every saved setting and where it comes from.)

```bash
# 1. Skip only the hardware / free-VRAM preflight (keeps your settings and the registry):
bash scripts/switch.sh --force <slug>

# 2. Skip the scripts entirely: boot the compose file with Docker.
#    Layout: models/<model>/<engine>/compose/<topology>/<quant>/<serving>.yml
#    MODEL_DIR is where your weights live; it's mounted into the container.

# for example, the Qwen3.6-27B dual-card default (serves on :8010)
MODEL_DIR=/path/to/models docker compose \
  -f models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml up -d

# or MiMo 9B on one card (serves on :8094)
MODEL_DIR=/path/to/models docker compose \
  -f models/mimo-v2.6-9b/llama-cpp/compose/single/bartowski-q8/full-vision.yml up -d

# check it's serving (use that compose's port), then stop it the same way:
curl -s http://localhost:8010/v1/models
docker compose -f <the-same-compose-file> down
```

`MODEL_DIR` is the only variable you must set; it defaults to the in-repo `models-cache/`.
Everything else has a default, and each compose's **header** documents its own overrides and its
port. Booting directly doesn't read your saved settings, so pass anything else you rely on the same
way, or hand compose all your settings with `envf="$(bash scripts/settings.sh compose-env-file)"` and `--env-file "$envf"` (remove the file afterwards: it holds your secrets too). The card
pages ([single](SINGLE_CARD.md) · [dual](DUAL_CARD.md) · [multi](MULTI_CARD.md)) link every slug's
compose file, and a successful `switch.sh` run prints the path it used.

> ⚠️ Booting directly skips the preflight that catches not-enough-VRAM and wrong-GPU-count
> mistakes. If the container exits, check `docker logs <container> 2>&1 | tail -50`.
