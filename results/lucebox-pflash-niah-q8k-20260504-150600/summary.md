# Dual-GPU PFlash Phase-Split Report

- PFlash GPU: `0`
- PFlash daemon ready: `0.60 s`
- keep ratio: `0.05`
- lookahead: `2`
- PFlash K cache: `q8_0`

## Resource Peak

| gpu | samples | peak mem MiB | peak temp C | avg power W | peak power W | avg util % | peak util % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 21024.00 | 54.00 | 190.07 | 229.25 | 82.00 | 100.00 |

## Cases

| case | source tokens | compressed tokens | ratio | PFlash s | PFlash tok/s | key retained | answer retained |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| niah_ctx131068 | 131068 | 6524 | 0.0498 | 10.65 | 12302.49 | yes | yes |
| niah_ctx199996 | 199996 | 0 | 0.0000 | 0.10 | 1952806.29 | no | no |
| niah_ctx259996 | 259996 | 0 | 0.0000 | 0.10 | 2510841.84 | no | no |

Files:
- pflash: `/opt/ai/github/club-3090/results/lucebox-pflash-niah-q8k-20260504-150600/pflash_daemon.log`
- monitor: `/opt/ai/github/club-3090/results/lucebox-pflash-niah-q8k-20260504-150600/gpu_monitor.csv`
