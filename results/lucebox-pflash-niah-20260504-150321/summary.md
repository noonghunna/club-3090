# Dual-GPU PFlash Phase-Split Report

- PFlash GPU: `0`
- PFlash daemon ready: `0.63 s`
- keep ratio: `0.05`
- lookahead: `2`
- PFlash K cache: `compute`

## Resource Peak

| gpu | samples | peak mem MiB | peak temp C | avg power W | peak power W | avg util % | peak util % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 21024.00 | 61.00 | 198.71 | 228.89 | 88.70 | 100.00 |

## Cases

| case | source tokens | compressed tokens | ratio | PFlash s | PFlash tok/s | key retained | answer retained |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| niah_ctx16372 | 16372 | 788 | 0.0481 | 1.08 | 15117.23 | yes | yes |
| niah_ctx32764 | 32764 | 1628 | 0.0497 | 1.80 | 18204.51 | yes | yes |
| niah_ctx65524 | 65524 | 3252 | 0.0496 | 4.37 | 15009.40 | yes | yes |
| niah_ctx131068 | 131068 | 6524 | 0.0498 | 10.80 | 12134.52 | yes | yes |
| niah_ctx199996 | 199996 | 0 | 0.0000 | 0.10 | 1956471.24 | no | no |
| niah_ctx259996 | 259996 | 0 | 0.0000 | 0.09 | 2847516.82 | no | no |

Files:
- pflash: `/opt/ai/github/club-3090/results/lucebox-pflash-niah-20260504-150321/pflash_daemon.log`
- monitor: `/opt/ai/github/club-3090/results/lucebox-pflash-niah-20260504-150321/gpu_monitor.csv`
