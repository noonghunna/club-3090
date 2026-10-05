"""Page state of an ExLlamaV3 CPU-MoE server's expert arena (club-3090#1542).

Run INSIDE the serving container (`exl3_arena_state` in exl3-arena.sh pipes this file to
`docker exec -i <container> python3 -`). It sums the `multiprocessing.spawn` workers, which hold
the arena, and leaves out the parent `main.py`: the parent maps the pinned arena too and carries
tens of GB of transparent huge pages on its own heap, so counting it reads "huge" whatever the
experts are on. Prints one key=value line.

Whether the experts sit on 2 MiB pages is ~19% of decode on exl3 (#1542), and the default arena
gets there only through a one-shot background collapse that can fail silently. verdict=4KiB means
numbers from this boot read low, not that the engine regressed.
"""
import os
import re

# A fixture tree stands in for /proc in scripts/tests/test-exl3-arena.sh.
PROC = os.environ.get("EXL3_ARENA_PROC", "/proc")

workers = rss = hugetlb = anon_thp = shmem_thp = 0
for pid in os.listdir(PROC):
    if not pid.isdigit():
        continue
    try:
        with open(f"{PROC}/{pid}/cmdline", "rb") as fh:
            if b"multiprocessing.spawn" not in fh.read():
                continue
        with open(f"{PROC}/{pid}/smaps_rollup", encoding="utf-8") as fh:
            roll = fh.read()
    except OSError:
        continue
    f = {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+) kB", roll, re.M)}
    workers += 1
    rss += f.get("Rss", 0)                     # excludes hugetlb pages
    hugetlb += f.get("Shared_Hugetlb", 0) + f.get("Private_Hugetlb", 0)
    anon_thp += f.get("AnonHugePages", 0)      # default (unpinned) arena, once collapsed
    shmem_thp += f.get("ShmemPmdMapped", 0)    # pinned memfd arena under shmem THP


def gib(kib):
    return f"{kib / 1048576:.1f}"


resident = rss + hugetlb
huge = hugetlb + anon_thp + shmem_thp
if not workers or resident < 1048576:
    # Under 1 GiB in the workers: no arena resident (not CPU-MoE, or still loading).
    print(f"workers={workers} verdict=none")
else:
    frac = huge / resident
    print(f"workers={workers} resident_gib={gib(resident)} huge_gib={gib(huge)} "
          f"hugetlb_gib={gib(hugetlb)} anon_thp_gib={gib(anon_thp)} shmem_thp_gib={gib(shmem_thp)} "
          f"frac={frac:.2f} verdict={'2MiB' if frac >= 0.5 else '4KiB'}")
