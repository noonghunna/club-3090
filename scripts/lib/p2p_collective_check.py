#!/usr/bin/env python3
"""
Collective-level P2P validation — the tier below `nvidia-smi topo -p2p` and copy benchmarks.

WHY THIS EXISTS
  A driver reporting `topo -p2p = OK`, a passing `p2pBandwidthLatencyTest`, and a successful
  `cuMemSetAccess` do NOT prove NCCL works. All three were true, on the same rig, while every
  collective deadlocked — and one NCCL setting made them return wrong data instead.

  The reason is structural: a copy benchmark *synchronizes after* the transfer, so it can ride
  completion-time flushing. NCCL's producer and consumer kernels stay resident and must observe
  each other's writes while both are still running. Any copy-then-sync test passes on a mapping
  where NCCL cannot work.

WHAT IT DOES
  Runs the same tiny all-reduce twice:
    ARM A — P2P as the driver currently has it
    ARM B — NCCL_P2P_DISABLE=1 (host-staged control; should ALWAYS pass)
  Small (4 B) and large (16 MiB) payloads, because NCCL selects different protocols by size
  (LL vs Simple) and they can fail differently.

  Ranks contribute 1.0 and 2.0, so every element must return exactly 3.0. Correctness is checked,
  not merely completion — a collective that finishes with the wrong answer is worse than one that
  hangs, and that failure mode is real (NCCL_P2P_USE_CUDA_MEMCPY=1 produced it).

  Arm B is what makes the result trustworthy: if the control also fails, the verdict is
  INCONCLUSIVE rather than a false P2P accusation.

SAFETY
  Read-only. ~16 MiB per GPU. Changes no configuration, writes no files. Each arm runs in its own
  process group under a hard timeout, so a hang is killed rather than left wedging the machine.

USAGE
  python3 p2p_collective_check.py            # uses GPUs 0,1
  GPUS=0,3 TIMEOUT_S=120 python3 p2p_collective_check.py

  Normally invoked via scripts/p2p-validate.sh, which runs it inside a container that has torch.

EXIT CODES
  0 healthy · 2 collectives hang · 3 collectives return wrong data · 4 inconclusive · 5 unusable
"""

import os
import signal
import subprocess
import sys

TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "90"))
GPUS = os.environ.get("GPUS", "0,1")
BASE_PORT = int(os.environ.get("BASE_PORT", "29517"))
SIZES = [("4 B    (small -> LL proto)", 1), ("16 MiB (large -> Simple proto)", 4 * 1024 * 1024)]

EXIT_HEALTHY, EXIT_HANG, EXIT_WRONG, EXIT_INCONCLUSIVE, EXIT_UNUSABLE = 0, 2, 3, 4, 5


# ----------------------------------------------------------------- worker side
def _worker(rank, world, port):
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    expected = float(world * (world + 1) // 2)  # 1+2 = 3.0 at world=2
    ok = True
    for label, n in SIZES:
        t = torch.full((n,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.float32)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        wrong = int((t != expected).sum().item())
        if rank == 0:
            got = t.flatten()[0].item()
            verdict = "OK" if wrong == 0 else f"WRONG ({wrong}/{n} elements)"
            print(f"    {label:<34} expect {expected}  got {got}  -> {verdict}", flush=True)
        ok = ok and wrong == 0

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("WORKER_VERDICT=" + ("PASS" if ok else "WRONG_DATA"), flush=True)
    sys.exit(0 if ok else 2)


def _run_worker():
    import torch.multiprocessing as mp
    mp.spawn(_worker, args=(2, int(os.environ["P2PCHECK_PORT"])), nprocs=2, join=True)


# ----------------------------------------------------------------- driver side
def _arm(name, extra_env, port):
    """Run one arm in its own process group. Returns PASS | WRONG | HANG | ERROR."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["P2PCHECK_WORKER"] = "1"
    env["P2PCHECK_PORT"] = str(port)          # distinct per arm: a killed arm can hold its port
    env.update(extra_env)

    print(f"\n  {name}")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_S)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        # kill the whole group — mp.spawn children outlive a bare proc.kill()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        print(f"    HUNG — no result in {TIMEOUT_S}s (killed)")
        return "HANG"

    for line in out.splitlines():
        if line.startswith("    ") or "Error" in line or "error" in line:
            print(line if line.startswith("    ") else f"    {line.strip()[:150]}")
    if "WORKER_VERDICT=PASS" in out:
        return "PASS"
    if "WORKER_VERDICT=WRONG_DATA" in out:
        return "WRONG"
    print(f"    (arm did not report a verdict; exit={rc})")
    return "ERROR"


def verdict(a, b):
    """Pure verdict logic → (message, exit_code). Kept separate and side-effect free
    because every FAILURE branch is unreachable on a healthy rig — the only way to
    verify them is a test. See scripts/tests/test-p2p-validate.sh."""
    if b != "PASS":
        return ("  INCONCLUSIVE — the P2P-off control also failed, so something unrelated is\n"
                "  wrong (install / driver / GPU visibility). This is NOT a P2P finding.",
                EXIT_INCONCLUSIVE)
    if a == "PASS":
        return ("  HEALTHY — collectives complete correctly with P2P as configured.",
                EXIT_HEALTHY)
    if a == "HANG":
        return ("  *** BUG — collectives HANG with P2P on, but pass with it off. ***\n"
                "  Workaround: NCCL_P2P_DISABLE=1 (or NCCL_P2P_LEVEL=PXB), or NVLINK_MODE=force_off.",
                EXIT_HANG)
    if a == "WRONG":
        return ("  *** SERIOUS — collectives COMPLETE WITH WRONG DATA with P2P on. ***\n"
                "  Silent corruption. Set NCCL_P2P_DISABLE=1 now and treat any multi-GPU output\n"
                "  from this rig as suspect until it is resolved.",
                EXIT_WRONG)
    return (f"  ARM A errored ({a}) — see output above.", EXIT_INCONCLUSIVE)


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def main():
    bar = "=" * 76
    print(bar); print("  P2P COLLECTIVE CHECK"); print(bar)

    gpus = _sh(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"])
    if not gpus:
        print("\n  ERROR: nvidia-smi unavailable.")
        return EXIT_UNUSABLE
    print("\n  " + "\n  ".join(gpus.splitlines()[:8]))
    if len(gpus.splitlines()) < 2:
        print("\n  Only one GPU visible — a collective needs two. Nothing to check.")
        return EXIT_UNUSABLE

    topo = _sh(["nvidia-smi", "topo", "-p2p", "r"])
    if topo:
        print("\n  topo -p2p r:")
        print("  " + "\n  ".join(l for l in topo.splitlines()[:6] if l.strip()))
    lic = _sh(["modinfo", "-F", "license", "nvidia"])
    if lic:
        flavour = "OPEN — P2P-capable" if "MIT" in lic else "proprietary — refuses P2P on GeForce"
        print(f"\n  driver module license: {lic}   ({flavour})")

    a = _arm("ARM A — P2P as your driver has it", {}, BASE_PORT)
    b = _arm("ARM B — NCCL_P2P_DISABLE=1 (control)", {"NCCL_P2P_DISABLE": "1"}, BASE_PORT + 1)

    print("\n" + bar)
    print(f"  ARM A (P2P as-is)        : {a}")
    print(f"  ARM B (P2P off, control) : {b}")
    print(bar)

    msg, rc = verdict(a, b)
    print(msg)
    print(bar)
    if rc != EXIT_HEALTHY:
        print("\n  Why the usual checks miss this: a copy benchmark synchronizes AFTER the")
        print("  transfer, so it can ride completion-time flushing. NCCL's producer and consumer")
        print("  kernels stay resident and must observe each other's writes mid-flight. Any")
        print("  copy-then-sync test therefore passes on a mapping where NCCL cannot work.")
    print()
    return rc


if __name__ == "__main__":
    if os.environ.get("P2PCHECK_WORKER"):
        _run_worker()
    else:
        sys.exit(main())
