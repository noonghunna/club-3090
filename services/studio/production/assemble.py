"""Post-production: the final ffmpeg mix.

`build_mix_command` is PURE (inputs -> argv) so it's unit-testable without running
ffmpeg; `assemble` runs it. One pass: concat the silent clips, lay each narration
WAV at its timeline offset (adelay), duck the music bed under the narration
(sidechaincompress), loudnorm to the delivery target, mux to one MP4. The exact
argv is recorded in the manifest for reproducibility.

Reuses the studio ffmpeg idioms (loudnorm I=…:TP=-1.5:LRA=11, sidechaincompress)
from tts/tts.py `_mixdown`.
"""
from __future__ import annotations

from .util import sh


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def build_mix_command(
    final_path: str,
    clips: list[str],
    narrations: list[tuple[str, int]],   # (wav_path, start_ms)
    bed: str | None,
    *,
    fps: int,
    lufs: float,
    bed_level_db: float,
    duck_ratio: int = 8,
) -> list[str]:
    """Build the single-pass ffmpeg argv. Pure — no side effects."""
    n = len(clips)
    m = len(narrations)
    if n == 0:
        raise ValueError("no clips to assemble")

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", c]
    for (p, _) in narrations:
        inputs += ["-i", p]
    if bed:
        inputs += ["-i", bed]

    fc: list[str] = []
    # 1) video concat
    fc.append("".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vid]")

    # 2) narration timeline (delay each VO to its start, mix together)
    narr_label = None
    if m:
        for j, (_, start) in enumerate(narrations):
            fc.append(f"[{n + j}:a]adelay={start}|{start}[n{j}]")
        if m == 1:
            narr_label = "[n0]"
        else:
            fc.append("".join(f"[n{j}]" for j in range(m)) + f"amix=inputs={m}:normalize=0[narr]")
            narr_label = "[narr]"

    # 3) bed + duck + final audio
    bidx = n + m
    if bed and m:
        # split narration: one copy keys the sidechain, one is mixed back in
        fc.append(f"{narr_label}asplit=2[nkey][nmix]")
        fc.append(f"[{bidx}:a]volume={db_to_linear(bed_level_db):.4f}[bedv]")
        fc.append(f"[bedv][nkey]sidechaincompress=threshold=0.03:ratio={duck_ratio}:attack=5:release=250[bedduck]")
        fc.append(f"[bedduck][nmix]amix=inputs=2:normalize=0[premix]")
        pre = "[premix]"
    elif bed:
        fc.append(f"[{bidx}:a]volume={db_to_linear(bed_level_db):.4f}[premix]")
        pre = "[premix]"
    elif m:
        pre = narr_label
    else:
        raise ValueError("nothing to put on the audio track (no narration, no bed)")

    fc.append(f"{pre}loudnorm=I={lufs}:TP=-1.5:LRA=11[a]")

    return (
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(fc),
         "-map", "[vid]", "-map", "[a]",
         "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", final_path]
    )


def assemble(final_path, clips, narrations, bed, *, fps, lufs, bed_level_db) -> list[str]:
    """Build + run the mix. Returns the exact argv used (for the manifest)."""
    cmd = build_mix_command(final_path, clips, narrations, bed,
                            fps=fps, lufs=lufs, bed_level_db=bed_level_db)
    sh(cmd, timeout=900)
    return cmd
