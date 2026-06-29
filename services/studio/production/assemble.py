"""Post-production: the final ffmpeg mix.

`build_mix_command` is PURE (inputs -> argv) so it's unit-testable without running
ffmpeg; `assemble` runs it.

Video: clips are joined with per-seam transitions — `dissolve` (xfade crossfade,
the default) or `cut`. A dissolve overlaps two clips by `transition_seconds`, so it
shortens the timeline; clip start times are computed crossfade-aware and the
narration is delayed to those (shifted) starts.

Audio: each narration WAV is laid at its shot's start (+ intra-shot offset), the
music bed is ducked under the narration (sidechaincompress, gentle defaults so it
glides instead of pumping), then loudnorm to the delivery target, one MP4. The exact
argv is recorded in the manifest. Reuses the studio ffmpeg idioms from tts.py.
"""
from __future__ import annotations

from .util import sh

# A "cut" seam inside an xfade chain is faked with a near-instant blend. It MUST be ≥ ~1 frame
# at the output fps — a sub-frame xfade duration (the old 1/30 s = 0.033 s, < 1 frame at 24 fps)
# silently produces a degenerate transition that collapses the whole chain to one clip's length
# (surfaced on a 6-shot LTX/Sulphur noir with cut seams, 2026-06-29). fps-aware below.
_CUT_FRAMES = 2.0   # 2 frames: imperceptible (reads as a cut) but valid for xfade


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def duck_ratio(duck_db: int) -> int:
    """Map a duck depth (dB) to a sidechaincompress ratio (gentle floor)."""
    return max(2, min(20, round(duck_db / 1.5)))


def _clip_starts(durations: list[float], xfades: list[float]) -> list[float]:
    """Output start time of each clip, given per-seam crossfade overlaps.

    xfades[k] is the crossfade seconds for the seam between clip k and k+1.
    """
    starts = [0.0]
    for k in range(1, len(durations)):
        starts.append(starts[-1] + durations[k - 1] - xfades[k - 1])
    return starts


def build_mix_command(
    final_path: str,
    clips: list[str],
    durations: list[float],
    transitions: list[tuple[str, float]],     # per seam (len = len(clips)-1): (type, seconds)
    narrations: list[tuple[str, int, float]],  # (wav_path, shot_index, intra_offset_s)
    bed: str | None,
    *,
    fps: int,
    lufs: float,
    bed_level_db: float,
    duck_db: int = 6,
) -> list[str]:
    """Build the single-pass ffmpeg argv. Pure — no side effects."""
    n = len(clips)
    if n == 0:
        raise ValueError("no clips to assemble")
    if len(durations) != n:
        raise ValueError("durations must match clips")
    if len(transitions) != max(0, n - 1):
        raise ValueError("transitions must have one entry per seam (clips-1)")

    # effective crossfade per seam: dissolve -> its seconds; cut -> a 2-frame blend at the
    # output fps (NOT sub-frame, which collapses the xfade chain). Never longer than the seam's
    # own dissolve seconds would be, and clamped below each clip so the offset math stays valid.
    cut_xf = _CUT_FRAMES / float(max(1, fps))
    xf = [secs if typ == "dissolve" else cut_xf for (typ, secs) in transitions]
    all_cut = all(typ == "cut" for (typ, _s) in transitions)
    starts = (
        [sum(durations[:k]) for k in range(n)] if (all_cut or n == 1)
        else _clip_starts(durations, xf)
    )

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", c]
    for (p, _i, _o) in narrations:
        inputs += ["-i", p]
    if bed:
        inputs += ["-i", bed]

    fc: list[str] = []
    # 1) video — concat (all hard cuts) or an xfade dissolve chain
    if all_cut or n == 1:
        fc.append("".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vid]")
    else:
        prev = "[0:v]"
        for k in range(1, n):
            out = "[vid]" if k == n - 1 else f"[vx{k}]"
            fc.append(
                f"{prev}[{k}:v]xfade=transition=dissolve:"
                f"duration={xf[k - 1]:.3f}:offset={starts[k]:.3f}{out}"
            )
            prev = out

    # 2) narration timeline — delay each VO to its (crossfade-aware) shot start
    m = len(narrations)
    narr_label = None
    if m:
        for j, (_p, shot_idx, intra) in enumerate(narrations):
            start_ms = max(0, int((starts[shot_idx] + intra) * 1000))
            fc.append(f"[{n + j}:a]adelay={start_ms}|{start_ms}[n{j}]")
        if m == 1:
            narr_label = "[n0]"
        else:
            fc.append("".join(f"[n{j}]" for j in range(m)) + f"amix=inputs={m}:normalize=0[narr]")
            narr_label = "[narr]"

    # 3) bed + (gentle) duck + final audio
    bidx = n + m
    if bed and m:
        fc.append(f"{narr_label}asplit=2[nkey][nmix]")
        fc.append(f"[{bidx}:a]volume={db_to_linear(bed_level_db):.4f}[bedv]")
        fc.append(
            f"[bedv][nkey]sidechaincompress=threshold=0.05:ratio={duck_ratio(duck_db)}:"
            f"attack=20:release=600[bedduck]"
        )
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


def assemble(final_path, clips, durations, transitions, narrations, bed, *,
             fps, lufs, bed_level_db, duck_db=6) -> list[str]:
    """Build + run the mix. Returns the exact argv used (for the manifest)."""
    cmd = build_mix_command(final_path, clips, durations, transitions, narrations, bed,
                            fps=fps, lufs=lufs, bed_level_db=bed_level_db, duck_db=duck_db)
    sh(cmd, timeout=900)
    return cmd
