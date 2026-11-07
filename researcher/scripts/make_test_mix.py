"""Join a folder of audio files in name order with ffmpeg acrossfade.

Truth boundaries are the middle of each crossfade. A crossfade of 0 is a hard cut.
The wav and the truth csv are local outputs and are gitignored.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_lab import config, io

ALLOWED_CROSSFADE_S = (0, 2, 4, 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Papkadagi treklarni bitta mix ga ulash")
    parser.add_argument("folder", help="Audio treklar papkasi")
    parser.add_argument("--crossfade-s", type=int, required=True, choices=ALLOWED_CROSSFADE_S)
    parser.add_argument("--out", required=True, help="Chiqish wav, masalan mix.wav")
    args = parser.parse_args(argv)
    try:
        out, truth = build_mix(Path(args.folder), args.crossfade_s, Path(args.out))
    except (io.AudioLabError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(out)
    print(truth)
    return 0


def build_mix(folder: Path, crossfade_s: int, out: Path) -> tuple[Path, Path]:
    if crossfade_s not in ALLOWED_CROSSFADE_S:
        raise io.AudioLabError("--crossfade-s faqat 0, 2, 4 yoki 6 bo'lishi mumkin.")
    tracks = _tracks(folder)
    if len(tracks) < 2:
        raise io.AudioLabError(f"Kamida 2 ta audio trek kerak: {folder}")
    durations = []
    for path in tracks:
        info = io.probe(path)
        duration = info.get("duration_s")
        if not duration or duration <= 0:
            raise io.AudioLabError(f"Davomiylik yo'q: {path.name}")
        if crossfade_s and duration <= crossfade_s:
            raise io.AudioLabError(
                f"{path.name} crossfade ({crossfade_s} s) dan uzun emas ({duration:.2f} s)."
            )
        durations.append(float(duration))
    out.parent.mkdir(parents=True, exist_ok=True)
    _render(tracks, crossfade_s, out)
    starts = _starts(durations, crossfade_s)
    truth = out.with_name(f"{out.stem}_truth.csv")
    with truth.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_s"])
        writer.writeheader()
        for start in starts:
            writer.writerow({"start_s": f"{start:.4f}"})
    made = io.probe(out).get("duration_s") or 0.0
    expected = sum(durations) - crossfade_s * (len(durations) - 1)
    if abs(made - expected) > 1.0:
        raise io.AudioLabError(
            f"Mix uzunligi kutilganidan farq qiladi: {made:.2f} s, kutilgan {expected:.2f} s."
        )
    return out, truth


def _tracks(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Papka topilmadi: {folder}")
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in config.AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name)


def _starts(durations: list[float], crossfade_s: int) -> list[float]:
    """Track starts. Later tracks begin at the middle of the crossfade."""
    starts = [0.0]
    output_len = durations[0]
    for duration in durations[1:]:
        starts.append(output_len - crossfade_s / 2.0)
        output_len = output_len + duration - crossfade_s
    return starts


def _render(tracks: list[Path], crossfade_s: int, out: Path) -> None:
    cmd = [io._tool("ffmpeg"), "-y"]
    for path in tracks:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            _filter(len(tracks), crossfade_s),
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not out.is_file():
        detail = (proc.stderr or b"").decode("utf-8", errors="replace")[-600:]
        raise io.AudioLabError(f"ffmpeg mix yig'a olmadi: {out.name}. {detail}")


def _filter(count: int, crossfade_s: int) -> str:
    parts = []
    labels = []
    for index in range(count):
        parts.append(
            f"[{index}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{index}]"
        )
        labels.append(f"[a{index}]")
    if crossfade_s == 0:
        joined = "".join(labels)
        parts.append(f"{joined}concat=n={count}:v=0:a=1[out]")
        return ";".join(parts)
    current = labels[0]
    for index in range(1, count):
        nxt = "[out]" if index == count - 1 else f"[m{index}]"
        parts.append(f"{current}{labels[index]}acrossfade=d={crossfade_s}:c1=tri:c2=tri{nxt}")
        current = nxt
    return ";".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
