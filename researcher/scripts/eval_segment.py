"""Score audio_lab segment against a mix_truth.csv. Prints precision, recall, and F1.

Uses segment_file, the same detector as `python audio_lab.py segment`.
`--expect-tracks` is off, so the track count is never forced.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_lab import config, io
from audio_lab.segment import boundary_times, score_boundaries, segment_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mix chegaralarini haqiqat bilan solishtirish")
    parser.add_argument("--mix", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--min-track", type=float, default=config.SEG_MIN_TRACK_S)
    args = parser.parse_args(argv)
    try:
        report(Path(args.mix), Path(args.truth), args.min_track)
    except (io.AudioLabError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def report(mix: Path, truth: Path, min_track_s: float) -> None:
    if not mix.is_file():
        raise FileNotFoundError(f"Fayl topilmadi: {mix}")
    if not truth.is_file():
        raise FileNotFoundError(f"Fayl topilmadi: {truth}")
    truths = _truth_boundaries(truth)
    result = segment_file(mix, min_track_s=min_track_s)
    predicted = boundary_times(result)
    print(f"mix: {mix}")
    print(f"truth: {truth}")
    print(f"min_track_s: {min_track_s}")
    print(f"k: {result['threshold_k']}  attempts: {result['threshold_attempts']}")
    if result.get("warning"):
        print(result["warning"])
    print(f"truth_boundaries: {len(truths)}  detected: {len(predicted)}")
    print()
    print(f"{'tol_s':>6}  {'precision':>9}  {'recall':>8}  {'f1':>8}")
    for tolerance in (3.0, 5.0):
        score = score_boundaries(predicted, truths, tolerance)
        print(
            f"{tolerance:6.1f}  {score['precision']:9.4f}  {score['recall']:8.4f}  "
            f"{_f1(score['precision'], score['recall']):8.4f}"
        )
    print()
    print(f"{'time_s':>8}  {'strength':<10}  {'transition_in':<12}  {'nearest_s':>10}  {'error_s':>8}")
    for row in result["boundaries"]:
        nearest, error = _nearest(row["time_s"], truths)
        nearest_text = "" if nearest is None else f"{nearest:.1f}"
        error_text = "" if error is None else f"{error:.1f}"
        print(
            f"{row['time_s']:8.1f}  {row['strength']:<10}  {row['transition']['kind']:<12}  "
            f"{nearest_text:>10}  {error_text:>8}"
        )
    missed = [truth for truth in truths if all(abs(guess - truth) > 5.0 for guess in predicted)]
    if missed:
        print()
        print("±5 s ichida topilmagan haqiqat:", ", ".join(f"{item:.1f}" for item in missed))


def _truth_boundaries(path: Path) -> list[float]:
    starts = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "start_s" not in reader.fieldnames:
            raise ValueError(f"truth csv da start_s ustuni yo'q: {path.name}")
        for row in reader:
            if row.get("start_s"):
                starts.append(float(row["start_s"]))
    # The first cue is the file start, not a boundary between tracks.
    return [start for start in starts if start > 0.5]


def _nearest(time_s: float, truths: list[float]) -> tuple[float | None, float | None]:
    if not truths:
        return None, None
    nearest = min(truths, key=lambda truth: abs(truth - time_s))
    return nearest, abs(nearest - time_s)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


if __name__ == "__main__":
    raise SystemExit(main())
