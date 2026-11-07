"""Audio Audit Lab command line. Local files only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio_lab import config, io
from audio_lab.compare import compare_track
from audio_lab.profile import build_profile
from audio_lab.chapters import parse_tracklist
from audio_lab.report import analyze_to_disk, long_file_warning, write_segment_bundle
from audio_lab.segment import read_manual_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audio Audit Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", required=True, help="Audio fayl yoki papka")
    analyze.add_argument("--out", default=str(config.DEFAULT_OUT_DIR))
    analyze.add_argument("--clip", nargs=2, type=float, metavar=("START", "END"))
    analyze.add_argument("--stems", action="store_true")
    analyze.add_argument("--tags", action="store_true")
    analyze.add_argument("--sr", type=int, default=config.FEATURE_SR)
    analyze.add_argument("--refresh", action="store_true")
    analyze.add_argument("--segment", action="store_true")
    analyze.add_argument("--min-track", type=float, default=config.SEG_MIN_TRACK_S)
    analyze.add_argument("--expect-tracks", default=None)

    profile = sub.add_parser("profile")
    profile.add_argument("--dir", required=True, dest="directory")
    profile.add_argument("--name", required=True)
    profile.add_argument("--stems", action="store_true")
    profile.add_argument("--tags", action="store_true")
    profile.add_argument("--sr", type=int, default=config.FEATURE_SR)
    profile.add_argument("--refresh", action="store_true")
    profile.add_argument("--per-track", action="store_true")

    segment = sub.add_parser("segment")
    segment.add_argument("--input", required=True)
    segment.add_argument("--out", default=str(config.DEFAULT_OUT_DIR))
    segment.add_argument("--expect-tracks", default=None)
    segment.add_argument("--min-track", type=float, default=config.SEG_MIN_TRACK_S)
    segment.add_argument("--boundaries", default=None)
    segment.add_argument("--tracklist-text", default=None)
    segment.add_argument("--export-segments", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("--track", required=True)
    compare.add_argument("--profile", required=True)
    compare.add_argument("--clip", nargs=2, type=float, metavar=("START", "END"))
    compare.add_argument("--stems", action="store_true")
    compare.add_argument("--tags", action="store_true")
    compare.add_argument("--sr", type=int, default=config.FEATURE_SR)

    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "segment":
            return _segment(args)
        if args.command == "profile":
            path = build_profile(
                args.directory,
                args.name,
                sr=args.sr,
                stems=args.stems,
                tags=args.tags,
                refresh=args.refresh,
                per_track=args.per_track,
            )
            print(path)
            return 0
        text = compare_track(
            args.track,
            args.profile,
            sr=args.sr,
            clip=tuple(args.clip) if args.clip else None,
            stems=args.stems,
            tags=args.tags,
        )
        print(text)
        return 0
    except (io.AudioLabError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _analyze(args) -> int:
    source = Path(args.input)
    clip = tuple(args.clip) if args.clip else None
    if source.is_dir():
        files = [
            path
            for path in sorted(source.iterdir())
            if path.is_file() and path.suffix.lower() in config.AUDIO_EXTENSIONS
        ]
    elif source.is_file():
        files = [source]
    else:
        raise io.AudioLabError(f"Fayl topilmadi: {source}")
    if not files:
        raise io.AudioLabError(f"Audio fayl topilmadi: {source}")
    for path in files:
        info = io.probe(path)
        warning = long_file_warning(info.get("duration_s"), args.segment)
        if warning:
            print(warning, file=sys.stderr)
        payload = analyze_to_disk(
            path,
            out_dir=args.out,
            sr=args.sr,
            clip=clip,
            stems=args.stems,
            tags=args.tags,
            refresh=args.refresh,
            segment=args.segment,
            min_track_s=args.min_track,
            expect_tracks=_expect(args.expect_tracks),
        )
        print(payload["file_name"], payload["hash8"])
    return 0


def _segment(args) -> int:
    source = Path(args.input)
    if not source.is_file():
        raise io.AudioLabError(f"Fayl topilmadi: {source}")
    manual = read_manual_csv(args.boundaries) if args.boundaries else None
    tracklist = None
    if manual is None and args.tracklist_text:
        tracklist = parse_tracklist(Path(args.tracklist_text).read_text(encoding="utf-8"))
    dest = write_segment_bundle(
        source,
        out_dir=args.out,
        min_track_s=args.min_track,
        expect_tracks=_expect(args.expect_tracks),
        manual_starts=manual,
        tracklist=tracklist,
        export_segments=args.export_segments,
    )
    saved = json.loads((dest / "segment.json").read_text(encoding="utf-8"))
    if saved.get("warning"):
        print(saved["warning"], file=sys.stderr)
    print(dest)
    return 0


def _expect(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    low, high = text.split("-", 1)
    return int(low), int(high)


if __name__ == "__main__":
    raise SystemExit(main())
