"""Audio Audit Lab command line. Local files only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audio_lab import config, io
from audio_lab.compare import compare_track
from audio_lab.profile import build_profile
from audio_lab.report import analyze_to_disk


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

    profile = sub.add_parser("profile")
    profile.add_argument("--dir", required=True, dest="directory")
    profile.add_argument("--name", required=True)
    profile.add_argument("--stems", action="store_true")
    profile.add_argument("--tags", action="store_true")
    profile.add_argument("--sr", type=int, default=config.FEATURE_SR)
    profile.add_argument("--refresh", action="store_true")

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
        if args.command == "profile":
            path = build_profile(
                args.directory,
                args.name,
                sr=args.sr,
                stems=args.stems,
                tags=args.tags,
                refresh=args.refresh,
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
        payload = analyze_to_disk(
            path,
            out_dir=args.out,
            sr=args.sr,
            clip=clip,
            stems=args.stems,
            tags=args.tags,
            refresh=args.refresh,
        )
        print(payload["file_name"], payload["hash8"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
