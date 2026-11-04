"""Phase 1 CLI: collect a niche, compute metrics, store them, write a CSV.

Claude, Notion, and Telegram are later phases. --no-claude is the Phase 1 mode.
The other skip flags are accepted and stored with the audit parameters.
"""

from datetime import datetime, timezone
import argparse
import logging
import os
import re

import requests
from dotenv import load_dotenv

from collect import QuotaExceeded, QuotaTracker, YouTubeAPIError
from collect import get_channel_uploads, get_channels, get_videos, search_niche_videos
from config import (
    DEFAULT_DAYS,
    DEFAULT_INTEREST,
    DEFAULT_MAX_AGE,
    DEFAULT_MAX_TOP_SHARE,
    DEFAULT_MAX_VIDEOS_PER_CHANNEL,
    DEFAULT_MIN_AVG,
    DEFAULT_MIN_MEDIAN,
    DEFAULT_MIN_MEDIAN_AVG_RATIO,
    DEFAULT_PAGES,
    DEFAULT_REGION,
    DEFAULT_DB_PATH,
    DEFAULT_OUTPUT_DIR,
    ENV_PATH,
    ROOT,
)
from metrics import (
    age_days,
    average_views,
    compute_channel_metrics,
    compute_niche_metrics,
    found_by_for_channel,
    needs_upload_sample,
    parse_datetime,
)
from store import find_audit, save_audit, sort_channels, write_channels_csv

logger = logging.getLogger(__name__)
PREVIEW_ROWS = 10


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YouTube niche bo'yicha yosh va ko'rishi baland kanallarni yig'adi (Faza 1)."
    )
    parser.add_argument("niche", help="Tekshiriladigan niche")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Qidiruv oynasi, kun")
    parser.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE, help="Kanal yoshi chegarasi, kun")
    parser.add_argument("--min-avg", type=int, default=DEFAULT_MIN_AVG, help="O'rtacha ko'rish chegarasi")
    parser.add_argument("--min-median", type=int, default=DEFAULT_MIN_MEDIAN, help="Median ko'rish chegarasi")
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_MEDIAN_AVG_RATIO, help="Median / o'rtacha chegarasi")
    parser.add_argument("--interest", type=int, default=DEFAULT_INTEREST, help="Shaxsiy qiziqish, 1-5")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="search.list sahifalari")
    parser.add_argument("--region", default=DEFAULT_REGION, help="regionCode")
    parser.add_argument("--no-claude", action="store_true", help="Faqat yig'ish va hisoblash")
    parser.add_argument("--no-notion", action="store_true", help="Notion'ga yozilmaydi")
    parser.add_argument("--no-telegram", action="store_true", help="Telegram'ga yuborilmaydi")
    parser.add_argument("--refresh", action="store_true", help="Oldin tekshirilgan niche'ni qayta tekshiradi")
    return parser


def _validate(args: argparse.Namespace) -> str | None:
    if not args.niche or not args.niche.strip():
        return "Niche bo'sh bo'lmasligi kerak."
    if args.days <= 0:
        return "--days musbat bo'lishi kerak."
    if args.max_age <= 0:
        return "--max-age musbat bo'lishi kerak."
    if args.min_avg < 0:
        return "--min-avg manfiy bo'lmasligi kerak."
    if args.min_median < 0:
        return "--min-median manfiy bo'lmasligi kerak."
    if args.min_ratio < 0:
        return "--min-ratio manfiy bo'lmasligi kerak."
    if args.pages <= 0:
        return "--pages musbat bo'lishi kerak."
    if not 1 <= args.interest <= 5:
        return "--interest 1 dan 5 gacha bo'lishi kerak."
    return None


def niche_slug(niche: str) -> str:
    text = re.sub(r"\s+", "_", niche.strip().lower())
    text = re.sub(r"[^\w\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "niche"


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


def _display_path(path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _fmt_int(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.0f}"


def _fmt_decimal(value, digits: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.{digits}f}"


def _percent(value) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def _clip(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _print_table(channels: list[dict]) -> None:
    ordered = sort_channels(channels)
    if not ordered:
        print("Kanal topilmadi.")
        return
    labels = (
        "Kanal",
        "Yosh",
        "O'rtacha",
        "Median",
        "Oxirgi10 median",
        "Ko'rish/kun",
        "Topilgan",
        "Outlier",
    )
    print(
        f"{labels[0]:<24} {labels[1]:>6} {labels[2]:>10} {labels[3]:>10} "
        f"{labels[4]:>16} {labels[5]:>12} {labels[6]:>10} {labels[7]:>8}"
    )
    for row in ordered[:PREVIEW_ROWS]:
        flag = "ha" if row.get("is_outlier") else "yo'q"
        print(
            f"{_clip(row.get('title') or '', 24):<24} "
            f"{_fmt_int(row.get('age_days')):>6} "
            f"{_fmt_int(row.get('avg_views')):>10} "
            f"{_fmt_int(row.get('median_views')):>10} "
            f"{_fmt_int(row.get('recent10_median_views')):>16} "
            f"{_fmt_int(row.get('views_per_day')):>12} "
            f"{(row.get('found_by') or '-'):>10} "
            f"{flag:>8}"
        )
    rest = len(ordered) - min(len(ordered), PREVIEW_ROWS)
    if rest:
        print(f"Yana {rest} ta kanal CSV da.")


def print_report(niche: str, summary: dict, channels: list[dict], csv_path, db_path) -> None:
    print(f"Niche: {niche}")
    print(f"Tekshirilgan kanallar: {summary['channels_analyzed']}")
    print(f"Yangi kanallar: {summary['new_channels']}")
    print(f"Outlier: {summary['outliers']} ({summary['outlier_ratio'] * 100:.0f}%)")
    print(f"Outlier ulushi (hammasi): {_percent(summary['outlier_ratio'])}")
    print(f"Outlier ulushi (faqat relevance): {_percent(summary.get('outlier_ratio_by_source'))}")
    if summary.get("relevance_channels", 0) < 10:
        print("Ogohlantirish: relevance tanlama kichik.")
    print(f"Outlier ko'rish/kun (median): {_fmt_int(summary['median_outlier_views_per_day'])}")
    print(f"Odatiy davomiylik (daq): {_fmt_decimal(summary['typical_duration_min'], 1)}")
    print(f"Odatiy yuklash (hafta): {_fmt_decimal(summary['typical_uploads_per_week'], 2)}")
    print()
    _print_table(channels)
    print()
    print(f"CSV: {_display_path(csv_path)}")
    print(f"Baza: {_display_path(db_path)}")


def _params(args: argparse.Namespace) -> dict:
    return {
        "days": args.days,
        "max_age": args.max_age,
        "min_avg": args.min_avg,
        "min_median": args.min_median,
        "min_ratio": args.min_ratio,
        "max_top_share": DEFAULT_MAX_TOP_SHARE,
        "pages": args.pages,
        "max_videos_per_channel": DEFAULT_MAX_VIDEOS_PER_CHANNEL,
        "region": args.region,
        "interest": args.interest,
        "no_claude": args.no_claude,
        "no_notion": args.no_notion,
        "no_telegram": args.no_telegram,
        "refresh": args.refresh,
    }


def gather(
    niche: str,
    *,
    days: int,
    pages: int,
    region: str,
    max_age: float,
    min_avg: float,
    min_median: float,
    min_median_avg_ratio: float,
    max_top_share: float,
    max_videos: int,
    api_key: str,
    tracker: QuotaTracker,
    session: requests.Session,
    now: datetime,
) -> tuple[list[dict], list[dict]]:
    """Search the niche and compute a row for every channel that comes back.

    Upload samples are requested only for channels that already pass the age
    and average-views filter, to save quota. On quota exhaustion the rows
    finished so far are attached to the exception.
    """
    channel_rows: list[dict] = []
    video_rows: list[dict] = []
    try:
        video_ids, video_sources = search_niche_videos(
            niche,
            days,
            pages,
            region,
            api_key=api_key,
            tracker=tracker,
            session=session,
            now=now,
        )
        if not video_ids:
            return [], []
        searched = get_videos(video_ids, api_key=api_key, tracker=tracker, session=session)
        channel_ids = []
        channel_video_ids: dict[str, list[str]] = {}
        seen: set[str] = set()
        for video in searched:
            channel_id = video.get("channel_id")
            video_id = video.get("video_id")
            if channel_id and video_id:
                channel_video_ids.setdefault(channel_id, []).append(video_id)
            if channel_id and channel_id not in seen:
                seen.add(channel_id)
                channel_ids.append(channel_id)
        if not channel_ids:
            return [], []
        channels = get_channels(channel_ids, api_key=api_key, tracker=tracker, session=session)
        logger.info("Unique channels from search: %s", len(channels))
        for channel in channels:
            published_at = channel.get("published_at")
            age = age_days(parse_datetime(published_at), now)
            avg = average_views(channel.get("view_count"), channel.get("video_count"))
            uploads = None
            if needs_upload_sample(age, avg, max_age=max_age, min_avg=min_avg):
                uploads = get_channel_uploads(
                    channel["channel_id"],
                    n=max_videos,
                    api_key=api_key,
                    tracker=tracker,
                    session=session,
                    uploads_playlist=channel.get("uploads_playlist") or "",
                )
            channel_rows.append(
                compute_channel_metrics(
                    channel_id=channel["channel_id"],
                    title=channel.get("title") or "",
                    url=channel.get("url") or "",
                    published_at=published_at,
                    video_count=channel.get("video_count"),
                    view_count=channel.get("view_count"),
                    subs=channel.get("subs"),
                    uploads=uploads,
                    now=now,
                    max_age=max_age,
                    min_avg=min_avg,
                    min_median=min_median,
                    max_top_share=max_top_share,
                    min_median_avg_ratio=min_median_avg_ratio,
                    found_by=found_by_for_channel(
                        channel_video_ids.get(channel["channel_id"], []),
                        video_sources,
                    ),
                )
            )
            for video in uploads or []:
                video_rows.append(
                    {
                        "channel_id": channel["channel_id"],
                        "video_id": video["video_id"],
                        "title": video.get("title"),
                        "views": video.get("views"),
                        "duration_min": video.get("duration_min"),
                        "published_at": video.get("published_at"),
                        "thumbnail_url": video.get("thumbnail_url"),
                    }
                )
        return channel_rows, video_rows
    except QuotaExceeded as exc:
        exc.channels = list(channel_rows)
        exc.videos = list(video_rows)
        raise


def main(
    argv: list[str] | None = None,
    *,
    now: datetime | None = None,
    db_path=None,
    output_dir=None,
) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    problem = _validate(args)
    if problem:
        print(problem)
        return 1

    niche = args.niche.strip()
    load_dotenv(ENV_PATH)
    database = db_path or DEFAULT_DB_PATH
    folder = output_dir or DEFAULT_OUTPUT_DIR
    moment = _aware(now or datetime.now(timezone.utc))

    if not args.refresh:
        existing = find_audit(database, niche)
        if existing and existing["summary"].get("complete", True):
            print(f"Bu niche oldin tekshirilgan ({existing['created_at']}). To'xtatildi.")
            print("Qayta tekshirish uchun --refresh qo'ying.")
            return 0

    api_key = os.environ.get("YT_API_KEY", "").strip()
    if not api_key:
        print("YT_API_KEY topilmadi. Kalitni researcher/.env fayliga yozing (.env.example ga qarang).")
        return 1

    if not args.no_claude:
        print("Claude tahlili keyingi fazada. Hozir faqat yig'ish va hisoblash bajariladi.")

    tracker = QuotaTracker()
    session = requests.Session()
    try:
        interrupted = False
        try:
            channels, videos = gather(
                niche,
                days=args.days,
                pages=args.pages,
                region=args.region,
                max_age=args.max_age,
                min_avg=args.min_avg,
                min_median=args.min_median,
                min_median_avg_ratio=args.min_ratio,
                max_top_share=DEFAULT_MAX_TOP_SHARE,
                max_videos=DEFAULT_MAX_VIDEOS_PER_CHANNEL,
                api_key=api_key,
                tracker=tracker,
                session=session,
                now=moment,
            )
        except QuotaExceeded as exc:
            channels, videos = exc.channels, exc.videos
            interrupted = True
            if not channels:
                logger.error("YouTube Data API quota exceeded before any channel was finished")
                print("YouTube kvotasi tugadi. Saqlanadigan natija yig'ilmagan.")
                return 1
        except YouTubeAPIError as exc:
            logger.error("YouTube API error: %s", exc)
            print("YouTube API xatosi.")
            print(str(exc))
            return 1

        summary = compute_niche_metrics(channels, max_age=args.max_age)
        summary["complete"] = not interrupted
        created_at = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        csv_path = folder / f"{niche_slug(niche)}_{moment.astimezone().date().isoformat()}.csv"
        save_audit(
            database,
            niche=niche,
            created_at=created_at,
            params=_params(args),
            quota_used=tracker.used,
            summary=summary,
            channels=channels,
            videos=videos,
        )
        write_channels_csv(csv_path, channels)
        logger.info("Saved audit for %r (quota %s)", niche, tracker.used)
        print_report(niche, summary, channels, csv_path, database)
        if interrupted:
            print("YouTube kvotasi tugadi. Shu paytgacha yig'ilgan natija saqlandi.")
            return 1
        return 0
    finally:
        session.close()
        print(f"Sarflangan kvota: {tracker.used}")


if __name__ == "__main__":
    raise SystemExit(main())
