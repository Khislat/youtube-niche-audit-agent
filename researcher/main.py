"""Phase 1 CLI: collect niches, compute metrics, flags, suggestions, and CSV exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import logging
import os
import re

import requests
from dotenv import load_dotenv

from collect import QuotaExceeded, QuotaTracker, YouTubeAPIError
from collect import get_channel_uploads, get_channels, get_videos, search_niche_videos
from config import (
    CSV_FLAG_DISCLAIMER,
    DEFAULT_AUTO_SUGGEST,
    DEFAULT_DAYS,
    DEFAULT_INTEREST,
    DEFAULT_MAX_AGE,
    DEFAULT_MAX_QUOTA,
    DEFAULT_MAX_TOP_SHARE,
    DEFAULT_MAX_VIDEOS_PER_CHANNEL,
    DEFAULT_MIN_AVG,
    DEFAULT_MIN_MEDIAN,
    DEFAULT_MIN_MEDIAN_AVG_RATIO,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAGES,
    DEFAULT_REGION,
    DEFAULT_SUGGEST_MIN_CHANNELS,
    DEFAULT_DB_PATH,
    ENV_PATH,
    ESTIMATED_QUOTA_PER_NICHE,
    MAX_SUGGESTIONS,
    ROOT,
    TOP_OUTLIER_VIDEOS,
)
from flags import channel_flag_metrics, video_flags
from metrics import (
    age_days,
    average_views,
    compute_channel_metrics,
    compute_niche_metrics,
    found_by_for_channel,
    needs_upload_sample,
    parse_datetime,
    pick_top_videos,
    views_per_day,
)
from suggest import build_suggestions
from store import (
    audited_niche_names,
    find_audit,
    save_audit,
    sort_channels,
    sort_compare_rows,
    write_channels_csv,
    write_compare_csv,
    write_suggestions_csv,
    write_top_videos_csv,
)

logger = logging.getLogger(__name__)
PREVIEW_ROWS = 10


@dataclass
class AuditOutcome:
    niche: str
    status: str
    quota_used: int
    compare_row: dict | None = None
    suggestions: list[dict] | None = None
    error: str | None = None


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YouTube niche bo'yicha kanallarni yig'adi, outlier'larni va belgilarni hisoblaydi."
    )
    parser.add_argument("niche", nargs="?", help="Tekshiriladigan niche")
    parser.add_argument("--file", help="Har qatorda bitta niche (bo'sh va # qatorlar o'tkaziladi)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Qidiruv oynasi, kun")
    parser.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE, help="Kanal yoshi chegarasi, kun")
    parser.add_argument("--min-avg", type=int, default=DEFAULT_MIN_AVG, help="O'rtacha ko'rish chegarasi")
    parser.add_argument("--min-median", type=int, default=DEFAULT_MIN_MEDIAN, help="Median ko'rish chegarasi")
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_MEDIAN_AVG_RATIO, help="Median / o'rtacha chegarasi")
    parser.add_argument("--min-channels", type=int, default=DEFAULT_SUGGEST_MIN_CHANNELS, help="Taklif uchun min kanal")
    parser.add_argument("--interest", type=int, default=DEFAULT_INTEREST, help="Shaxsiy qiziqish, 1-5")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="search.list sahifalari")
    parser.add_argument("--region", default=DEFAULT_REGION, help="regionCode")
    parser.add_argument("--max-quota", type=int, default=DEFAULT_MAX_QUOTA, help="Shu ishga tushirish kvota limiti")
    parser.add_argument("--suggest", action="store_true", help="Outlier videolardan taklif CSV yozadi")
    parser.add_argument("--auto-suggest", type=int, default=DEFAULT_AUTO_SUGGEST, metavar="N", help="Eng yaxshi N taklifni audit qiladi")
    parser.add_argument("--no-claude", action="store_true", help="Faqat yig'ish va hisoblash")
    parser.add_argument("--no-notion", action="store_true", help="Notion'ga yozilmaydi")
    parser.add_argument("--no-telegram", action="store_true", help="Telegram'ga yuborilmaydi")
    parser.add_argument("--refresh", action="store_true", help="Oldin tekshirilgan niche'ni qayta tekshiradi")
    return parser


def _validate(args: argparse.Namespace) -> str | None:
    if args.file and args.niche:
        return "Niche va --file bir vaqtda ishlatilmaydi."
    if not args.file and (not args.niche or not args.niche.strip()):
        return "Niche yoki --file kerak."
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
    if args.max_quota <= 0:
        return "--max-quota musbat bo'lishi kerak."
    if args.min_channels <= 0:
        return "--min-channels musbat bo'lishi kerak."
    if args.auto_suggest < 0:
        return "--auto-suggest manfiy bo'lmasligi kerak."
    if not 1 <= args.interest <= 5:
        return "--interest 1 dan 5 gacha bo'lishi kerak."
    return None


def parse_niche_file(path) -> list[str]:
    niches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        niches.append(text)
    return niches


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
        "Oxirgi10",
        "Ko'rish/kun",
        "AI%",
        "Brend%",
        "Topilgan",
        "Outlier",
    )
    print(
        f"{labels[0]:<22} {labels[1]:>5} {labels[2]:>9} {labels[3]:>9} "
        f"{labels[4]:>9} {labels[5]:>11} {labels[6]:>5} {labels[7]:>6} "
        f"{labels[8]:>9} {labels[9]:>8}"
    )
    for row in ordered[:PREVIEW_ROWS]:
        flag = "ha" if row.get("is_outlier") else "yo'q"
        print(
            f"{_clip(row.get('title') or '', 22):<22} "
            f"{_fmt_int(row.get('age_days')):>5} "
            f"{_fmt_int(row.get('avg_views')):>9} "
            f"{_fmt_int(row.get('median_views')):>9} "
            f"{_fmt_int(row.get('recent10_median_views')):>9} "
            f"{_fmt_int(row.get('views_per_day')):>11} "
            f"{_percent(row.get('ai_flag_share')):>5} "
            f"{_percent(row.get('brand_flag_share')):>6} "
            f"{(row.get('found_by') or '-'):>9} "
            f"{flag:>8}"
        )
    rest = len(ordered) - min(len(ordered), PREVIEW_ROWS)
    if rest:
        print(f"Yana {rest} ta kanal CSV da.")


def print_report(
    niche: str,
    summary: dict,
    channels: list[dict],
    csv_path,
    db_path,
    *,
    top_videos_path=None,
    suggestions_path=None,
) -> None:
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
    print(CSV_FLAG_DISCLAIMER)
    print()
    _print_table(channels)
    print()
    print(f"CSV: {_display_path(csv_path)}")
    if top_videos_path is not None:
        print(f"Top videolar: {_display_path(top_videos_path)}")
    if suggestions_path is not None:
        print(f"Takliflar: {_display_path(suggestions_path)}")
    print(f"Baza: {_display_path(db_path)}")


def _params(args: argparse.Namespace, *, source_file: str | None = None) -> dict:
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
        "max_quota": args.max_quota,
        "suggest": args.suggest,
        "auto_suggest": args.auto_suggest,
        "min_channels": args.min_channels,
        "source_file": source_file,
        "no_claude": args.no_claude,
        "no_notion": args.no_notion,
        "no_telegram": args.no_telegram,
        "refresh": args.refresh,
    }


def _compare_row(niche: str, summary: dict, quota_used: int) -> dict:
    note = ""
    if summary.get("relevance_channels", 0) < 10:
        note = "(kichik tanlama)"
    return {
        "niche": niche,
        "channels_analyzed": summary.get("channels_analyzed"),
        "new_channels": summary.get("new_channels"),
        "outliers": summary.get("outliers"),
        "outlier_ratio": summary.get("outlier_ratio"),
        "outlier_ratio_by_source": summary.get("outlier_ratio_by_source"),
        "relevance_channels": summary.get("relevance_channels"),
        "median_outlier_views_per_day": summary.get("median_outlier_views_per_day"),
        "typical_duration_min": summary.get("typical_duration_min"),
        "typical_uploads_per_week": summary.get("typical_uploads_per_week"),
        "outlier_ai_flag_pct": summary.get("outlier_ai_flag_pct"),
        "outlier_brand_flag_pct": summary.get("outlier_brand_flag_pct"),
        "quota_used": quota_used,
        "note": note,
    }


def _print_compare_table(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    print("Solishtirma jadval:")
    for row in sort_compare_rows(rows):
        label = row["niche"]
        if row.get("note"):
            label = f"{label} {row['note']}"
        print(
            f"- {label}: outlier (relevance) {_percent(row.get('outlier_ratio_by_source'))}, "
            f"median ko'rish/kun {_fmt_int(row.get('median_outlier_views_per_day'))}, "
            f"kvota {row.get('quota_used')}"
        )


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
            metrics_row = compute_channel_metrics(
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
            if uploads:
                metrics_row.update(channel_flag_metrics(uploads))
            channel_rows.append(metrics_row)
            if metrics_row.get("is_outlier") and uploads:
                for rank, video in enumerate(pick_top_videos(uploads, TOP_OUTLIER_VIDEOS), start=1):
                    published = video.get("published_at")
                    video_age = age_days(parse_datetime(published), now)
                    flags = video_flags(video)
                    video_id = video["video_id"]
                    video_rows.append(
                        {
                            "channel_id": channel["channel_id"],
                            "channel_title": metrics_row["title"],
                            "video_id": video_id,
                            "title": video.get("title"),
                            "views": video.get("views"),
                            "duration_min": video.get("duration_min"),
                            "published_at": published,
                            "thumbnail_url": video.get("thumbnail_url"),
                            "description_snippet": video.get("description_snippet"),
                            "tags": video.get("tags") or [],
                            "top_rank": rank,
                            "age_days": video_age,
                            "views_per_day": views_per_day(video.get("views"), video_age),
                            "video_url": f"https://www.youtube.com/watch?v={video_id}",
                            "ai_terms": flags["ai_terms"],
                            "brand_terms": flags["brand_terms"],
                        }
                    )
        return channel_rows, video_rows
    except QuotaExceeded as exc:
        exc.channels = list(channel_rows)
        exc.videos = list(video_rows)
        raise


def _top_video_csv_rows(video_rows: list[dict]) -> list[dict]:
    ordered = sorted(
        video_rows,
        key=lambda row: (
            (row.get("channel_title") or "").lower(),
            row.get("top_rank") or 99,
        ),
    )
    export: list[dict] = []
    for row in ordered:
        export.append(
            {
                "channel_title": row.get("channel_title"),
                "top_rank": row.get("top_rank"),
                "title": row.get("title"),
                "views": row.get("views"),
                "duration_min": row.get("duration_min"),
                "published_at": row.get("published_at"),
                "age_days": row.get("age_days"),
                "views_per_day": row.get("views_per_day"),
                "video_url": row.get("video_url"),
                "thumbnail_url": row.get("thumbnail_url"),
                "ai_terms": row.get("ai_terms"),
                "brand_terms": row.get("brand_terms"),
            }
        )
    return export


def run_audit(
    niche: str,
    args: argparse.Namespace,
    *,
    tracker: QuotaTracker,
    session: requests.Session,
    moment: datetime,
    database,
    folder,
    exclude_niches: set[str],
    source_file: str | None,
    depth: int,
) -> AuditOutcome:
    quota_before = tracker.used
    if not args.refresh:
        existing = find_audit(database, niche)
        if existing and existing["summary"].get("complete", True):
            print(f"'{niche}' oldin tekshirilgan, o'tkazildi ({existing['created_at']}).")
            return AuditOutcome(niche=niche, status="skipped", quota_used=0)

    if tracker.used + ESTIMATED_QUOTA_PER_NICHE > args.max_quota:
        print(f"Kvota chegarasi: joriy {tracker.used} + taxminiy {ESTIMATED_QUOTA_PER_NICHE} > {args.max_quota}.")
        return AuditOutcome(niche=niche, status="quota_cap", quota_used=0)

    api_key = os.environ.get("YT_API_KEY", "").strip()
    if not api_key:
        return AuditOutcome(niche=niche, status="error", quota_used=0, error="YT_API_KEY topilmadi")

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
            logger.error("YouTube quota exceeded before any channel was finished")
            return AuditOutcome(
                niche=niche,
                status="error",
                quota_used=tracker.used - quota_before,
                error="YouTube kvotasi tugadi, natija yig'ilmagan",
            )
    except YouTubeAPIError as exc:
        logger.error("YouTube API error for %r: %s", niche, exc)
        return AuditOutcome(
            niche=niche,
            status="error",
            quota_used=tracker.used - quota_before,
            error=str(exc),
        )

    summary = compute_niche_metrics(channels, max_age=args.max_age)
    summary["complete"] = not interrupted
    suggestions: list[dict] = []
    want_suggest = args.suggest or (args.auto_suggest > 0 and depth == 0)
    if want_suggest and videos and not interrupted:
        suggestions = build_suggestions(
            videos,
            niche=niche,
            min_channels=args.min_channels,
            max_results=MAX_SUGGESTIONS,
            exclude_niches=exclude_niches,
        )

    date_label = moment.astimezone().date().isoformat()
    slug = niche_slug(niche)
    csv_path = folder / f"{slug}_{date_label}.csv"
    top_videos_path = folder / f"{slug}_{date_label}_top_videos.csv"
    suggestions_path = folder / f"{slug}_{date_label}_suggestions.csv"
    created_at = moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    save_audit(
        database,
        niche=niche,
        created_at=created_at,
        params=_params(args, source_file=source_file),
        quota_used=tracker.used - quota_before,
        summary=summary,
        channels=channels,
        videos=videos,
        suggestions=suggestions if suggestions else None,
    )
    write_channels_csv(csv_path, channels)
    if videos:
        write_top_videos_csv(top_videos_path, _top_video_csv_rows(videos))
    else:
        top_videos_path = None
    if suggestions:
        write_suggestions_csv(suggestions_path, suggestions)
        print("Takliflar:")
        for item in suggestions[:10]:
            print(f"  - {item['term']} ({item['channel_count']} kanal): {item['example_title']}")
    else:
        suggestions_path = None if not want_suggest else suggestions_path

    print_report(
        niche,
        summary,
        channels,
        csv_path,
        database,
        top_videos_path=top_videos_path,
        suggestions_path=suggestions_path if suggestions else None,
    )
    if interrupted:
        print("YouTube kvotasi tugadi. Shu paytgacha yig'ilgan natija saqlandi.")
    quota_used = tracker.used - quota_before
    print(f"Sarflangan kvota (shu niche): {quota_used}")
    return AuditOutcome(
        niche=niche,
        status="partial" if interrupted else "ok",
        quota_used=quota_used,
        compare_row=_compare_row(niche, summary, quota_used),
        suggestions=suggestions,
    )


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

    load_dotenv(ENV_PATH)
    database = db_path or DEFAULT_DB_PATH
    folder = output_dir or DEFAULT_OUTPUT_DIR
    moment = _aware(now or datetime.now(timezone.utc))

    if args.file:
        niches = parse_niche_file(Path(args.file))
        source_file = args.file
    else:
        niches = [args.niche.strip()]
        source_file = None

    if not niches:
        print("Ishlatiladigan niche topilmadi.")
        return 1

    if not args.no_claude:
        print("Claude tahlili keyingi fazada. Hozir faqat yig'ish va hisoblash bajariladi.")

    exclude_niches = {name.strip().lower() for name in audited_niche_names(database)}
    exclude_niches.update(name.strip().lower() for name in niches)

    tracker = QuotaTracker()
    session = requests.Session()
    compare_rows: list[dict] = []
    exit_code = 0
    queue: list[tuple[str, int]] = [(niche, 0) for niche in niches]
    seen: set[str] = set()

    try:
        while queue:
            niche, depth = queue.pop(0)
            key = niche.strip().lower()
            if key in seen:
                continue
            seen.add(key)

            outcome = run_audit(
                niche,
                args,
                tracker=tracker,
                session=session,
                moment=moment,
                database=database,
                folder=folder,
                exclude_niches=exclude_niches,
                source_file=source_file,
                depth=depth,
            )

            if outcome.status == "quota_cap":
                exit_code = 1
                break
            if outcome.status == "error":
                logger.error("Niche %r failed: %s", niche, outcome.error)
                print(f"Xato ({niche}): {outcome.error}")
                exit_code = 1
                continue
            if outcome.status == "skipped":
                continue
            if outcome.compare_row:
                compare_rows.append(outcome.compare_row)
            if outcome.status == "partial":
                exit_code = 1

            if depth == 0 and args.auto_suggest > 0 and outcome.suggestions:
                for item in outcome.suggestions[: args.auto_suggest]:
                    term = item["term"].strip()
                    if term.lower() not in seen:
                        queue.append((term, 1))

        if compare_rows:
            compare_path = folder / f"compare_{moment.astimezone().date().isoformat()}.csv"
            write_compare_csv(compare_path, compare_rows)
            _print_compare_table(compare_rows)
            print(f"Solishtirma CSV: {_display_path(compare_path)}")
        print(f"Jami sarflangan kvota: {tracker.used}")
        return exit_code
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
