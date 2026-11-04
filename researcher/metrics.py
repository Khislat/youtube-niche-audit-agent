"""Channel and niche metrics.

Numbers are computed here and nowhere else. Channel age, lifetime video count,
lifetime views, average views, views per day, and subscriber count come from
channels.list. The last uploads sample supplies median views, top-video share,
average duration, and upload cadence.

uploads_per_week is sample size divided by the weeks between the oldest and
newest dated upload in that sample. One upload, or several with the same
timestamp, has no cadence (None). views_per_day is None when age is zero.
"""

from datetime import datetime, timezone
import re

_DURATION_RE = re.compile(
    r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$"
)


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an RFC3339 timestamp. Naive values are treated as UTC."""
    if not value or not str(value).strip():
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def age_days(published_at: datetime | None, now: datetime) -> float | None:
    """Whole-day age as a float. Future timestamps clamp to zero."""
    if published_at is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = (now - published_at).total_seconds() / 86400
    if days < 0:
        return 0.0
    return days


def iso8601_duration_minutes(value: str | None) -> float | None:
    """Convert an ISO 8601 duration such as PT1H2M3S into minutes."""
    if not value:
        return None
    match = _DURATION_RE.fullmatch(value.strip())
    if not match:
        return None
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if days == hours == minutes == seconds == 0 and value.strip() in {"P", "PT"}:
        return None
    return days * 1440 + hours * 60 + minutes + seconds / 60


def average_views(total_views: int | None, videos: int | None) -> float | None:
    if total_views is None or videos is None or videos <= 0:
        return None
    return total_views / videos


def views_per_day(total_views: int | None, age: float | None) -> float | None:
    if total_views is None or age is None or age <= 0:
        return None
    return total_views / age


def median(values: list[float]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2 == 1:
        return float(clean[mid])
    return (clean[mid - 1] + clean[mid]) / 2


def top_video_share(view_counts: list[float]) -> float | None:
    """Most-viewed upload divided by the sum of upload views in the sample."""
    if not view_counts:
        return None
    total = sum(view_counts)
    if total <= 0:
        return None
    return max(view_counts) / total


def average_duration_min(durations: list[float | None]) -> float | None:
    clean = [float(value) for value in durations if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def uploads_per_week(published: list[datetime]) -> float | None:
    dated = sorted(item for item in published if item is not None)
    if len(dated) < 2:
        return None
    span_days = (dated[-1] - dated[0]).total_seconds() / 86400
    if span_days <= 0:
        return None
    return len(dated) / (span_days / 7)


def needs_upload_sample(
    age: float | None,
    avg: float | None,
    *,
    max_age: float,
    min_avg: float,
) -> bool:
    """First filter: only young channels with a high average get an upload sample."""
    if age is None or avg is None:
        return False
    return age <= max_age and avg >= min_avg


def median_avg_ratio(median_views: float | None, avg: float | None) -> float | None:
    """Median divided by lifetime average. Undefined when the average is missing or zero."""
    if median_views is None or avg is None or avg <= 0:
        return None
    return median_views / avg


def recent10_median_views(uploads: list[dict], n: int = 10) -> float | None:
    """Median views of the newest uploads. Fewer than n uses whatever is available."""
    if not uploads:
        return None

    def newest_first(video: dict):
        published = parse_datetime(video.get("published_at"))
        if published is None:
            return (1, 0.0)
        return (0, -published.timestamp())

    newest = sorted(uploads, key=newest_first)[:n]
    return median([float(video["views"]) for video in newest])


def found_by_for_channel(video_ids: list[str], video_sources: dict[str, set[str]]) -> str:
    """`views`, `relevance`, or `both` from the searches that returned this channel's videos."""
    labels: set[str] = set()
    for video_id in video_ids:
        labels.update(video_sources.get(video_id, set()))
    has_views = "views" in labels
    has_relevance = "relevance" in labels
    if has_views and has_relevance:
        return "both"
    if has_views:
        return "views"
    if has_relevance:
        return "relevance"
    return ""


def channel_is_outlier(
    age: float | None,
    avg: float | None,
    median_views: float | None,
    share: float | None,
    *,
    max_age: float,
    min_avg: float,
    min_median: float,
    max_top_share: float,
    min_median_avg_ratio: float,
) -> bool:
    if age is None or avg is None or median_views is None or share is None or avg <= 0:
        return False
    return (
        age <= max_age
        and avg >= min_avg
        and median_views >= min_median
        and (median_views / avg) >= min_median_avg_ratio
        and share <= max_top_share
    )


def compute_channel_metrics(
    *,
    channel_id: str,
    title: str,
    url: str,
    published_at: str | None,
    video_count: int | None,
    view_count: int | None,
    subs: int | None,
    uploads: list[dict] | None,
    now: datetime,
    max_age: float,
    min_avg: float,
    min_median: float,
    max_top_share: float,
    min_median_avg_ratio: float,
    found_by: str = "",
) -> dict:
    """Build one channel row. `uploads` is the last-N sample, or None if not fetched."""
    age = age_days(parse_datetime(published_at), now)
    avg = average_views(view_count, video_count)
    sample = uploads or []
    view_counts = [float(video["views"]) for video in sample]
    durations = [video.get("duration_min") for video in sample]
    dates = [parse_datetime(video.get("published_at")) for video in sample]
    median_views = median(view_counts)
    share = top_video_share(view_counts)
    ratio = median_avg_ratio(median_views, avg)
    return {
        "channel_id": channel_id,
        "title": title,
        "url": url,
        "age_days": age,
        "videos": video_count,
        "total_views": view_count,
        "avg_views": avg,
        "median_views": median_views,
        "recent10_median_views": recent10_median_views(sample),
        "median_avg_ratio": ratio,
        "found_by": found_by,
        "top_share": share,
        "views_per_day": views_per_day(view_count, age),
        "avg_duration_min": average_duration_min(durations),
        "uploads_per_week": uploads_per_week([item for item in dates if item is not None]),
        "subs": subs,
        "is_outlier": channel_is_outlier(
            age,
            avg,
            median_views,
            share,
            max_age=max_age,
            min_avg=min_avg,
            min_median=min_median,
            max_top_share=max_top_share,
            min_median_avg_ratio=min_median_avg_ratio,
        ),
    }


def compute_niche_metrics(channels: list[dict], *, max_age: float) -> dict:
    """Niche rollup. `channels_analyzed` is every channel returned by the search."""
    outliers = [channel for channel in channels if channel.get("is_outlier")]
    analyzed = len(channels)
    # Relevance sample is every channel the relevance search returned,
    # including ones viewCount also returned ("both"). That sample is the
    # less biased outlier rate. Channels found only by viewCount stay out of it.
    relevance = [
        channel for channel in channels if channel.get("found_by") in {"relevance", "both"}
    ]
    relevance_outliers = [channel for channel in relevance if channel.get("is_outlier")]
    return {
        "channels_analyzed": analyzed,
        "outliers": len(outliers),
        "outlier_ratio": (len(outliers) / analyzed) if analyzed else 0.0,
        "relevance_channels": len(relevance),
        "outlier_ratio_by_source": (
            (len(relevance_outliers) / len(relevance)) if relevance else None
        ),
        "new_channels": sum(
            1
            for channel in channels
            if channel.get("age_days") is not None and channel["age_days"] <= max_age
        ),
        "median_outlier_views_per_day": median(
            [
                channel["views_per_day"]
                for channel in outliers
                if channel.get("views_per_day") is not None
            ]
        ),
        "typical_duration_min": median(
            [
                channel["avg_duration_min"]
                for channel in outliers
                if channel.get("avg_duration_min") is not None
            ]
        ),
        "typical_uploads_per_week": median(
            [
                channel["uploads_per_week"]
                for channel in outliers
                if channel.get("uploads_per_week") is not None
            ]
        ),
    }
