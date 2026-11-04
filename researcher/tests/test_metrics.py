"""Synthetic checks for every Phase 1 formula."""

from datetime import datetime, timedelta, timezone
import sqlite3

from config import (
    DEFAULT_DAYS,
    DEFAULT_MAX_AGE,
    DEFAULT_MAX_TOP_SHARE,
    DEFAULT_MAX_VIDEOS_PER_CHANNEL,
    DEFAULT_MIN_AVG,
    DEFAULT_MIN_MEDIAN,
    DEFAULT_MIN_MEDIAN_AVG_RATIO,
    DEFAULT_PAGES,
    SCORE_WEIGHTS,
)
from metrics import (
    age_days,
    average_duration_min,
    average_views,
    channel_is_outlier,
    compute_channel_metrics,
    compute_niche_metrics,
    found_by_for_channel,
    iso8601_duration_minutes,
    median,
    needs_upload_sample,
    parse_datetime,
    top_video_share,
    uploads_per_week,
    views_per_day,
)
from store import init_db

NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)
LIMITS = {
    "max_age": DEFAULT_MAX_AGE,
    "min_avg": DEFAULT_MIN_AVG,
    "min_median": DEFAULT_MIN_MEDIAN,
    "max_top_share": DEFAULT_MAX_TOP_SHARE,
    "min_median_avg_ratio": DEFAULT_MIN_MEDIAN_AVG_RATIO,
}


def _uploads():
    views = [8000, 9000, 10000, 11000, 12000]
    durations = [40, 50, 60, 40, 60]
    start = datetime(2026, 10, 7, tzinfo=timezone.utc)
    rows = []
    for index, (view_count, duration) in enumerate(zip(views, durations)):
        published = start + timedelta(days=7 * index)
        rows.append(
            {
                "views": view_count,
                "duration_min": duration,
                "published_at": published.isoformat(),
            }
        )
    return rows


def _channel(**overrides):
    payload = {
        "channel_id": "ch",
        "title": "Channel",
        "url": "https://www.youtube.com/channel/ch",
        "published_at": "2026-07-27T00:00:00Z",
        "video_count": 10,
        "view_count": 200_000,
        "subs": 1500,
        "uploads": _uploads(),
        "now": NOW,
        **LIMITS,
    }
    payload.update(overrides)
    return compute_channel_metrics(**payload)


def test_config_defaults():
    assert DEFAULT_DAYS == 180
    assert DEFAULT_MAX_AGE == 365
    assert DEFAULT_MIN_AVG == 5000
    # Phase 1.1 raised the median floor from 1000 to 3000 and added the median/average ratio.
    assert DEFAULT_MIN_MEDIAN == 3000
    assert DEFAULT_MIN_MEDIAN_AVG_RATIO == 0.15
    assert DEFAULT_MAX_TOP_SHARE == 0.6
    assert DEFAULT_PAGES == 1
    assert DEFAULT_MAX_VIDEOS_PER_CHANNEL == 50
    assert set(SCORE_WEIGHTS) == {
        "demand",
        "supply_gap",
        "ai_feasibility",
        "differentiation",
        "interest",
    }
    assert set(SCORE_WEIGHTS.values()) == {1.0}


def test_duration_parser():
    assert iso8601_duration_minutes("PT1H2M3S") == 62.05
    assert iso8601_duration_minutes("PT45M") == 45
    assert iso8601_duration_minutes("PT90S") == 1.5
    assert iso8601_duration_minutes("PT2H") == 120
    assert iso8601_duration_minutes("P1DT2H") == 1560
    assert iso8601_duration_minutes("PT0S") == 0
    assert iso8601_duration_minutes("PT") is None
    assert iso8601_duration_minutes("soon") is None
    assert iso8601_duration_minutes(None) is None


def test_age_days_and_average_views():
    published = parse_datetime("2026-07-27T00:00:00Z")
    assert age_days(published, NOW) == 100
    assert age_days(parse_datetime("2026-12-01T00:00:00Z"), NOW) == 0
    assert age_days(None, NOW) is None
    assert average_views(200_000, 10) == 20_000
    assert average_views(10, 0) is None
    assert average_views(None, 10) is None


def test_views_per_day_does_not_divide_by_zero():
    assert views_per_day(100_000, 100) == 1000
    assert views_per_day(100_000, 0) is None
    assert views_per_day(100_000, None) is None


def test_median_odd_and_even():
    assert median([1, 100, 3]) == 3
    assert median([1, 2, 3, 4]) == 2.5
    assert median([]) is None


def test_top_share_and_duration_and_cadence():
    assert top_video_share([10, 30, 60]) == 0.6
    assert top_video_share([0, 0]) is None
    assert top_video_share([]) is None
    assert average_duration_min([40, None, 60]) == 50
    assert average_duration_min([None]) is None

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    weekly = [start, start + timedelta(days=7), start + timedelta(days=14)]
    assert uploads_per_week(weekly) == 1.5
    assert uploads_per_week([start]) is None
    assert uploads_per_week([start, start]) is None


def test_channel_metrics_use_lifetime_stats_and_upload_sample():
    row = _channel()
    assert row["age_days"] == 100
    assert row["videos"] == 10
    assert row["total_views"] == 200_000
    assert row["avg_views"] == 20_000
    assert row["views_per_day"] == 2_000
    assert row["median_views"] == 10_000
    assert row["recent10_median_views"] == 10_000
    assert row["median_avg_ratio"] == 0.5
    assert row["top_share"] == 0.24
    assert row["avg_duration_min"] == 50
    assert row["uploads_per_week"] == 1.25
    assert row["subs"] == 1500
    assert row["is_outlier"] is True


def test_outlier_boundaries():
    assert _channel()["is_outlier"] is True
    assert _channel(published_at="2020-01-01T00:00:00Z")["is_outlier"] is False
    assert _channel(view_count=10_000)["is_outlier"] is False  # avg 1000 < 5000
    low_median = _channel(uploads=[{"views": 100, "duration_min": 30, "published_at": "2026-11-01T00:00:00Z"}])
    assert low_median["median_views"] == 100
    assert low_median["is_outlier"] is False
    concentrated = _channel(
        uploads=[
            {"views": 1000, "duration_min": 30, "published_at": "2026-10-01T00:00:00Z"},
            {"views": 1000, "duration_min": 30, "published_at": "2026-10-15T00:00:00Z"},
            {"views": 8000, "duration_min": 30, "published_at": "2026-11-01T00:00:00Z"},
        ]
    )
    assert concentrated["top_share"] == 0.8
    assert concentrated["is_outlier"] is False

    # Scaled from 2000/2000/6000 so the median still clears the new 3000 floor; share stays 0.6.
    on_the_line = _channel(
        uploads=[
            {"views": 4000, "duration_min": 40, "published_at": "2026-10-01T00:00:00Z"},
            {"views": 4000, "duration_min": 40, "published_at": "2026-10-15T00:00:00Z"},
            {"views": 12000, "duration_min": 40, "published_at": "2026-11-01T00:00:00Z"},
        ]
    )
    assert on_the_line["top_share"] == 0.6
    assert on_the_line["is_outlier"] is True
    assert _channel(subs=None)["subs"] is None
    assert _channel(subs=0)["subs"] == 0
    assert _channel(uploads=None)["is_outlier"] is False
    assert _channel(video_count=0, uploads=None)["avg_views"] is None


def test_exact_threshold_passes():
    assert channel_is_outlier(
        365,
        5000,
        1000,
        0.6,
        max_age=365,
        min_avg=5000,
        min_median=1000,
        max_top_share=0.6,
        min_median_avg_ratio=0.15,
    )
    assert needs_upload_sample(365, 5000, max_age=365, min_avg=5000)
    assert not needs_upload_sample(366, 5000, max_age=365, min_avg=5000)
    assert not needs_upload_sample(100, 4999, max_age=365, min_avg=5000)
    assert not needs_upload_sample(None, 5000, max_age=365, min_avg=5000)


def test_zero_age_has_no_views_per_day():
    row = _channel(published_at="2026-11-04T00:00:00Z")
    assert row["age_days"] == 0
    assert row["views_per_day"] is None


def test_niche_rollup():
    channels = [
        {"is_outlier": True, "age_days": 100, "views_per_day": 1000, "avg_duration_min": 40, "uploads_per_week": 1},
        {"is_outlier": True, "age_days": 200, "views_per_day": 3000, "avg_duration_min": 60, "uploads_per_week": 3},
        {"is_outlier": False, "age_days": 50, "views_per_day": 10, "avg_duration_min": 10, "uploads_per_week": 7},
        {"is_outlier": False, "age_days": 400, "views_per_day": 9, "avg_duration_min": 10, "uploads_per_week": 1},
    ]
    summary = compute_niche_metrics(channels, max_age=365)
    assert summary["channels_analyzed"] == 4
    assert summary["outliers"] == 2
    assert summary["outlier_ratio"] == 0.5
    assert summary["new_channels"] == 3
    assert summary["median_outlier_views_per_day"] == 2000
    assert summary["typical_duration_min"] == 50
    assert summary["typical_uploads_per_week"] == 2


def test_lucky_videos_are_not_outliers():
    # Both new rules fail: median 1085 < 3000 and 1085/16857 < 0.15.
    # Top share stays under 0.6, so the old share rule is not what rejects it.
    uploads = [
        {"views": 500, "duration_min": 40, "published_at": "2026-10-01T00:00:00Z"},
        {"views": 1085, "duration_min": 40, "published_at": "2026-10-15T00:00:00Z"},
        {"views": 2000, "duration_min": 40, "published_at": "2026-11-01T00:00:00Z"},
    ]
    row = _channel(video_count=1, view_count=16857, uploads=uploads)
    assert row["median_views"] == 1085
    assert row["avg_views"] == 16857
    assert abs(row["median_avg_ratio"] - (1085 / 16857)) < 1e-12
    assert row["top_share"] <= 0.6
    assert row["is_outlier"] is False


def test_steady_channel_is_outlier():
    uploads = [
        {"views": 4000, "duration_min": 40, "published_at": "2026-10-01T00:00:00Z"},
        {"views": 8000, "duration_min": 40, "published_at": "2026-10-08T00:00:00Z"},
        {"views": 8000, "duration_min": 40, "published_at": "2026-10-15T00:00:00Z"},
        {"views": 8000, "duration_min": 40, "published_at": "2026-10-22T00:00:00Z"},
        {"views": 12000, "duration_min": 40, "published_at": "2026-11-01T00:00:00Z"},
    ]
    row = _channel(uploads=uploads)
    assert row["age_days"] == 100
    assert row["avg_views"] == 20_000
    assert row["median_views"] == 8000
    assert row["top_share"] == 0.3
    assert row["median_avg_ratio"] == 0.4
    assert row["is_outlier"] is True


def test_zero_average_does_not_raise():
    row = _channel(view_count=0, video_count=10)
    assert row["avg_views"] == 0
    assert row["median_avg_ratio"] is None
    assert row["is_outlier"] is False


def test_found_by_both_when_channel_is_in_each_search():
    sources = {"v1": {"views"}, "v2": {"relevance"}, "v3": {"views", "relevance"}}
    assert found_by_for_channel(["v1", "v2"], sources) == "both"
    assert found_by_for_channel(["v3"], sources) == "both"
    assert found_by_for_channel(["v1"], sources) == "views"
    assert found_by_for_channel(["v2"], sources) == "relevance"


def test_recent10_median_with_fewer_than_ten_uploads():
    uploads = [
        {"views": 10, "duration_min": 10, "published_at": "2026-11-01T00:00:00Z"},
        {"views": 30, "duration_min": 10, "published_at": "2026-11-03T00:00:00Z"},
        {"views": 20, "duration_min": 10, "published_at": "2026-11-02T00:00:00Z"},
    ]
    row = _channel(uploads=uploads)
    assert row["recent10_median_views"] == 20


def test_recent10_median_uses_the_newest_ten():
    uploads = []
    for index in range(12):
        uploads.append(
            {
                "views": 1000 if index < 2 else 5000 + index,
                "duration_min": 30,
                "published_at": f"2026-01-{index + 1:02d}T00:00:00Z",
            }
        )
    row = _channel(uploads=uploads)
    assert row["recent10_median_views"] == 5006.5


def test_outlier_ratio_by_source_keeps_the_relevance_sample():
    channels = [
        {"is_outlier": True, "age_days": 10, "found_by": "both", "views_per_day": 1, "avg_duration_min": 1, "uploads_per_week": 1},
        {"is_outlier": False, "age_days": 10, "found_by": "relevance", "views_per_day": 1, "avg_duration_min": 1, "uploads_per_week": 1},
        {"is_outlier": True, "age_days": 10, "found_by": "views", "views_per_day": 1, "avg_duration_min": 1, "uploads_per_week": 1},
    ]
    summary = compute_niche_metrics(channels, max_age=365)
    assert summary["outlier_ratio"] == 2 / 3
    assert summary["relevance_channels"] == 2
    assert summary["outlier_ratio_by_source"] == 0.5


def test_old_database_migration_keeps_existing_rows(tmp_path):
    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE channels (
            audit_id INTEGER NOT NULL,
            channel_id TEXT NOT NULL,
            title TEXT,
            url TEXT,
            age_days REAL,
            videos INTEGER,
            total_views INTEGER,
            avg_views REAL,
            median_views REAL,
            top_share REAL,
            views_per_day REAL,
            avg_duration_min REAL,
            uploads_per_week REAL,
            subs INTEGER,
            is_outlier INTEGER NOT NULL,
            PRIMARY KEY (audit_id, channel_id)
        )
        """
    )
    connection.execute(
        "INSERT INTO channels (audit_id, channel_id, title, is_outlier) VALUES (1, 'old', 'Old row', 1)"
    )
    connection.commit()
    connection.close()

    init_db(db_path)
    init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(channels)")]
        stored = connection.execute(
            "SELECT title, is_outlier, found_by FROM channels WHERE channel_id = 'old'"
        ).fetchone()
    finally:
        connection.close()
    assert "recent10_median_views" in columns
    assert "found_by" in columns
    assert "median_avg_ratio" in columns
    assert stored == ("Old row", 1, None)


def test_empty_niche():
    summary = compute_niche_metrics([], max_age=365)
    assert summary["channels_analyzed"] == 0
    assert summary["outlier_ratio"] == 0
    assert summary["median_outlier_views_per_day"] is None
    assert summary["typical_duration_min"] is None
    assert summary["typical_uploads_per_week"] is None
    assert summary["outlier_ratio_by_source"] is None
    assert summary["relevance_channels"] == 0
