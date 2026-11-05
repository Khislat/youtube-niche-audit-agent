"""SQLite and CSV persistence."""

from pathlib import Path
import csv
import io
import sqlite3

from store import find_audit, init_db, recent_audits, save_audit, write_channels_csv


def _channel(channel_id, title, *, outlier, views_per_day):
    return {
        "channel_id": channel_id,
        "title": title,
        "url": f"https://www.youtube.com/channel/{channel_id}",
        "age_days": 100,
        "videos": 10,
        "total_views": 100_000,
        "avg_views": 10_000,
        "median_views": 8_000,
        "top_share": 0.2,
        "views_per_day": views_per_day,
        "avg_duration_min": 45,
        "uploads_per_week": 2,
        "subs": None,
        "is_outlier": outlier,
    }


def test_save_find_and_recent(tmp_path: Path):
    db_path = tmp_path / "researcher.db"
    init_db(db_path)
    summary = {"channels_analyzed": 1, "outliers": 1, "complete": True}
    params = {"days": 180}
    save_audit(
        db_path,
        niche="cozy fantasy ambience",
        created_at="2026-11-04T00:00:00+00:00",
        params=params,
        quota_used=104,
        summary=summary,
        channels=[_channel("ch1", "First", outlier=True, views_per_day=20)],
        videos=[
            {
                "channel_id": "ch1",
                "video_id": "v1",
                "title": "Night",
                "views": 10,
                "duration_min": 40,
                "published_at": "2026-10-01T00:00:00Z",
                "thumbnail_url": "https://img.example/a.jpg",
            },
            {
                "channel_id": "ch1",
                "video_id": "v1",
                "title": "Night duplicate",
                "views": 11,
                "duration_min": 41,
                "published_at": "2026-10-01T00:00:00Z",
                "thumbnail_url": "https://img.example/a.jpg",
            },
        ],
    )
    found = find_audit(db_path, "cozy fantasy ambience")
    assert found is not None
    assert found["quota_used"] == 104
    assert found["params"]["days"] == 180
    assert found["summary"]["outliers"] == 1
    assert find_audit(db_path, "other") is None

    save_audit(
        db_path,
        niche="rain",
        created_at="2026-11-05T00:00:00+00:00",
        params=params,
        quota_used=10,
        summary=summary,
        channels=[],
        videos=[],
    )
    recent = recent_audits(db_path, 2)
    assert [item["niche"] for item in recent] == ["rain", "cozy fantasy ambience"]
    assert recent_audits(db_path, 0) == []

    connection = sqlite3.connect(db_path)
    try:
        stored_subs, stored_flag = connection.execute(
            "SELECT subs, is_outlier FROM channels WHERE channel_id = 'ch1'"
        ).fetchone()
        video_count = connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        connection.execute("SELECT audit_id, term FROM expansions")
    finally:
        connection.close()
    assert stored_subs is None
    assert stored_flag == 1
    assert video_count == 1


def test_old_videos_table_migration(tmp_path: Path):
    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE videos (
            audit_id INTEGER NOT NULL,
            channel_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            title TEXT,
            views INTEGER,
            duration_min REAL,
            published_at TEXT,
            thumbnail_url TEXT,
            PRIMARY KEY (audit_id, video_id)
        )
        """
    )
    connection.commit()
    connection.close()
    init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(videos)")]
    finally:
        connection.close()
    assert "description_snippet" in columns
    assert "top_rank" in columns


def test_csv_puts_outliers_first(tmp_path: Path):
    path = tmp_path / "out.csv"
    write_channels_csv(
        path,
        [
            _channel("low", "Quiet", outlier=False, views_per_day=99999),
            _channel("top", "Bright", outlier=True, views_per_day=5),
        ],
    )
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        text = "\n".join(lines[1:])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["Kanal"] == "Bright"
    assert rows[0]["Outlier"] == "ha"
    assert rows[1]["Kanal"] == "Quiet"
    assert rows[1]["Outlier"] == "yo'q"
    assert rows[1]["Obunachi"] == ""
