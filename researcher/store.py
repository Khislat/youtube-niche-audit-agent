"""SQLite persistence and the Phase 1 CSV export."""

from pathlib import Path
import csv
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    niche TEXT NOT NULL,
    created_at TEXT NOT NULL,
    params_json TEXT NOT NULL,
    quota_used INTEGER NOT NULL,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
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
    recent10_median_views REAL,
    found_by TEXT,
    median_avg_ratio REAL,
    ai_flag_share REAL,
    brand_flag_share REAL,
    ai_terms TEXT,
    brand_terms TEXT,
    PRIMARY KEY (audit_id, channel_id)
);

CREATE TABLE IF NOT EXISTS videos (
    audit_id INTEGER NOT NULL,
    channel_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    title TEXT,
    views INTEGER,
    duration_min REAL,
    published_at TEXT,
    thumbnail_url TEXT,
    description_snippet TEXT,
    tags_json TEXT,
    ai_terms TEXT,
    brand_terms TEXT,
    top_rank INTEGER,
    PRIMARY KEY (audit_id, video_id)
);

CREATE TABLE IF NOT EXISTS expansions (
    audit_id INTEGER NOT NULL,
    term TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS suggestions (
    audit_id INTEGER NOT NULL,
    term TEXT NOT NULL,
    channel_count INTEGER NOT NULL,
    example_title TEXT
);

CREATE INDEX IF NOT EXISTS idx_audits_niche ON audits (niche);
"""

# Added in phase 1.1. ALTER keeps rows already stored in an older database.
NEW_CHANNEL_COLUMNS = (
    ("recent10_median_views", "REAL"),
    ("found_by", "TEXT"),
    ("median_avg_ratio", "REAL"),
    ("ai_flag_share", "REAL"),
    ("brand_flag_share", "REAL"),
    ("ai_terms", "TEXT"),
    ("brand_terms", "TEXT"),
)

NEW_VIDEO_COLUMNS = (
    ("description_snippet", "TEXT"),
    ("tags_json", "TEXT"),
    ("ai_terms", "TEXT"),
    ("brand_terms", "TEXT"),
    ("top_rank", "INTEGER"),
)

CSV_COLUMNS = [
    ("title", "Kanal"),
    ("url", "Havola"),
    ("channel_id", "Kanal ID"),
    ("age_days", "Yosh (kun)"),
    ("videos", "Videolar"),
    ("total_views", "Jami ko'rish"),
    ("avg_views", "O'rtacha ko'rish"),
    ("median_views", "Median ko'rish"),
    ("recent10_median_views", "recent10_median_views"),
    ("median_avg_ratio", "median_avg_ratio"),
    ("top_share", "Top ulush"),
    ("views_per_day", "Ko'rish/kun"),
    ("avg_duration_min", "O'rtacha davomiylik (daq)"),
    ("uploads_per_week", "Yuklash/hafta"),
    ("subs", "Obunachi"),
    ("found_by", "found_by"),
    ("ai_flag_share", "ai_flag_share"),
    ("brand_flag_share", "brand_flag_share"),
    ("ai_terms", "ai_terms"),
    ("brand_terms", "brand_terms"),
    ("is_outlier", "Outlier"),
]

TOP_VIDEO_COLUMNS = [
    ("channel_title", "kanal"),
    ("top_rank", "o'rin (1-5)"),
    ("title", "sarlavha"),
    ("views", "ko'rish"),
    ("duration_min", "davomiylik (daq)"),
    ("published_at", "chiqqan sana"),
    ("age_days", "necha kun oldin"),
    ("views_per_day", "ko'rish/kun"),
    ("video_url", "video havolasi"),
    ("thumbnail_url", "thumbnail havolasi"),
    ("ai_terms", "AI belgilari"),
    ("brand_terms", "brend belgilari"),
]

SUGGESTION_COLUMNS = [
    ("term", "ibora"),
    ("channel_count", "kanallar soni"),
    ("example_title", "misol sarlavha"),
]

COMPARE_COLUMNS = [
    ("niche", "niche"),
    ("channels_analyzed", "kanallar"),
    ("new_channels", "yangi kanallar"),
    ("outliers", "outlier"),
    ("outlier_ratio", "outlier % (hammasi)"),
    ("outlier_ratio_by_source", "outlier % (relevance)"),
    ("relevance_channels", "relevance kanallar soni"),
    ("median_outlier_views_per_day", "outlier median ko'rish/kun"),
    ("typical_duration_min", "odatiy davomiylik"),
    ("typical_uploads_per_week", "odatiy video/hafta"),
    ("outlier_ai_flag_pct", "outlier'larda AI belgisi %"),
    ("outlier_brand_flag_pct", "outlier'larda brend belgisi %"),
    ("quota_used", "kvota"),
    ("note", "eslatma"),
]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _migrate_table(connection: sqlite3.Connection, table: str, columns: tuple[tuple[str, str], ...]) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        return
    existing = {row[1] for row in rows}
    for name, column_type in columns:
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")


def _migrate_channels(connection: sqlite3.Connection) -> None:
    _migrate_table(connection, "channels", NEW_CHANNEL_COLUMNS)


def _migrate_videos(connection: sqlite3.Connection) -> None:
    _migrate_table(connection, "videos", NEW_VIDEO_COLUMNS)


def init_db(db_path: Path) -> None:
    connection = _connect(db_path)
    try:
        connection.executescript(SCHEMA)
        _migrate_channels(connection)
        _migrate_videos(connection)
        connection.commit()
    finally:
        connection.close()


def _audit_from_row(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["params"] = json.loads(data.pop("params_json"))
    data["summary"] = json.loads(data.pop("summary_json"))
    return data


def save_audit(
    db_path: Path,
    *,
    niche: str,
    created_at: str,
    params: dict,
    quota_used: int,
    summary: dict,
    channels: list[dict],
    videos: list[dict],
    suggestions: list[dict] | None = None,
) -> int:
    """Insert one audit plus its channels and videos. Returns the audit id."""
    init_db(db_path)
    unique_videos: dict[str, dict] = {}
    for video in videos:
        video_id = video.get("video_id")
        if video_id:
            unique_videos[video_id] = video

    connection = _connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO audits (niche, created_at, params_json, quota_used, summary_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    niche,
                    created_at,
                    json.dumps(params, ensure_ascii=False),
                    quota_used,
                    json.dumps(summary, ensure_ascii=False),
                ),
            )
            audit_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO channels (
                    audit_id, channel_id, title, url, age_days, videos, total_views,
                    avg_views, median_views, top_share, views_per_day, avg_duration_min,
                    uploads_per_week, subs, is_outlier, recent10_median_views,
                    found_by, median_avg_ratio, ai_flag_share, brand_flag_share,
                    ai_terms, brand_terms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        audit_id,
                        channel["channel_id"],
                        channel.get("title"),
                        channel.get("url"),
                        channel.get("age_days"),
                        channel.get("videos"),
                        channel.get("total_views"),
                        channel.get("avg_views"),
                        channel.get("median_views"),
                        channel.get("top_share"),
                        channel.get("views_per_day"),
                        channel.get("avg_duration_min"),
                        channel.get("uploads_per_week"),
                        channel.get("subs"),
                        1 if channel.get("is_outlier") else 0,
                        channel.get("recent10_median_views"),
                        channel.get("found_by"),
                        channel.get("median_avg_ratio"),
                        channel.get("ai_flag_share"),
                        channel.get("brand_flag_share"),
                        channel.get("ai_terms"),
                        channel.get("brand_terms"),
                    )
                    for channel in channels
                ],
            )
            connection.executemany(
                """
                INSERT INTO videos (
                    audit_id, channel_id, video_id, title, views, duration_min,
                    published_at, thumbnail_url, description_snippet, tags_json,
                    ai_terms, brand_terms, top_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        audit_id,
                        video.get("channel_id"),
                        video_id,
                        video.get("title"),
                        video.get("views"),
                        video.get("duration_min"),
                        video.get("published_at"),
                        video.get("thumbnail_url"),
                        video.get("description_snippet"),
                        json.dumps(video.get("tags") or [], ensure_ascii=False),
                        video.get("ai_terms"),
                        video.get("brand_terms"),
                        video.get("top_rank"),
                    )
                    for video_id, video in unique_videos.items()
                ],
            )
            if suggestions:
                connection.executemany(
                    """
                    INSERT INTO suggestions (audit_id, term, channel_count, example_title)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            audit_id,
                            item["term"],
                            item["channel_count"],
                            item.get("example_title"),
                        )
                        for item in suggestions
                    ],
                )
        return audit_id
    finally:
        connection.close()


def find_audit(db_path: Path, niche: str) -> dict | None:
    """Return the latest audit for this exact niche, or None."""
    init_db(db_path)
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM audits WHERE niche = ? ORDER BY id DESC LIMIT 1",
            (niche,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return _audit_from_row(row)


def audited_niche_names(db_path: Path) -> set[str]:
    init_db(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute("SELECT DISTINCT niche FROM audits").fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def recent_audits(db_path: Path, n: int) -> list[dict]:
    if n <= 0:
        return []
    init_db(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM audits ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        connection.close()
    return [_audit_from_row(row) for row in rows]


def sort_channels(channels: list[dict]) -> list[dict]:
    def sort_key(row: dict):
        views_per_day = row.get("views_per_day")
        avg_views = row.get("avg_views")
        return (
            0 if row.get("is_outlier") else 1,
            -(views_per_day if views_per_day is not None else -1),
            -(avg_views if avg_views is not None else -1),
            (row.get("title") or "").lower(),
        )

    return sorted(channels, key=sort_key)


def _csv_value(row: dict, key: str):
    value = row.get(key)
    if key == "is_outlier":
        return "ha" if value else "yo'q"
    if value is None:
        return ""
    if isinstance(value, float):
        if key in {"top_share", "median_avg_ratio", "ai_flag_share", "brand_flag_share"}:
            return f"{value:.4f}"
        return f"{value:.2f}"
    return value


def _write_csv(path: Path, columns: list[tuple[str, str]], rows: list[dict], *, preamble: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [label for _, label in columns]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if preamble:
            handle.write(f"# {preamble}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({label: row.get(key, "") for key, label in columns})


def write_channels_csv(path: Path, channels: list[dict], *, disclaimer: str | None = None) -> None:
    """Write the channel table with outliers first."""
    from config import CSV_FLAG_DISCLAIMER

    rows = []
    for row in sort_channels(channels):
        rows.append({key: _csv_value(row, key) for key, _ in CSV_COLUMNS})
    _write_csv(path, CSV_COLUMNS, rows, preamble=disclaimer or CSV_FLAG_DISCLAIMER)


def write_top_videos_csv(path: Path, rows: list[dict]) -> None:
    _write_csv(path, TOP_VIDEO_COLUMNS, rows)


def write_suggestions_csv(path: Path, rows: list[dict]) -> None:
    _write_csv(path, SUGGESTION_COLUMNS, rows)


def _compare_value(key: str, value):
    if value is None:
        return ""
    if key in {"outlier_ratio", "outlier_ratio_by_source", "outlier_ai_flag_pct", "outlier_brand_flag_pct"}:
        return f"{float(value) * 100:.0f}%"
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def sort_compare_rows(rows: list[dict]) -> list[dict]:
    def sort_key(row: dict):
        relevance = row.get("outlier_ratio_by_source")
        median_vpd = row.get("median_outlier_views_per_day")
        return (
            -(relevance if relevance is not None else -1),
            -(median_vpd if median_vpd is not None else -1),
            (row.get("niche") or "").lower(),
        )

    return sorted(rows, key=sort_key)


def write_compare_csv(path: Path, rows: list[dict]) -> None:
    prepared = []
    for row in sort_compare_rows(rows):
        item = {key: _compare_value(key, row.get(key)) for key, _ in COMPARE_COLUMNS if key != "note"}
        item["note"] = row.get("note") or ""
        prepared.append(item)
    _write_csv(path, COMPARE_COLUMNS, prepared)
