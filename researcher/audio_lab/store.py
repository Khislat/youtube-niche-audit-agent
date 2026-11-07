"""SQLite cache of audio analyses. Separate from the Phase 1 researcher database."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from audio_lab import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    report_path TEXT NOT NULL,
    duration_s REAL,
    summary_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audio_analyses_hash
    ON audio_analyses (file_hash, params_hash);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def params_sha256(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def insert_analysis(
    file_hash: str,
    params_hash: str,
    report_path: Path,
    duration_s: float | None,
    summary: dict,
    db_path: Path | None = None,
) -> None:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audio_analyses
                (file_hash, params_hash, created_at, report_path, duration_s, summary_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_hash,
                params_hash,
                created_at,
                str(report_path),
                duration_s,
                json.dumps(summary, ensure_ascii=False, sort_keys=True),
            ),
        )
