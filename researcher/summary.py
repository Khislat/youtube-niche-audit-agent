"""Build a Markdown rollup from Phase 1.2 audits in researcher.db."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import sqlite3

from config import DEFAULT_DB_PATH, DEFAULT_OUTPUT_DIR

MAX_LINES = 400


def _connect(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_json(text: str) -> dict:
    return json.loads(text) if text else {}


def _is_eski_qoida(niche: str, params: dict) -> bool:
    if "(eski qoida)" in niche.casefold():
        return True
    return params.get("min_ratio") is None


def _is_phase_12(summary: dict) -> bool:
    return "outlier_ai_flag_pct" in summary


def _fmt_int(value) -> str:
    if value is None:
        return "-"
    return f"{int(round(float(value))):,}"


def _fmt_ratio(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _fmt_pct_from_ratio(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def _load_audits(connection) -> list[dict]:
    rows = connection.execute(
        "SELECT id, niche, created_at, quota_used, params_json, summary_json FROM audits ORDER BY id"
    ).fetchall()
    audits: list[dict] = []
    for row in rows:
        params = _parse_json(row["params_json"])
        summary = _parse_json(row["summary_json"])
        audits.append(
            {
                "id": row["id"],
                "niche": row["niche"],
                "created_at": row["created_at"],
                "quota_used": row["quota_used"],
                "params": params,
                "summary": summary,
            }
        )
    return audits


def _latest_phase_12_by_niche(audits: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Return the newest qualifying audit per niche and skip reasons for excluded latest rows."""
    by_niche: dict[str, list[dict]] = {}
    for audit in audits:
        by_niche.setdefault(audit["niche"], []).append(audit)

    selected: dict[str, dict] = {}
    skipped_notes: list[str] = []

    for niche, items in sorted(by_niche.items()):
        latest = max(items, key=lambda item: item["id"])
        params = latest["params"]
        summary = latest["summary"]
        if _is_eski_qoida(niche, params):
            skipped_notes.append(
                f"{niche}: oxirgi audit (id {latest['id']}) eski qoida / Faza 1.0 — o'tkazildi"
            )
            continue
        if not _is_phase_12(summary):
            skipped_notes.append(
                f"{niche}: oxirgi audit (id {latest['id']}) Faza 1.2 emas — o'tkazildi"
            )
            continue
        selected[niche] = latest
    return selected, skipped_notes


def _compare_rows(selected: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for niche, audit in selected.items():
        summary = audit["summary"]
        rows.append(
            {
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
                "quota_used": audit["quota_used"],
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["outlier_ratio_by_source"] if row["outlier_ratio_by_source"] is not None else -1),
            -(row["median_outlier_views_per_day"] if row["median_outlier_views_per_day"] is not None else -1),
            row["niche"].lower(),
        )
    )
    return rows


def _cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|")


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return lines


def _load_outliers(connection, audit_id: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT title, age_days, videos, avg_views, median_views, median_avg_ratio, top_share,
               recent10_median_views, views_per_day, found_by, ai_flag_share, brand_flag_share,
               ai_terms, brand_terms, channel_id
        FROM channels
        WHERE audit_id = ? AND is_outlier = 1
        ORDER BY views_per_day DESC, avg_views DESC, title COLLATE NOCASE
        """,
        (audit_id,),
    ).fetchall()


def _load_top_videos(connection, audit_id: int, channel_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT title, views, duration_min, published_at, top_rank
        FROM videos
        WHERE audit_id = ? AND channel_id = ? AND top_rank IS NOT NULL
        ORDER BY top_rank ASC
        """,
        (audit_id, channel_id),
    ).fetchall()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _video_age_days(published_at: str | None, *, at: datetime) -> float | None:
    published = _parse_dt(published_at)
    if published is None:
        return None
    moment = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    return max((moment - published).total_seconds() / 86400, 0.0)


def _load_suggestions(connection, audit_id: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT term, channel_count
        FROM suggestions
        WHERE audit_id = ?
        ORDER BY channel_count DESC, term COLLATE NOCASE
        LIMIT 10
        """,
        (audit_id,),
    ).fetchall()


def _incomplete_audits(audits: list[dict]) -> list[str]:
    lines: list[str] = []
    for audit in audits:
        if audit["summary"].get("complete") is False:
            lines.append(f"{audit['niche']} (audit id {audit['id']}, {audit['created_at']}) — to'liq emas")
    return lines


def _niche_section(connection, audit: dict) -> list[str]:
    niche = audit["niche"]
    audit_id = audit["id"]
    audit_moment = _parse_dt(audit["created_at"]) or datetime.now(timezone.utc)
    lines = [f"## {niche}", ""]

    outliers = _load_outliers(connection, audit_id)
    if not outliers:
        lines.append("Outlier yo'q.")
        lines.append("")
    else:
        lines.append("### Outlier kanallar")
        lines.append("")
        outlier_headers = [
            "kanal",
            "yosh",
            "video",
            "o'rtacha",
            "median",
            "med/ort",
            "top ulush",
            "oxirgi10 med.",
            "ko'rish/kun",
            "found_by",
            "AI ulush",
            "brend ulush",
            "AI term",
            "brend term",
        ]
        outlier_rows = []
        for channel in outliers:
            outlier_rows.append(
                [
                    channel["title"] or "",
                    _fmt_ratio(channel["age_days"]),
                    _fmt_int(channel["videos"]),
                    _fmt_int(channel["avg_views"]),
                    _fmt_int(channel["median_views"]),
                    _fmt_ratio(channel["median_avg_ratio"]),
                    _fmt_ratio(channel["top_share"]),
                    _fmt_int(channel["recent10_median_views"]),
                    _fmt_int(channel["views_per_day"]),
                    channel["found_by"] or "-",
                    _fmt_ratio(channel["ai_flag_share"]),
                    _fmt_ratio(channel["brand_flag_share"]),
                    channel["ai_terms"] or "",
                    channel["brand_terms"] or "",
                ]
            )
        lines.extend(_md_table(outlier_headers, outlier_rows))
        lines.append("")

        lines.append("### Top 5 videolar (outlier kanallar)")
        lines.append("")
        for channel in outliers:
            videos = _load_top_videos(connection, audit_id, channel["channel_id"])
            if not videos:
                continue
            lines.append(f"**{channel['title']}**")
            for video in videos:
                age = _video_age_days(video["published_at"], at=audit_moment)
                lines.append(
                    f"- {video['title']} | {_fmt_int(video['views'])} | "
                    f"{_fmt_ratio(video['duration_min'])} | {_fmt_ratio(age)}"
                )
            lines.append("")

    suggestions = _load_suggestions(connection, audit_id)
    lines.append("### Niche taklifi (10 ta)")
    lines.append("")
    if suggestions:
        sug_rows = [[row["term"], _fmt_int(row["channel_count"])] for row in suggestions]
        lines.extend(_md_table(["ibora", "kanallar"], sug_rows))
    else:
        lines.append("Taklif yo'q.")
    lines.append("")
    return lines


def build_report(db_path=DEFAULT_DB_PATH) -> tuple[list[str], str]:
    connection = _connect(db_path)
    try:
        audits = _load_audits(connection)
        selected, skipped_latest = _latest_phase_12_by_niche(audits)
        compare = _compare_rows(selected)

        report_date = date.today().isoformat()
        lines: list[str] = [
            f"# Faza 1.2 umumiy hisobot ({report_date})",
            "",
            "Manba: `data/researcher.db` — har niche uchun eng oxirgi Faza 1.2 audit.",
            "",
            "## Solishtirma",
            "",
        ]

        compare_headers = [
            "niche",
            "kanallar",
            "yangi",
            "outlier",
            "outlier % (hammasi)",
            "outlier % (rel.)",
            "rel. kanallar",
            "outlier med. ko'rish/kun",
            "odatiy daq",
            "odatiy /hafta",
            "outlier AI %",
            "outlier brend %",
            "kvota",
        ]
        compare_data = []
        for row in compare:
            compare_data.append(
                [
                    row["niche"],
                    _fmt_int(row["channels_analyzed"]),
                    _fmt_int(row["new_channels"]),
                    _fmt_int(row["outliers"]),
                    _fmt_pct_from_ratio(row["outlier_ratio"]),
                    _fmt_pct_from_ratio(row["outlier_ratio_by_source"]),
                    _fmt_int(row["relevance_channels"]),
                    _fmt_int(row["median_outlier_views_per_day"]),
                    _fmt_ratio(row["typical_duration_min"]),
                    _fmt_ratio(row["typical_uploads_per_week"]),
                    _fmt_pct_from_ratio(row["outlier_ai_flag_pct"]),
                    _fmt_pct_from_ratio(row["outlier_brand_flag_pct"]),
                    _fmt_int(row["quota_used"]),
                ]
            )
        lines.extend(_md_table(compare_headers, compare_data))
        lines.append("")

        niche_order = [row["niche"] for row in compare]
        niche_blocks: list[tuple[str, list[str]]] = []
        for niche in niche_order:
            niche_blocks.append((niche, _niche_section(connection, selected[niche])))

        total_quota = sum(audit["quota_used"] for audit in selected.values())
        dropped: list[str] = []

        def assemble() -> list[str]:
            body = list(lines)
            for niche, block in niche_blocks:
                body.extend(block)
            body.append("## Yakun")
            body.append("")
            body.append(f"Jami sarflangan kvota (hisobotdagi auditlar): {_fmt_int(total_quota)}")
            incomplete = _incomplete_audits(audits)
            footer_items = skipped_latest + incomplete
            if dropped:
                body.append("")
                body.append("Quyidagi niche bo'limlari 400 qator chegarasi tufayli kiritilmadi:")
                for name in dropped:
                    body.append(f"- {name}")
            if footer_items:
                body.append("")
                body.append("### O'tkazilgan / to'liq emas")
                for item in footer_items:
                    body.append(f"- {item}")
            elif not dropped:
                body.append("")
                body.append("O'tkazilgan yoki xato bilan tugagan niche bazada qayd etilmagan.")
            return body

        while len(assemble()) > MAX_LINES and niche_blocks:
            dropped.append(niche_blocks.pop()[0])

        lines = assemble()
        if len(lines) > MAX_LINES:
            lines = lines[: MAX_LINES - 1]
            lines.append("… (400 qator chegarasi — qisqartirildi)")

        return lines, report_date
    finally:
        connection.close()


def main() -> int:
    lines, report_date = build_report()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_OUTPUT_DIR / f"summary_{report_date}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print(f"Qatorlar: {len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
