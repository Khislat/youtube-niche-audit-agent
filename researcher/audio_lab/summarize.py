"""Compact reading of one report.json. No new measurements."""

from __future__ import annotations

import json
from pathlib import Path

from audio_lab import config, io
from audio_lab.segment import FEATURE_DISTANCE_NOTE, groups_by_distance

BANDS = ("sub_bass", "bass", "low_mid", "mid", "high_mid", "high")
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
# Numeric columns compared across groups. Label, metric path.
COMPARE = (
    ("LUFS", "loudness.integrated_lufs"),
    ("LRA", "loudness.lra"),
    ("dinamik diapazon", "dynamic_range_db"),
    ("centroid", "spectral.centroid_hz_median"),
    ("lowpass", "spectral.lowpass_cutoff_hz"),
    ("pauza ulushi", "pause.pause_ratio"),
    ("eng uzun pauza", "pause.longest_pause_s"),
    ("onset/s", "onsets.onsets_per_sec"),
    ("repetition", "repetition.repetition_score"),
    ("noise_floor", "artifacts.noise_floor_db"),
    ("flatness 2-8 kHz", "artifacts.flatness_2_8khz"),
    ("hpss garmonik", "artifacts.hpss_harmonic_share"),
    ("hpss residual", "artifacts.hpss_residual_share"),
    ("spektr oqimi", "artifacts.flux_mean"),
    ("spektr oqimi std", "artifacts.flux_std"),
    *( (band, f"bands.share_pct.{band}") for band in BANDS ),
)


def summarize_report(report_path: str | Path, out_path: str | Path | None = None) -> Path:
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Hisobot topilmadi: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dest = Path(out_path) if out_path else report_path.with_name("summary.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(report, report_path), encoding="utf-8")
    return dest


def latest_report(directory: str | Path | None = None) -> Path:
    root = Path(directory or config.DEFAULT_OUT_DIR)
    found = [path for path in root.rglob("report.json") if path.is_file()]
    if not found:
        raise FileNotFoundError(f"report.json topilmadi: {root}")
    return max(found, key=lambda path: path.stat().st_mtime)


def render(report: dict, report_path: Path | None = None) -> str:
    """One compact markdown summary. Callers keep it within 120 lines."""
    params = report.get("params") or {}
    samples = report.get("samples") or {}
    clips = list(samples.get("clips") or [])
    rows = clips or [_whole_file_row(report)]
    lines = [_title(report, params, report_path), "", FEATURE_DISTANCE_NOTE, ""]
    lines.extend(_table(rows))
    if _stems_on(rows, params):
        lines.extend(_stems_section(rows))
    if _tags_on(rows, params):
        lines.extend(_tags_section(rows))
    distance = samples.get("feature_distance") or []
    if distance and clips:
        lines.extend(_groups_section(clips, distance))
    lines.extend(_limits(len(rows)))
    return "\n".join(lines).rstrip() + "\n"


def _title(report: dict, params: dict, report_path: Path | None) -> str:
    mode = "sample-times" if params.get("sample_times") else "butun fayl"
    duration = _duration(report, report_path)
    duration_text = "hisobotda yo'q" if duration is None else f"{_num(duration)} s"
    stems = "yoqilgan" if params.get("stems") else "o'chiq"
    tags = "yoqilgan" if params.get("tags") else "o'chiq"
    return (
        f"# {report.get('file_name') or 'audio'} | {duration_text} | "
        f"sr {params.get('sr', '—')} | {mode} | stems {stems} | tags {tags}"
    )


def _duration(report: dict, report_path: Path | None):
    stored = report.get("source_duration_s")
    if isinstance(stored, (int, float)):
        return float(stored)
    if not report.get("samples"):
        value = _value(report.get("metrics") or {}, "duration_s")
        if isinstance(value, (int, float)):
            return float(value)
    name = report.get("file_name")
    if name:
        candidate = config.ROOT / "data" / "audio_inputs" / name
        if candidate.is_file():
            probed = io.probe(candidate).get("duration_s")
            if isinstance(probed, (int, float)):
                return float(probed)
    return None


def _whole_file_row(report: dict) -> dict:
    duration = _value(report.get("metrics") or {}, "duration_s") or 0
    return {
        "index": 1,
        "start_s": 0,
        "end_s": duration,
        "metrics": report.get("metrics") or {},
        "stems": report.get("stems") or {"status": "off"},
        "tags": report.get("tags") or {"status": "off"},
    }


def table_header() -> str:
    names = [
        "start-end",
        "LUFS o'lchov",
        "LRA o'lchov",
        "centroid_hz o'lchov",
        "sub_bass% o'lchov",
        "bass% o'lchov",
        "low_mid% o'lchov",
        "mid% o'lchov",
        "high_mid% o'lchov",
        "high% o'lchov",
        "lowpass_hz taxmin",
        "pauza% o'lchov",
        "uzun pauza s o'lchov",
        "onset/s proksi",
        "tempo taxmin",
        "key taxmin",
        "repetition proksi",
        "noise_floor_db o'lchov",
        "flatness_2_8k o'lchov",
        "hpss_h proksi",
        "hpss_r proksi",
        "flux_mean proksi",
        "flux_std proksi",
    ]
    return "| " + " | ".join(names) + " |"


def table_rule() -> str:
    return "| " + " | ".join("---" for _ in range(23)) + " |"


def table_cells(row: dict) -> list[str]:
    metrics = row.get("metrics") or {}
    pause = _value(metrics, "pause.pause_ratio")
    pause_pct = "" if not isinstance(pause, (int, float)) else _num(float(pause) * 100)
    return [
        f"{_num(row.get('start_s'))}-{_num(row.get('end_s'))}",
        _plain(metrics, "loudness.integrated_lufs"),
        _level_span(metrics),
        _plain(metrics, "spectral.centroid_hz_median"),
        *[_plain(metrics, f"bands.share_pct.{band}") for band in BANDS],
        _plain(metrics, "spectral.lowpass_cutoff_hz"),
        pause_pct,
        _plain(metrics, "pause.longest_pause_s"),
        _plain(metrics, "onsets.onsets_per_sec"),
        _reliable_text(metrics, "tempo", "tempo.bpm"),
        _reliable_text(metrics, "key", "key.label"),
        _plain(metrics, "repetition.repetition_score"),
        _plain(metrics, "artifacts.noise_floor_db"),
        _plain(metrics, "artifacts.flatness_2_8khz"),
        _plain(metrics, "artifacts.hpss_harmonic_share"),
        _plain(metrics, "artifacts.hpss_residual_share"),
        _plain(metrics, "artifacts.flux_mean"),
        _plain(metrics, "artifacts.flux_std"),
    ]


def _table(rows: list[dict]) -> list[str]:
    lines = ["## Parchalar", "", table_header(), table_rule()]
    for row in rows:
        lines.append("| " + " | ".join(table_cells(row)) + " |")
    lines.append("")
    return lines


def _level_span(metrics: dict) -> str:
    lra = _value(metrics, "loudness.lra")
    if isinstance(lra, (int, float)):
        return _num(lra)
    return _plain(metrics, "dynamic_range_db")


def _stems_on(rows: list[dict], params: dict) -> bool:
    if not params.get("stems"):
        return False
    return any((row.get("stems") or {}).get("status") == "ok" for row in rows)


def _tags_on(rows: list[dict], params: dict) -> bool:
    if not params.get("tags"):
        return False
    return any((row.get("tags") or {}).get("status") == "ok" for row in rows)


def _stems_section(rows: list[dict]) -> list[str]:
    header = "| parcha | " + " | ".join(STEMS) + " |"
    rule = "| " + " | ".join("---" for _ in range(1 + len(STEMS))) + " |"
    lines = ["## Stem ulushi", "", header, rule]
    for row in rows:
        stems = ((row.get("stems") or {}).get("stems")) or {}
        cells = []
        for name in STEMS:
            value = _value(stems.get(name) or {}, "energy_share")
            cells.append("" if not isinstance(value, (int, float)) else _num(value))
        lines.append("| " + " | ".join([str(row.get("index"))] + cells) + " |")
    lines.append("")
    lines.append("drums yoki guitar >0.05 bo'lsa, bu stem artefakti bo'lishi mumkin (Demucs bleeding).")
    lines.append("")
    return lines


def _tags_section(rows: list[dict]) -> list[str]:
    scored: dict[str, float] = {}
    for row in rows:
        tags = row.get("tags") or {}
        for head in ("genre", "instrument", "mood"):
            for item in tags.get(head) or []:
                label = item.get("label")
                raw = item.get("probability") or {}
                value = raw.get("value") if isinstance(raw, dict) else raw
                if not label or not isinstance(value, (int, float)):
                    continue
                key = f"{head}: {label}"
                scored[key] = max(scored.get(key, 0.0), float(value))
    top = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    lines = ["## Teglar", ""]
    if not top:
        lines.append("Teg yo'q.")
    else:
        for label, probability in top:
            lines.append(f"- {label}: {_num(probability)} ehtimollik")
    lines.append("")
    return lines


def _groups_section(clips: list[dict], distance: list[list[float]]) -> list[str]:
    indexes = [int(clip.get("index") or i) for i, clip in enumerate(clips, start=1)]
    groups = groups_by_distance(distance, indexes)
    threshold = groups["distance_threshold"]
    lines = [
        "## Guruhlar farqi",
        "",
        f"Dendrogram kesimi: kosinus masofa {threshold} (average linkage). "
        f"[{groups['badge']}]",
        "",
    ]
    for cluster in groups["clusters"]:
        lines.append(f"- guruh {cluster['id']}: {cluster['tracks']}")
    pair = _two_largest(groups["clusters"])
    if pair is None:
        lines.append("")
        return lines
    left, right = pair
    ranked = _metric_gaps(clips, left["tracks"], right["tracks"])
    lines.append("")
    lines.append(
        f"Eng katta farq, guruh {left['id']} / guruh {right['id']} / farq %:"
    )
    if not ranked:
        lines.append("- solishtiradigan son yo'q")
    for label, a, b, gap in ranked[:5]:
        lines.append(f"- {label}: {_num(a)} / {_num(b)} / {_num(gap)}%")
    lines.append("")
    return lines


def _two_largest(clusters: list[dict]) -> tuple[dict, dict] | None:
    if len(clusters) < 2:
        return None
    ordered = sorted(clusters, key=lambda cluster: (-len(cluster["tracks"]), cluster["id"]))
    return ordered[0], ordered[1]


def _metric_gaps(clips: list[dict], left_ids: list[int], right_ids: list[int]) -> list[tuple]:
    by_index = {int(clip.get("index")): clip for clip in clips}
    ranked = []
    for label, path in COMPARE:
        left = _mean(by_index, left_ids, path)
        right = _mean(by_index, right_ids, path)
        if left is None or right is None:
            continue
        denom = max(abs(left), abs(right))
        gap = 0.0 if denom < 1e-9 else abs(left - right) / denom * 100.0
        ranked.append((label, left, right, gap))
    ranked.sort(key=lambda item: item[3], reverse=True)
    return ranked


def _mean(by_index: dict, ids: list[int], path: str) -> float | None:
    values = []
    for index in ids:
        value = _value((by_index.get(index) or {}).get("metrics") or {}, path)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _limits(n_rows: int) -> list[str]:
    lines = [
        "## Cheklovlar",
        "",
        "- Tempo va tonallik ambientda ishonchsiz; tempoda yarim yoki ikki baravar noaniqlik bor. Taxmin.",
        "- Stemlar ajratish natijasi, aralashma va artefakt bor; pianino ishonchsizroq; torli va pad 'other' ga tushadi. Taxmin.",
        "- Teglar ehtimollik, o'lchov emas.",
        "- Bu tahlil \"nega yoqadi\" degan savolga javob bermaydi.",
    ]
    if n_rows < 8:
        lines.append("- n<8 parcha bo'lsa guruhlar taxminiy.")
    lines.append("")
    return lines


def _reliable_text(metrics: dict, block: str, path: str) -> str:
    node = metrics.get(block) or {}
    if node.get("reliable") is not True:
        return "ishonchsiz"
    return _plain(metrics, path)


def _plain(metrics: dict, path: str) -> str:
    value = _value(metrics, path)
    if not isinstance(value, (int, float)):
        return "" if value is None else str(value)
    return _num(value)


def _value(node: dict, path: str):
    if not path:
        return None
    current = node
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, dict) and "value" in current:
        return current.get("value")
    return current


def _num(value) -> str:
    if not isinstance(value, (int, float)):
        return ""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"
