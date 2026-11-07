"""Compare one track with a style profile. Numbers only, no written advice."""

from __future__ import annotations

import json
from pathlib import Path

from audio_lab import config
from audio_lab.profile import scalar_metrics


def compare_track(
    track: str | Path,
    profile_path: str | Path,
    out_dir: str | Path | None = None,
    sr: int = config.FEATURE_SR,
    clip: tuple[float, float] | None = None,
    stems: bool = False,
    tags: bool = False,
    refresh: bool = False,
    db_path: Path | None = None,
) -> str:
    from audio_lab.report import analyze_to_disk

    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    out = Path(out_dir or config.DEFAULT_OUT_DIR)
    report = analyze_to_disk(
        track,
        out_dir=out,
        sr=sr,
        clip=clip,
        stems=stems,
        tags=tags,
        refresh=refresh,
        db_path=db_path,
    )
    values = dict(scalar_metrics(report.get("metrics", {})))
    stem_block = (report.get("stems") or {}).get("stems") or {}
    for stem_name, stem_metrics in stem_block.items():
        values.update(dict(scalar_metrics(stem_metrics, f"stems.{stem_name}")))
    rows = []
    for path, stats in profile.get("metrics", {}).items():
        metric = values.get(path)
        if metric is None or type(metric["value"]) not in (int, float):
            continue
        value = float(metric["value"])
        median = float(stats["median"])
        p25 = float(stats["p25"])
        p75 = float(stats["p75"])
        iqr = p75 - p25
        if iqr == 0:
            deviation = 0.0 if value == median else None
        else:
            deviation = (value - median) / iqr
        flagged = abs(deviation) > 1.5 if deviation is not None else True
        rows.append(
            {
                "metric": path,
                "value": round(value, 4),
                "median": round(median, 4),
                "p25": round(p25, 4),
                "p75": round(p75, 4),
                "deviation": None if deviation is None else round(deviation, 4),
                "flagged": flagged,
            }
        )
    rows.sort(key=lambda row: -(abs(row["deviation"]) if row["deviation"] is not None else 1e9))
    text = _table(Path(track).name, profile.get("name", ""), rows)
    safe_track = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(track).stem)[:60]
    safe_profile = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(profile.get("name", "profile")))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{safe_track}_{safe_profile}.md").write_text(text, encoding="utf-8")
    return text


def _table(track_name: str, profile_name: str, rows: list[dict]) -> str:
    lines = [
        f"# Solishtirish: {track_name} / {profile_name}",
        "",
        "| metrika | bizning qiymat | profil median | p25-p75 | og'ish | chegaradan tashqari |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        deviation = "" if row["deviation"] is None else f"{row['deviation']}"
        flag = "ha" if row["flagged"] else "yo'q"
        lines.append(
            f"| {row['metric']} | {row['value']} | {row['median']} | "
            f"{row['p25']}–{row['p75']} | {deviation} | {flag} |"
        )
    lines.append("")
    return "\n".join(lines)
