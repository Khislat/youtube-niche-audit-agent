"""Aggregate scalar metrics from several files or saved reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio_lab import config, io

WARNING = "kichik tanlama, xulosa ehtiyotkorlik bilan"


def scalar_metrics(node, prefix: str = ""):
    """Yield (path, metric) for metrics whose value is a single number."""
    if isinstance(node, dict) and {"value", "unit", "kind", "note"} <= set(node):
        value = node["value"]
        if type(value) in (int, float):
            yield prefix, node
        return
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from scalar_metrics(value, child)


def build_profile(
    directory: str | Path,
    name: str,
    out_dir: str | Path | None = None,
    sr: int = config.FEATURE_SR,
    stems: bool = False,
    tags: bool = False,
    refresh: bool = False,
    db_path: Path | None = None,
    per_track: bool = False,
) -> Path:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Papka topilmadi: {directory}")
    out = Path(out_dir or config.DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    reports = _reports_in(directory, out, sr, stems, tags, refresh, db_path, per_track)
    if not reports:
        raise FileNotFoundError(f"Profil uchun hisobot yoki audio topilmadi: {directory}")
    payload = _aggregate_tracks(reports, name) if per_track else _aggregate(reports, name)
    safe = _safe_name(name)
    json_path = out / f"style_profile_{safe}.json"
    md_path = out / f"style_profile_{safe}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return json_path


def _reports_in(directory, out, sr, stems, tags, refresh, db_path, per_track: bool) -> list[dict]:
    found = sorted(directory.rglob("report.json"))
    if found:
        return [json.loads(path.read_text(encoding="utf-8")) for path in found]
    from audio_lab.report import analyze_to_disk

    audio = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in config.AUDIO_EXTENSIONS
    ]
    return [
        analyze_to_disk(
            path,
            out_dir=out,
            sr=sr,
            clip=None,
            stems=stems,
            tags=tags,
            refresh=refresh,
            db_path=db_path,
            segment=per_track,
        )
        for path in audio
    ]


def _aggregate(reports: list[dict], name: str) -> dict:
    buckets: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    tag_counts: dict[str, dict[str, int]] = {}
    tag_files = 0
    for report in reports:
        for path, item in scalar_metrics(report.get("metrics", {})):
            buckets.setdefault(path, []).append(float(item["value"]))
            meta.setdefault(path, {"unit": item["unit"], "kind": item["kind"]})
        stems = (report.get("stems") or {}).get("stems") or {}
        for stem_name, stem_metrics in stems.items():
            for metric_name, item in scalar_metrics(stem_metrics, f"stems.{stem_name}"):
                buckets.setdefault(metric_name, []).append(float(item["value"]))
                meta.setdefault(metric_name, {"unit": item["unit"], "kind": item["kind"]})
        tags = report.get("tags") or {}
        if tags.get("status") == "ok":
            tag_files += 1
            for head in ("genre", "instrument", "mood"):
                for row in tags.get(head) or []:
                    label = row.get("label")
                    if not label:
                        continue
                    tag_counts.setdefault(head, {})
                    tag_counts[head][label] = tag_counts[head].get(label, 0) + 1
    metrics = {}
    for path, values in buckets.items():
        arr = np.asarray(values, dtype=np.float64)
        metrics[path] = {
            "n": int(arr.size),
            "median": round(float(np.median(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "unit": meta[path]["unit"],
            "kind": meta[path]["kind"],
        }
    n_files = len(reports)
    return {
        "name": name,
        "n_files": n_files,
        "warning": WARNING if n_files < 5 else None,
        "metrics": metrics,
        "tags": {"n_files": tag_files, "counts": tag_counts},
    }


def _aggregate_tracks(reports: list[dict], name: str) -> dict:
    """Same median table as a file profile, but each row is one track."""
    track_reports = []
    per_video = []
    style_groups = []
    transitions = {"hard": 0, "gap": 0, "crossfade": 0}
    durations = []
    for report in reports:
        segments = report.get("segments") or {}
        tracks = segments.get("tracks") or []
        per_video.append(len(tracks))
        style_n = (segments.get("style_groups") or {}).get("n")
        if style_n is not None:
            style_groups.append(int(style_n))
        for track in tracks:
            kind = track.get("transition_in")
            if kind in transitions:
                transitions[kind] += 1
            durations.append(track.get("duration_s"))
            track_reports.append({"metrics": track.get("metrics") or {}, "stems": track.get("stems") or {}})
    if not track_reports:
        raise io.AudioLabError(
            "per-track uchun treklar yo'q. Avval analyze --segment ishga tushiring."
        )
    payload = _aggregate(track_reports, name)
    n_tracks = len(track_reports)
    payload["n_files"] = len(reports)
    total_transitions = sum(transitions.values())
    shares = {
        kind: round(count / total_transitions, 4) if total_transitions else 0.0
        for kind, count in transitions.items()
    }
    payload["n_tracks"] = n_tracks
    payload["per_track"] = True
    payload["warning"] = WARNING if n_tracks < config.SEG_PROFILE_MIN_TRACKS else None
    payload["track_stats"] = {
        "tracks_per_video": _distribution(per_video),
        "duration_s": _distribution(durations),
        "style_groups": _distribution(style_groups),
        "transition_shares": shares,
        "transition_kind": "proxy",
    }
    return payload


def _distribution(values) -> dict | None:
    clean = [float(value) for value in values if type(value) in (int, float)]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def _markdown(payload: dict) -> str:
    lines = [f"# Uslub profili: {payload['name']}", ""]
    if payload["warning"]:
        lines.append(payload["warning"])
        lines.append("")
    lines.append(f"Fayllar: {payload['n_files']}")
    if payload.get("n_tracks") is not None:
        lines.append(f"Treklar: {payload['n_tracks']}")
    lines.append("")
    lines.append("| metrika | n | median | p25 | p75 | min | max | kind |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for path, item in sorted(payload["metrics"].items()):
        lines.append(
            f"| {path} | {item['n']} | {item['median']} | {item['p25']} | {item['p75']} | "
            f"{item['min']} | {item['max']} | {item['kind']} |"
        )
    lines.append("")
    stats = payload.get("track_stats")
    if stats:
        lines.append("## Treklar bo'yicha")
        lines.append("")
        for label, block in (
            ("videodagi trek soni", stats.get("tracks_per_video")),
            ("trek uzunligi, s", stats.get("duration_s")),
            ("style_groups", stats.get("style_groups")),
        ):
            if not block:
                continue
            lines.append(
                f"- {label}: n={block['n']}, median={block['median']}, "
                f"min={block['min']}, max={block['max']}"
            )
        shares = stats.get("transition_shares") or {}
        lines.append(
            "- o'tish ulushi [PROKSI]: "
            + ", ".join(f"{kind}={share}" for kind, share in sorted(shares.items()))
        )
        lines.append("")
    counts = payload["tags"]["counts"]
    if counts:
        lines.append("## Teglar chastotasi")
        lines.append("")
        for head, labels in sorted(counts.items()):
            for label, count in sorted(labels.items(), key=lambda pair: (-pair[1], pair[0])):
                lines.append(f"- {head} / {label}: {count}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return cleaned or "profile"
