"""Write JSON, CSV, PNG, and Markdown for one analysis."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_lab import config, io, measure, stems, store, tags

STRENGTH_NOTE = (
    "strength chegaraning kuchini ko'rsatadi, to'g'riligini emas; "
    "A va B real mixlarda soxta chegaralar ham `high` bo'lgan."
)
BOUNDARY_TITLE = (
    "Chegara nomzodlari (ishonchsiz, tracklist yoki qo'lda tuzatish tavsiya etiladi)"
)

BADGE = {
    "measured": "[O'LCHANGAN]",
    "estimate": "[TAXMINIY]",
    "proxy": "[PROKSI]",
    "probabilistic": "[EHTIMOLLIK]",
}
ROWS = (
    ("duration_s", "Davomiylik"),
    ("sample_rate_in", "Asl namuna chastotasi"),
    ("channels", "Kanallar"),
    ("loudness.integrated_lufs", "Integrallangan balandlik"),
    ("loudness.lra", "Loudness range"),
    ("loudness.true_peak_dbtp", "True peak"),
    ("dynamic_range_db", "Dinamik diapazon"),
    ("bands.share_pct.sub_bass", "Sub-bass ulushi"),
    ("bands.share_pct.bass", "Bass ulushi"),
    ("bands.share_pct.low_mid", "Past-o'rta ulushi"),
    ("bands.share_pct.mid", "O'rta ulushi"),
    ("bands.share_pct.high_mid", "Yuqori-o'rta ulushi"),
    ("bands.share_pct.high", "Yuqori ulushi"),
    ("spectral.centroid_hz_median", "Spektr centroid (median)"),
    ("spectral.centroid_hz_p25", "Spektr centroid p25"),
    ("spectral.centroid_hz_p75", "Spektr centroid p75"),
    ("spectral.rolloff85_hz", "Rolloff 85%"),
    ("spectral.flatness_median", "Spektr tekisligi"),
    ("spectral.lowpass_cutoff_hz", "Past-o'tkazish chegarasi"),
    ("onsets.onsets_per_sec", "Onset tezligi"),
    ("onsets.ioi_median_s", "Onset oralig'i medianasi"),
    ("onsets.ioi_cv", "Onset oralig'i CV"),
    ("pause.pause_ratio", "Pauza ulushi"),
    ("pause.longest_pause_s", "Eng uzun pauza"),
    ("pause.pauses_over_1s_count", "1 soniyadan uzun pauzalar"),
    ("tempo.bpm", "Tempo"),
    ("tempo.confidence", "Tempo ishonchi"),
    ("key.label", "Tonallik"),
    ("key.margin", "Tonallik margin"),
    ("repetition.repetition_score", "Takrorlanish"),
    ("repetition.top_lag_s", "Takrorlanish lagi"),
    ("stereo.width", "Stereo kenglik"),
    ("stereo.channel_correlation", "Kanallar korrelyatsiyasi"),
    ("noise_floor_dbfs", "Sokin joylar (5% RMS, proksi)"),
    ("artifacts.noise_floor_db", "Sokin 10% RMS"),
    ("artifacts.flatness_2_8khz", "Tekislik 2–8 kHz"),
    ("artifacts.hpss_harmonic_share", "HPSS garmonik ulush"),
    ("artifacts.hpss_residual_share", "HPSS residual ulush"),
    ("artifacts.flux_mean", "Spektr oqimi o'rtacha"),
    ("artifacts.flux_std", "Spektr oqimi std"),
)
LIMITATIONS = """## Cheklovlar

- Tempo va tonallik ambient yoki ritmsiz musiqada ishonchsiz; tempoda yarim yoki ikki baravar noaniqlik bor. [TAXMINIY]
- Stemlar ML ajratish natijasi, aralashma (bleeding) va artefakt bor; pianino stemi ishonchsizroq; violonchel/torli/pad 'other' stemiga tushadi va alohida ajratilmaydi. [TAXMINIY]
- Teglar ehtimollik, o'lchov emas. [EHTIMOLLIK]
- Bu tahlil "nega yoqadi" degan savolga javob bermaydi.
"""


def long_file_warning(duration_s: float | None, segment: bool) -> str | None:
    """A long mix is only a warning. Segmentation stays off until asked."""
    if segment or duration_s is None or duration_s <= config.SEG_LONG_FILE_S:
        return None
    return (
        "Fayl 15 daqiqadan uzun. Butun fayl o'rtachasi treklarni aralashtiradi. "
        "analyze --segment ni qo'lda yoqing; segmentatsiya o'zi yoqilmaydi."
    )


def analyze_to_disk(
    path: str | Path,
    out_dir: str | Path | None = None,
    sr: int = config.FEATURE_SR,
    clip: tuple[float, float] | None = None,
    stems: bool = False,
    tags: bool = False,
    refresh: bool = False,
    db_path: Path | None = None,
    segment: bool = False,
    min_track_s: float = config.SEG_MIN_TRACK_S,
    expect_tracks: tuple[int, int] | None = None,
    export_segments: bool = False,
    manual_starts: list[float] | None = None,
    tracklist: list[dict] | None = None,
    sample_times: list[float] | None = None,
    clip_len_s: float | None = None,
) -> dict:
    path = Path(path)
    out = Path(out_dir or config.DEFAULT_OUT_DIR)
    file_hash = store.file_sha256(path)
    params = {
        "sr": sr,
        "clip_start_s": None if clip is None else clip[0],
        "clip_end_s": None if clip is None else clip[1],
        "stems": bool(stems),
        "tags": bool(tags),
    }
    sampling = sample_times is not None
    if sampling and clip is not None:
        raise io.AudioLabError("--sample-times va --clip birga ishlamaydi.")
    if sampling and segment:
        raise io.AudioLabError("--sample-times va --segment birga ishlamaydi.")
    if sampling:
        if clip_len_s is None:
            clip_len_s = 240.0
        if clip_len_s <= 0:
            raise io.AudioLabError("--clip-len musbat bo'lishi kerak.")
        params["sample_times"] = [round(float(value), 4) for value in sample_times]
        params["clip_len_s"] = float(clip_len_s)
    if segment:
        params["segment"] = True
        params["min_track_s"] = min_track_s
        params["expect_tracks"] = (
            None if expect_tracks is None else f"{expect_tracks[0]}-{expect_tracks[1]}"
        )
    params_hash = store.params_sha256(params)
    dest = out / f"{_safe_stem(path)}_{file_hash[:8]}"
    report_path = dest / "report.json"
    if not refresh and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("file_hash") == file_hash and existing.get("params_hash") == params_hash:
            return existing
    samples = None
    if sampling:
        info = io.probe(path)
        duration_s = float(info.get("duration_s") or 0.0)
        measured = {}
        stem_result = {"status": "off"}
        tag_result = {"status": "off"}
        samples = _analyze_samples(
            path,
            duration_s=duration_s,
            starts=list(sample_times or []),
            clip_len_s=float(clip_len_s or 0.0),
            sr=sr,
            stems=stems,
            tags=tags,
        )
    else:
        measured = measure.analyze_file(path, sr=sr, clip=clip)
        measured.pop("params", None)
        stem_result = stems_module(path, clip) if stems else {"status": "off"}
        tag_result = tags_module(path, clip) if tags else {"status": "off"}
        duration_s = None
    payload = {
        "file_name": path.name,
        "file_hash": file_hash,
        "hash8": file_hash[:8],
        "params_hash": params_hash,
        "params": params,
        "versions": _versions(),
        "metrics": measured,
        "stems": stem_result,
        "tags": tag_result,
        "source_duration_s": None if duration_s is None else round(float(duration_s), 4),
    }
    if samples is not None:
        payload["samples"] = samples
    dest.mkdir(parents=True, exist_ok=True)
    segmentation = None
    if segment:
        segmentation = _segment_tracks(
            path,
            sr=sr,
            stems=stems,
            tags=tags,
            min_track_s=min_track_s,
            expect_tracks=expect_tracks,
            manual_starts=manual_starts,
            tracklist=tracklist,
            export_segments=export_segments,
            dest=dest,
        )
        payload["segments"] = segmentation
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not sampling:
        _write_csv(dest / "timeline.csv", measured, stem_result)
        _write_plots(dest / "plots", path, clip, sr, measured, stem_result)
    (dest / "report.md").write_text(_markdown(payload), encoding="utf-8")
    if not sampling:
        duration = _lookup(measured, "duration_s")
        duration_s = duration["value"] if isinstance(duration, dict) else None
    store.insert_analysis(
        file_hash,
        params_hash,
        report_path,
        duration_s,
        _summary(measured, stem_result),
        db_path=db_path,
    )
    return payload


def stems_module(path: Path, clip):
    return stems.separate(path, clip)


def tags_module(path: Path, clip):
    return tags.label(path, clip)


def parse_sample_times(text: str) -> list[float]:
    """Clock list: `5:00,15:00` is minutes:seconds. Three parts are h:mm:ss."""
    if not text or not text.strip():
        raise io.AudioLabError("--sample-times bo'sh.")
    times = []
    for piece in text.split(","):
        piece = piece.strip()
        parts = piece.split(":")
        if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
            raise io.AudioLabError(f"Vaqt formati noto'g'ri: {piece}")
        numbers = [int(part) for part in parts]
        if len(numbers) == 2:
            hours, minutes, seconds = 0, numbers[0], numbers[1]
        else:
            hours, minutes, seconds = numbers
            if minutes > 59:
                raise io.AudioLabError(f"Vaqt formati noto'g'ri: {piece}")
        if seconds > 59:
            raise io.AudioLabError(f"Vaqt formati noto'g'ri: {piece}")
        times.append(float(hours * 3600 + minutes * 60 + seconds))
    return times


def clip_windows(duration_s: float, starts: list[float], clip_len_s: float):
    """Keep clips that fit. A window that passes the file end is skipped."""
    kept = []
    skipped = []
    for start in starts:
        end = float(start) + float(clip_len_s)
        if start < 0 or end > float(duration_s) + 1e-3:
            skipped.append(
                {
                    "start_s": round(float(start), 4),
                    "end_s": round(end, 4),
                    "warning": (
                        f"Parcha o'tkazib yuborildi: {float(start):.4f} s + {float(clip_len_s):.4f} s "
                        f"fayl chegarasidan oshadi ({float(duration_s):.4f} s)."
                    ),
                }
            )
        else:
            kept.append((float(start), end))
    return kept, skipped


def _analyze_samples(path, duration_s, starts, clip_len_s, sr, stems, tags) -> dict:
    from audio_lab.segment import clip_diversity

    kept, skipped = clip_windows(duration_s, starts, clip_len_s)
    clips = []
    waves = []
    for index, (start, end) in enumerate(kept, start=1):
        window = (start, end)
        measured = measure.analyze_file(path, sr=sr, clip=window)
        measured.pop("params", None)
        clips.append(
            {
                "index": index,
                "start_s": round(start, 4),
                "end_s": round(end, 4),
                "metrics": measured,
                "stems": stems_module(path, window) if stems else {"status": "off"},
                "tags": tags_module(path, window) if tags else {"status": "off"},
            }
        )
        waves.append(io.load_mono(path, sr=config.SEG_SR, start_s=start, end_s=end))
    diversity = clip_diversity(waves, config.SEG_SR) if waves else {
        "distance": [],
        "style_groups": {"n": 0, "kind": "proxy", "clusters": [], "note": "Parcha yo'q."},
    }
    return {
        "clip_len_s": clip_len_s,
        "clips": clips,
        "skipped": skipped,
        "feature_distance": diversity["distance"],
        "style_groups": diversity["style_groups"],
    }


def write_segment_bundle(
    path: str | Path,
    out_dir: str | Path | None = None,
    min_track_s: float = config.SEG_MIN_TRACK_S,
    expect_tracks: tuple[int, int] | None = None,
    manual_starts: list[float] | None = None,
    tracklist: list[dict] | None = None,
    export_segments: bool = False,
) -> Path:
    """Boundary report only. Per-track LUFS and spectra stay on analyze --segment."""
    from audio_lab import segment as segment_mod

    path = Path(path)
    out = Path(out_dir or config.DEFAULT_OUT_DIR)
    file_hash = store.file_sha256(path)
    dest = out / f"{_safe_stem(path)}_{file_hash[:8]}"
    dest.mkdir(parents=True, exist_ok=True)
    detected = segment_mod.segment_file(
        path,
        min_track_s=min_track_s,
        expect_tracks=expect_tracks,
        manual_starts=manual_starts,
        tracklist=tracklist,
    )
    rows = [_segment_row(track) for track in detected["tracks"]]
    _write_segment_csv(dest / "segments.csv", rows)
    _write_segment_plots(dest / "plots", detected)
    if export_segments:
        segment_mod.export_segments(path, detected, dest / "segments")
    (dest / "segment.json").write_text(
        json.dumps(segment_mod.json_ready(detected), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dest


def _segment_tracks(
    path: Path,
    sr: int,
    stems: bool,
    tags: bool,
    min_track_s: float,
    expect_tracks,
    manual_starts,
    tracklist,
    export_segments: bool,
    dest: Path,
) -> dict:
    from audio_lab import segment as segment_mod

    detected = segment_mod.segment_file(
        path,
        min_track_s=min_track_s,
        expect_tracks=expect_tracks,
        manual_starts=manual_starts,
        tracklist=tracklist,
    )
    lufs_values = []
    rows = []
    for track in detected["tracks"]:
        end_s = track["end_s"] if track["end_s"] > track["start_s"] else track["start_s"] + 0.05
        clip = (float(track["start_s"]), float(end_s))
        measured = measure.analyze_file(path, sr=sr, clip=clip)
        measured.pop("params", None)
        track["metrics"] = measured
        track["stems"] = stems_module(path, clip) if stems else {"status": "off"}
        track["tags"] = tags_module(path, clip) if tags else {"status": "off"}
        lufs_values.append(_metric_value(measured, "loudness.integrated_lufs"))
        rows.append(_segment_row(track))
    detected["summary"]["neighbor_lufs_diff_db"] = _neighbor_lufs(lufs_values)
    _fill_cluster_means(detected)
    _write_segment_csv(dest / "segments.csv", rows)
    _write_segment_plots(dest / "plots", detected)
    if export_segments:
        segment_mod.export_segments(path, detected, dest / "segments")
    return segment_mod.json_ready(detected)


def _neighbor_lufs(values: list) -> dict:
    diffs = []
    for earlier, later in zip(values, values[1:]):
        if type(earlier) in (int, float) and type(later) in (int, float):
            diffs.append(abs(float(later) - float(earlier)))
    median = None if not diffs else round(float(np.median(diffs)), 4)
    return {
        "value": median,
        "unit": "dB",
        "kind": "measured",
        "note": "Qo'shni treklar integrallangan LUFS farqi medianasi.",
    }


def _fill_cluster_means(detected: dict) -> None:
    groups = detected.get("style_groups") or {}
    by_index = {track["index"]: track for track in detected["tracks"]}
    for cluster in groups.get("clusters") or []:
        members = [
            by_index[index]
            for index in cluster.get("tracks") or []
            if index in by_index and by_index[index].get("metrics")
        ]
        if not members:
            continue
        cluster["mean_metrics"] = {
            "lufs": _mean_metric(members, "loudness.integrated_lufs", "LUFS"),
            "centroid_hz": _mean_metric(members, "spectral.centroid_hz_median", "Hz"),
            "onsets_per_sec": _mean_metric(members, "onsets.onsets_per_sec", "1/s"),
        }


def _mean_metric(tracks: list[dict], path: str, unit: str) -> dict:
    values = []
    for track in tracks:
        value = _metric_value(track.get("metrics") or {}, path)
        if type(value) in (int, float):
            values.append(float(value))
    mean = None if not values else round(float(np.mean(values)), 4)
    return {
        "value": mean,
        "unit": unit,
        "kind": "proxy",
        "note": "Klasterdagi treklar o'rtachasi.",
    }


def _metric_value(node: dict, path: str):
    item = _lookup(node, path)
    if not isinstance(item, dict):
        return ""
    value = item.get("value")
    return "" if value is None else value


_SEGMENT_COLUMNS = (
    "index",
    "start_s",
    "end_s",
    "duration_s",
    "strength",
    "evidence",
    "transition_in",
    "transition_in_len_s",
    "key",
    "centroid",
    "lufs",
    "onsets_per_sec",
)


def _segment_row(track: dict) -> dict:
    metrics = track.get("metrics") or {}
    row = {
        "index": track["index"],
        "start_s": track["start_s"],
        "end_s": track["end_s"],
        "duration_s": track["duration_s"],
        "strength": track.get("strength") or "",
        "evidence": "|".join(track.get("evidence") or []),
        "transition_in": track.get("transition_in") or "",
        "transition_in_len_s": (
            "" if track.get("transition_in_len_s") is None else track["transition_in_len_s"]
        ),
        "key": _metric_value(metrics, "key.label") if metrics else "",
        "centroid": _metric_value(metrics, "spectral.centroid_hz_median") if metrics else "",
        "lufs": _metric_value(metrics, "loudness.integrated_lufs") if metrics else "",
        "onsets_per_sec": _metric_value(metrics, "onsets.onsets_per_sec") if metrics else "",
    }
    stems = track.get("stems") or {}
    if stems.get("status") == "ok":
        for name, stem in (stems.get("stems") or {}).items():
            row[f"stem_{name}_share"] = _metric_value(stem, "energy_share")
    return row


def _write_segment_csv(path: Path, rows: list[dict]) -> None:
    extra = []
    for row in rows:
        for key in row:
            if key not in _SEGMENT_COLUMNS and key not in extra:
                extra.append(key)
    fieldnames = list(_SEGMENT_COLUMNS) + extra
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {STRENGTH_NOTE}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_segment_plots(folder: Path, result: dict) -> None:
    from audio_lab.segment import feature_distance_matrix

    folder.mkdir(parents=True, exist_ok=True)
    novelty = result.get("novelty")
    fig, ax = plt.subplots(figsize=(8, 3))
    if novelty is not None and len(novelty):
        ax.plot(np.arange(len(novelty)) * config.SEG_HOP_S, np.asarray(novelty), color="#222")
    colors = {"high": "#1b7f3a", "medium": "#d98a00", "low": "#b00020"}
    for row in result.get("boundaries") or []:
        ax.axvline(row["time_s"], color=colors.get(row["strength"], "#666666"), alpha=0.85)
    ax.set_xlabel("s")
    ax.set_ylabel("novelty")
    fig.tight_layout()
    fig.savefig(folder / "novelty.png")
    plt.close(fig)

    kind_color = {"hard": "#3d5a80", "gap": "#98c1d9", "crossfade": "#ee6c4d"}
    fig, ax = plt.subplots(figsize=(8, 2.2))
    for track in result.get("tracks") or []:
        ax.barh(
            0,
            track["duration_s"],
            left=track["start_s"],
            color=kind_color.get(track.get("transition_in") or "", "#cccccc"),
            edgecolor="white",
        )
    ax.set_xlabel("s")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(folder / "transitions.png")
    plt.close(fig)

    _labels, matrix = feature_distance_matrix(result)
    fig, ax = plt.subplots(figsize=(4, 4))
    if matrix.size:
        image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=max(1.0, float(np.max(matrix))))
        fig.colorbar(image, ax=ax)
    ax.set_xlabel("trek")
    ax.set_ylabel("trek")
    fig.tight_layout()
    fig.savefig(folder / "feature_distance.png")
    plt.close(fig)


def _track_lines(segments: dict) -> list[str]:
    summary = segments.get("summary") or {}
    count = summary.get("n_tracks") or {}
    shown = count.get("value", segments.get("n_tracks"))
    lines = [
        f"## {BOUNDARY_TITLE}",
        "",
        STRENGTH_NOTE,
        "",
        f"Treklar soni: {shown}. Manba: {segments.get('source')}.",
    ]
    if segments.get("warning"):
        lines.append(segments["warning"])
    agreement = segments.get("tracklist_agreement")
    if agreement:
        lines.append(
            "Tracklist solishtirish (±5 s): "
            f"precision {agreement.get('precision')}, recall {agreement.get('recall')}."
        )
    lines.append("")
    lines.append("| index | start_s | end_s | strength | transition_in |")
    lines.append("| --- | --- | --- | --- | --- |")
    for track in segments.get("tracks") or []:
        lines.append(
            f"| {track['index']} | {track['start_s']} | {track['end_s']} | "
            f"{track.get('strength') or ''} | {track.get('transition_in') or ''} |"
        )
    groups = segments.get("style_groups") or {}
    diff = (summary.get("neighbor_lufs_diff_db") or {}).get("value")
    lines.append("")
    lines.append(f"style_groups: {groups.get('n')} [PROKSI]")
    lines.append(f"Qo'shni LUFS farqi (median): {diff}")
    return lines


def _versions() -> dict:
    import librosa
    import matplotlib as mpl
    import numpy
    import scipy

    ffmpeg = "unknown"
    try:
        proc = subprocess.run(
            [io._tool("ffmpeg"), "-version"],
            capture_output=True,
            check=False,
            text=True,
        )
        ffmpeg = proc.stdout.splitlines()[0] if proc.stdout else "unknown"
    except io.AudioLabError:
        pass
    return {
        "ffmpeg": ffmpeg,
        "librosa": librosa.__version__,
        "matplotlib": mpl.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def _write_csv(path: Path, measured: dict, stem_result: dict) -> None:
    rows = (_lookup(measured, "timelines_1s") or {}).get("value") or []
    stem_columns = {}
    if stem_result.get("status") == "ok":
        for name, stem in (stem_result.get("stems") or {}).items():
            series = (stem.get("timeline_rms_dbfs") or {}).get("value") or []
            stem_columns[name] = {int(round(item["t_s"])): item.get("rms_dbfs") for item in series}
    fieldnames = [
        "t_s",
        "lufs",
        "rms_dbfs",
        "sub_bass",
        "bass",
        "low_mid",
        "mid",
        "high_mid",
        "high",
        "centroid_hz",
        "onset_count",
        * [f"{name}_rms_dbfs" for name in stem_columns],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames if not key.endswith("_rms_dbfs") or key in row}
            second = int(row.get("t_s", 0))
            for name, series in stem_columns.items():
                out[f"{name}_rms_dbfs"] = series.get(second)
            writer.writerow(out)


def _write_plots(folder: Path, path: Path, clip, sr: int, measured: dict, stem_result: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _plot_loudness(folder / "loudness.png", measured)
    _plot_bands(folder / "bands.png", measured)
    _plot_spectrogram(folder / "spectrogram.png", path, clip, sr)
    _plot_ssm(folder / "ssm.png", measured)
    if stem_result.get("status") == "ok":
        _plot_stems(folder / "stems.png", stem_result)


def _plot_loudness(path: Path, measured: dict) -> None:
    short = (_lookup(measured, "loudness.short_term_lufs") or {}).get("value") or []
    rms = (_lookup(measured, "rms_timeline_dbfs") or {}).get("value") or []
    fig, ax = plt.subplots(figsize=(8, 3))
    if short:
        ax.plot([row["t_s"] for row in short], [row["lufs"] for row in short], label="LUFS")
    if rms:
        ax.plot([row["t_s"] for row in rms], [row["dbfs"] for row in rms], label="RMS dBFS")
    ax.set_xlabel("s")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_bands(path: Path, measured: dict) -> None:
    rows = (_lookup(measured, "bands.timeline_10s") or {}).get("value") or []
    fig, ax = plt.subplots(figsize=(8, 3))
    if rows:
        times = [row["t_s"] for row in rows]
        names = [name for name, _low, _high in config.BANDS]
        stack = np.vstack([[row.get(name) or 0 for row in rows] for name in names])
        ax.stackplot(times, stack, labels=names)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_spectrogram(path: Path, audio_path: Path, clip, sr: int) -> None:
    import librosa

    start_s, end_s = (None, None) if clip is None else clip
    y = io.load_mono(audio_path, sr=sr, start_s=start_s, end_s=end_s)
    stft = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(db, origin="lower", aspect="auto")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_ssm(path: Path, measured: dict) -> None:
    matrix = (_lookup(measured, "ssm") or {}).get("value") or []
    fig, ax = plt.subplots(figsize=(4, 4))
    if matrix:
        ax.imshow(np.asarray(matrix), origin="lower", aspect="auto")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_stems(path: Path, stem_result: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, stem in (stem_result.get("stems") or {}).items():
        series = (stem.get("timeline_rms_dbfs") or {}).get("value") or []
        if series:
            ax.plot([row["t_s"] for row in series], [row["rms_dbfs"] for row in series], label=name)
    ax.legend(loc="best", fontsize=8)
    ax.set_ylabel("dBFS")
    ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _format_metric_display(value, path: str, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and "flatness" in path and 0 < abs(value) < 0.01:
        return f"{float(value):.4e} {unit}".strip()
    return f"{value} {unit}".strip()


def _markdown(payload: dict) -> str:
    metrics = payload["metrics"]
    tempo_reliable = metrics.get("tempo", {}).get("reliable")
    lines = [f"# Audio tahlil: {payload['file_name']}", "", "| Metrika | Qiymat | Belgi |", "| --- | --- | --- |"]
    for path, label in ROWS:
        if path in ("tempo.bpm", "tempo.confidence") and tempo_reliable is False:
            continue
        item = _lookup(metrics, path)
        if not item:
            continue
        value = item.get("value")
        unit = item.get("unit") or ""
        badge = BADGE.get(item.get("kind"), "")
        if path == "tempo.bpm" and tempo_reliable is False:
            shown = "ishonchsiz"
        else:
            shown = _format_metric_display(value, path, unit)
        lines.append(f"| {label} | {shown} | {badge} |")
    if tempo_reliable is False:
        lines.append("| Tempo | ishonchsiz | [TAXMINIY] |")
    stem_result = payload.get("stems") or {}
    if stem_result.get("status") == "ok":
        for name, stem in sorted((stem_result.get("stems") or {}).items()):
            share = (stem.get("energy_share") or {}).get("value")
            lines.append(f"| Stem {name} ulushi | {share} | [TAXMINIY] |")
    else:
        lines.append(f"| Stemlar | {stem_result.get('status', 'off')} | [TAXMINIY] |")
    tag_result = payload.get("tags") or {}
    if tag_result.get("status") == "ok":
        for head in ("genre", "instrument", "mood"):
            top = tag_result.get(head) or []
            if not top:
                continue
            label = top[0]["label"]
            prob = top[0]["probability"]["value"]
            lines.append(f"| Teg {head} | {label} {prob} | [EHTIMOLLIK] |")
    else:
        status = tag_result.get("status", "off")
        reason = tag_result.get("reason")
        if status == "skipped" and reason:
            status = f"skipped ({reason})"
        lines.append(f"| Teglar | {status} | [EHTIMOLLIK] |")
    lines.append("")
    if payload.get("segments"):
        lines.extend(_track_lines(payload["segments"]))
        lines.append("")
    if payload.get("samples"):
        lines.extend(_sample_lines(payload["samples"]))
        lines.append("")
    lines.append(LIMITATIONS)
    if tag_result.get("license"):
        lines.append(tag_result["license"])
        lines.append("")
    return "\n".join(lines)


def _sample_lines(samples: dict) -> list[str]:
    from audio_lab.segment import FEATURE_DISTANCE_NOTE
    from audio_lab.summarize import table_cells, table_header, table_rule

    lines = ["## Parchalar", "", FEATURE_DISTANCE_NOTE, ""]
    for item in samples.get("skipped") or []:
        lines.append(item.get("warning") or "")
    if samples.get("skipped"):
        lines.append("")
    lines.append(table_header())
    lines.append(table_rule())
    for clip in samples.get("clips") or []:
        lines.append("| " + " | ".join(table_cells(clip)) + " |")
    lines.append("")
    lines.append("Feature masofa [PROKSI]")
    lines.append("")
    distance = samples.get("feature_distance") or []
    if distance:
        header = "| | " + " | ".join(str(clip["index"]) for clip in samples.get("clips") or []) + " |"
        rule = "| --- | " + " | ".join("---" for _ in distance) + " |"
        lines.append(header)
        lines.append(rule)
        for index, row in enumerate(distance, start=1):
            cells = " | ".join(str(value) for value in row)
            lines.append(f"| {index} | {cells} |")
        lines.append("")
    groups = samples.get("style_groups") or {}
    badge = groups.get("badge") or "PROKSI, kichik tanlama"
    threshold = groups.get("distance_threshold")
    cut = "" if threshold is None else f", dendrogram kesimi {threshold}"
    lines.append(f"style_groups: {groups.get('n')} [{badge}]{cut}")
    for cluster in groups.get("clusters") or []:
        lines.append(f"- klaster {cluster.get('id')}: {cluster.get('tracks')}")
    return lines


def _lookup(node: dict, path: str):
    current = node
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _summary(measured: dict, stem_result: dict) -> dict:
    summary = {}
    for path, _label in ROWS:
        item = _lookup(measured, path)
        if item and type(item.get("value")) in (int, float):
            summary[path] = item["value"]
    if stem_result.get("status") == "ok":
        for name, stem in (stem_result.get("stems") or {}).items():
            share = (stem.get("energy_share") or {}).get("value")
            if type(share) in (int, float):
                summary[f"stems.{name}.energy_share"] = share
    return summary


def _safe_stem(path: Path) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    return (cleaned[:80] or "audio")
