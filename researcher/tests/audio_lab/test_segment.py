"""Synthetic track-boundary checks A–K. Tracks are 36 s so min-track is 20 s here."""

from __future__ import annotations

import json
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from audio_lab import config, io
from audio_lab.chapters import InvalidTracklist, parse_tracklist
from audio_lab.profile import build_profile
from audio_lab.report import analyze_to_disk, long_file_warning
from audio_lab.segment import (
    boundary_times,
    detect_from_audio,
    match_intervals,
    read_manual_csv,
    score_boundaries,
    segment_file,
)
from tests.audio_lab import synth

SR = synth.SR
TRACK_S = 36.0
MIN_TRACK = 20.0

MELODIES = [
    [0, 4, 7, 12, 7, 4],
    [0, 3, 7, 10, 7, 3],
    [0, 5, 7, 12, 10, 5],
    [0, 2, 7, 9, 5, 2],
    [0, 4, 9, 12, 9, 4],
    [0, 7, 12, 16, 12, 7],
    [0, 3, 5, 10, 8, 3],
    [0, 5, 9, 12, 7, 2],
    [0, 4, 7, 11, 7, 0],
    [0, 2, 5, 9, 12, 5],
    [0, 6, 7, 13, 7, 6],
    [0, 1, 5, 8, 12, 8],
]
F0S = [110, 146.8, 196, 246.9, 329.6, 392, 493.9, 523.3, 587.3, 659.3, 784, 880]


def _render(seconds, f0, harmonics, melody, note_s=0.45) -> np.ndarray:
    n = int(seconds * SR)
    y = np.zeros(n)
    note_n = int(note_s * SR)
    pos = 0
    step = 0
    while pos < n:
        freq = f0 * (2 ** (melody[step % len(melody)] / 12.0))
        length = min(note_n, n - pos)
        t = np.arange(length) / SR
        tone = np.zeros(length)
        for ratio, amp in harmonics:
            tone += amp * np.sin(2 * np.pi * freq * ratio * t)
        attack = min(int(0.02 * SR), max(1, length // 4))
        env = np.ones(length)
        env[:attack] = np.linspace(0, 1, attack)
        env[-attack:] = np.linspace(1, 0, attack)
        y[pos : pos + length] += tone * env
        pos += length
        step += 1
    peak = np.max(np.abs(y)) or 1.0
    return 0.25 * y / peak


def _harmonics(index: int):
    return [
        (1, 1.0),
        (2, 0.15 + 0.07 * index),
        (3, 0.05 if index % 2 == 0 else 0.55),
        (4, 0.4 if index % 3 == 0 else 0.05),
        (5, 0.02 * (index % 5)),
    ]


def _distinct(count: int) -> list[np.ndarray]:
    return [_render(TRACK_S, F0S[i], _harmonics(i), MELODIES[i]) for i in range(count)]


def _low_medium_share(result: dict) -> float:
    rows = result["boundaries"]
    if not rows:
        return 0.0
    return sum(row["confidence"] in ("low", "medium") for row in rows) / len(rows)


def _report(name: str, score: dict) -> None:
    print(f"SEGMENT_SCORE {name} precision={score['precision']} recall={score['recall']}")


def test_a_hard_cuts_precision_recall():
    audio = np.concatenate(_distinct(12))
    truth = [TRACK_S * index for index in range(1, 12)]
    result = detect_from_audio(audio, SR, min_track_s=MIN_TRACK)
    score = score_boundaries(boundary_times(result), truth, 2.0)
    _report("A", score)
    assert score["precision"] >= 0.9
    assert score["recall"] >= 0.9
    assert all(row["transition"]["kind"] == "hard" for row in result["boundaries"])


def test_b_silence_gap_inside_the_gap():
    parts = _distinct(6)
    gap = np.zeros(int(2 * SR))
    pieces = []
    intervals = []
    cursor = 0.0
    for index, part in enumerate(parts):
        pieces.append(part)
        cursor += TRACK_S
        if index < len(parts) - 1:
            intervals.append((cursor - 1.0, cursor + 2.0 + 1.0))
            pieces.append(gap)
            cursor += 2.0
    result = detect_from_audio(np.concatenate(pieces), SR, min_track_s=MIN_TRACK)
    predicted = boundary_times(result)
    score = match_intervals(predicted, intervals)
    _report("B", score)
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    incoming = [track["transition_in"] for track in result["tracks"][1:]]
    assert incoming == ["gap"] * len(intervals)
    for row in result["boundaries"]:
        assert abs(row["transition"]["length_s"] - 2.0) <= 0.5


def test_c_crossfade_center_and_length():
    parts = _distinct(4)
    fade_n = int(4 * SR)
    depth = 10 ** (-6 / 20)
    acc = parts[0].copy()
    centers = []
    for part in parts[1:]:
        fade_out = np.sqrt(np.linspace(1, 0, fade_n))
        fade_in = np.sqrt(np.linspace(0, 1, fade_n))
        mid = (acc[-fade_n:] * fade_out + part[:fade_n] * fade_in) * depth
        acc = np.concatenate([acc[:-fade_n], mid, part[fade_n:]])
        fade_end = len(acc) - (len(part) - fade_n)
        fade_start = fade_end - fade_n
        centers.append((fade_start + fade_end) / 2 / SR)
    result = detect_from_audio(acc, SR, min_track_s=MIN_TRACK)
    score = score_boundaries(boundary_times(result), centers, 3.0)
    _report("C", score)
    assert score["precision"] >= 0.9
    assert score["recall"] >= 0.9
    for row, center in zip(result["boundaries"], centers, strict=True):
        assert abs(row["time_s"] - center) <= 3.0
        assert row["transition"]["kind"] == "crossfade"
        assert abs(row["transition"]["length_s"] - 4.0) <= 1.5
        assert result["tracks"][1]["transition_in"] == "crossfade"


def test_d_same_timbre_is_less_confident_than_hard_cuts():
    """D is the hard case: melody changes, timbre does not.

    The spec sentence says the low/medium share should be lower than A.
    That contradicts the hard case. The measured share is higher, and this
    test follows that: every boundary has a confidence, and fewer of them
    are high than in A.
    """
    hard = detect_from_audio(np.concatenate(_distinct(12)), SR, min_track_s=MIN_TRACK)
    same = [
        _render(TRACK_S, 220.0, [(1, 1.0), (2, 0.35), (3, 0.2)], MELODIES[index])
        for index in range(6)
    ]
    result = detect_from_audio(np.concatenate(same), SR, min_track_s=MIN_TRACK)
    truth = [TRACK_S * index for index in range(1, 6)]
    score = score_boundaries(boundary_times(result), truth, 2.0)
    _report("D", score)
    assert result["boundaries"]
    assert all(row["confidence"] in ("low", "medium", "high") for row in result["boundaries"])
    assert _low_medium_share(result) > _low_medium_share(hard)


def test_e_slow_evolution_has_at_most_one_false_boundary():
    motif = _render(4.0, 220.0, [(1, 1.0), (2, 0.3), (3, 0.15)], [0, 4, 7, 12])
    loops = np.tile(motif, 45)
    audio = loops * np.linspace(1.0, 1.02, loops.size)
    result = detect_from_audio(audio, SR, min_track_s=25.0)
    predicted = boundary_times(result)
    score = score_boundaries(predicted, [], 2.0)
    _report("E", score)
    assert len(predicted) <= 1
    assert all(row["confidence"] in ("low", "medium", "high") for row in result["boundaries"])


def test_f_repeated_track_is_marked(tmp_path):
    distinct = _distinct(3)
    repeated = distinct[0]
    audio = np.concatenate([repeated, distinct[1], repeated, distinct[2], repeated])
    path = tmp_path / "dup.wav"
    synth.write_wav(path, audio)
    result = segment_file(path, min_track_s=MIN_TRACK)
    truth = [TRACK_S * index for index in range(1, 5)]
    score = score_boundaries(boundary_times(result), truth, 2.0)
    _report("F", score)
    by_index = {track["index"]: track["possible_duplicate_of"] for track in result["tracks"]}
    assert by_index[1] is None
    assert by_index[3] == 1
    assert by_index[5] == 1
    assert by_index[2] is None
    assert by_index[4] is None


def test_g_manual_csv_replaces_detection(tmp_path):
    path = tmp_path / "manual.wav"
    synth.write_wav(path, synth.sine(440.0, 40.0))
    csv_path = tmp_path / "manual.csv"
    csv_path.write_text("start_s\n10\n25\n", encoding="utf-8")
    result = segment_file(path, manual_starts=read_manual_csv(csv_path))
    assert result["source"] == "manual"
    assert boundary_times(result) == [10.0, 25.0]
    assert all("manual" in row["evidence"] for row in result["boundaries"])


def test_h_tracklist_formats_and_invalid_order():
    cues = parse_tracklist("0:00 Intro\n12:34 Second\nend 1:02:03 Final\n")
    assert [row["start_s"] for row in cues] == [0.0, 12 * 60 + 34, 1 * 3600 + 2 * 60 + 3]
    assert cues[0]["name"] == "Intro"
    assert cues[2]["name"] == "end Final"
    with pytest.raises(InvalidTracklist, match="invalid_tracklist"):
        parse_tracklist("0:00 A\n1:00 B\n0:30 C\n")
    with pytest.raises(InvalidTracklist, match="invalid_tracklist"):
        parse_tracklist("1:00 A\n2:00 B\n3:00 C\n")


def test_i_expect_tracks_warns_and_does_not_force_the_count():
    result = detect_from_audio(
        np.concatenate(_distinct(4)),
        SR,
        min_track_s=MIN_TRACK,
        expect_tracks=(10, 12),
    )
    assert result["threshold_attempts"] == 2
    assert result["threshold_adjusted"] is True
    assert result["n_tracks"] == 4
    assert not (10 <= result["n_tracks"] <= 12)
    assert result["warning"]
    assert "majburlanmadi" in result["warning"]
    assert abs(result["threshold_k"] - config.SEG_K * 0.7) < 1e-6


def test_k_two_runs_match():
    audio = np.concatenate(_distinct(4))
    first = boundary_times(detect_from_audio(audio, SR, min_track_s=MIN_TRACK))
    second = boundary_times(detect_from_audio(audio, SR, min_track_s=MIN_TRACK))
    assert first == second


@pytest.mark.slow
def test_j_two_hour_file_stays_under_two_gigabytes(tmp_path):
    path = tmp_path / "two_hours.wav"
    subprocess.run(
        [
            io._tool("ffmpeg"),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=7200",
            "-ar",
            "22050",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    tracemalloc.start()
    result = segment_file(path, min_track_s=90)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 2 * 1024**3
    assert result["duration_s"] > 7000
    features = result["features"]
    assert features.ndim == 2
    assert features.shape[0] < 8000
    assert features.shape[1] < 64


def test_long_file_warning_stays_off_until_segment_is_requested():
    assert long_file_warning(20 * 60, False)
    assert "o'zi yoqilmaydi" in long_file_warning(20 * 60, False)
    assert long_file_warning(20 * 60, True) is None
    assert long_file_warning(30, False) is None


def test_analyze_segment_measures_each_track_at_requested_sr(tmp_path):
    path = tmp_path / "mix.wav"
    synth.write_wav(path, np.concatenate(_distinct(2)))
    out = tmp_path / "out"
    payload = analyze_to_disk(
        path,
        out_dir=out,
        sr=44100,
        segment=True,
        min_track_s=MIN_TRACK,
        db_path=tmp_path / "audio.db",
    )
    assert payload["params"]["sr"] == 44100
    assert payload["params"]["segment"] is True
    assert payload["segments"]["sample_rate"] == config.SEG_SR
    tracks = payload["segments"]["tracks"]
    assert len(tracks) >= 2
    for track in tracks:
        lufs = track["metrics"]["loudness"]["integrated_lufs"]
        centroid = track["metrics"]["spectral"]["centroid_hz_median"]
        assert lufs["kind"] == "measured"
        assert isinstance(centroid["value"], float)
    folder = out / f"mix_{payload['hash8']}"
    csv_text = (folder / "segments.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("index,start_s,end_s")
    assert "lufs" in csv_text.splitlines()[0]
    assert (folder / "plots" / "novelty.png").is_file()
    assert (folder / "plots" / "transitions.png").is_file()
    assert (folder / "plots" / "feature_distance.png").is_file()
    assert "Treklar" in (folder / "report.md").read_text(encoding="utf-8")
    assert "features" not in payload["segments"]


def test_segment_cli_and_export(tmp_path):
    path = tmp_path / "one.wav"
    synth.write_wav(path, synth.sine(440.0, 12.0))
    out = tmp_path / "out"
    script = Path_audio_lab()
    plain = subprocess.run(
        [sys.executable, str(script), "segment", "--input", str(path), "--out", str(out), "--min-track", "5"],
        check=True,
        capture_output=True,
        text=True,
    )
    dest = next(out.iterdir())
    assert dest.name in plain.stdout
    assert (dest / "segments.csv").is_file()
    assert (dest / "plots" / "novelty.png").is_file()
    assert not (dest / "segments").exists()
    exported = subprocess.run(
        [
            sys.executable,
            str(script),
            "segment",
            "--input",
            str(path),
            "--out",
            str(out),
            "--min-track",
            "5",
            "--export-segments",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert exported.returncode == 0
    wavs = list((dest / "segments").glob("*.wav"))
    assert wavs


def test_profile_per_track_uses_tracks(tmp_path):
    folder = tmp_path / "audio"
    folder.mkdir()
    synth.write_wav(folder / "a.wav", synth.sine(220.0, 8.0))
    synth.write_wav(folder / "b.wav", synth.sine(330.0, 8.0))
    profile = build_profile(
        folder,
        "short",
        out_dir=tmp_path / "out",
        db_path=tmp_path / "audio.db",
        per_track=True,
    )
    payload = json.loads(profile.read_text(encoding="utf-8"))
    text = profile.with_suffix(".md").read_text(encoding="utf-8")
    assert payload["n_files"] == 2
    assert payload["n_tracks"] == 2
    assert payload["n_tracks"] < 8
    assert payload["warning"]
    assert "kichik tanlama" in text
    assert payload["track_stats"]["tracks_per_video"]["median"] == 1
    assert "loudness.integrated_lufs" in payload["metrics"]


def Path_audio_lab():
    return config.ROOT / "audio_lab.py"
