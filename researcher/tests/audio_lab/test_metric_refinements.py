"""Regression tests for noise floor, lowpass, repetition lag, dynamics, report tempo."""

from pathlib import Path

import numpy as np

from audio_lab import config, io
from audio_lab.measure import _dynamic_range_metric, _noise_floor_metric, analyze_file
from audio_lab.report import _format_metric_display, _markdown
from tests.audio_lab import synth


def test_noise_floor_metric_is_proxy_not_measured_shovqin():
    sr = config.FEATURE_SR
    y = synth.sine(440.0, 4.0, sr=sr, rms_dbfs=-30.0)
    from audio_lab.measure import _frame_dbfs

    result = _noise_floor_metric(_frame_dbfs(y, sr, config.FRAME_S))
    assert result["kind"] == "proxy"
    assert "haqiqiy akustik shovqin" in result["note"]
    assert "Sokin 10% RMS" in result["note"]


def test_noise_floor_none_when_only_digital_silence():
    sr = config.FEATURE_SR
    y = np.zeros(int(10.0 * sr), dtype=np.float64)
    from audio_lab.measure import _frame_dbfs

    frame_db = _frame_dbfs(y, sr, config.FRAME_S)
    result = _noise_floor_metric(frame_db)
    assert result["value"] is None
    assert "Raqamli sukunat" in result["note"]


def test_lowpass_cutoff_not_at_nyquist_for_sine(tmp_path):
    path = tmp_path / "tone.wav"
    y = synth.sine(440.0, 4.0)
    synth.write_wav(path, y)
    report = analyze_file(path)
    cutoff = report["spectral"]["lowpass_cutoff_hz"]["value"]
    assert cutoff is None or cutoff < config.FEATURE_SR / 2.0 - 100


def test_repetition_lag_at_search_floor_is_none(tmp_path):
    sr = config.FEATURE_SR
    duration_s = 120.0
    chunk = synth.sine(330.0, 4.0, sr=sr)
    parts = []
    while sum(len(p) for p in parts) / sr < duration_s:
        parts.append(chunk)
    y = np.concatenate(parts)[: int(duration_s * sr)]
    path = tmp_path / "long.wav"
    synth.write_wav(path, y)
    report = analyze_file(path)
    top = report["repetition"]["top_lag_s"]["value"]
    assert top is None or top > config.REP_MIN_LAG_S + 1e-3


def test_dynamic_range_uses_ebur128_short_term_when_available():
    short = [{"t_s": 3.0, "lufs": -20.0}, {"t_s": 4.0, "lufs": -10.0}] * 5
    rms = [{"t_s": 0.0, "dbfs": -30.0}, {"t_s": 1.0, "dbfs": -10.0}]
    metric = _dynamic_range_metric(short, rms)
    assert metric["value"] is not None
    assert metric["value"] == 10.0
    assert "ebur128" in metric["note"]


def test_report_hides_unreliable_tempo_number():
    payload = {
        "file_name": "x.wav",
        "metrics": {
            "tempo": {"reliable": False, "bpm": {"value": 120.0, "unit": "BPM", "kind": "estimate", "note": ""}},
            "duration_s": {"value": 10.0, "unit": "s", "kind": "measured", "note": ""},
        },
        "stems": {"status": "off"},
        "tags": {"status": "off"},
    }
    md = _markdown(payload)
    assert "120.0 BPM" not in md
    assert "| Tempo | ishonchsiz |" in md


def test_flatness_scientific_display():
    shown = _format_metric_display(3.2e-05, "spectral.flatness_median", "0-1")
    assert "e" in shown or "3.2" in shown


def test_ebur128_parser_reads_short_term():
    sample = (
        "[Parsed_ebur128_0 @ 0x0] t: 3.099977   TARGET:-23 LUFS    M: -44.0 S:-11.5     I: -48.0 LUFS\n"
        "Integrated loudness:\n    I:         -11.3 LUFS\n"
    )
    parsed = io._parse_ebur128(sample)
    assert len(parsed["short_term"]) == 1
    assert parsed["short_term"][0] == (3.099977, -11.5)
