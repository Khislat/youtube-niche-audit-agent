"""Synthetic checks for level, bands, flatness, loudness, and pauses."""

from pathlib import Path

from audio_lab.measure import analyze_file
from tests.audio_lab import synth

KINDS = {"measured", "estimate", "proxy", "probabilistic"}


def _write(tmp_path: Path, name: str, y) -> Path:
    path = tmp_path / name
    synth.write_wav(path, y)
    return path


def _walk_metrics(node):
    if isinstance(node, dict) and {"value", "unit", "kind", "note"} <= set(node):
        yield node
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_metrics(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_metrics(value)


def test_sine_centroid_onsets_and_unreliable_tempo(tmp_path):
    report = analyze_file(_write(tmp_path, "sine440.wav", synth.sine(440.0, 10.0)))
    centroid = report["spectral"]["centroid_hz_median"]["value"]
    assert abs(centroid - 440.0) / 440.0 <= 0.05
    assert report["onsets"]["onsets_per_sec"]["value"] < 0.3
    assert report["tempo"]["reliable"] is False
    for item in _walk_metrics(report):
        assert item["kind"] in KINDS


def test_band_shares_follow_the_sine(tmp_path):
    bass = analyze_file(_write(tmp_path, "b100.wav", synth.sine(100.0, 8.0)))
    high_mid = analyze_file(_write(tmp_path, "h4000.wav", synth.sine(4000.0, 8.0)))
    assert bass["bands"]["share_pct"]["bass"]["value"] > 90.0
    assert high_mid["bands"]["share_pct"]["high_mid"]["value"] > 90.0


def test_flatness_noise_versus_sine(tmp_path):
    noise = analyze_file(_write(tmp_path, "noise.wav", synth.white_noise(8.0)))
    tone = analyze_file(_write(tmp_path, "tone.wav", synth.sine(440.0, 8.0)))
    assert noise["spectral"]["flatness_median"]["value"] > 0.5
    assert tone["spectral"]["flatness_median"]["value"] < 0.05


def test_lufs_tracks_a_10db_level_change(tmp_path):
    quiet = analyze_file(_write(tmp_path, "m20.wav", synth.sine(440.0, 10.0, rms_dbfs=-20.0)))
    loud = analyze_file(_write(tmp_path, "m10.wav", synth.sine(440.0, 10.0, rms_dbfs=-10.0)))
    delta = loud["loudness"]["integrated_lufs"]["value"] - quiet["loudness"]["integrated_lufs"]["value"]
    assert abs(delta - 10.0) <= 0.5


def test_pause_ratio_and_longest_gap(tmp_path):
    report = analyze_file(_write(tmp_path, "gap.wav", synth.tone_silence_tone()))
    assert abs(report["pause"]["pause_ratio"]["value"] - 0.2) <= 0.03
    assert abs(report["pause"]["longest_pause_s"]["value"] - 5.0) <= 0.3
