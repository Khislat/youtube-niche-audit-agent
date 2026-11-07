"""Tempo, repetition, and key on synthetic material."""

from audio_lab.measure import analyze_file
from tests.audio_lab import synth


def _report(tmp_path, name, y):
    path = tmp_path / name
    synth.write_wav(path, y)
    return analyze_file(path)


def test_click_tempo_and_onset_rate(tmp_path):
    for bpm in (60.0, 90.0):
        report = _report(tmp_path, f"click_{int(bpm)}.wav", synth.clicks(bpm, 20.0))
        assert abs(report["tempo"]["bpm"]["value"] - bpm) <= 2.0
        expected = bpm / 60.0
        got = report["onsets"]["onsets_per_sec"]["value"]
        assert abs(got - expected) / expected <= 0.10


def test_repeated_phrase_scores_higher_and_lag_is_4s(tmp_path):
    looped = _report(tmp_path, "loop.wav", synth.repeated_phrase())
    fresh = _report(tmp_path, "fresh.wav", synth.random_melody())
    assert looped["repetition"]["repetition_score"]["value"] > fresh["repetition"]["repetition_score"]["value"]
    assert abs(looped["repetition"]["top_lag_s"]["value"] - 4.0) <= 0.2


def test_major_and_minor_arpeggios(tmp_path):
    c_major = _report(
        tmp_path,
        "cmaj.wav",
        synth.arpeggio((261.6256, 329.6276, 391.9954, 523.2511)),
    )
    a_minor = _report(
        tmp_path,
        "amin.wav",
        synth.arpeggio((220.0, 261.6256, 329.6276, 440.0)),
    )
    assert c_major["key"]["label"]["value"] == "C major"
    assert a_minor["key"]["label"]["value"] == "A minor"
    assert c_major["key"]["margin"]["value"] > 0
    assert a_minor["key"]["margin"]["value"] > 0
