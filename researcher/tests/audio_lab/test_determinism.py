"""The same file and parameters yield the same rounded numbers."""

from audio_lab.measure import analyze_file
from tests.audio_lab import synth


def test_two_passes_match_at_four_decimals(tmp_path):
    path = tmp_path / "once.wav"
    synth.write_wav(path, synth.synthetic_music(duration_s=8.0))
    first = analyze_file(path)
    second = analyze_file(path)
    assert first == second
