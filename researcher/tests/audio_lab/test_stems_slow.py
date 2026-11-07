"""Slow Demucs check. Skipped unless CC0 fixtures and Demucs weights are present."""

from pathlib import Path

import pytest

from audio_lab.stems import separate

DATA = Path(__file__).parent / "data"
PIANO = DATA / "piano.wav"
DRUMS = DATA / "drums.wav"


@pytest.mark.slow
def test_piano_and_drum_stem_shares():
    if not PIANO.is_file() or not DRUMS.is_file():
        pytest.skip("CC0 piano.wav va drums.wav yo'q")
    piano = separate(PIANO)
    drums = separate(DRUMS)
    if piano.get("status") == "skipped" or drums.get("status") == "skipped":
        pytest.skip(piano.get("reason") or drums.get("reason") or "Demucs skipped")
    piano_share = piano["stems"]["drums"]["energy_share"]["value"]
    drum_share = drums["stems"]["drums"]["energy_share"]["value"]
    assert piano_share < 0.05
    assert drum_share > 0.5
