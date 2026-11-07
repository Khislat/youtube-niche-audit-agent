"""Added noise raises the quiet-frame floor and the 2–8 kHz flatness."""

from pathlib import Path

from audio_lab.measure import analyze_file
from audio_lab.spectra import compare_spectra
from tests.audio_lab import synth


def test_added_noise_raises_flatness_and_floor(tmp_path: Path):
    clean_path = tmp_path / "clean.wav"
    noisy_path = tmp_path / "noisy.wav"
    clean = synth.sine(3000.0, 3.0, rms_dbfs=-12.0)
    noise = synth.white_noise(3.0, rms_dbfs=-18.0, seed=1)
    synth.write_wav(clean_path, clean)
    synth.write_wav(noisy_path, clean + noise)

    clean_art = analyze_file(clean_path, sr=synth.SR)["artifacts"]
    noisy_art = analyze_file(noisy_path, sr=synth.SR)["artifacts"]

    assert clean_art["noise_floor_db"]["kind"] == "measured"
    assert clean_art["flatness_2_8khz"]["kind"] == "measured"
    assert clean_art["hpss_harmonic_share"]["kind"] == "proxy"
    assert clean_art["flux_mean"]["kind"] == "proxy"
    assert noisy_art["flatness_2_8khz"]["value"] > clean_art["flatness_2_8khz"]["value"]
    assert noisy_art["noise_floor_db"]["value"] > clean_art["noise_floor_db"]["value"]
    assert clean_art["quiet_spectrum"]["kind"] == "measured"
    assert any(row["center_hz"] for row in clean_art["quiet_spectrum"]["value"])


def test_compare_spectra_writes_two_curves(tmp_path: Path):
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    synth.write_wav(first, synth.sine(440.0, 2.0))
    synth.write_wav(second, synth.sine(880.0, 2.0))
    dest = compare_spectra([first, second], tmp_path / "spectra.png", clip=(0.0, 2.0))
    assert dest.is_file()
    assert dest.stat().st_size > 1000
