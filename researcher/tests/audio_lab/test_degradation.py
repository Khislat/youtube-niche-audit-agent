"""Low-pass and added noise move the spectral estimates in the expected direction."""

import subprocess
from pathlib import Path

from audio_lab.measure import analyze_file
from tests.audio_lab import synth


def _ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def test_lowpass_and_noise_move_the_estimates(tmp_path: Path):
    original = tmp_path / "music.wav"
    lowpassed = tmp_path / "lowpass.wav"
    noisy = tmp_path / "noisy.wav"
    synth.write_wav(original, synth.synthetic_music())
    # Steep 3 kHz low-pass. ffmpeg's default 2-pole does not fall 60 dB
    # before Nyquist, so the -60 dB cutoff estimate cannot land near 3 kHz.
    _ffmpeg(
        [
            "-i",
            str(original),
            "-af",
            "firequalizer=gain='if(lt(f,3000),0,-80)':delay=0.1",
            str(lowpassed),
        ]
    )
    _ffmpeg(
        [
            "-i",
            str(original),
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=white:sample_rate=22050:amplitude=0.08:duration=16",
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:normalize=0",
            str(noisy),
        ]
    )
    base = analyze_file(original)
    darker = analyze_file(lowpassed)
    harsher = analyze_file(noisy)
    assert darker["spectral"]["centroid_hz_median"]["value"] < base["spectral"]["centroid_hz_median"]["value"]
    assert darker["spectral"]["rolloff85_hz"]["value"] < base["spectral"]["rolloff85_hz"]["value"]
    cutoff = darker["spectral"]["lowpass_cutoff_hz"]["value"]
    assert 2000.0 <= cutoff <= 4500.0
    assert harsher["noise_floor_dbfs"]["value"] > base["noise_floor_dbfs"]["value"]
    assert harsher["spectral"]["flatness_median"]["value"] > base["spectral"]["flatness_median"]["value"]
