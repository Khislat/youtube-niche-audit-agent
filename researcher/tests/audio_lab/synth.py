"""Synthetic audio built with numpy. Fixed seeds keep the tests deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050

# One octave of equal temperament starting at C4.
SCALE_HZ = np.array(
    [261.6256, 293.6648, 329.6276, 349.2282, 391.9954, 440.0, 493.8833, 523.2511],
    dtype=np.float64,
)


def sine(freq_hz: float, duration_s: float, sr: int = SR, rms_dbfs: float = -12.0) -> np.ndarray:
    """Sine whose RMS level is `rms_dbfs` (full scale = amplitude 1)."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    rms = 10 ** (rms_dbfs / 20.0)
    amplitude = rms * np.sqrt(2.0)
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def white_noise(duration_s: float, sr: int = SR, rms_dbfs: float = -12.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(int(round(duration_s * sr)))
    rms = math_rms(y)
    target = 10 ** (rms_dbfs / 20.0)
    if rms > 0:
        y = y * (target / rms)
    return y


def tone_silence_tone(sr: int = SR) -> np.ndarray:
    """10 s tone, 5 s digital silence, 10 s tone."""
    tone = sine(440.0, 10.0, sr=sr, rms_dbfs=-12.0)
    gap = np.zeros(int(round(5.0 * sr)), dtype=np.float64)
    return np.concatenate([tone, gap, tone])


def clicks(bpm: float, duration_s: float, sr: int = SR) -> np.ndarray:
    n = int(round(duration_s * sr))
    y = np.zeros(n, dtype=np.float64)
    click_n = int(round(0.01 * sr))
    decay = np.exp(-np.linspace(0.0, 8.0, click_n))
    burst = decay * np.sin(2.0 * np.pi * 2000.0 * np.arange(click_n) / sr)
    peak = np.max(np.abs(burst))
    if peak > 0:
        burst = 0.8 * burst / peak
    step = 60.0 / bpm
    t = 0.05
    while t < duration_s - 0.02:
        i = int(round(t * sr))
        end = min(n, i + click_n)
        y[i:end] += burst[: end - i]
        t += step
    return y


def phrase(rng: np.random.Generator, sr: int = SR, n_notes: int = 8, note_s: float = 0.5) -> np.ndarray:
    parts = []
    for _ in range(n_notes):
        freq = float(rng.choice(SCALE_HZ))
        n = int(round(note_s * sr))
        t = np.arange(n, dtype=np.float64) / sr
        env = np.ones(n, dtype=np.float64)
        attack = int(0.02 * sr)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[-attack:] = np.linspace(1.0, 0.0, attack)
        parts.append(0.3 * env * np.sin(2.0 * np.pi * freq * t))
    return np.concatenate(parts)


def repeated_phrase(repeats: int = 6, sr: int = SR, seed: int = 7) -> np.ndarray:
    one = phrase(np.random.default_rng(seed), sr=sr)
    return np.tile(one, repeats)


def random_melody(phrases: int = 6, sr: int = SR, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([phrase(rng, sr=sr) for _ in range(phrases)])


def arpeggio(freqs: tuple[float, ...], sr: int = SR, note_s: float = 0.5, cycles: int = 8) -> np.ndarray:
    parts = []
    for _ in range(cycles):
        for freq in freqs:
            n = int(round(note_s * sr))
            t = np.arange(n, dtype=np.float64) / sr
            env = np.ones(n, dtype=np.float64)
            attack = max(1, int(0.01 * sr))
            env[:attack] = np.linspace(0.0, 1.0, attack)
            env[-attack:] = np.linspace(1.0, 0.0, attack)
            parts.append(0.35 * env * np.sin(2.0 * np.pi * freq * t))
    return np.concatenate(parts)


def synthetic_music(duration_s: float = 16.0, sr: int = SR, seed: int = 3) -> np.ndarray:
    """A few tones, including energy above 3 kHz, plus quiet noise."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    y = (
        0.22 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.16 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.12 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 4000.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 6000.0 * t)
        + 0.06 * np.sin(2.0 * np.pi * 8000.0 * t)
    )
    rng = np.random.default_rng(seed)
    y = y + 0.004 * rng.standard_normal(n)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.8 * y / peak
    return y


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, dtype=np.float64), sr, subtype="FLOAT")


def math_rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y * y)))
