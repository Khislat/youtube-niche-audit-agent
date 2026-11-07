"""Demucs htdemucs_6s stem energy. Runs only when --stems is set.

Weights are not downloaded here. If they are missing, the stage is skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from audio_lab import config, io
from audio_lab.measure import metric

MODEL_NAME = "htdemucs_6s"
WEIGHT_SIGNATURE = "5c90dfd2"
HONESTY = (
    "Stemlar ML ajratish natijasi, aralashma (bleeding) va artefakt bor; "
    "pianino stemi ishonchsizroq; violonchel/torli/pad 'other' stemiga tushadi "
    "va alohida ajratilmaydi."
)
CHUNK_S = 20.0
OVERLAP_S = 2.0


def separate(path: Path, clip: tuple[float, float] | None = None) -> dict:
    if not _demucs_installed():
        return _skipped("Demucs o'rnatilmagan.")
    if not _weights_cached():
        return _skipped(
            "htdemucs_6s og'irliklari topilmadi. O'rnatish paytida bir marta yuklab oling "
            "(README). Tahlil paytida tarmoqqa chiqilmaydi."
        )
    try:
        return _separate(Path(path), clip)
    except Exception as exc:
        return _skipped(f"Demucs ishlamadi: {exc}")


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "reason": reason, "note": HONESTY}


def _demucs_installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("demucs") is not None
    except Exception:
        return False


def _weights_cached() -> bool:
    folders = []
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        folders.append(Path(torch_home) / "hub" / "checkpoints")
    folders.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    for folder in folders:
        if not folder.is_dir():
            continue
        for item in folder.iterdir():
            if WEIGHT_SIGNATURE in item.name:
                return True
    return False


def _separate(path: Path, clip: tuple[float, float] | None) -> dict:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(MODEL_NAME)
    model.cpu()
    model.eval()
    start_s, end_s = (None, None) if clip is None else clip
    info = io.probe(path)
    duration = info["duration_s"] or 0.0
    if start_s is not None and end_s is not None:
        duration = max(0.0, end_s - start_s)
        origin = start_s
    else:
        origin = 0.0
    sr = int(model.samplerate)
    channels = int(model.audio_channels)
    names = list(model.sources)
    totals = {name: {"sum_sq": 0.0, "samples": 0, "active": 0, "frames": 0, "timeline": []} for name in names}
    position = 0.0
    while position < duration - 1e-6:
        chunk_end = min(duration, position + CHUNK_S)
        audio = io.load_float(
            path,
            channels=channels,
            sr=sr,
            start_s=origin + position,
            end_s=origin + chunk_end,
        )
        wav = torch.from_numpy(np.ascontiguousarray(audio.T)).float()
        ref = wav.mean(0)
        wav = wav - ref.mean()
        wav = wav / (ref.std() + 1e-8)
        with torch.no_grad():
            sources = apply_model(
                model,
                wav[None],
                device="cpu",
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        sources = sources * ref.std() + ref.mean()
        keep_from, keep_to = _keep_bounds(position, chunk_end, duration, sr, sources.shape[-1])
        for name, source in zip(names, sources):
            mono = source.mean(0)[keep_from:keep_to].detach().cpu().numpy()
            _accumulate(totals[name], mono, sr, position + keep_from / sr)
        position = chunk_end - OVERLAP_S if chunk_end < duration else duration
        if chunk_end >= duration:
            break
    energies = {}
    for name, bucket in totals.items():
        energies[name] = bucket["sum_sq"]
    total_energy = sum(energies.values())
    stems = {}
    for name, bucket in totals.items():
        rms = 0.0 if bucket["samples"] == 0 else (bucket["sum_sq"] / bucket["samples"]) ** 0.5
        rms_db = None if rms <= 0 else 20.0 * float(np.log10(rms))
        share = 0.0 if total_energy <= 0 else bucket["sum_sq"] / total_energy
        active = 0.0 if bucket["frames"] == 0 else bucket["active"] / bucket["frames"]
        stems[name] = {
            "rms_db": metric(rms_db, "dBFS", "estimate", HONESTY),
            "energy_share": metric(share, "ratio", "estimate", HONESTY),
            "active_ratio": metric(
                active,
                "ratio",
                "estimate",
                HONESTY + " 1 s kadrlar ulushi, stem RMS > -50 dBFS.",
            ),
            "timeline_rms_dbfs": metric(
                bucket["timeline"],
                "dBFS",
                "estimate",
                "1 s qadamli stem RMS. " + HONESTY,
            ),
        }
    return {"status": "ok", "model": MODEL_NAME, "note": HONESTY, "stems": stems}


def _keep_bounds(position: float, chunk_end: float, duration: float, sr: int, n_samples: int):
    start = 0 if position <= 1e-6 else int(round(OVERLAP_S * sr))
    end = n_samples
    if chunk_end < duration - 1e-6:
        end = max(start, n_samples - int(round(OVERLAP_S * sr)))
    return start, min(end, n_samples)


def _accumulate(bucket: dict, mono: np.ndarray, sr: int, time_origin: float) -> None:
    if mono.size == 0:
        return
    bucket["sum_sq"] += float(np.dot(mono, mono))
    bucket["samples"] += int(mono.size)
    frame = sr
    for index, start in enumerate(range(0, mono.size - frame + 1, frame)):
        block = mono[start : start + frame]
        rms = float(np.sqrt(np.dot(block, block) / frame))
        db = None if rms <= 0 else 20.0 * float(np.log10(rms))
        bucket["frames"] += 1
        if db is not None and db > config.PAUSE_DBFS:
            bucket["active"] += 1
        bucket["timeline"].append({"t_s": round(time_origin + index, 4), "rms_dbfs": db})
