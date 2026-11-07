"""Long-term and quiet spectra of several files, drawn on one figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_lab import config, io
from audio_lab.measure import ltas_and_quiet

COMPARE_SR = config.SEG_CLARITY_SR


def compare_spectra(
    paths: list[str | Path],
    out: str | Path,
    clip: tuple[float, float] | None = None,
) -> Path:
    if len(paths) < 1:
        raise io.AudioLabError("Kamida bitta audio fayl kerak.")
    start_s = None if clip is None else clip[0]
    end_s = None if clip is None else clip[1]
    if clip is not None and end_s <= start_s:
        raise io.AudioLabError("--clip oxiri boshidan keyin bo'lishi kerak.")
    series = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise io.AudioLabError(f"Fayl topilmadi: {path}")
        info = io.probe(path)
        duration = info.get("duration_s") or 0.0
        if clip is None and duration > config.SEG_LONG_FILE_S:
            raise io.AudioLabError(
                f"{path.name} 15 daqiqadan uzun. --clip START END bilan taxminan 240 s bo'lak bering."
            )
        y = io.load_mono(path, COMPARE_SR, start_s=start_s, end_s=end_s)
        ltas, quiet = ltas_and_quiet(y, COMPARE_SR)
        series.append((path.name, ltas, quiet))
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _draw(series, dest, clip)
    return dest


def _draw(series: list[tuple[str, list, list]], dest: Path, clip: tuple[float, float] | None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, ltas, quiet in series:
        axes[0].plot(_hz(ltas), _db(ltas), label=name)
        axes[1].plot(_hz(quiet), _db(quiet), label=name)
    span = "butun fayl" if clip is None else f"{clip[0]:.0f}–{clip[1]:.0f} s"
    axes[0].set_title(f"LTAS, 1/3-oktava ({span}, {COMPARE_SR} Hz)")
    axes[1].set_title("Sokin 10% kadrlar spektri")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_ylabel("dB")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    axes[1].set_xlabel("Hz")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def _hz(rows: list[dict]) -> list[float]:
    return [float(row["center_hz"]) for row in rows if row.get("db") is not None]


def _db(rows: list[dict]) -> list[float]:
    return [float(row["db"]) for row in rows if row.get("db") is not None]
