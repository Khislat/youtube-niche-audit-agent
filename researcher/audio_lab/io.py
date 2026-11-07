"""Load local audio with ffmpeg. No network and no remote URLs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from audio_lab import config


class AudioLabError(RuntimeError):
    """A local-file or ffmpeg problem the user can act on."""


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for folder in ("/opt/homebrew/bin", "/usr/local/bin"):
        candidate = Path(folder) / name
        if candidate.is_file():
            return str(candidate)
    raise AudioLabError(
        "ffmpeg topilmadi. Audio Audit Lab uchun ffmpeg o'rnatilgan bo'lishi kerak."
    )


def ensure_ffmpeg() -> None:
    _tool("ffmpeg")
    _tool("ffprobe")


def probe(path: Path) -> dict:
    """Return original sample rate, channel count, and duration in seconds."""
    ensure_ffmpeg()
    path = Path(path)
    if not path.is_file():
        raise AudioLabError(f"Fayl topilmadi: {path}")
    cmd = [
        _tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLabError(f"ffprobe o'qiy olmadi: {path.name}. {err}")
    try:
        payload = json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AudioLabError(f"ffprobe javobi JSON emas: {path.name}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise AudioLabError(f"Audio oqim yo'q: {path.name}")
    stream = streams[0]
    duration_raw = (payload.get("format") or {}).get("duration")
    duration = float(duration_raw) if duration_raw not in (None, "N/A") else None
    return {
        "sample_rate_in": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration_s": duration,
    }


def _input_args(path: Path, start_s: float | None, end_s: float | None) -> list[str]:
    args = ["-nostdin", "-hide_banner"]
    if start_s is not None:
        args += ["-ss", f"{start_s:.6f}"]
    args += ["-i", str(path)]
    if start_s is not None and end_s is not None:
        args += ["-t", f"{max(0.0, end_s - start_s):.6f}"]
    elif end_s is not None:
        args += ["-t", f"{end_s:.6f}"]
    return args


def load_float(
    path: Path,
    channels: int,
    sr: int,
    start_s: float | None = None,
    end_s: float | None = None,
) -> np.ndarray:
    """Decode a clip to float64 with shape (samples, channels)."""
    ensure_ffmpeg()
    path = Path(path)
    cmd = [_tool("ffmpeg"), "-v", "error", *_input_args(path, start_s, end_s)]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLabError(f"Audiyni ochib bo'lmadi: {path.name}. {err}")
    frame = channels * 4
    usable = len(proc.stdout) - (len(proc.stdout) % frame)
    if usable == 0:
        raise AudioLabError(f"Audio bo'sh: {path.name}")
    audio = np.frombuffer(proc.stdout[:usable], dtype="<f4").astype(np.float64, copy=False)
    return audio.reshape(-1, channels)


def load_mono(
    path: Path,
    sr: int = config.FEATURE_SR,
    start_s: float | None = None,
    end_s: float | None = None,
) -> np.ndarray:
    """Decode a clip to mono float32 at `sr` for feature extraction."""
    ensure_ffmpeg()
    path = Path(path)
    cmd = [_tool("ffmpeg"), "-v", "error", *_input_args(path, start_s, end_s)]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLabError(f"Audiyni ochib bo'lmadi: {path.name}. {err}")
    if not proc.stdout:
        raise AudioLabError(f"Audio bo'sh: {path.name}")
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64, copy=False)
    if audio.size == 0:
        raise AudioLabError(f"Audio bo'sh: {path.name}")
    return audio


def iter_stereo(
    path: Path,
    start_s: float | None = None,
    end_s: float | None = None,
    chunk_frames: int = 22050 * 10,
):
    """Stream original-rate stereo (or mono duplicated by ffmpeg -ac 2) in chunks.

    The caller passes the real channel count separately. This decoder is only
    used when the file has two or more channels.
    """
    ensure_ffmpeg()
    cmd = [_tool("ffmpeg"), "-v", "error", *_input_args(Path(path), start_s, end_s)]
    cmd += ["-ac", "2", "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = 2 * 4
    try:
        while True:
            raw = proc.stdout.read(chunk_frames * frame_bytes)
            if not raw:
                break
            usable = len(raw) - (len(raw) % frame_bytes)
            if usable == 0:
                continue
            block = np.frombuffer(raw[:usable], dtype="<f4").astype(np.float64, copy=False)
            yield block.reshape(-1, 2)
    finally:
        if proc.stdout:
            proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise AudioLabError(f"Stereo oqimni o'qib bo'lmadi. {err.strip()}")


def measure_ebur128(
    path: Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict:
    """Integrated LUFS, LRA, true peak, and short-term loudness from ffmpeg.

    The filter runs inside ffmpeg, so a one-hour file is not loaded as one array.
    """
    ensure_ffmpeg()
    cmd = [_tool("ffmpeg"), "-v", "info", *_input_args(Path(path), start_s, end_s)]
    cmd += ["-af", "ebur128=framelog=verbose:peak=true", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    text = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise AudioLabError(
            f"ffmpeg ebur128 xato: {Path(path).name}. {text[-500:].strip()}"
        )
    parsed = _parse_ebur128(text)
    if parsed["integrated_lufs"] is None:
        raise AudioLabError(f"ffmpeg ebur128 natija bermadi: {Path(path).name}")
    return parsed


def _parse_ebur128(text: str) -> dict:
    summary_at = text.rfind("Integrated loudness:")
    frame_text = text if summary_at < 0 else text[:summary_at]
    summary = text[summary_at:] if summary_at >= 0 else ""
    short_term: list[tuple[float, float]] = []
    for line in frame_text.splitlines():
        if " S:" not in line or "t:" not in line:
            continue
        # t: 1.0 ... S: -20.1 ...
        try:
            t_token = line.split("t:", 1)[1].split()[0]
            s_token = line.split("S:", 1)[1].split()[0]
            t_s = float(t_token)
            s_lufs = float(s_token)
        except (IndexError, ValueError):
            continue
        short_term.append((t_s, s_lufs))
    return {
        "integrated_lufs": _summary_float(summary, "I:", "LUFS"),
        "lra_lu": _summary_float(summary, "LRA:", "LU"),
        "true_peak_dbtp": _summary_float(summary, "Peak:", "dBFS"),
        "short_term": short_term,
    }


def _summary_float(summary: str, label: str, unit: str) -> float | None:
    for line in summary.splitlines():
        stripped = line.strip()
        if not stripped.startswith(label):
            continue
        if unit not in stripped:
            continue
        token = stripped.split()[1]
        try:
            return float(token)
        except ValueError:
            return None
    return None
