"""Find track boundaries without storing a full self-similarity matrix.

Novelty uses a checkerboard kernel whose support is a diagonal band
(half-width 20, 30 and 60 s). A two-hour file stays a 1 s feature sequence.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import librosa
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from audio_lab import config, io


def detect_from_audio(
    y: np.ndarray,
    sr: int = config.SEG_SR,
    *,
    min_track_s: float = config.SEG_MIN_TRACK_S,
    expect_tracks: tuple[int, int] | None = None,
) -> dict:
    features, silence_runs, level = _signal_features(y, sr)
    return _decide(
        features,
        silence_runs,
        level,
        duration_s=float(len(y) / sr) if sr else 0.0,
        min_track_s=min_track_s,
        expect_tracks=expect_tracks,
    )


def segment_file(
    path: str | Path,
    *,
    min_track_s: float = config.SEG_MIN_TRACK_S,
    expect_tracks: tuple[int, int] | None = None,
    manual_starts: list[float] | None = None,
    tracklist: list[dict] | None = None,
    chunk_s: float = config.SEG_CHUNK_S,
) -> dict:
    """Segment a local file. Manual cues replace detection. A tracklist is the
    cut list and is also scored against the automatic boundaries (±5 s).
    """
    path = Path(path)
    info = io.probe(path)
    duration = float(info["duration_s"] or 0.0)
    auto = None
    if manual_starts is None:
        auto = _detect_file(path, duration, min_track_s, expect_tracks, chunk_s)
    if manual_starts is not None:
        result = _cuts_from_starts(manual_starts, duration, source="manual", evidence=["manual"])
    elif tracklist is not None:
        result = _cuts_from_starts(
            [row["start_s"] for row in tracklist],
            duration,
            source="tracklist",
            evidence=["tracklist"],
            names=[row.get("name") or "" for row in tracklist],
        )
        if auto is not None:
            score = score_boundaries(
                [row["time_s"] for row in auto["boundaries"]],
                [row["start_s"] for row in tracklist][1:],
                tolerance_s=config.SEG_TRACKLIST_TOL_S,
            )
            result["tracklist_agreement"] = score
            result["auto_boundaries"] = auto["boundaries"]
    else:
        result = auto
    result["duration_s"] = round(duration, 4)
    result["sample_rate"] = config.SEG_SR
    _mark_duplicates(result)
    result["style_groups"] = _style_groups(result)
    result["summary"] = _summary(result)
    return result


def score_boundaries(predicted: list[float], truth: list[float], tolerance_s: float) -> dict:
    """Greedy one-to-one match. Times are in seconds."""
    used: set[int] = set()
    matched = 0
    for target in truth:
        for index, guess in enumerate(predicted):
            if index in used:
                continue
            if abs(guess - target) <= tolerance_s:
                used.add(index)
                matched += 1
                break
    precision = 1.0 if not predicted else matched / len(predicted)
    if not truth:
        recall = 1.0 if not predicted else 0.0
    else:
        recall = matched / len(truth)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matched": matched,
        "predicted": len(predicted),
        "truth": len(truth),
        "kind": "measured",
    }


def match_intervals(predicted: list[float], intervals: list[tuple[float, float]]) -> dict:
    """A prediction matches when it falls inside an interval. Each interval once."""
    used: set[int] = set()
    matched = 0
    for start, end in intervals:
        for index, guess in enumerate(predicted):
            if index in used:
                continue
            if start <= guess <= end:
                used.add(index)
                matched += 1
                break
    precision = 1.0 if not predicted else matched / len(predicted)
    recall = 1.0 if not intervals else matched / len(intervals)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matched": matched,
        "predicted": len(predicted),
        "truth": len(intervals),
    }


def _detect_file(path, duration, min_track_s, expect_tracks, chunk_s) -> dict:
    overlap = config.SEG_CHUNK_OVERLAP_S
    blocks = []
    position = 0.0
    while position < duration - 1e-6:
        end = min(duration, position + chunk_s)
        audio = io.load_mono(path, sr=config.SEG_SR, start_s=position, end_s=end)
        features, silence_runs, level = _signal_features(audio, config.SEG_SR)
        blocks.append((position, features, silence_runs, level))
        if end >= duration - 1e-3:
            break
        position = end - overlap
    features, silence_runs, level = _stitch(blocks, overlap)
    return _decide(
        features,
        silence_runs,
        level,
        duration_s=duration,
        min_track_s=min_track_s,
        expect_tracks=expect_tracks,
    )


def _stitch(blocks, overlap_s: float):
    if not blocks:
        empty = np.zeros((0, 1))
        return empty, [], np.zeros(0)
    features = [blocks[0][1]]
    silence = list(blocks[0][2])
    level = [blocks[0][3]]
    drop = int(round(overlap_s / config.SEG_HOP_S))
    for position, feat, runs, lev in blocks[1:]:
        if feat.shape[0] > drop:
            features.append(feat[drop:])
        if lev.size > drop:
            level.append(lev[drop:])
        for start, end in runs:
            silence.append((start + position, end + position))
    merged = _merge_runs(silence)
    feat = np.vstack(features) if features else np.zeros((0, 1))
    lev = np.concatenate(level) if level else np.zeros(0)
    return feat, merged, lev


def _merge_runs(runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not runs:
        return []
    ordered = sorted(runs)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _signal_features(y: np.ndarray, sr: int):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    features = _frame_features(y, sr)
    silence_runs = _silence_runs(y, sr)
    level = _short_term_db(y, sr, n_frames=features.shape[0])
    return features, silence_runs, level


def _frame_features(y: np.ndarray, sr: int) -> np.ndarray:
    if len(y) < int(config.SEG_WINDOW_S * sr):
        return np.zeros((0, 33))
    hop = config.HOP_LENGTH
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=config.N_FFT, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=config.N_FFT, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=config.N_FFT, hop_length=hop)
    width = min(chroma.shape[1], mfcc.shape[1], contrast.shape[1], rms.shape[1])
    stack = np.vstack([chroma[:, :width], mfcc[:, :width], contrast[:, :width], rms[:, :width]])
    times = librosa.frames_to_time(np.arange(width), sr=sr, hop_length=hop)
    duration = len(y) / sr
    rows = []
    start = 0.0
    while start + config.SEG_WINDOW_S <= duration + 1e-6:
        mask = (times >= start) & (times < start + config.SEG_WINDOW_S)
        if np.any(mask):
            rows.append(stack[:, mask].mean(axis=1))
        start += config.SEG_HOP_S
    if not rows:
        return np.zeros((0, stack.shape[0]))
    return np.vstack(rows)


def _silence_runs(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    frame = max(1, int(round(config.FRAME_S * sr)))
    count = len(y) // frame
    if count == 0:
        return []
    shaped = y[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(shaped * shaped, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    quiet = db < config.PAUSE_DBFS
    runs = []
    start = None
    for index, flag in enumerate(quiet):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            _keep_silence(runs, start, index)
            start = None
    if start is not None:
        _keep_silence(runs, start, count)
    return runs


def _keep_silence(runs, start: int, end: int) -> None:
    length = (end - start) * config.FRAME_S
    if length > config.SEG_SILENCE_MIN_S:
        runs.append((start * config.FRAME_S, end * config.FRAME_S))


def _short_term_db(y: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    window = int(round(config.SEG_LEVEL_WINDOW_S * sr))
    hop = int(round(config.SEG_HOP_S * sr))
    if window <= 0 or len(y) < window or n_frames == 0:
        return np.zeros(n_frames)
    values = []
    for start in range(0, len(y) - window + 1, hop):
        block = y[start : start + window]
        rms = float(np.sqrt(np.dot(block, block) / window))
        values.append(-120.0 if rms <= 0 else 20.0 * np.log10(rms))
        if len(values) >= n_frames:
            break
    level = np.full(n_frames, -120.0)
    level[: len(values)] = values
    return level


def _decide(features, silence_runs, level, duration_s, min_track_s, expect_tracks) -> dict:
    normalized = _zscore(features)
    novelty = _multiscale_novelty(normalized)
    silence_score = _silence_score(silence_runs, novelty.shape[0])
    dip_runs = _dip_runs(level, novelty)
    dip_score = _dip_score(dip_runs, novelty.shape[0])
    novelty_n = _unit_peak(novelty)
    score = (
        config.SEG_W_NOVELTY * novelty_n
        + config.SEG_W_SILENCE * silence_score
        + config.SEG_W_DIP * dip_score
    )
    peaks, k_used, attempts, adjusted = _pick_peaks(
        score, duration_s, min_track_s, expect_tracks
    )
    boundaries = []
    for time_s, peak_score in peaks:
        evidence = _evidence(time_s, novelty_n, silence_runs, dip_runs)
        boundaries.append(
            {
                "time_s": round(float(time_s), 4),
                "score": round(float(peak_score), 4),
                "confidence": _confidence(peak_score, score),
                "evidence": evidence,
                "transition": _transition(time_s, silence_runs, dip_runs),
            }
        )
    result = _tracks_from_boundaries(boundaries, duration_s, source="auto")
    result["threshold_k"] = round(float(k_used), 4)
    result["threshold_attempts"] = attempts
    result["threshold_adjusted"] = adjusted
    result["warning"] = _expect_warning(result["n_tracks"], expect_tracks, adjusted)
    result["features"] = normalized
    result["novelty"] = np.round(novelty_n, 4)
    result["score"] = np.round(score, 4)
    return result


def _zscore(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (features - mean) / scale


def _multiscale_novelty(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 5:
        return np.zeros(features.shape[0])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    vectors = features / norms
    curves = []
    for half in config.SEG_KERNEL_HALF_S:
        half_i = int(half)
        if vectors.shape[0] <= 2 * half_i:
            continue
        curves.append(_novelty(vectors, half_i))
    if not curves:
        curves.append(_novelty(vectors, max(2, vectors.shape[0] // 6)))
    length = min(curve.shape[0] for curve in curves)
    stacked = np.vstack([_zscore_1d(curve[:length]) for curve in curves])
    return stacked.mean(axis=0)


def _zscore_1d(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def _novelty(vectors: np.ndarray, half: int) -> np.ndarray:
    """Checkerboard correlation. Kernel is rank-1, so this is a diagonal band.

    novelty[t] = sum_d (sum_{i=-L..L} w[i] x[t+i, d])^2
    with w = gaussian * (+1 left of center, -1 right of center).
    """
    axis = np.arange(-half, half + 1, dtype=np.float64)
    sigma = max(half / 3.0, 1.0)
    gaussian = np.exp(-0.5 * (axis / sigma) ** 2)
    sign = np.where(axis < 0, 1.0, np.where(axis > 0, -1.0, 0.0))
    kernel = gaussian * sign
    flipped = kernel[::-1]
    total = np.zeros(vectors.shape[0], dtype=np.float64)
    for column in range(vectors.shape[1]):
        mixed = np.convolve(vectors[:, column], flipped, mode="same")
        total += mixed * mixed
    return total


def _unit_peak(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.min(values)
    peak = float(np.max(shifted))
    if peak <= 1e-8:
        return np.zeros_like(shifted)
    return shifted / peak


def _silence_score(runs, n_frames: int) -> np.ndarray:
    score = np.zeros(n_frames, dtype=np.float64)
    for start, end in runs:
        center = 0.5 * (start + end)
        for index in range(n_frames):
            time_s = index * config.SEG_HOP_S
            if start <= time_s <= end:
                half = max(0.5 * (end - start), 1e-6)
                score[index] = max(score[index], 1.0 - abs(time_s - center) / half)
    return score


def _dip_runs(level: np.ndarray, novelty: np.ndarray) -> list[tuple[float, float]]:
    if level.size < config.SEG_DIP_MIN_S + 2:
        return []
    window = max(3, int(config.SEG_DIP_MEDIAN_S))
    if window % 2 == 0:
        window += 1
    padded = np.pad(level, window // 2, mode="edge")
    median = np.array(
        [np.median(padded[index : index + window]) for index in range(level.size)]
    )
    below = level <= median - config.SEG_DIP_DB
    runs = []
    start = None
    for index, flag in enumerate(below):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            _keep_dip(runs, start, index, novelty)
            start = None
    if start is not None:
        _keep_dip(runs, start, level.size, novelty)
    return runs


def _keep_dip(runs, start: int, end: int, novelty: np.ndarray) -> None:
    length = (end - start) * config.SEG_HOP_S
    if length < config.SEG_DIP_MIN_S or length > config.SEG_DIP_MAX_S:
        return
    left = max(0, start - 1)
    right = min(novelty.shape[0], end + 1)
    if right <= left:
        return
    segment = novelty[left:right]
    # A dip counts only when a novelty peak sits on it.
    if segment.size >= 3 and np.max(segment) >= np.median(novelty) and _has_local_peak(segment):
        runs.append((start * config.SEG_HOP_S, end * config.SEG_HOP_S))
    elif segment.size and float(np.max(segment)) >= 0.45:
        runs.append((start * config.SEG_HOP_S, end * config.SEG_HOP_S))


def _has_local_peak(values: np.ndarray) -> bool:
    for index in range(1, len(values) - 1):
        if values[index] >= values[index - 1] and values[index] > values[index + 1]:
            return True
    return False


def _dip_score(runs, n_frames: int) -> np.ndarray:
    score = np.zeros(n_frames, dtype=np.float64)
    for start, end in runs:
        for index in range(n_frames):
            time_s = index * config.SEG_HOP_S
            if start <= time_s <= end:
                score[index] = 1.0
    return score


def _pick_peaks(score, duration_s, min_track_s, expect_tracks):
    k = float(config.SEG_K)
    peaks = _peaks_at_k(score, duration_s, min_track_s, k)
    attempts = 1
    adjusted = False
    count = len(peaks) + 1 if duration_s > 0 else 0
    if expect_tracks is not None and not _inside(count, expect_tracks):
        low, high = expect_tracks
        k = k * (1.0 + config.SEG_K_RETRY_SCALE) if count > high else k * (1.0 - config.SEG_K_RETRY_SCALE)
        peaks = _peaks_at_k(score, duration_s, min_track_s, k)
        attempts = 2
        adjusted = True
    return peaks, k, attempts, adjusted


def _inside(count: int, expect_tracks: tuple[int, int]) -> bool:
    low, high = expect_tracks
    return low <= count <= high


def _peaks_at_k(score: np.ndarray, duration_s: float, min_track_s: float, k: float):
    if score.size == 0:
        return []
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    if mad < 1e-8:
        mad = 1e-8
    threshold = median + k * mad
    candidates = []
    for index in range(1, score.size - 1):
        if score[index] < threshold:
            continue
        if score[index] < score[index - 1] or score[index] < score[index + 1]:
            continue
        # The 2 s window is centered one second after its start index.
        time_s = index * config.SEG_HOP_S + config.SEG_WINDOW_S / 2.0
        if time_s < min_track_s or time_s > duration_s - min_track_s:
            continue
        candidates.append((time_s, float(score[index])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept = []
    for time_s, peak_score in candidates:
        if all(abs(time_s - other) >= min_track_s for other, _score in kept):
            kept.append((time_s, peak_score))
    kept.sort(key=lambda item: item[0])
    return kept


def _confidence(peak_score: float, score: np.ndarray) -> str:
    median = float(np.median(score)) if score.size else 0.0
    mad = float(np.median(np.abs(score - median))) if score.size else 1.0
    if mad < 1e-8:
        mad = 1e-8
    strength = (peak_score - median) / mad
    if strength >= config.SEG_CONF_HIGH_MAD:
        return "high"
    if strength >= config.SEG_CONF_MED_MAD:
        return "medium"
    return "low"


def _evidence(time_s, novelty, silence_runs, dip_runs) -> list[str]:
    evidence = []
    index = int(round(time_s / config.SEG_HOP_S))
    index = min(max(index, 0), max(0, novelty.shape[0] - 1))
    if novelty.shape[0] and novelty[index] >= 0.35:
        evidence.append("novelty")
    if _near_run(time_s, silence_runs, margin=1.0):
        evidence.append("silence")
    if _near_run(time_s, dip_runs, margin=1.0):
        evidence.append("dip")
    if not evidence:
        evidence.append("novelty")
    return evidence


def _near_run(time_s: float, runs, margin: float) -> bool:
    return any(start - margin <= time_s <= end + margin for start, end in runs)


def _transition(time_s, silence_runs, dip_runs) -> dict:
    silence = _containing(time_s, silence_runs, margin=1.0)
    dip = _containing(time_s, dip_runs, margin=1.0)
    # A real gap wins over a level dip. Crossfade has a dip and no silence.
    if silence is not None:
        kind, length = "gap", silence[1] - silence[0]
    elif dip is not None:
        kind, length = "crossfade", dip[1] - dip[0]
    else:
        kind, length = "hard", 0.0
    return {"kind": kind, "length_s": round(float(length), 4)}


def _containing(time_s, runs, margin):
    best = None
    for start, end in runs:
        if start - margin <= time_s <= end + margin:
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
    return best


def _tracks_from_boundaries(boundaries, duration_s, source: str) -> dict:
    cuts = [0.0, *[row["time_s"] for row in boundaries], round(float(duration_s), 4)]
    tracks = []
    for index in range(len(cuts) - 1):
        start_s = cuts[index]
        end_s = cuts[index + 1]
        incoming = boundaries[index - 1] if index else None
        tracks.append(
            {
                "index": index + 1,
                "start_s": round(float(start_s), 4),
                "end_s": round(float(end_s), 4),
                "duration_s": round(float(end_s - start_s), 4),
                "confidence": None if incoming is None else incoming["confidence"],
                "evidence": ["start"] if incoming is None else list(incoming["evidence"]),
                "transition_in": None if incoming is None else incoming["transition"]["kind"],
                "transition_in_len_s": None if incoming is None else incoming["transition"]["length_s"],
                "possible_duplicate_of": None,
                "name": "",
            }
        )
    return {
        "source": source,
        "boundaries": boundaries,
        "tracks": tracks,
        "n_tracks": len(tracks),
        "threshold_k": None,
        "threshold_attempts": 0,
        "threshold_adjusted": False,
        "warning": None,
        "features": None,
        "novelty": None,
        "score": None,
    }


def _cuts_from_starts(starts, duration, source, evidence, names=None) -> dict:
    cleaned = sorted({round(float(value), 4) for value in starts if 0 < float(value) < duration})
    boundaries = []
    for time_s in cleaned:
        boundaries.append(
            {
                "time_s": time_s,
                "score": None,
                "confidence": "high",
                "evidence": list(evidence),
                "transition": {"kind": "hard", "length_s": 0.0},
            }
        )
    result = _tracks_from_boundaries(boundaries, duration, source=source)
    if names:
        for track, name in zip(result["tracks"], names):
            track["name"] = name
    return result


def _expect_warning(count: int, expect_tracks, adjusted: bool):
    if not adjusted or expect_tracks is None or _inside(count, expect_tracks):
        return None
    low, high = expect_tracks
    return (
        f"Topilgan trek soni {count}, kutilgan oralik {low}-{high}. "
        "Chegara bir marta qayta sozlandi. Son majburlanmadi."
    )


def _mark_duplicates(result: dict) -> None:
    features = result.get("features")
    tracks = result["tracks"]
    if features is None or features.size == 0:
        return
    means = []
    sequences = []
    for track in tracks:
        start, end = _interior_frames(track, features.shape[0])
        block = features[start:end]
        if block.size == 0:
            means.append(None)
            sequences.append(None)
            continue
        means.append(block.mean(axis=0))
        sequences.append(_resample_chroma(block[:, :12], 32))
    for index, track in enumerate(tracks):
        if means[index] is None:
            continue
        for earlier in range(index):
            if means[earlier] is None:
                continue
            mean_cos = _cosine(means[index], means[earlier])
            seq_cos = _cosine(sequences[index], sequences[earlier])
            if mean_cos > config.SEG_DUP_MEAN_COSINE and seq_cos >= config.SEG_DUP_CHROMA_COSINE:
                track["possible_duplicate_of"] = tracks[earlier]["index"]
                break


def _interior_frames(track: dict, n_frames: int) -> tuple[int, int]:
    """Frames whose 2 s window sits inside the track, so a cut does not leak."""
    start = int(np.ceil(float(track["start_s"]) / config.SEG_HOP_S))
    end = int(np.floor((float(track["end_s"]) - config.SEG_WINDOW_S) / config.SEG_HOP_S)) + 1
    start = min(max(0, start), n_frames)
    end = min(max(start, end), n_frames)
    if end - start < 4:
        start = min(n_frames, max(0, int(round(float(track["start_s"]) / config.SEG_HOP_S))))
        end = min(n_frames, max(start + 1, int(round(float(track["end_s"]) / config.SEG_HOP_S))))
    return start, end


def _resample_chroma(chroma: np.ndarray, length: int) -> np.ndarray:
    if chroma.size == 0:
        return np.zeros(length * 12)
    if chroma.shape[0] == 1:
        stretched = np.repeat(chroma, length, axis=0)
    else:
        source = np.linspace(0.0, 1.0, chroma.shape[0])
        target = np.linspace(0.0, 1.0, length)
        stretched = np.vstack([np.interp(target, source, chroma[:, column]) for column in range(chroma.shape[1])]).T
    flat = stretched.reshape(-1)
    return flat


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def _style_groups(result: dict) -> dict:
    features = result.get("features")
    tracks = result["tracks"]
    groups = {"n": 1, "kind": "proxy", "clusters": [], "note": "Treklararo xilma-xillik, proksi."}
    if features is None or len(tracks) == 0:
        return groups
    vectors = []
    usable = []
    for track in tracks:
        start = max(0, int(round(track["start_s"] / config.SEG_HOP_S)))
        end = max(start + 1, int(round(track["end_s"] / config.SEG_HOP_S)))
        block = features[start: min(end, features.shape[0])]
        if block.size == 0:
            continue
        vectors.append(block.mean(axis=0))
        usable.append(track["index"])
    if len(vectors) < 3:
        groups["clusters"] = [{"id": 1, "tracks": usable, "mean_features": None}]
        groups["n"] = 1 if vectors else 0
        return groups
    matrix = np.vstack(vectors)
    distances = pdist(matrix, metric="cosine")
    if not np.isfinite(distances).all() or float(np.max(distances)) < 1e-6:
        groups["clusters"] = [{"id": 1, "tracks": usable}]
        return groups
    linked = linkage(distances, method="average")
    best_labels = np.ones(len(vectors), dtype=int)
    best_score = -1.0
    dense = squareform(distances)
    for clusters in range(2, len(vectors)):
        labels = fcluster(linked, clusters, criterion="maxclust")
        score = _silhouette(dense, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    grouped: dict[int, list[int]] = {}
    for track_index, label in zip(usable, best_labels):
        grouped.setdefault(int(label), []).append(track_index)
    groups["n"] = len(grouped)
    groups["silhouette"] = round(float(best_score), 4)
    groups["clusters"] = [
        {"id": cluster_id, "tracks": members}
        for cluster_id, members in sorted(grouped.items())
    ]
    return groups


def _silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for index in range(len(labels)):
        same = labels == labels[index]
        same[index] = False
        other_labels = [label for label in set(labels) if label != labels[index]]
        if not np.any(same) or not other_labels:
            continue
        cohesion = float(np.mean(distances[index, same]))
        separation = min(float(np.mean(distances[index, labels == label])) for label in other_labels)
        width = max(cohesion, separation)
        if width <= 1e-8:
            scores.append(0.0)
        else:
            scores.append((separation - cohesion) / width)
    if not scores:
        return -1.0
    return float(np.mean(scores))


def _summary(result: dict) -> dict:
    durations = [track["duration_s"] for track in result["tracks"]]
    transitions = {"hard": 0, "gap": 0, "crossfade": 0}
    for track in result["tracks"]:
        kind = track.get("transition_in")
        if kind in transitions:
            transitions[kind] += 1
    arr = np.asarray(durations, dtype=np.float64) if durations else np.zeros(1)
    return {
        "n_tracks": {"value": result["n_tracks"], "unit": "count", "kind": "estimate", "note": "Topilgan trek soni."},
        "duration_median_s": {
            "value": None if not durations else round(float(np.median(arr)), 4),
            "unit": "s",
            "kind": "measured",
            "note": "Trek uzunliklari medianasi.",
        },
        "duration_min_s": {
            "value": None if not durations else round(float(np.min(arr)), 4),
            "unit": "s",
            "kind": "measured",
            "note": "Eng qisqa trek.",
        },
        "duration_max_s": {
            "value": None if not durations else round(float(np.max(arr)), 4),
            "unit": "s",
            "kind": "measured",
            "note": "Eng uzun trek.",
        },
        "transitions": transitions,
        "style_groups": result.get("style_groups", {}).get("n"),
    }


def read_manual_csv(path: str | Path) -> list[float]:
    starts = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        sample = handle.read()
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].lower()
    if "start_s" in header:
        reader = csv.DictReader(lines)
        for row in reader:
            if row.get("start_s"):
                starts.append(float(row["start_s"]))
        return starts
    for line in lines:
        starts.append(float(line.split(",")[0].strip()))
    return starts


def boundary_times(result: dict) -> list[float]:
    return [row["time_s"] for row in result["boundaries"]]


def json_ready(result: dict) -> dict:
    """Drop feature arrays so the report can be stored as JSON."""
    return {key: value for key, value in result.items() if key not in ("features", "novelty", "score")}


def feature_distance_matrix(result: dict) -> tuple[list[int], np.ndarray]:
    """Pairwise cosine distance of per-track mean features. Proxy."""
    features = result.get("features")
    tracks = result.get("tracks") or []
    if features is None or features.size == 0 or not tracks:
        return [], np.zeros((0, 0))
    means = []
    labels = []
    for track in tracks:
        start, end = _interior_frames(track, features.shape[0])
        block = features[start:end]
        if block.size == 0:
            continue
        means.append(block.mean(axis=0))
        labels.append(int(track["index"]))
    count = len(means)
    distance = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(count):
            distance[left, right] = 1.0 - _cosine(means[left], means[right])
    return labels, distance


def export_segments(path: str | Path, result: dict, directory: str | Path) -> list[Path]:
    """Cut each track with ffmpeg. Stream-copy first, then PCM if copy fails."""
    path = Path(path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    ffmpeg = io._tool("ffmpeg")
    for track in result["tracks"]:
        target = directory / f"{int(track['index']):02d}.wav"
        bounds = ["-ss", str(track["start_s"]), "-to", str(track["end_s"]), "-i", str(path)]
        copied = subprocess.run(
            [ffmpeg, "-y", *bounds, "-c", "copy", str(target)],
            capture_output=True,
            check=False,
        )
        if copied.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            encoded = subprocess.run(
                [ffmpeg, "-y", *bounds, "-c:a", "pcm_s16le", str(target)],
                capture_output=True,
                check=False,
            )
            if encoded.returncode != 0:
                detail = (encoded.stderr or b"").decode("utf-8", errors="replace")[-400:]
                raise io.AudioLabError(f"Trek kesilmadi: {target.name}. {detail}")
        written.append(target)
    return written
