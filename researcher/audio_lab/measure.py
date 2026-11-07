"""Signal measurements. Every metric carries value, unit, kind, and note."""

from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import pearsonr

from audio_lab import config, io


def metric(value, unit: str, kind: str, note: str) -> dict:
    return {"value": value, "unit": unit, "kind": kind, "note": note}


def analyze_file(
    path: str | Path,
    sr: int = config.FEATURE_SR,
    clip: tuple[float, float] | None = None,
) -> dict:
    """Measure one local file. `clip` is (start, end) in seconds."""
    path = Path(path)
    start_s, end_s = (None, None) if clip is None else clip
    if clip is not None and end_s <= start_s:
        raise io.AudioLabError("Clip oxiri boshidan keyin bo'lishi kerak.")
    info = io.probe(path)
    y = io.load_mono(path, sr=sr, start_s=start_s, end_s=end_s)
    loudness = io.measure_ebur128(path, start_s=start_s, end_s=end_s)
    stereo = _stereo(path, info["channels"], start_s, end_s)
    measured = _measure(y, sr, info, loudness, stereo)
    measured["params"] = {
        "sr": sr,
        "clip_start_s": start_s,
        "clip_end_s": end_s,
    }
    return _json_ready(measured)


def _measure(y: np.ndarray, sr: int, info: dict, loudness: dict, stereo: dict) -> dict:
    duration_s = float(len(y) / sr) if sr else 0.0
    frame_db = _frame_dbfs(y, sr, config.FRAME_S)
    spectrum = _spectrum(y, sr)
    onset_times = _onset_times(y, sr)
    tempo = _tempo(y, sr, onset_times, duration_s)
    key = _key(y, sr)
    repetition = _repetition(y, sr)
    bands = _bands(spectrum, sr)
    short_term = _short_term_series(loudness["short_term"], duration_s)
    rms_1s = _rms_timeline(y, sr)
    return {
        "duration_s": metric(
            duration_s, "s", "measured", "Tahlil qilingan parcha uzunligi."
        ),
        "sample_rate_in": metric(
            info.get("sample_rate_in"),
            "Hz",
            "measured",
            "Faylning asl namuna chastotasi (feature uchun qayta namunalashdan oldin).",
        ),
        "channels": metric(
            info.get("channels"),
            "count",
            "measured",
            "Asl fayldagi kanallar soni.",
        ),
        "loudness": {
            "integrated_lufs": metric(
                loudness["integrated_lufs"],
                "LUFS",
                "measured",
                "ffmpeg ebur128 integrallangan balandlik.",
            ),
            "lra": metric(
                loudness["lra_lu"],
                "LU",
                "measured",
                "ffmpeg ebur128 loudness range.",
            ),
            "true_peak_dbtp": metric(
                loudness["true_peak_dbtp"],
                "dBTP",
                "measured",
                "ffmpeg ebur128 true peak.",
            ),
            "short_term_lufs": metric(
                short_term,
                "LUFS",
                "measured",
                "Qisqa muddatli balandlik: 3 s oyna, 1 s qadam. To'liq oyna uchun t>=3 s.",
            ),
        },
        "rms_timeline_dbfs": metric(
            rms_1s,
            "dBFS",
            "measured",
            "1 soniyalik oynalarning RMS darajasi.",
        ),
        "dynamic_range_db": metric(
            _dynamic_range(short_term),
            "dB",
            "measured",
            "Qisqa muddatli balandlikning p95 minus p5.",
        ),
        "bands": bands,
        "spectral": _spectral_metrics(spectrum, sr),
        "onsets": _onset_metrics(onset_times, duration_s),
        "pause": _pause_metrics(frame_db),
        "tempo": tempo,
        "key": key,
        "repetition": repetition,
        "stereo": stereo,
        "ssm": metric(
            repetition.pop("_ssm"),
            "0-1",
            "proxy",
            "O'z-o'ziga o'xshashlik matritsasining kichraytirilgan ko'rinishi (grafik uchun).",
        ),
        "noise_floor_dbfs": metric(
            _noise_floor(frame_db),
            "dBFS",
            "estimate",
            "Eng sokin 5% kadrlarning o'rtacha darajasi. Signal ichidagi sokin joyni shovqin deb o'qishi mumkin.",
        ),
        "timelines_1s": metric(
            _timeline_1s(y, sr, spectrum, onset_times, short_term, rms_1s),
            "mixed",
            "measured",
            "1 s qadam: LUFS, RMS, 6 band ulushi, centroid, onset soni.",
        ),
    }


def _frame_dbfs(y: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    frame = max(1, int(round(frame_s * sr)))
    count = len(y) // frame
    if count == 0:
        return np.array([], dtype=np.float64)
    shaped = y[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(shaped * shaped, axis=1))
    rms = np.maximum(rms, 1e-12)
    return 20.0 * np.log10(rms)


def _spectrum(y: np.ndarray, sr: int) -> dict:
    stft = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    power = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
    return {"power": power, "freqs": freqs, "magnitude": stft}


def _bands(spectrum: dict, sr: int) -> dict:
    power = spectrum["power"]
    freqs = spectrum["freqs"]
    total = float(power.sum())
    nyquist = sr / 2.0
    shares = {}
    notes_high = (
        f"Feature namuna chastotasi {sr} Hz, Nyquist {nyquist:.0f} Hz. "
        "6000–16000 Hz bandining Nyquistdan yuqori qismi bu signalda yo'q."
    )
    for name, low, high in config.BANDS:
        mask = (freqs >= low) & (freqs < high)
        share = 0.0 if total <= 0 else 100.0 * float(power[mask].sum()) / total
        note = "STFT quvvati ulushi, butun parcha."
        if name == "high":
            note = notes_high
        shares[name] = metric(share, "%", "measured", note)
    timeline = _band_timeline(power, freqs, sr, config.BAND_WINDOW_S)
    return {
        "share_pct": shares,
        "timeline_10s": metric(
            timeline,
            "%",
            "measured",
            "Har 10 soniyalik oynadagi band ulushlari.",
        ),
    }


def _band_timeline(power: np.ndarray, freqs: np.ndarray, sr: int, window_s: float) -> list:
    hop = config.HOP_LENGTH
    times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=hop)
    if times.size == 0:
        return []
    duration = float(times[-1])
    rows = []
    t0 = 0.0
    while t0 < duration - 1e-9:
        t1 = t0 + window_s
        mask_t = (times >= t0) & (times < t1)
        if np.any(mask_t):
            block = power[:, mask_t]
            total = float(block.sum())
            row = {"t_s": t0}
            for name, low, high in config.BANDS:
                mask_f = (freqs >= low) & (freqs < high)
                row[name] = 0.0 if total <= 0 else 100.0 * float(block[mask_f].sum()) / total
            rows.append(row)
        t0 += window_s
    return rows


def _spectral_metrics(spectrum: dict, sr: int) -> dict:
    power = spectrum["power"]
    magnitude = spectrum["magnitude"]
    centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        roll_percent=0.85,
    )[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    return {
        "centroid_hz_median": metric(
            _finite_stat(centroid, np.median),
            "Hz",
            "measured",
            "Kadr centroidlarining medianasi.",
        ),
        "centroid_hz_p25": metric(
            _finite_stat(centroid, lambda v: np.percentile(v, 25)),
            "Hz",
            "measured",
            "Kadr centroidlarining 25-percentili.",
        ),
        "centroid_hz_p75": metric(
            _finite_stat(centroid, lambda v: np.percentile(v, 75)),
            "Hz",
            "measured",
            "Kadr centroidlarining 75-percentili.",
        ),
        "rolloff85_hz": metric(
            _finite_stat(rolloff, np.median),
            "Hz",
            "measured",
            "85% spektral rolloffning kadrlar medianasi.",
        ),
        "flatness_median": metric(
            _finite_stat(flatness, np.median),
            "0-1",
            "measured",
            "Spektr tekisligining kadrlar medianasi. 1 ga yaqin — shovqinga o'xshash.",
        ),
        "lowpass_cutoff_hz": metric(
            _lowpass_cutoff(power, spectrum["freqs"]),
            "Hz",
            "estimate",
            "Taxminiy past-o'tkazish chegarasi: silliq spektr cho'qqisidan 60 dB pastga tushgan eng yuqori chastota. Bo'sh binlar o'rtachani yerga tortmasligi uchun tayanch — cho'qqi.",
        ),
    }


def _lowpass_cutoff(power: np.ndarray, freqs: np.ndarray) -> float | None:
    """Highest frequency whose smoothed spectrum stays within 60 dB of the peak.

    A median over about 200 Hz ignores single-bin leakage. The reference is the
    peak, because the mean of every bin is dominated by empty bins.
    """
    psd = power.mean(axis=1)
    if freqs.size < 3 or not np.any(psd > 0):
        return None
    db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    width = max(3, int(round(200.0 / max(bin_hz, 1e-6))))
    if width % 2 == 0:
        width += 1
    width = min(width, db.size if db.size % 2 else db.size - 1)
    smooth = median_filter(db, size=max(3, width), mode="nearest")
    usable = freqs >= 20.0
    if not np.any(usable):
        return None
    # Peak of the smoothed spectrum. A mean over every bin sits near the
    # empty-bin floor, so "60 dB down" would never cross inside Nyquist.
    reference = float(np.max(smooth[usable]))
    above = usable & (smooth >= reference - config.LOWPASS_DROP_DB)
    if not np.any(above):
        return float(freqs[np.argmax(usable)])
    return float(freqs[np.max(np.where(above)[0])])


def _onset_times(y: np.ndarray, sr: int) -> np.ndarray:
    hop = config.ONSET_HOP_LENGTH
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop,
        units="time",
        backtrack=False,
    )
    return np.asarray(times, dtype=np.float64)


def _onset_metrics(times: np.ndarray, duration_s: float) -> dict:
    note = (
        "librosa onset. Pianino pedal va uzun notalarda onsetlar kam aniqlanishi mumkin."
    )
    per_sec = 0.0 if duration_s <= 0 else float(len(times) / duration_s)
    if len(times) >= 2:
        ioi = np.diff(times)
        ioi_median = float(np.median(ioi))
        mean = float(np.mean(ioi))
        if len(ioi) >= 2 and mean > 0:
            ioi_cv = float(np.std(ioi, ddof=1) / mean)
        else:
            ioi_cv = None
    else:
        ioi_median = None
        ioi_cv = None
    return {
        "onsets_per_sec": metric(per_sec, "1/s", "proxy", note),
        "ioi_median_s": metric(ioi_median, "s", "proxy", "Onsetlar orasidagi vaqtning medianasi. " + note),
        "ioi_cv": metric(
            ioi_cv,
            "ratio",
            "proxy",
            "Onsetlar orasidagi vaqtning variatsiya koeffitsienti (rubato uchun proksi). " + note,
        ),
    }


def _pause_metrics(frame_db: np.ndarray) -> dict:
    note = f"Kadr RMS < {config.PAUSE_DBFS:.0f} dBFS (mutlaq, cho'qqiga nisbatan emas)."
    if frame_db.size == 0:
        return {
            "pause_ratio": metric(None, "ratio", "measured", note),
            "longest_pause_s": metric(None, "s", "measured", note),
            "pauses_over_1s_count": metric(None, "count", "measured", note),
        }
    quiet = frame_db < config.PAUSE_DBFS
    ratio = float(np.mean(quiet))
    longest_frames = 0
    over_1s = 0
    run = 0
    for flag in quiet:
        if flag:
            run += 1
        elif run:
            longest_frames = max(longest_frames, run)
            if run * config.FRAME_S > 1.0:
                over_1s += 1
            run = 0
    if run:
        longest_frames = max(longest_frames, run)
        if run * config.FRAME_S > 1.0:
            over_1s += 1
    return {
        "pause_ratio": metric(ratio, "ratio", "measured", note),
        "longest_pause_s": metric(longest_frames * config.FRAME_S, "s", "measured", note),
        "pauses_over_1s_count": metric(over_1s, "count", "measured", note),
    }


def _tempo(y: np.ndarray, sr: int, onset_times: np.ndarray, duration_s: float) -> dict:
    hop = config.ONSET_HOP_LENGTH
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    gram = librosa.feature.tempogram(onset_envelope=envelope, sr=sr, hop_length=hop)
    profile = gram.mean(axis=1)
    bpms = librosa.tempo_frequencies(n_bins=len(profile), hop_length=hop, sr=sr)
    valid = (bpms >= config.TEMPO_BPM_MIN) & (bpms <= config.TEMPO_BPM_MAX) & np.isfinite(bpms)
    strength = np.where(valid, profile, 0.0)
    peaks = []
    for index in range(1, len(strength) - 1):
        if strength[index] <= 0:
            continue
        if strength[index] >= strength[index - 1] and strength[index] > strength[index + 1]:
            peaks.append(index)
    if not peaks and np.any(strength > 0):
        peaks = [int(np.argmax(strength))]
    peaks.sort(key=lambda i: strength[i], reverse=True)
    candidates = [float(bpms[i]) for i in peaks[:5]]
    track_bpm = _beat_track_bpm(y, sr)
    if track_bpm is not None:
        candidates.append(track_bpm)
    unique = _unique_bpm(candidates)
    best = float(strength[peaks[0]]) if peaks else 0.0
    second = float(strength[peaks[1]]) if len(peaks) > 1 else 0.0
    confidence = 0.0 if best <= 0 else (best - second) / best
    onset_bpm = None
    if duration_s > 0 and len(onset_times) >= 2:
        onset_bpm = float(len(onset_times) / duration_s * 60.0)
    chosen = unique[0] if unique else None
    if onset_bpm and unique:
        nearest = min(unique, key=lambda bpm: abs(bpm - onset_bpm))
        if abs(nearest - onset_bpm) / onset_bpm <= 0.15:
            chosen = nearest
    onsets_per_sec = 0.0 if duration_s <= 0 else len(onset_times) / duration_s
    reliable = (
        chosen is not None
        and confidence >= config.TEMPO_RELIABLE_MIN_CONFIDENCE
        and onsets_per_sec >= config.TEMPO_RELIABLE_MIN_ONSETS_PER_SEC
    )
    unreliable = (
        "Ritmsiz/ambient musiqada tempo ishonchsiz; yarim/ikki baravar noaniqligi bor."
    )
    note = (
        "librosa tempogram cho'qqisi ustunligi va beat_track nomzodlari. "
        "Onset tezligiga yaqin nomzod yarim/ikki baravar noaniqligini kamaytirish uchun tanlanadi."
    )
    if not reliable:
        note = note + " " + unreliable
    return {
        "bpm": metric(chosen, "BPM", "estimate", note),
        "confidence": metric(
            confidence,
            "0-1",
            "estimate",
            "Tempogramdagi eng yuqori cho'qqi ustunligi: (birinchi − ikkinchi) / birinchi.",
        ),
        "candidates_bpm": metric(unique, "BPM", "estimate", "Tempogram va beat_track nomzodlari."),
        "reliable": reliable,
    }


def _beat_track_bpm(y: np.ndarray, sr: int) -> float | None:
    try:
        tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        return None
    values = np.atleast_1d(np.asarray(tempo, dtype=np.float64))
    if values.size == 0 or not np.isfinite(values[0]) or values[0] <= 0:
        return None
    return float(values[0])


def _unique_bpm(values: list[float]) -> list[float]:
    ordered: list[float] = []
    for value in values:
        if not math.isfinite(value):
            continue
        if any(abs(value - kept) < 0.5 for kept in ordered):
            continue
        ordered.append(float(value))
    return ordered


def _key(y: np.ndarray, sr: int) -> dict:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean = chroma.mean(axis=1)
    total = float(mean.sum())
    if total <= 0:
        distribution = np.zeros(12)
    else:
        distribution = mean / total
    scores: list[tuple[float, str, str]] = []
    for shift in range(12):
        for mode, profile in (("major", config.MAJOR_PROFILE), ("minor", config.MINOR_PROFILE)):
            rotated = np.roll(np.asarray(profile, dtype=np.float64), shift)
            corr_raw = pearsonr(distribution, rotated)
            corr = getattr(corr_raw, "statistic", None)
            if corr is None:
                corr = corr_raw[0]
            corr_value = float(corr) if corr is not None and math.isfinite(float(corr)) else 0.0
            name = config.PITCH_CLASS_NAMES[shift]
            scores.append((corr_value, name, mode))
    scores.sort(key=lambda item: item[0], reverse=True)
    best_corr, best_name, best_mode = scores[0]
    second_corr = scores[1][0]
    margin = best_corr - second_corr
    reliable = margin >= config.KEY_RELIABLE_MIN_MARGIN
    label = f"{best_name} {best_mode}"
    note = "Krumhansl-Schmuckler, chroma o'rtachasi bilan korrelyatsiya. Margin — eng yaxshi va ikkinchi natija farqi."
    if not reliable:
        note += " Margin kichik: tonallik ishonchsiz."
    return {
        "label": metric(label, "key", "estimate", note),
        "key": metric(best_name, "pitch class", "estimate", note),
        "mode": metric(best_mode, "mode", "estimate", note),
        "margin": metric(margin, "correlation", "estimate", note),
        "reliable": reliable,
    }


def _repetition(y: np.ndarray, sr: int) -> dict:
    note = "2 s chroma+MFCC oynalarining o'z-o'ziga o'xshashligi. Diagonaldan uzoq lag'larning o'rtacha kosinus o'xshashligi."
    win = int(round(config.REP_WINDOW_S * sr))
    if len(y) < win * 2:
        empty = {
            "repetition_score": metric(None, "0-1", "proxy", note + " Parcha juda qisqa."),
            "top_lag_s": metric(None, "s", "proxy", note + " Parcha juda qisqa."),
            "_ssm": [],
        }
        return empty
    duration_s = len(y) / sr
    hop_s = config.REP_HOP_S
    max_windows = int((duration_s - config.REP_WINDOW_S) / hop_s) + 1
    if max_windows > config.REP_MAX_WINDOWS:
        hop_s = (duration_s - config.REP_WINDOW_S) / (config.REP_MAX_WINDOWS - 1)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )
    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=config.HOP_LENGTH
    )
    starts = np.arange(0.0, duration_s - config.REP_WINDOW_S + 1e-9, hop_s)
    vectors = []
    for t0 in starts:
        mask = (times >= t0) & (times < t0 + config.REP_WINDOW_S)
        if not np.any(mask):
            continue
        vectors.append(np.concatenate([chroma[:, mask].mean(axis=1), mfcc[:, mask].mean(axis=1)]))
    if len(vectors) < 2:
        return {
            "repetition_score": metric(None, "0-1", "proxy", note),
            "top_lag_s": metric(None, "s", "proxy", note),
            "_ssm": [],
        }
    matrix = np.vstack(vectors)
    chroma_part = _l2_rows(matrix[:, :12])
    mfcc_part = matrix[:, 12:]
    mfcc_part = mfcc_part - mfcc_part.mean(axis=0)
    scale = mfcc_part.std(axis=0)
    scale[scale < 1e-6] = 1.0
    mfcc_part = _l2_rows(mfcc_part / scale)
    # Equal weight: a loud MFCC coefficient must not hide a repeated chroma pattern.
    matrix = _l2_rows(np.concatenate([chroma_part, mfcc_part], axis=1))
    similarity = matrix @ matrix.T
    lag_s = []
    lag_mean = []
    for lag in range(1, similarity.shape[0]):
        lag_s.append(lag * hop_s)
        lag_mean.append(float(np.mean(np.diag(similarity, k=lag))))
    lag_s_arr = np.asarray(lag_s)
    lag_mean_arr = np.asarray(lag_mean)
    far = lag_s_arr >= config.REP_MIN_LAG_S
    if not np.any(far):
        score = None
        top = None
    else:
        score = float(np.mean(lag_mean_arr[far]))
        best = float(np.max(lag_mean_arr[far]))
        # A repeated phrase is equally similar at 4 s, 8 s, 12 s. The period
        # is the shortest lag that reaches that peak, not the first numeric max.
        tied = far & (lag_mean_arr >= best - 0.02)
        top = float(lag_s_arr[int(np.argmax(tied))])
    step = max(1, math.ceil(similarity.shape[0] / 200))
    preview = similarity[::step, ::step]
    return {
        "repetition_score": metric(score, "0-1", "proxy", note),
        "top_lag_s": metric(top, "s", "proxy", "Eng yuqori o'xshashlikdagi lag (oyna uzunligidan uzoq)."),
        "_ssm": preview.tolist(),
    }


def _stereo(path: Path, channels: int, start_s: float | None, end_s: float | None) -> dict:
    width_note = "Side energiya / mid energiya. 0 — farq yo'q."
    corr_note = "Chap va o'ng kanalning nol-o'rtacha korrelyatsiyasi."
    if channels < 2:
        return {
            "width": metric(0.0, "ratio", "measured", width_note + " Bitta kanal: side energiya 0."),
            "channel_correlation": metric(
                None,
                "correlation",
                "measured",
                corr_note + " Bitta kanalda korrelyatsiya aniqlanmaydi.",
            ),
        }
    sum_l2 = 0.0
    sum_r2 = 0.0
    sum_lr = 0.0
    sum_mid2 = 0.0
    sum_side2 = 0.0
    for block in io.iter_stereo(path, start_s=start_s, end_s=end_s):
        left = block[:, 0]
        right = block[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        sum_l2 += float(np.dot(left, left))
        sum_r2 += float(np.dot(right, right))
        sum_lr += float(np.dot(left, right))
        sum_mid2 += float(np.dot(mid, mid))
        sum_side2 += float(np.dot(side, side))
    width = None if sum_mid2 <= 0 else sum_side2 / sum_mid2
    denom = math.sqrt(sum_l2 * sum_r2)
    corr = None if denom <= 0 else sum_lr / denom
    return {
        "width": metric(width, "ratio", "measured", width_note),
        "channel_correlation": metric(corr, "correlation", "measured", corr_note),
    }


def _noise_floor(frame_db: np.ndarray) -> float | None:
    if frame_db.size == 0:
        return None
    count = max(1, int(math.ceil(frame_db.size * config.NOISE_FLOOR_QUIET_FRACTION)))
    quiet = np.partition(frame_db, count - 1)[:count]
    return float(np.mean(quiet))


def _short_term_series(samples: list[tuple[float, float]], duration_s: float) -> list[dict]:
    if not samples or duration_s <= 0:
        return []
    by_t = {round(t, 1): value for t, value in samples}
    rows = []
    t = config.SHORT_TERM_WINDOW_S
    while t <= duration_s + 0.05:
        nearest = min(by_t, key=lambda item: abs(item - t), default=None)
        if nearest is not None and abs(nearest - t) <= 0.15:
            rows.append({"t_s": float(round(t, 1)), "lufs": by_t[nearest]})
        t += config.SHORT_TERM_STEP_S
    return rows


def _dynamic_range(short_term: list[dict]) -> float | None:
    values = [row["lufs"] for row in short_term if row.get("lufs") is not None]
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 95) - np.percentile(arr, 5))


def _rms_timeline(y: np.ndarray, sr: int) -> list[dict]:
    frame = int(round(config.RMS_STEP_S * sr))
    if frame <= 0:
        return []
    rows = []
    for index, start in enumerate(range(0, len(y) - frame + 1, frame)):
        block = y[start : start + frame]
        rms = math.sqrt(float(np.dot(block, block) / frame))
        db = None if rms <= 0 else 20.0 * math.log10(rms)
        rows.append({"t_s": float(index), "dbfs": db})
    return rows


def _timeline_1s(y, sr, spectrum, onset_times, short_term, rms_1s) -> dict:
    """1 s grid used later by timeline.csv. Each series is one metric."""
    power = spectrum["power"]
    freqs = spectrum["freqs"]
    hop = config.HOP_LENGTH
    frame_times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(
        S=spectrum["magnitude"], sr=sr, n_fft=config.N_FFT, hop_length=hop
    )[0]
    n_sec = int(math.floor(len(y) / sr))
    lufs_by_t = {int(round(row["t_s"])): row["lufs"] for row in short_term}
    rms_by_t = {int(round(row["t_s"])): row["dbfs"] for row in rms_1s}
    rows = []
    for second in range(n_sec):
        mask = (frame_times >= second) & (frame_times < second + 1)
        block = power[:, mask] if np.any(mask) else None
        total = 0.0 if block is None else float(block.sum())
        row = {
            "t_s": second,
            "lufs": lufs_by_t.get(second),
            "rms_dbfs": rms_by_t.get(second),
            "centroid_hz": None if not np.any(mask) else float(np.nanmean(centroid[mask])),
            "onset_count": int(np.sum((onset_times >= second) & (onset_times < second + 1))),
        }
        for name, low, high in config.BANDS:
            if block is None or total <= 0:
                row[name] = None
            else:
                band = (freqs >= low) & (freqs < high)
                row[name] = 100.0 * float(block[band].sum()) / total
        rows.append(row)
    return rows


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return matrix / norms


def _finite_stat(values: np.ndarray, fn) -> float | None:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return None
    result = float(fn(clean))
    if not math.isfinite(result):
        return None
    return result


def _json_ready(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, 4)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    raise TypeError(f"JSON ga o'tmaydi: {type(value)}")
