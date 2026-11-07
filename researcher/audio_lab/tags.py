"""Optional Essentia tags. Skipped without an error when Essentia or models are absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_lab import config, io
from audio_lab.measure import metric

LICENSE = (
    "Essentia modellari CC BY-NC-SA 4.0 (tijoriy litsenziya alohida mavjud). "
    "Tijoriy foydalanishdan oldin shartlarni tekshiring."
)
CHUNK_S = 30.0
TAG_SR = 16000
HEADS = (
    ("genre", "genre_discogs400-discogs-effnet-1.pb", "genre_discogs400-discogs-effnet-1.json"),
    (
        "instrument",
        "mtg_jamendo_instrument-discogs-effnet-1.pb",
        "mtg_jamendo_instrument-discogs-effnet-1.json",
    ),
    (
        "mood",
        "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        "mtg_jamendo_moodtheme-discogs-effnet-1.json",
    ),
)
EMBEDDING_FILE = "discogs-effnet-bs64-1.pb"


def label(path: Path, clip: tuple[float, float] | None = None) -> dict:
    try:
        import essentia.standard  # noqa: F401
    except Exception:
        return _skipped("Essentia o'rnatilmagan yoki bu platformada ishlamaydi.")
    model_dir = config.MODEL_DIR
    needed = [model_dir / EMBEDDING_FILE]
    for _name, graph, meta in HEADS:
        needed.append(model_dir / graph)
        needed.append(model_dir / meta)
    missing = [item.name for item in needed if not item.is_file()]
    if missing:
        return _skipped("Essentia modellari topilmadi: " + ", ".join(missing))
    try:
        return _label(Path(path), clip, model_dir)
    except Exception as exc:
        return _skipped(f"Essentia ishlamadi: {exc}")


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "reason": reason, "license": LICENSE}


def _label(path: Path, clip: tuple[float, float] | None, model_dir: Path) -> dict:
    import json
    from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

    start_s, end_s = (None, None) if clip is None else clip
    audio = io.load_mono(path, sr=TAG_SR, start_s=start_s, end_s=end_s).astype(np.float32)
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_dir / EMBEDDING_FILE),
        output="PartitionedCall:1",
    )
    heads = []
    for name, graph, meta in HEADS:
        payload = json.loads((model_dir / meta).read_text(encoding="utf-8"))
        classes = payload.get("classes") or payload.get("labels") or []
        predictor = TensorflowPredict2D(
            graphFilename=str(model_dir / graph),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        heads.append((name, classes, predictor))
    chunk = int(CHUNK_S * TAG_SR)
    if audio.size < chunk:
        slices = [audio] if audio.size >= TAG_SR else []
    else:
        slices = [audio[start : start + chunk] for start in range(0, audio.size - chunk + 1, chunk)]
    collected = {name: [] for name, _classes, _predictor in heads}
    for piece in slices:
        embeddings = embedding_model(piece)
        for name, _classes, predictor in heads:
            predictions = np.asarray(predictor(embeddings), dtype=np.float64)
            if predictions.ndim == 1:
                vector = predictions
            else:
                vector = predictions.mean(axis=0)
            collected[name].append(vector)
    result = {"status": "ok", "license": LICENSE, "chunk_s": CHUNK_S}
    for name, classes, _predictor in heads:
        result[name] = _top10(collected[name], classes)
    return result


def _top10(vectors: list[np.ndarray], classes: list) -> list[dict]:
    if not vectors:
        return []
    stack = np.vstack(vectors)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    order = np.argsort(mean)[::-1][:10]
    rows = []
    for index in order:
        label_name = classes[index] if index < len(classes) else str(int(index))
        rows.append(
            {
                "label": label_name,
                "probability": metric(
                    float(mean[index]),
                    "probability",
                    "probabilistic",
                    "30 s bo'laklar bo'yicha o'rtacha ehtimollik. " + LICENSE,
                ),
                "std": metric(
                    float(std[index]),
                    "probability",
                    "probabilistic",
                    "Bo'laklar orasidagi standart og'ish (barqarorlik).",
                ),
            }
        )
    return rows
