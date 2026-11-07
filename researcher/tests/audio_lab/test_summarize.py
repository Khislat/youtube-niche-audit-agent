"""Compact summary stays short and omits stems or tags that were not run."""

import json

from audio_lab.summarize import render, summarize_report


def _metric(value, unit, kind="measured"):
    return {"value": value, "unit": unit, "kind": kind, "note": ""}


def _clip(index, start, end, *, tempo_ok=False, key_ok=True):
    return {
        "index": index,
        "start_s": start,
        "end_s": end,
        "metrics": {
            "loudness": {
                "integrated_lufs": _metric(-14.0, "LUFS"),
                "lra": _metric(6.0, "LU"),
            },
            "spectral": {
                "centroid_hz_median": _metric(400 + index, "Hz"),
                "lowpass_cutoff_hz": _metric(12000, "Hz", "estimate"),
            },
            "bands": {
                "share_pct": {
                    name: _metric(10.0, "%")
                    for name in ("sub_bass", "bass", "low_mid", "mid", "high_mid", "high")
                }
            },
            "pause": {
                "pause_ratio": _metric(0.02, "ratio"),
                "longest_pause_s": _metric(0.4, "s"),
            },
            "onsets": {"onsets_per_sec": _metric(1.5, "1/s", "proxy")},
            "repetition": {"repetition_score": _metric(0.3, "0-1", "proxy")},
            "tempo": {"bpm": _metric(90, "BPM", "estimate"), "reliable": tempo_ok},
            "key": {"label": _metric("C major", "key", "estimate"), "reliable": key_ok},
            "dynamic_range_db": _metric(None, "dB"),
        },
        "stems": {"status": "off"},
        "tags": {"status": "off"},
    }


def _report():
    return {
        "file_name": "synthetic.wav",
        "source_duration_s": 60,
        "params": {"sr": 22050, "sample_times": [0, 20, 40], "stems": False, "tags": False},
        "metrics": {},
        "samples": {
            "clips": [_clip(1, 0, 20), _clip(2, 20, 40), _clip(3, 40, 60)],
            "feature_distance": [
                [0.0, 0.2, 1.5],
                [0.2, 0.0, 1.4],
                [1.5, 1.4, 0.0],
            ],
        },
    }


def test_summary_fits_and_lists_columns_without_stems_or_tags(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    dest = summarize_report(path, tmp_path / "summary.md")
    text = dest.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 120
    for column in (
        "LUFS o'lchov",
        "LRA o'lchov",
        "centroid_hz o'lchov",
        "sub_bass%",
        "bass%",
        "low_mid%",
        "mid%",
        "high_mid%",
        "high%",
        "lowpass_hz taxmin",
        "pauza% o'lchov",
        "uzun pauza s o'lchov",
        "onset/s proksi",
        "tempo taxmin",
        "key taxmin",
        "repetition proksi",
    ):
        assert column in text
    assert "ishonchsiz" in text
    assert "C major" in text
    assert "chroma 12" in text
    assert "Dendrogram kesimi: kosinus masofa 1.0" in text
    assert "## Stem" not in text
    assert "## Teglar" not in text
    assert "n<8 parcha bo'lsa guruhlar taxminiy." in text
    assert "timeline" not in text.lower()


def test_summary_includes_stems_and_tags_when_they_ran():
    report = _report()
    report["params"]["stems"] = True
    report["params"]["tags"] = True
    for clip in report["samples"]["clips"]:
        clip["stems"] = {
            "status": "ok",
            "stems": {
                name: {"energy_share": _metric(0.2 if name == "drums" else 0.01, "share", "estimate")}
                for name in ("drums", "bass", "other", "vocals", "guitar", "piano")
            },
        }
        clip["tags"] = {
            "status": "ok",
            "genre": [{"label": "Ambient", "probability": _metric(0.4, "probability", "probabilistic")}],
            "instrument": [],
            "mood": [],
        }
    text = render(report)
    assert "## Stem ulushi" in text
    assert "Demucs bleeding" in text
    assert "## Teglar" in text
    assert "Ambient" in text
    assert len(text.splitlines()) <= 120
