"""Profile median and compare flags on synthetic clicks."""

from audio_lab.compare import compare_track
from audio_lab.profile import build_profile
from tests.audio_lab import synth


def _row(text: str, metric: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"| {metric} |"):
            return line
    raise AssertionError(metric)


def test_profile_of_five_and_outlier_flag(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    out = tmp_path / "out"
    db = tmp_path / "audio.db"
    for index in range(5):
        synth.write_wav(base / f"t{index}.wav", synth.clicks(60.0, 12.0))
    profile_path = build_profile(base, "click", out_dir=out, db_path=db)
    text = (profile_path.with_suffix(".md")).read_text(encoding="utf-8")
    assert "kichik tanlama" not in text

    same = compare_track(base / "t0.wav", profile_path, out_dir=out, db_path=db)
    assert "tavsiya" not in same.lower()
    onset_same = _row(same, "onsets.onsets_per_sec")
    assert onset_same.endswith("| yo'q |")
    assert "| 0.0 |" in onset_same or "| 0 |" in onset_same

    fast = tmp_path / "fast.wav"
    synth.write_wav(fast, synth.clicks(240.0, 12.0))
    flagged = compare_track(fast, profile_path, out_dir=out, db_path=db)
    assert _row(flagged, "onsets.onsets_per_sec").endswith("| ha |")

    report_md = next((out).rglob("report.md"))
    report_text = report_md.read_text(encoding="utf-8")
    assert "Cheklovlar" in report_text
    for badge in ("[O'LCHANGAN]", "[TAXMINIY]", "[PROKSI]", "[EHTIMOLLIK]"):
        assert badge in report_text
    assert any(out.rglob("timeline.csv"))
    for name in ("loudness.png", "bands.png", "spectrogram.png", "ssm.png"):
        assert any(path.name == name for path in out.rglob(name))


def test_small_profile_warns(tmp_path):
    folder = tmp_path / "two"
    folder.mkdir()
    synth.write_wav(folder / "a.wav", synth.clicks(60.0, 8.0))
    synth.write_wav(folder / "b.wav", synth.clicks(90.0, 8.0))
    path = build_profile(folder, "small", out_dir=tmp_path / "out", db_path=tmp_path / "db.sqlite")
    text = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "kichik tanlama, xulosa ehtiyotkorlik bilan" in text
