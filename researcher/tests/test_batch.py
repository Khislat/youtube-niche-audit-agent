"""Multi-niche batch, quota cap, and compare table (mock API)."""

from datetime import datetime, timezone
from pathlib import Path

from main import main, parse_niche_file
from store import sort_compare_rows

NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)


class Api:
    def __init__(self):
        self.calls = []
        self.fail_for = set()

    def handle(self, url, params):
        self.calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("/search"):
            return FakeResponse(200, {"items": []})
        return FakeResponse(200, {"items": []})


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = "{}"
        self.url = "https://example.com"

    def json(self):
        return self._payload


def _install(monkeypatch, api: Api):
    monkeypatch.setattr("main.load_dotenv", lambda *_a, **_k: False)
    monkeypatch.setenv("YT_API_KEY", "k")

    def fake_get(_session, url, params=None, timeout=None, **_kwargs):
        return api.handle(url, params or {})

    monkeypatch.setattr("collect.requests.Session.get", fake_get)


def test_parse_niche_file(tmp_path: Path):
    path = tmp_path / "niches.txt"
    path.write_text(
        "# comment\n\ncozy fantasy ambience\n\n# another\njapanese zen focus music\n",
        encoding="utf-8",
    )
    assert parse_niche_file(path) == ["cozy fantasy ambience", "japanese zen focus music"]


def test_batch_continues_after_error(monkeypatch, tmp_path, capsys):
    api = Api()
    _install(monkeypatch, api)
    niche_file = tmp_path / "niches.txt"
    niche_file.write_text("good niche\nbad niche\n", encoding="utf-8")

    def gather(niche, **kwargs):
        if niche == "bad niche":
            raise __import__("collect").YouTubeAPIError("boom")
        return [], []

    monkeypatch.setattr("main.gather", gather)
    code = main(
        ["--file", str(niche_file), "--no-claude", "--max-quota", "5000"],
        now=NOW,
        db_path=tmp_path / "researcher.db",
        output_dir=tmp_path / "out",
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "Xato (bad niche)" in out
    assert "good niche" in out


def test_max_quota_stops_before_next_niche(monkeypatch, tmp_path, capsys):
    api = Api()
    _install(monkeypatch, api)
    niche_file = tmp_path / "niches.txt"
    niche_file.write_text("one\n two\n three\n".replace(" ", ""), encoding="utf-8")

    def fake_gather(niche, **kwargs):
        kwargs["tracker"].add(200)
        return [], []

    monkeypatch.setattr("main.gather", fake_gather)
    code = main(
        ["--file", str(niche_file), "--no-claude", "--max-quota", "300"],
        now=NOW,
        db_path=tmp_path / "researcher.db",
        output_dir=tmp_path / "out",
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "Kvota chegarasi" in out


def test_compare_sort_and_small_sample_note():
    rows = sort_compare_rows(
        [
            {
                "niche": "b",
                "outlier_ratio_by_source": 0.2,
                "median_outlier_views_per_day": 100,
                "note": "",
            },
            {
                "niche": "a",
                "outlier_ratio_by_source": 0.5,
                "median_outlier_views_per_day": 10,
                "note": "(kichik tanlama)",
            },
        ]
    )
    assert rows[0]["niche"] == "a"
    assert rows[1]["niche"] == "b"
