"""Phase 1 pipeline against a fake YouTube API. No live key required."""

from datetime import datetime, timezone
import csv
import json
import sqlite3

from main import main
from store import find_audit, recent_audits

NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)
SECRET = "test-key-SECRET"
NICHE = "cozy fantasy ambience"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.url = "https://www.googleapis.com/youtube/v3/test"

    def json(self):
        return self._payload


def _video(video_id, channel_id, title, published_at, views, duration):
    return {
        "id": video_id,
        "snippet": {
            "title": title,
            "channelId": channel_id,
            "publishedAt": published_at,
            "thumbnails": {"high": {"url": f"https://img.example/{video_id}.jpg"}},
        },
        "statistics": {"viewCount": str(views)},
        "contentDetails": {"duration": duration},
    }


def _channel(channel_id, title, published_at, playlist, *, hidden=False, subs=1500):
    stats = {
        "viewCount": "100000",
        "videoCount": "10",
        "hiddenSubscriberCount": hidden,
    }
    stats["subscriberCount"] = "999" if hidden else str(subs)
    return {
        "id": channel_id,
        "snippet": {"title": title, "publishedAt": published_at},
        "statistics": stats,
        "contentDetails": {"relatedPlaylists": {"uploads": playlist}},
    }


STANDARD_UPLOADS = [
    ("up1", "One", "2026-10-07T00:00:00Z", 8000, "PT40M"),
    ("up2", "Two", "2026-10-14T00:00:00Z", 9000, "PT50M"),
    ("up3", "Three", "2026-10-21T00:00:00Z", 10000, "PT60M"),
    ("up4", "Four", "2026-10-28T00:00:00Z", 11000, "PT40M"),
    ("up5", "Five", "2026-11-04T00:00:00Z", 12000, "PT60M"),
]


class Api:
    def __init__(self, mode="standard"):
        self.mode = mode
        self.calls = []
        self.enabled = True
        self.fail_search = False
        self.fail_playlist_number = None
        self.playlist_calls = 0
        self.empty_search = False
        self.searches = []

    def handle(self, url, params):
        if not self.enabled:
            raise AssertionError(f"API called: {url}")
        self.calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("/search"):
            self.searches.append(params)
            if self.fail_search:
                return self._quota()
            if self.empty_search:
                return FakeResponse(200, {"items": []})
            if self.mode == "partial":
                ids = ["vid_a", "vid_b"]
            else:
                ids = ["vid_young", "vid_old"]
            return FakeResponse(200, {"items": [{"id": {"videoId": video_id}} for video_id in ids]})
        if url.endswith("/channels"):
            if self.mode == "partial":
                items = [
                    _channel("ch_a", "Alpha", "2026-07-27T00:00:00Z", "UUa"),
                    _channel("ch_b", "Beta", "2026-08-01T00:00:00Z", "UUb"),
                ]
            else:
                items = [
                    _channel("ch_young", "Young Dreams", "2026-07-27T00:00:00Z", "UUyoung"),
                    _channel("ch_old", "Old Archive", "2020-01-01T00:00:00Z", "UUold", hidden=True),
                ]
            return FakeResponse(200, {"items": items})
        if url.endswith("/playlistItems"):
            self.playlist_calls += 1
            if self.fail_playlist_number == self.playlist_calls:
                return self._quota()
            playlist_id = params["playlistId"]
            if playlist_id == "UUyoung":
                ids = [item[0] for item in STANDARD_UPLOADS]
            elif playlist_id == "UUa":
                ids = [f"a{i}" for i in range(1, 6)]
            elif playlist_id == "UUb":
                ids = [f"b{i}" for i in range(1, 6)]
            else:
                raise AssertionError(f"unexpected playlist {playlist_id}")
            return FakeResponse(
                200,
                {"items": [{"contentDetails": {"videoId": video_id}} for video_id in ids]},
            )
        if url.endswith("/videos"):
            return FakeResponse(
                200,
                {"items": [self._video_for(video_id) for video_id in params["id"].split(",") if video_id]},
            )
        raise AssertionError(url)

    def _video_for(self, video_id):
        for item in STANDARD_UPLOADS:
            if item[0] == video_id:
                return _video(item[0], "ch_young", item[1], item[2], item[3], item[4])
        channel_id = {
            "vid_young": "ch_young",
            "vid_old": "ch_old",
            "vid_a": "ch_a",
            "vid_b": "ch_b",
        }.get(video_id)
        if channel_id is None:
            channel_id = "ch_a" if video_id.startswith("a") else "ch_b"
        views = 1000 if video_id.startswith("vid_") else 10000
        return _video(video_id, channel_id, video_id, "2026-10-01T00:00:00Z", views, "PT40M")

    def _quota(self):
        return FakeResponse(
            403,
            {
                "error": {
                    "code": 403,
                    "message": "quotaExceeded",
                    "errors": [{"reason": "quotaExceeded"}],
                }
            },
        )


def _install(monkeypatch, api: Api):
    monkeypatch.setattr("main.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("YT_API_KEY", SECRET)

    def fake_get(_session, url, params=None, timeout=None, **_kwargs):
        return api.handle(url, params or {})

    monkeypatch.setattr("collect.requests.Session.get", fake_get)


def _run(tmp_path, args):
    return main(
        args,
        now=NOW,
        db_path=tmp_path / "researcher.db",
        output_dir=tmp_path / "out",
    )


def test_phase1_csv_sqlite_quota_and_cached_rerun(monkeypatch, tmp_path, capsys, caplog):
    api = Api()
    _install(monkeypatch, api)
    code = _run(tmp_path, [NICHE, "--no-claude"])
    out = capsys.readouterr().out

    assert code == 0
    # Two search.list calls (viewCount, then relevance) replace the old single search.
    assert api.calls == ["search", "search", "videos", "channels", "playlistItems", "videos"]
    assert [item["order"] for item in api.searches] == ["viewCount", "relevance"]
    assert api.playlist_calls == 1
    assert api.searches[0]["q"] == NICHE
    assert api.searches[0]["type"] == "video"
    assert api.searches[0]["videoDuration"] == "long"
    assert api.searches[0]["regionCode"] == "US"
    assert api.searches[0]["maxResults"] == 50
    assert api.searches[0]["key"] == SECRET
    assert api.searches[1]["publishedAfter"] == api.searches[0]["publishedAfter"]
    assert SECRET not in out
    assert SECRET not in caplog.text
    assert "Claude tahlili keyingi fazada" not in out
    assert "Tekshirilgan kanallar: 2" in out
    assert "Yangi kanallar: 1" in out
    assert "Outlier: 1 (50%)" in out
    assert "Outlier ulushi (hammasi): 50%" in out
    assert "Outlier ulushi (faqat relevance): 50%" in out
    assert "tanlama kichik" in out
    assert "Outlier ko'rish/kun (median): 1,000" in out
    assert "Oxirgi10" in out
    assert "Topilgan" in out
    assert "Sarflangan kvota (shu niche): 204" in out
    assert "Jami sarflangan kvota: 204" in out

    saved = find_audit(tmp_path / "researcher.db", NICHE)
    assert saved["quota_used"] == 204
    assert saved["created_at"] == "2026-11-04T00:00:00+00:00"
    assert saved["summary"]["complete"] is True
    assert saved["summary"]["channels_analyzed"] == 2
    assert saved["summary"]["outliers"] == 1
    assert saved["summary"]["new_channels"] == 1
    assert saved["summary"]["outlier_ratio"] == 0.5
    assert saved["summary"]["median_outlier_views_per_day"] == 1000
    assert saved["summary"]["typical_duration_min"] == 50
    assert saved["summary"]["typical_uploads_per_week"] == 1.25
    assert saved["params"]["no_claude"] is True
    assert saved["params"]["days"] == 180
    assert saved["params"]["min_median"] == 3000
    assert saved["params"]["min_ratio"] == 0.15
    assert saved["summary"]["outlier_ratio_by_source"] == 0.5

    connection = sqlite3.connect(tmp_path / "researcher.db")
    connection.row_factory = sqlite3.Row
    try:
        young = connection.execute("SELECT * FROM channels WHERE channel_id = 'ch_young'").fetchone()
        old = connection.execute("SELECT * FROM channels WHERE channel_id = 'ch_old'").fetchone()
        video_count = connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        connection.execute("SELECT term FROM expansions")
    finally:
        connection.close()
    assert young["is_outlier"] == 1
    assert young["subs"] == 1500
    assert young["videos"] == 10
    assert young["total_views"] == 100000
    assert young["avg_views"] == 10000
    assert young["median_views"] == 10000
    assert young["views_per_day"] == 1000
    assert young["age_days"] == 100
    assert young["avg_duration_min"] == 50
    assert young["uploads_per_week"] == 1.25
    assert young["found_by"] == "both"
    assert young["recent10_median_views"] == 10000
    assert young["median_avg_ratio"] == 1
    assert abs(young["top_share"] - 0.24) < 1e-9
    assert old["is_outlier"] == 0
    assert old["subs"] is None
    assert old["median_views"] is None
    assert video_count == 5
    assert young["ai_flag_share"] is not None

    files = [
        path
        for path in (tmp_path / "out").glob("cozy_fantasy_ambience_*.csv")
        if "_top_videos" not in path.name and "_suggestions" not in path.name and not path.name.startswith("compare_")
    ]
    assert len(files) == 1
    import io

    text = files[0].read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        text = "\n".join(lines[1:])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["Kanal"] == "Young Dreams"
    assert rows[0]["Outlier"] == "ha"
    assert rows[1]["Kanal"] == "Old Archive"
    assert rows[1]["Outlier"] == "yo'q"
    assert rows[1]["Obunachi"] == ""
    assert rows[0]["found_by"] == "both"
    assert "recent10_median_views" in rows[0]
    assert "median_avg_ratio" in rows[0]

    api.enabled = False
    code = _run(tmp_path, [NICHE, "--no-claude"])
    cached = capsys.readouterr().out
    assert code == 0
    assert "oldin tekshirilgan" in cached
    assert "Sarflangan kvota" not in cached
    assert len(recent_audits(tmp_path / "researcher.db", 10)) == 1

    api.enabled = True
    code = _run(tmp_path, [NICHE, "--no-claude", "--refresh"])
    refreshed = capsys.readouterr().out
    assert code == 0
    assert "Sarflangan kvota (shu niche): 204" in refreshed
    assert len(recent_audits(tmp_path / "researcher.db", 10)) == 2


def test_without_no_claude_still_only_collects(monkeypatch, tmp_path, capsys):
    api = Api()
    _install(monkeypatch, api)
    code = _run(tmp_path, [NICHE])
    out = capsys.readouterr().out
    assert code == 0
    assert "Claude tahlili keyingi fazada" in out
    assert "Sarflangan kvota (shu niche): 204" in out
    assert "Jami sarflangan kvota: 204" in out
    assert find_audit(tmp_path / "researcher.db", NICHE)["summary"]["outliers"] == 1


def test_empty_search_is_a_finished_audit(monkeypatch, tmp_path, capsys):
    api = Api()
    api.empty_search = True
    _install(monkeypatch, api)
    code = _run(tmp_path, ["quiet room", "--no-claude"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Kanal topilmadi." in out
    assert "Sarflangan kvota (shu niche): 200" in out
    assert find_audit(tmp_path / "researcher.db", "quiet room")["summary"]["complete"] is True

    api.enabled = False
    code = _run(tmp_path, ["quiet room", "--no-claude"])
    assert code == 0
    assert "oldin tekshirilgan" in capsys.readouterr().out


def test_quota_on_search_does_not_save(monkeypatch, tmp_path, capsys):
    api = Api()
    api.fail_search = True
    _install(monkeypatch, api)
    code = _run(tmp_path, [NICHE, "--no-claude"])
    out = capsys.readouterr().out
    assert code == 1
    assert "yig'ilmagan" in out
    assert "Jami sarflangan kvota: 0" in out
    assert find_audit(tmp_path / "researcher.db", NICHE) is None


def test_quota_midway_saves_partial_and_does_not_block_retry(monkeypatch, tmp_path, capsys):
    api = Api(mode="partial")
    api.fail_playlist_number = 2
    _install(monkeypatch, api)
    code = _run(tmp_path, ["soft rain", "--no-claude"])
    out = capsys.readouterr().out
    assert code == 1
    assert api.playlist_calls == 2
    assert api.calls == ["search", "search", "videos", "channels", "playlistItems", "videos", "playlistItems"]
    assert "kvotasi tugadi" in out
    assert "saqlandi" in out
    assert "Sarflangan kvota (shu niche): 204" in out
    assert "Jami sarflangan kvota: 204" in out
    partial = find_audit(tmp_path / "researcher.db", "soft rain")
    assert partial["summary"]["complete"] is False
    assert partial["summary"]["channels_analyzed"] == 1

    api.fail_playlist_number = None
    code = _run(tmp_path, ["soft rain", "--no-claude"])
    retried = capsys.readouterr().out
    assert code == 0
    assert "oldin tekshirilgan" not in retried
    assert "Sarflangan kvota (shu niche): 206" in retried
    finished = find_audit(tmp_path / "researcher.db", "soft rain")
    assert finished["summary"]["complete"] is True
    assert finished["summary"]["outliers"] == 2


def test_missing_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("main.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.delenv("YT_API_KEY", raising=False)
    code = _run(tmp_path, ["rain"])
    out = capsys.readouterr().out
    assert code == 1
    assert "YT_API_KEY" in out
    assert "Sarflangan kvota" not in out


def test_rejects_bad_arguments(capsys):
    assert main(["ambience", "--interest", "9"]) == 1
    assert main(["ambience", "--days", "0"]) == 1
    assert main(["   "]) == 1
    out = capsys.readouterr().out
    assert "1 dan 5" in out
    assert "--days" in out
    assert "Niche yoki --file kerak" in out
