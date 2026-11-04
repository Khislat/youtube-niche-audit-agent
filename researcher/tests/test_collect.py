"""YouTube client behavior with mocked HTTP. No live API key."""

from datetime import datetime, timezone
import json

import pytest
import requests

from collect import (
    QuotaExceeded,
    QuotaTracker,
    YouTubeAPIError,
    get_channel_uploads,
    get_channels,
    get_videos,
    search_niche_videos,
    search_videos,
)

NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)
SECRET = "SUPERSECRET"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.url = "https://www.googleapis.com/youtube/v3/test"

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


def _patch_get(monkeypatch, handler):
    def fake_get(_session, url, params=None, timeout=None, **_kwargs):
        return handler(url, params or {})

    monkeypatch.setattr("collect.requests.Session.get", fake_get)


def test_search_contract_and_published_after(monkeypatch):
    seen = {}

    def handler(url, params):
        seen.update(params)
        return FakeResponse(200, {"items": [{"id": {"videoId": "abc"}}]})

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    ids = search_videos(
        "cozy",
        10,
        1,
        "US",
        api_key=SECRET,
        tracker=tracker,
        now=NOW,
    )
    assert ids == ["abc"]
    assert tracker.used == 100
    assert seen["type"] == "video"
    assert seen["order"] == "viewCount"
    assert seen["videoDuration"] == "long"
    assert seen["part"] == "snippet"
    assert seen["maxResults"] == 50
    assert seen["regionCode"] == "US"
    assert seen["publishedAfter"] == "2026-10-25T00:00:00Z"
    assert seen["key"] == SECRET


def test_niche_search_merges_view_count_and_relevance(monkeypatch):
    # Phase 1.1 calls search.list twice. Each call is 100 quota; duplicate ids collapse.
    orders = []

    def handler(url, params):
        orders.append(params["order"])
        if params["order"] == "viewCount":
            items = [{"id": {"videoId": "a"}}, {"id": {"videoId": "b"}}]
        else:
            assert params["type"] == "video"
            assert params["videoDuration"] == "long"
            assert params["publishedAfter"] == "2026-10-25T00:00:00Z"
            items = [{"id": {"videoId": "b"}}, {"id": {"videoId": "c"}}]
        return FakeResponse(200, {"items": items})

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    ids, sources = search_niche_videos("q", 10, 1, "US", api_key="k", tracker=tracker, now=NOW)
    assert orders == ["viewCount", "relevance"]
    assert ids == ["a", "b", "c"]
    assert sources["a"] == {"views"}
    assert sources["b"] == {"views", "relevance"}
    assert sources["c"] == {"relevance"}
    assert tracker.used == 200


def test_blank_region_is_omitted_and_paging_stops(monkeypatch):
    calls = []

    def handler(url, params):
        calls.append(params)
        if len(calls) == 1:
            return FakeResponse(
                200,
                {"items": [{"id": {"videoId": "a"}}], "nextPageToken": "NEXT"},
            )
        return FakeResponse(200, {"items": [{"id": {"videoId": "a"}}, {"id": {"videoId": "b"}}]})

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    ids = search_videos("q", 10, 2, "  ", api_key="k", tracker=tracker, now=NOW)
    assert ids == ["a", "b"]
    assert "regionCode" not in calls[0]
    assert calls[1]["pageToken"] == "NEXT"
    assert tracker.used == 200


def test_search_does_not_request_a_missing_next_page(monkeypatch):
    calls = {"n": 0}

    def handler(url, params):
        calls["n"] += 1
        return FakeResponse(200, {"items": []})

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    assert search_videos("q", 10, 3, "US", api_key="k", tracker=tracker, now=NOW) == []
    assert calls["n"] == 1
    assert tracker.used == 100


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def handler(url, params):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse(500, {"error": {"code": 500}})
        return FakeResponse(200, {"items": [{"id": {"videoId": "ok"}}]})

    _patch_get(monkeypatch, handler)
    monkeypatch.setattr("collect.time.sleep", lambda seconds: sleeps.append(seconds))
    tracker = QuotaTracker()
    ids = search_videos("q", 10, 1, "US", api_key=SECRET, tracker=tracker, now=NOW)
    assert ids == ["ok"]
    assert calls["n"] == 3
    assert sleeps == [1, 2]
    assert tracker.used == 100


def test_retries_exhausted(monkeypatch):
    calls = {"n": 0}

    def handler(url, params):
        calls["n"] += 1
        return FakeResponse(503, {"error": {"code": 503}})

    _patch_get(monkeypatch, handler)
    monkeypatch.setattr("collect.time.sleep", lambda _seconds: None)
    tracker = QuotaTracker()
    with pytest.raises(YouTubeAPIError):
        search_videos("q", 10, 1, "US", api_key=SECRET, tracker=tracker, now=NOW)
    assert calls["n"] == 3
    assert tracker.used == 0


def test_quota_exceeded_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def handler(url, params):
        calls["n"] += 1
        return FakeResponse(
            403,
            {"error": {"code": 403, "errors": [{"reason": "quotaExceeded"}], "message": "quota"}},
        )

    _patch_get(monkeypatch, handler)
    monkeypatch.setattr("collect.time.sleep", lambda _seconds: (_ for _ in ()).throw(AssertionError("slept")))
    tracker = QuotaTracker()
    with pytest.raises(QuotaExceeded):
        search_videos("q", 10, 1, "US", api_key=SECRET, tracker=tracker, now=NOW)
    assert calls["n"] == 1
    assert tracker.used == 0


def test_error_text_redacts_the_api_key(monkeypatch, caplog):
    def handler(url, params):
        raise requests.Timeout(f"failed https://www.googleapis.com/youtube/v3/search?key={SECRET}")

    _patch_get(monkeypatch, handler)
    monkeypatch.setattr("collect.time.sleep", lambda _seconds: None)
    with pytest.raises(YouTubeAPIError) as caught:
        search_videos("q", 10, 1, "US", api_key=SECRET, tracker=QuotaTracker(), now=NOW)
    assert SECRET not in str(caught.value)
    assert SECRET not in caplog.text


def test_bad_key_is_not_a_quota_stop(monkeypatch):
    calls = {"n": 0}

    def handler(url, params):
        calls["n"] += 1
        return FakeResponse(
            403,
            {
                "error": {
                    "code": 403,
                    "message": f"Bad key {SECRET}",
                    "errors": [{"reason": "keyInvalid"}],
                }
            },
        )

    _patch_get(monkeypatch, handler)
    with pytest.raises(YouTubeAPIError) as caught:
        search_videos("q", 10, 1, "US", api_key=SECRET, tracker=QuotaTracker(), now=NOW)
    assert calls["n"] == 1
    assert SECRET not in str(caught.value)
    assert "quota exceeded" not in str(caught.value).lower()


def test_videos_are_batched_by_50_with_max_results(monkeypatch):
    batches = []

    def handler(url, params):
        batches.append(params["id"].split(","))
        assert params["maxResults"] == 50
        assert params["part"] == "snippet,statistics,contentDetails"
        return FakeResponse(200, {"items": []})

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    assert get_videos([f"v{i}" for i in range(51)], api_key="k", tracker=tracker) == []
    assert [len(batch) for batch in batches] == [50, 1]
    assert tracker.used == 2
    assert get_videos([], api_key="k", tracker=tracker) == []
    assert tracker.used == 2


def test_hidden_subscribers_are_dropped(monkeypatch):
    def handler(url, params):
        assert params["maxResults"] == 50
        return FakeResponse(
            200,
            {
                "items": [
                    {
                        "id": "hidden",
                        "snippet": {"title": "Hidden", "publishedAt": "2026-01-01T00:00:00Z"},
                        "statistics": {"hiddenSubscriberCount": True, "subscriberCount": "999", "videoCount": "2", "viewCount": "10"},
                        "contentDetails": {"relatedPlaylists": {"uploads": "UUhidden"}},
                    },
                    {
                        "id": "open",
                        "snippet": {"title": "Open", "publishedAt": "2026-01-01T00:00:00Z"},
                        "statistics": {"hiddenSubscriberCount": False, "subscriberCount": "10", "videoCount": "2", "viewCount": "10"},
                        "contentDetails": {"relatedPlaylists": {"uploads": "UUopen"}},
                    },
                ]
            },
        )

    _patch_get(monkeypatch, handler)
    channels = get_channels(["hidden", "open"], api_key="k", tracker=QuotaTracker())
    assert channels[0]["subs"] is None
    assert channels[0]["uploads_playlist"] == "UUhidden"
    assert channels[0]["url"] == "https://www.youtube.com/channel/hidden"
    assert channels[1]["subs"] == 10


def test_upload_sample_is_capped_and_hydrated(monkeypatch):
    calls = []

    def handler(url, params):
        calls.append((url, params))
        if url.endswith("/playlistItems"):
            assert params["maxResults"] == 2
            return FakeResponse(
                200,
                {
                    "items": [
                        {"contentDetails": {"videoId": f"up{i}"}}
                        for i in range(1, 6)
                    ]
                },
            )
        return FakeResponse(
            200,
            {
                "items": [
                    {
                        "id": video_id,
                        "snippet": {
                            "title": video_id,
                            "channelId": "ch",
                            "publishedAt": "2026-10-01T00:00:00Z",
                            "thumbnails": {"high": {"url": "https://img.example/a.jpg"}},
                        },
                        "statistics": {"viewCount": "15"},
                        "contentDetails": {"duration": "PT30M"},
                    }
                    for video_id in params["id"].split(",")
                ]
            },
        )

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    videos = get_channel_uploads(
        "ch",
        n=2,
        api_key="k",
        tracker=tracker,
        uploads_playlist="UUch",
    )
    assert [video["video_id"] for video in videos] == ["up1", "up2"]
    assert videos[0]["views"] == 15
    assert videos[0]["duration_min"] == 30
    assert tracker.used == 2
    assert calls[0][0].endswith("/playlistItems")
    assert calls[1][0].endswith("/videos")


def test_known_missing_playlist_does_not_call_the_api(monkeypatch):
    def handler(url, params):
        raise AssertionError(url)

    _patch_get(monkeypatch, handler)
    tracker = QuotaTracker()
    assert get_channel_uploads("ch", api_key="k", tracker=tracker, uploads_playlist="") == []
    assert tracker.used == 0
