"""YouTube Data API v3 client. No scraping.

Quota: search.list costs 100 units; videos.list, channels.list, and
playlistItems.list cost 1 unit each. The tracker is incremented only after a
successful response. A 403 quotaExceeded (or dailyLimitExceeded) stops the run.
Transient network errors and HTTP 5xx are tried up to 3 times with backoff.
"""

from datetime import datetime, timedelta, timezone
import logging
import os
import re
import time

import requests

from metrics import iso8601_duration_minutes

logger = logging.getLogger(__name__)

API_ROOT = "https://www.googleapis.com/youtube/v3"
SEARCH_LIST_COST = 100
LIST_COST = 1
# viewCount finds channels that already have a hit. relevance is the less biased pass.
SEARCH_ORDERS = (
    ("viewCount", "views"),
    ("relevance", "relevance"),
)
PAGE_SIZE = 50
ATTEMPTS = 3
REQUEST_TIMEOUT = 30

_KEY_RE = re.compile(r"(key=)[^&\s\"']+")


class QuotaExceeded(Exception):
    """The YouTube Data API rejected a call because the quota is exhausted."""

    def __init__(self) -> None:
        super().__init__("YouTube Data API quota exceeded")
        self.channels: list[dict] = []
        self.videos: list[dict] = []


class YouTubeAPIError(Exception):
    """A YouTube Data API call failed for a reason other than quota."""


class QuotaTracker:
    def __init__(self) -> None:
        self.used = 0

    def add(self, units: int) -> None:
        self.used += units


_default_tracker = QuotaTracker()


def _require_key(api_key: str | None) -> str:
    key = (api_key if api_key is not None else os.environ.get("YT_API_KEY", "")).strip()
    if not key:
        raise YouTubeAPIError("YT_API_KEY is not set")
    return key


def _redact(text: str, api_key: str | None) -> str:
    if api_key:
        text = text.replace(api_key, "***")
    return _KEY_RE.sub(r"\1***", text)


def _tracker_or_default(tracker: QuotaTracker | None) -> QuotaTracker:
    return tracker if tracker is not None else _default_tracker


def _is_quota_error(response: requests.Response) -> bool:
    try:
        body = response.json()
    except ValueError:
        return False
    error = body.get("error") or {}
    reasons = [item.get("reason", "") for item in error.get("errors") or []]
    message = str(error.get("message") or "")
    blob = " ".join([*reasons, message])
    return "quotaExceeded" in blob or "dailyLimitExceeded" in blob


def _get(
    session: requests.Session,
    url: str,
    params: dict,
    *,
    cost: int,
    tracker: QuotaTracker,
    api_key: str,
    label: str,
) -> dict:
    last_message = f"YouTube API call failed: {label}"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_message = _redact(str(exc), api_key)
            if attempt == ATTEMPTS:
                raise YouTubeAPIError(last_message) from None
            logger.warning(
                "Temporary network error on %s (%s), retry %s/%s",
                label,
                last_message,
                attempt,
                ATTEMPTS,
            )
            time.sleep(2 ** (attempt - 1))
            continue

        if response.status_code >= 500:
            last_message = f"YouTube API HTTP {response.status_code} on {label}"
            if attempt == ATTEMPTS:
                raise YouTubeAPIError(last_message)
            logger.warning(
                "Temporary YouTube API error on %s (HTTP %s), retry %s/%s",
                label,
                response.status_code,
                attempt,
                ATTEMPTS,
            )
            time.sleep(2 ** (attempt - 1))
            continue

        if response.status_code == 403 and _is_quota_error(response):
            logger.error("YouTube Data API quota exceeded on %s", label)
            raise QuotaExceeded()

        if response.status_code >= 400:
            snippet = _redact(response.text[:300], api_key)
            raise YouTubeAPIError(f"YouTube API HTTP {response.status_code} on {label}: {snippet}")

        tracker.add(cost)
        logger.info("YouTube %s +%s quota (total %s)", label, cost, tracker.used)
        try:
            return response.json()
        except ValueError:
            raise YouTubeAPIError(f"YouTube API returned non-JSON on {label}") from None

    raise YouTubeAPIError(last_message)


def _unique(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _chunks(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _thumbnail_url(snippet: dict) -> str:
    thumbnails = snippet.get("thumbnails") or {}
    for key in ("maxres", "standard", "high", "medium", "default"):
        url = (thumbnails.get(key) or {}).get("url")
        if url:
            return url
    return ""


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_video(item: dict) -> dict | None:
    video_id = item.get("id")
    if not video_id:
        return None
    snippet = item.get("snippet") or {}
    stats = item.get("statistics") or {}
    details = item.get("contentDetails") or {}
    views = _int_or_none(stats.get("viewCount"))
    description = snippet.get("description") or ""
    tags = snippet.get("tags") or []
    return {
        "video_id": video_id,
        "channel_id": snippet.get("channelId") or "",
        "title": snippet.get("title") or "",
        "published_at": snippet.get("publishedAt") or "",
        "views": 0 if views is None else views,
        "duration_min": iso8601_duration_minutes(details.get("duration")),
        "thumbnail_url": _thumbnail_url(snippet),
        "description_snippet": description[:300],
        "tags": list(tags) if isinstance(tags, list) else [],
    }


def _subs(stats: dict) -> int | None:
    hidden = stats.get("hiddenSubscriberCount")
    if hidden is True or hidden == "true":
        return None
    return _int_or_none(stats.get("subscriberCount"))


def _normalize_channel(item: dict) -> dict | None:
    channel_id = item.get("id")
    if not channel_id:
        return None
    snippet = item.get("snippet") or {}
    stats = item.get("statistics") or {}
    content = item.get("contentDetails") or {}
    playlists = content.get("relatedPlaylists") or {}
    return {
        "channel_id": channel_id,
        "title": snippet.get("title") or "",
        "url": f"https://www.youtube.com/channel/{channel_id}",
        "published_at": snippet.get("publishedAt") or "",
        "video_count": _int_or_none(stats.get("videoCount")),
        "view_count": _int_or_none(stats.get("viewCount")),
        "subs": _subs(stats),
        "uploads_playlist": playlists.get("uploads") or "",
    }


def search_videos(
    query: str,
    days: int,
    pages: int,
    region: str,
    *,
    api_key: str | None = None,
    tracker: QuotaTracker | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
    order: str = "viewCount",
) -> list[str]:
    """Return video ids from one search.list ordering."""
    key = _require_key(api_key)
    quota = _tracker_or_default(tracker)
    client = session or requests.Session()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    published_after = (moment - timedelta(days=days)).astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    ids: list[str] = []
    page_token = None
    for page in range(pages):
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": order,
            "videoDuration": "long",
            "publishedAfter": published_after,
            "maxResults": PAGE_SIZE,
            "key": key,
        }
        if region.strip():
            params["regionCode"] = region.strip()
        if page_token:
            params["pageToken"] = page_token
        data = _get(
            client,
            f"{API_ROOT}/search",
            params,
            cost=SEARCH_LIST_COST,
            tracker=quota,
            api_key=key,
            label=f"search.list {order} page {page + 1}",
        )
        for item in data.get("items") or []:
            video_id = (item.get("id") or {}).get("videoId")
            if video_id:
                ids.append(video_id)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    logger.info("Search %s returned %s video ids", order, len(_unique(ids)))
    return _unique(ids)


def search_niche_videos(
    query: str,
    days: int,
    pages: int,
    region: str,
    *,
    api_key: str | None = None,
    tracker: QuotaTracker | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> tuple[list[str], dict[str, set[str]]]:
    """Search by view count and by relevance. Ids are merged; sources stay per video.

    Each search.list call costs 100 units. A video found by both orders is listed once.
    """
    client = session or requests.Session()
    merged: list[str] = []
    seen: set[str] = set()
    sources: dict[str, set[str]] = {}
    for order, label in SEARCH_ORDERS:
        found = search_videos(
            query,
            days,
            pages,
            region,
            api_key=api_key,
            tracker=tracker,
            session=client,
            now=now,
            order=order,
        )
        for video_id in found:
            sources.setdefault(video_id, set()).add(label)
            if video_id not in seen:
                seen.add(video_id)
                merged.append(video_id)
    logger.info("Merged search ids: %s", len(merged))
    return merged, sources


def get_videos(
    ids: list[str],
    *,
    api_key: str | None = None,
    tracker: QuotaTracker | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch snippet, statistics, and contentDetails in batches of 50."""
    unique_ids = _unique(ids)
    if not unique_ids:
        return []
    key = _require_key(api_key)
    quota = _tracker_or_default(tracker)
    client = session or requests.Session()
    videos: list[dict] = []
    for batch in _chunks(unique_ids, PAGE_SIZE):
        data = _get(
            client,
            f"{API_ROOT}/videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
                "maxResults": PAGE_SIZE,
                "key": key,
            },
            cost=LIST_COST,
            tracker=quota,
            api_key=key,
            label="videos.list",
        )
        for item in data.get("items") or []:
            video = _normalize_video(item)
            if video:
                videos.append(video)
    return videos


def get_channels(
    ids: list[str],
    *,
    api_key: str | None = None,
    tracker: QuotaTracker | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch snippet, statistics, and contentDetails in batches of 50."""
    unique_ids = _unique(ids)
    if not unique_ids:
        return []
    key = _require_key(api_key)
    quota = _tracker_or_default(tracker)
    client = session or requests.Session()
    channels: list[dict] = []
    for batch in _chunks(unique_ids, PAGE_SIZE):
        data = _get(
            client,
            f"{API_ROOT}/channels",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
                "maxResults": PAGE_SIZE,
                "key": key,
            },
            cost=LIST_COST,
            tracker=quota,
            api_key=key,
            label="channels.list",
        )
        for item in data.get("items") or []:
            channel = _normalize_channel(item)
            if channel:
                channels.append(channel)
    return channels


def get_channel_uploads(
    channel_id: str,
    n: int = 50,
    *,
    api_key: str | None = None,
    tracker: QuotaTracker | None = None,
    session: requests.Session | None = None,
    uploads_playlist: str | None = None,
) -> list[dict]:
    """Load up to `n` latest uploads, then hydrate them with videos.list.

    Pass `uploads_playlist` from a channel already fetched by get_channels so
    this does not spend another channels.list unit.
    """
    if n <= 0:
        return []
    key = _require_key(api_key)
    quota = _tracker_or_default(tracker)
    client = session or requests.Session()
    # None means the playlist id is unknown and may be looked up.
    # An empty string means the caller already knows there is no uploads playlist.
    if uploads_playlist is None:
        found = get_channels(
            [channel_id],
            api_key=key,
            tracker=quota,
            session=client,
        )
        playlist_id = (found[0].get("uploads_playlist") or "") if found else ""
    else:
        playlist_id = uploads_playlist
    if not playlist_id:
        logger.info("Channel %s has no uploads playlist", channel_id)
        return []

    collected: list[str] = []
    page_token = None
    while len(collected) < n:
        page_size = min(PAGE_SIZE, n - len(collected))
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": page_size,
            "key": key,
        }
        if page_token:
            params["pageToken"] = page_token
        data = _get(
            client,
            f"{API_ROOT}/playlistItems",
            params,
            cost=LIST_COST,
            tracker=quota,
            api_key=key,
            label="playlistItems.list",
        )
        for item in data.get("items") or []:
            video_id = (item.get("contentDetails") or {}).get("videoId")
            if video_id:
                collected.append(video_id)
            if len(collected) >= n:
                break
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return get_videos(_unique(collected), api_key=key, tracker=quota, session=client)
