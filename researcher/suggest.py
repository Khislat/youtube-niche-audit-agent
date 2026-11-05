"""Niche phrase suggestions from outlier video titles and tags (no LLM)."""

from __future__ import annotations

import re
from collections import defaultdict

from flags import load_keywords

_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _niche_tokens(niche: str) -> set[str]:
    return set(_normalize_tokens(niche))


def _clean_phrase(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    if any(len(token) <= 3 for token in tokens):
        return None
    return " ".join(tokens)


def _ngrams(tokens: list[str], size: int) -> list[str]:
    if len(tokens) < size:
        return []
    phrases: list[str] = []
    for index in range(len(tokens) - size + 1):
        chunk = tokens[index : index + size]
        phrase = _clean_phrase(chunk)
        if phrase:
            phrases.append(phrase)
    return phrases


def _video_text(video: dict) -> tuple[str, str]:
    title = (video.get("title") or "").strip()
    tags = video.get("tags") or []
    tag_text = " ".join(str(tag) for tag in tags if tag)
    return title, tag_text


def build_suggestions(
    top_videos: list[dict],
    *,
    niche: str,
    min_channels: int = 2,
    max_results: int = 20,
    exclude_niches: set[str] | None = None,
    keywords: dict | None = None,
) -> list[dict]:
    """Return up to `max_results` phrases seen on at least `min_channels` outlier channels."""
    data = keywords or load_keywords()
    stop = set(data.get("stopwords") or [])
    stop.update(_niche_tokens(niche))
    exclude = {item.strip().lower() for item in (exclude_niches or set()) if item.strip()}

    # channel_id -> set of phrases seen on that channel (count once per channel)
    channel_phrases: dict[str, set[str]] = defaultdict(set)
    phrase_example: dict[str, str] = {}

    for video in top_videos:
        channel_id = video.get("channel_id") or ""
        if not channel_id:
            continue
        title, tag_text = _video_text(video)
        tokens: list[str] = []
        for source in (title, tag_text):
            for token in _normalize_tokens(source):
                if token in stop:
                    continue
                tokens.append(token)
        phrases: set[str] = set()
        for size in (1, 2, 3):
            phrases.update(_ngrams(tokens, size))
        for phrase in phrases:
            channel_phrases[channel_id].add(phrase)
            phrase_example.setdefault(phrase, title or phrase)

    counts: dict[str, int] = defaultdict(int)
    for phrases in channel_phrases.values():
        for phrase in phrases:
            counts[phrase] += 1

    rows: list[dict] = []
    for phrase, channel_count in counts.items():
        if channel_count < min_channels:
            continue
        if phrase in exclude:
            continue
        rows.append(
            {
                "term": phrase,
                "channel_count": channel_count,
                "example_title": phrase_example.get(phrase, ""),
            }
        )
    rows.sort(key=lambda row: (-row["channel_count"], row["term"]))
    return rows[:max_results]
