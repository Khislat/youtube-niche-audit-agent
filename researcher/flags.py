"""Keyword scanner for AI and brand mentions in titles, descriptions, and tags."""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import KEYWORDS_PATH

_TERM_CACHE: dict | None = None


def load_keywords(path: Path | None = None) -> dict:
    global _TERM_CACHE
    if _TERM_CACHE is not None and path is None:
        return _TERM_CACHE
    source = path or KEYWORDS_PATH
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    _TERM_CACHE = data
    return data


def _compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for term in terms:
        text = term.strip().lower()
        if not text:
            continue
        pattern = re.compile(r"\b" + re.escape(text) + r"\b", re.IGNORECASE)
        compiled.append((text, pattern))
    return compiled


def _text_parts(parts) -> str:
    chunks: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            chunks.extend(str(item) for item in part if item)
        else:
            text = str(part).strip()
            if text:
                chunks.append(text)
    return " ".join(chunks)


def scan(text_parts, *, keywords: dict | None = None) -> dict[str, list[str]]:
    """Return matched ai and brand terms (whole words/phrases only)."""
    blob = _text_parts(text_parts)
    if not blob:
        return {"ai": [], "brand": []}
    data = keywords or load_keywords()
    ai_hits: list[str] = []
    brand_hits: list[str] = []
    for label, raw_terms in (("ai", data.get("ai_terms") or []), ("brand", data.get("brand_terms") or [])):
        seen: set[str] = set()
        for term, pattern in _compile_terms(raw_terms):
            if pattern.search(blob) and term not in seen:
                seen.add(term)
                if label == "ai":
                    ai_hits.append(term)
                else:
                    brand_hits.append(term)
    return {"ai": ai_hits, "brand": brand_hits}


def join_terms(terms: list[str], limit: int = 5) -> str:
    unique: list[str] = []
    for term in terms:
        if term not in unique:
            unique.append(term)
        if len(unique) >= limit:
            break
    return ", ".join(unique)


def channel_flag_metrics(uploads: list[dict], *, keywords: dict | None = None) -> dict:
    """Share of uploads in the sample that mention AI or brand terms."""
    if not uploads:
        return {
            "ai_flag_share": None,
            "brand_flag_share": None,
            "ai_terms": "",
            "brand_terms": "",
        }
    ai_count = 0
    brand_count = 0
    ai_terms: list[str] = []
    brand_terms: list[str] = []
    for video in uploads:
        tags = video.get("tags") or []
        hits = scan(
            [video.get("title"), video.get("description_snippet"), tags],
            keywords=keywords,
        )
        if hits["ai"]:
            ai_count += 1
            ai_terms.extend(hits["ai"])
        if hits["brand"]:
            brand_count += 1
            brand_terms.extend(hits["brand"])
    total = len(uploads)
    return {
        "ai_flag_share": ai_count / total,
        "brand_flag_share": brand_count / total,
        "ai_terms": join_terms(ai_terms),
        "brand_terms": join_terms(brand_terms),
    }


def video_flags(video: dict, *, keywords: dict | None = None) -> dict[str, str]:
    hits = scan(
        [video.get("title"), video.get("description_snippet"), video.get("tags") or []],
        keywords=keywords,
    )
    return {
        "ai_terms": join_terms(hits["ai"]),
        "brand_terms": join_terms(hits["brand"]),
    }
