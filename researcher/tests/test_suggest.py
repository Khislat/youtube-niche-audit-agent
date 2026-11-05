from suggest import build_suggestions


def _video(channel_id, title, tags=None):
    return {
        "channel_id": channel_id,
        "title": title,
        "tags": tags or [],
    }


def test_suggest_counts_channels_not_occurrences():
    videos = [
        _video("a", "cozy dragon fire ambience", ["dragon"]),
        _video("a", "cozy dragon fire ambience again", ["dragon"]),
        _video("b", "cozy dragon fire mix", ["dragon"]),
        _video("c", "other music", []),
    ]
    rows = build_suggestions(
        videos,
        niche="cozy fantasy ambience",
        min_channels=2,
        exclude_niches=set(),
    )
    terms = {row["term"]: row["channel_count"] for row in rows}
    assert terms.get("dragon") == 2
    assert "music" not in terms
    assert "cozy" not in terms
    assert "fantasy" not in terms


def test_suggest_requires_min_channels():
    videos = [
        _video("a", "alpha beta gamma delta"),
        _video("b", "epsilon zeta eta theta"),
    ]
    rows = build_suggestions(videos, niche="alpha", min_channels=2, exclude_niches=set())
    assert rows == []
