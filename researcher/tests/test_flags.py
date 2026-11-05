from flags import channel_flag_metrics, load_keywords, scan, video_flags


def test_scan_ai_and_brand():
    assert scan(["Made with Suno"]) == {"ai": ["suno"], "brand": []}
    assert scan(["audio mix"]) == {"ai": [], "brand": []}
    assert scan(["Harry Potter Ambience"]) == {"ai": [], "brand": ["harry potter"]}
    assert scan(["HARRY potter"]) == {"ai": [], "brand": ["harry potter"]}


def test_channel_and_video_flags():
    uploads = [
        {
            "title": "Made with Suno",
            "description_snippet": "calm",
            "tags": ["music"],
        },
        {
            "title": "Plain",
            "description_snippet": "",
            "tags": [],
        },
    ]
    metrics = channel_flag_metrics(uploads)
    assert metrics["ai_flag_share"] == 0.5
    assert metrics["brand_flag_share"] == 0.0
    assert "suno" in metrics["ai_terms"]

    flags = video_flags(uploads[0])
    assert flags["ai_terms"] == "suno"


def test_keywords_file_loads():
    data = load_keywords()
    assert "ai_terms" in data
    assert "brand_terms" in data
    assert "stopwords" in data
