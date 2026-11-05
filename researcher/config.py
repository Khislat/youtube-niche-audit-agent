"""Default settings for the niche researcher."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "data" / "researcher.db"
DEFAULT_OUTPUT_DIR = ROOT / "output"
ENV_PATH = ROOT / ".env"

DEFAULT_DAYS = 180
DEFAULT_MAX_AGE = 365
DEFAULT_MIN_AVG = 5000
# Raised from 1000 so one lucky video cannot clear the median floor.
DEFAULT_MIN_MEDIAN = 3000
# median_views / avg_views. Below this, views are too concentrated to be a steady outlier.
DEFAULT_MIN_MEDIAN_AVG_RATIO = 0.15
DEFAULT_MAX_TOP_SHARE = 0.6
DEFAULT_PAGES = 1
DEFAULT_MAX_VIDEOS_PER_CHANNEL = 50
DEFAULT_REGION = "US"
DEFAULT_INTEREST = 3

# Phase 1.2 batch and suggestions
DEFAULT_MAX_QUOTA = 8000
ESTIMATED_QUOTA_PER_NICHE = 250
DEFAULT_SUGGEST_MIN_CHANNELS = 2
DEFAULT_AUTO_SUGGEST = 0
TOP_OUTLIER_VIDEOS = 5
MAX_SUGGESTIONS = 20
KEYWORDS_PATH = ROOT / "keywords.json"
CSV_FLAG_DISCLAIMER = (
    "Izoh: belgi yo'qligi AI yoki brend ishlatilmaganini isbotlamaydi; "
    "ko'p kanallar buni yozmaydi."
)

# Phase 2 score weights. Every criterion starts with an equal weight.
SCORE_WEIGHTS = {
    "demand": 1.0,
    "supply_gap": 1.0,
    "ai_feasibility": 1.0,
    "differentiation": 1.0,
    "interest": 1.0,
}
