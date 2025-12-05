from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "Daily_AI_Assistant_Usage_Behavior_Dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, METRICS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

TARGET_COL = "satisfaction_rating"

# Base columns in raw data
TIME_COL = "timestamp"

CATEGORICAL_FEATURES = [
    "device",
    "usage_category",
    "assistant_model",
]

NUMERIC_FEATURES = [
    "prompt_length",
    "session_length_minutes",
    "tokens_used",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42
