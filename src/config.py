from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

START_DATE = "2000-01-01"
DEFAULT_LAGS = [1, 2, 3]
DEFAULT_ROLLING = [3, 6]
DEFAULT_SPLITS = 5
SEED = 42