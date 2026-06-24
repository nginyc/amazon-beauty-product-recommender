"""Central path configuration for the webapp.

Everything is resolved relative to the repository root (the parent of /app),
so the app works regardless of the current working directory.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Raw Amazon Reviews 2023 files (the app is asked to read these directly).
REVIEWS_JSONL = DATA_DIR / "Beauty_and_Personal_Care.jsonl"
META_JSONL = DATA_DIR / "meta_Beauty_and_Personal_Care.jsonl"

# ID maps produced by preprocessing: original token -> integer used in atomic files.
USER_MAP_JSON = DATA_DIR / "user_map.json"   # amazon_user_hash -> int
ITEM_MAP_JSON = DATA_DIR / "item_map.json"   # amazon_asin      -> int

# Default model checkpoint the adaptor is preset to load.
DEFAULT_MODEL_PATH = REPO_ROOT / "train" / "saved" / "LightGCN-Jun-23-2026_23-36-49.pth"
