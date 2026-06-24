"""Build the ID maps the app actually needs — run ONCE before using the app.

Why this exists
---------------
`prepare-beauty-atomic-files.ipynb` assigns each user/item an integer id by:

    user_map = {uid: i + 1 for i, uid in enumerate(sorted(set(reviews.user_id)))}
    item_map = {pid: i + 1 for i, pid in enumerate(sorted(set(reviews.parent_asin)))}

over the FULL `reviews.csv` (both categories), and those `i + 1` integers are the
tokens stored in `beauty.*.inter` / `beauty.item` (and therefore what the trained
model knows). That mapping was never saved. The pre-existing `data/user_map.json`
and `data/item_map.json` are a DIFFERENT, 0-indexed, much smaller mapping and do
NOT correspond to the model — using them breaks every lookup.

This script reproduces the exact 1-indexed enumeration and writes compact sidecar
maps restricted to the users/items that appear in the beauty dataset:

    data/beauty/app_user_map.json : { amazon_user_hash : atomic_uid }
    data/beauty/app_item_map.json : { amazon_asin       : atomic_iid }

Run:
    python app/build_id_maps.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402

REVIEWS_CSV = DATA_DIR / "reviews.csv"
BEAUTY_DIR = DATA_DIR / "beauty"
APP_USER_MAP = BEAUTY_DIR / "app_user_map.json"
APP_ITEM_MAP = BEAUTY_DIR / "app_item_map.json"

CHUNK = 2_000_000


def _sorted_unique(column: str) -> list[str]:
    """sorted(set(reviews.csv[column])) — chunked to keep peak memory down."""
    seen: set[str] = set()
    n = 0
    for chunk in pd.read_csv(REVIEWS_CSV, usecols=[column], chunksize=CHUNK):
        vals = chunk[column].dropna().astype(str).values
        seen.update(vals)
        n += len(chunk)
        print(f"  ...{column}: scanned {n:,} rows, {len(seen):,} unique", end="\r")
    print()
    return sorted(seen)


def _atomic_ids(path, col: str) -> set[int]:
    """Collect the integer tokens present in a beauty atomic file."""
    ids: set[int] = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=[col], chunksize=500_000):
        ids.update(int(x) for x in chunk[col].dropna().tolist())
    return ids


def main() -> None:
    if not REVIEWS_CSV.exists():
        raise SystemExit(f"Missing {REVIEWS_CSV}")

    # --- users -------------------------------------------------------------
    print("Reconstructing user enumeration from reviews.csv ...")
    users_sorted = _sorted_unique("user_id")          # atomic uid = index + 1
    beauty_uids = _atomic_ids(BEAUTY_DIR / "beauty.user", "user_id:token")
    user_map = {
        users_sorted[uid - 1]: uid
        for uid in beauty_uids
        if 1 <= uid <= len(users_sorted)
    }
    APP_USER_MAP.write_text(json.dumps(user_map))
    print(f"Wrote {APP_USER_MAP} ({len(user_map):,} users)")
    del users_sorted

    # --- items -------------------------------------------------------------
    print("Reconstructing item enumeration from reviews.csv ...")
    items_sorted = _sorted_unique("parent_asin")      # atomic iid = index + 1
    beauty_iids = _atomic_ids(BEAUTY_DIR / "beauty.item", "item_id:token")
    item_map = {
        items_sorted[iid - 1]: iid
        for iid in beauty_iids
        if 1 <= iid <= len(items_sorted)
    }
    APP_ITEM_MAP.write_text(json.dumps(item_map))
    print(f"Wrote {APP_ITEM_MAP} ({len(item_map):,} items)")

    # --- sanity check ------------------------------------------------------
    missing_u = len(beauty_uids) - len(user_map)
    missing_i = len(beauty_iids) - len(item_map)
    if missing_u or missing_i:
        print(f"WARNING: {missing_u} uids / {missing_i} iids had no source string "
              "(reviews.csv may differ from the file used in preprocessing).")
    else:
        print("OK: every beauty user/item resolved to an original Amazon id.")


if __name__ == "__main__":
    main()
