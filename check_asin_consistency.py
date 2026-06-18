# ======================================================================
# ASIN / parent_asin consistency checks
# Dataset: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace hub)
# Date window: 2020-01-01 .. 2022-12-31 (inclusive), UTC
#
# Designed for Google Colab. Paste each "CELL" into its own cell,
# or just run the whole file.
# ======================================================================

# ----------------------------------------------------------------------
# CELL 1 — install deps
# ----------------------------------------------------------------------
# !pip install -q -U datasets

# ----------------------------------------------------------------------
# CELL 2 — config
# ----------------------------------------------------------------------
from datasets import load_dataset
from datetime import datetime, timezone

# Pick the category you are analysing. Examples:
#   "All_Beauty", "Amazon_Fashion", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"
CATEGORY = "All_Beauty"

# Inclusive date window. Amazon-Reviews-2023 timestamps are UNIX milliseconds.
START = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END   = datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
START_MS = int(START.timestamp() * 1000)
END_MS   = int(END.timestamp() * 1000) + 999  # include the final second fully

EXPECTED = 33  # the number we are testing against

print(f"Category : {CATEGORY}")
print(f"Window   : {START.date()} .. {END.date()}  (UTC)")
print(f"Window ms: {START_MS} .. {END_MS}")


def to_ms(ts):
    """Normalise a timestamp to milliseconds (handles s / ms / us)."""
    ts = int(ts)
    digits = len(str(abs(ts)))
    if digits <= 11:      # seconds
        return ts * 1000
    if digits >= 15:      # microseconds
        return ts // 1000
    return ts             # milliseconds


# ----------------------------------------------------------------------
# CELL 3 — stream the REVIEWS table, filter by date, collect stats
# ----------------------------------------------------------------------
# Streaming avoids downloading the whole category into RAM.
reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    f"raw_review_{CATEGORY}",
    split="full",
    streaming=True,
    trust_remote_code=True,
)

review_asins = set()          # unique asin in window
review_parent_asins = set()   # unique parent_asin in window
n_rows = 0                    # rows in window
n_match = 0                  # rows where asin == parent_asin
n_total_seen = 0             # rows scanned (any date)

for r in reviews:
    n_total_seen += 1
    ts = r.get("timestamp")
    if ts is None:
        continue
    ms = to_ms(ts)
    if ms < START_MS or ms > END_MS:
        continue

    asin = r.get("asin")
    pasin = r.get("parent_asin")
    n_rows += 1
    if asin is not None:
        review_asins.add(asin)
    if pasin is not None:
        review_parent_asins.add(pasin)
    if asin == pasin:
        n_match += 1

    if n_total_seen % 500_000 == 0:
        print(f"  scanned {n_total_seen:,} rows... (in-window so far: {n_rows:,})")

print(f"\nScanned {n_total_seen:,} review rows total; {n_rows:,} fall in the window.")


# ----------------------------------------------------------------------
# CELL 4 — Q1: are there 33 unique `asin` in the reviews window? What are they?
# ----------------------------------------------------------------------
print("=" * 60)
print("Q1: unique `asin` in user reviews (2020-01-01 .. 2022-12-31)")
print("=" * 60)
print(f"Unique asin count : {len(review_asins)}")
print(f"Is it {EXPECTED}?      : {len(review_asins) == EXPECTED}")
print("The unique asin values:")
for a in sorted(review_asins):
    print(f"  {a}")


# ----------------------------------------------------------------------
# CELL 5 — stream the ITEM METADATA table, collect parent_asin
# ----------------------------------------------------------------------
# Metadata has NO timestamp, so the date window cannot apply to it directly.
# We compare against the parent_asins observed in the windowed reviews.
meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    f"raw_meta_{CATEGORY}",
    split="full",
    streaming=True,
    trust_remote_code=True,
)

meta_parent_asins = set()                 # all parent_asin in metadata
meta_parent_in_window = set()             # metadata parent_asin that also appear in windowed reviews

for m in meta:
    p = m.get("parent_asin")
    if p is None:
        continue
    meta_parent_asins.add(p)
    if p in review_parent_asins:
        meta_parent_in_window.add(p)

print(f"Total unique parent_asin in metadata table: {len(meta_parent_asins):,}")


# ----------------------------------------------------------------------
# CELL 6 — Q2: 33 unique parent_asin in reviews AND metadata? Same as the asin set?
# ----------------------------------------------------------------------
print("=" * 60)
print("Q2: unique `parent_asin` in reviews & metadata, vs the asin set")
print("=" * 60)
print(f"Unique parent_asin in WINDOWED REVIEWS        : {len(review_parent_asins)}  (== {EXPECTED}? {len(review_parent_asins) == EXPECTED})")
print(f"Of those, present in the METADATA table       : {len(meta_parent_in_window)}  (== {EXPECTED}? {len(meta_parent_in_window) == EXPECTED})")

missing_in_meta = review_parent_asins - meta_parent_asins
if missing_in_meta:
    print(f"  parent_asin in reviews but MISSING from metadata: {sorted(missing_in_meta)}")
else:
    print("  every windowed-review parent_asin exists in the metadata table.")

same_as_asin = review_parent_asins == review_asins
print(f"\nIs the parent_asin set IDENTICAL to the asin set? {same_as_asin}")
if not same_as_asin:
    only_asin = review_asins - review_parent_asins
    only_parent = review_parent_asins - review_asins
    print(f"  only in asin set        : {sorted(only_asin)}")
    print(f"  only in parent_asin set : {sorted(only_parent)}")


# ----------------------------------------------------------------------
# CELL 7 — Q3: does EVERY windowed review row have asin == parent_asin?
# ----------------------------------------------------------------------
print("=" * 60)
print("Q3: do all windowed review rows have asin == parent_asin?")
print("=" * 60)
print(f"Windowed review rows           : {n_rows:,}")
print(f"Rows with asin == parent_asin  : {n_match:,}")
print(f"All rows match?                : {n_match == n_rows}")
if n_match != n_rows:
    print(f"  rows where they DIFFER       : {n_rows - n_match:,}")
