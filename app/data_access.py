"""Live, on-demand readers for the two large jsonl files.

Both files are multi-GB, so we stream line by line and use a cheap substring
pre-filter (`needle in line`) before paying for `json.loads`. Results are cached
by the Streamlit layer, so a given user / item set is only scanned once per run.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, Optional

from paths import REVIEWS_JSONL, META_JSONL


def scan_user_reviews(
    amazon_hash: str,
    max_lines: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> list[dict]:
    """Return all reviews written by `amazon_hash`, newest first.

    Each record: {asin, rating, title, text, timestamp, helpful_vote, verified}.
    `max_lines` caps how far into the file we read (None = whole file).
    """
    if not amazon_hash:
        return []

    out: list[dict] = []
    with open(REVIEWS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if progress is not None and (i & 0x3FFFF) == 0:  # every ~256k lines
                progress(i)
            # Cheap reject: the 28-char hash is effectively unique in a line.
            if amazon_hash not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("user_id") != amazon_hash:
                continue
            out.append(
                {
                    "asin": d.get("parent_asin") or d.get("asin"),
                    "rating": d.get("rating"),
                    "title": d.get("title", ""),
                    "text": d.get("text", ""),
                    "timestamp": d.get("timestamp", 0),
                    "helpful_vote": d.get("helpful_vote", 0),
                    "verified": d.get("verified_purchase", False),
                }
            )

    out.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    return out


def _first_image_url(images: list[dict]) -> Optional[str]:
    """Pick a displayable image URL from a meta `images` list."""
    if not images:
        return None
    for key in ("large", "hi_res", "thumb"):
        for img in images:
            url = img.get(key)
            if url:
                return url
    return None


def fetch_item_meta(
    asins: Iterable[str],
    max_lines: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> dict[str, dict]:
    """Return {asin: {title, image, average_rating, store, price}} for `asins`.

    Stops early once every requested ASIN has been found.
    """
    needed = {a for a in asins if a}
    if not needed:
        return {}

    found: dict[str, dict] = {}
    with open(META_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if not needed:
                break
            if progress is not None and (i & 0x3FFFF) == 0:
                progress(i)
            # Cheap reject: skip lines that mention none of the wanted ASINs.
            if not any(a in line for a in needed):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = d.get("parent_asin")
            if asin not in needed:
                continue
            found[asin] = {
                "title": d.get("title", ""),
                "image": _first_image_url(d.get("images", [])),
                "average_rating": d.get("average_rating"),
                "rating_number": d.get("rating_number"),
                "store": d.get("store"),
                "price": d.get("price"),
            }
            needed.discard(asin)

    return found
