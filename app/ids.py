"""User / item ID translation.

The raw jsonl files key on the original Amazon strings:
  - user_id : a long hash, e.g. "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ"
  - parent_asin / asin : e.g. "B00Z03RC80"

Preprocessing (`prepare-beauty-atomic-files.ipynb`) remapped each to an integer
via `{x: i + 1 for i, x in enumerate(sorted(set(...)))}` over the full reviews.csv,
and those integers are the tokens stored in the RecBole atomic files. RecBole then
assigns its own internal contiguous ids on top of the tokens.

That preprocessing mapping was never saved (and the pre-existing data/user_map.json
/ data/item_map.json are a *different*, 0-indexed map that does NOT match the model).
So we rely on the sidecar maps produced by `app/build_id_maps.py`:

    data/beauty/app_user_map.json : { amazon_user_hash : atomic_uid }
    data/beauty/app_item_map.json : { amazon_asin       : atomic_iid }

Run `python app/build_id_maps.py` once to create them.
"""
from __future__ import annotations

import functools
import json
import random
from typing import Optional

from paths import DATA_DIR

APP_USER_MAP = DATA_DIR / "beauty" / "app_user_map.json"
APP_ITEM_MAP = DATA_DIR / "beauty" / "app_item_map.json"


class MapsMissing(RuntimeError):
    """Raised when the sidecar id maps haven't been built yet."""


def maps_ready() -> bool:
    return APP_USER_MAP.exists() and APP_ITEM_MAP.exists()


def _require(path) -> None:
    if not path.exists():
        raise MapsMissing(
            f"Missing {path.name}. Build the id maps once with:\n"
            f"    python app/build_id_maps.py"
        )


@functools.lru_cache(maxsize=1)
def _user_map() -> dict[str, int]:  # amazon hash -> atomic uid
    _require(APP_USER_MAP)
    with open(APP_USER_MAP) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _item_map() -> dict[str, int]:  # amazon asin -> atomic iid
    _require(APP_ITEM_MAP)
    with open(APP_ITEM_MAP) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _int_to_hash() -> dict[int, str]:
    return {v: k for k, v in _user_map().items()}


@functools.lru_cache(maxsize=1)
def _int_to_asin() -> dict[int, str]:
    return {v: k for k, v in _item_map().items()}


def sample_user_ints(pool: int = 2000) -> list[int]:
    """Random atomic-integer user ids for the 'Random user' button."""
    values = list(_user_map().values())
    if not values:
        return [0]
    return random.sample(values, min(pool, len(values)))


def token_to_asin(token: str) -> Optional[str]:
    """RecBole/atomic item token (integer string) -> original Amazon ASIN."""
    try:
        return _int_to_asin().get(int(token))
    except (TypeError, ValueError):
        return None


def hash_for_mapped_int(mapped_int: int) -> Optional[str]:
    """Atomic integer user id -> original Amazon user hash."""
    return _int_to_hash().get(mapped_int)


def candidate_user_tokens(identifier: str) -> list[str]:
    """RecBole user token(s) to try, supporting BOTH input forms.

      - an original Amazon user hash  -> looked up in app_user_map.json
      - an atomic integer id          -> used directly as the token
    """
    identifier = identifier.strip()
    tokens: list[str] = []
    if identifier in _user_map():            # Amazon hash
        tokens.append(str(_user_map()[identifier]))
    if identifier.isdigit():                 # already an atomic integer / token
        tokens.append(identifier)
    seen: set[str] = set()
    return [t for t in tokens if not (t in seen or seen.add(t))]


def amazon_hash_for(identifier: str, token: str | None) -> Optional[str]:
    """Resolve the Amazon user hash used to scan the reviews file."""
    identifier = identifier.strip()
    if identifier in _user_map():
        return identifier
    if token is not None:
        try:
            return hash_for_mapped_int(int(token))
        except (TypeError, ValueError):
            return None
    return None
