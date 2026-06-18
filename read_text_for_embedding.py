"""
Read 'features' and 'description' text from the Amazon meta JSONL file,
in preparation for text embedding with bge-base-en-v1.5.

Usage:
    python read_text_for_embedding.py

Notes:
- The .jsonl file is one JSON object per line (~2.8 GB), so we STREAM it
  line-by-line instead of loading the whole thing into memory.
- In this dataset, BOTH 'features' and 'description' are LISTS OF STRINGS
  (often empty: [] ). We join each list into a single text block.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(
    "/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/"
    "meta_Beauty_and_Personal_Care.jsonl"
)
OUTPUT_PATH = Path(
    "/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/"
    "beauty_text_for_embedding.jsonl"
)

# Set to an int (e.g. 5000) for a quick try; set to None to process everything.
LIMIT = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_to_text(value) -> str:
    """Both 'features' and 'description' are lists of strings here.
    Join them into one string. Handle the rare case where it's already
    a plain string or None."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return "\n".join(parts)
    return str(value).strip()


def iter_records(path: Path):
    """Yield parsed JSON objects from a JSONL file, one line at a time."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines but don't crash the whole run.
                print(f"  [warn] skipped malformed JSON on line {line_no}")
                continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    total = 0
    kept = 0
    feat_empty = 0
    desc_empty = 0
    both_empty = 0

    print(f"Reading: {DATA_PATH}")
    print(f"Writing: {OUTPUT_PATH}\n")

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for obj in iter_records(DATA_PATH):
            total += 1

            features_text = list_to_text(obj.get("features"))
            description_text = list_to_text(obj.get("description"))

            if not features_text:
                feat_empty += 1
            if not description_text:
                desc_empty += 1
            if not features_text and not description_text:
                both_empty += 1
                # Nothing to embed for this product -> skip it.
                if LIMIT is not None and total >= LIMIT:
                    break
                continue

            # Combined field is what you'll typically feed to the embedder.
            combined_text = "\n".join(
                t for t in (features_text, description_text) if t
            )

            record = {
                "parent_asin": obj.get("parent_asin"),
                "title": obj.get("title"),
                "features_text": features_text,
                "description_text": description_text,
                "combined_text": combined_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if LIMIT is not None and total >= LIMIT:
                break

    # ---- Summary -----------------------------------------------------------
    print("Done.")
    print(f"  records read         : {total}")
    print(f"  records kept (output): {kept}")
    print(f"  features empty       : {feat_empty} ({100*feat_empty/total:.1f}%)")
    print(f"  description empty    : {desc_empty} ({100*desc_empty/total:.1f}%)")
    print(f"  both empty (skipped) : {both_empty} ({100*both_empty/total:.1f}%)")

    # Peek at the first kept record so you can eyeball the text.
    print("\nFirst kept record preview:")
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        first = f.readline()
        if first:
            rec = json.loads(first)
            print("  title          :", (rec["title"] or "")[:80])
            print("  features_text  :", rec["features_text"][:200].replace("\n", " ⏎ "))
            print("  description    :", rec["description_text"][:200].replace("\n", " ⏎ "))


if __name__ == "__main__":
    main()
