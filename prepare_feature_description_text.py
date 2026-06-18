#!/usr/bin/env python3
"""
Profile and preprocess Amazon metadata `features` and `description` text.

This script streams a large JSONL file line by line, so it is suitable for
multi-GB metadata files on a laptop.
"""

import argparse
import csv
import html
import json
import re
from pathlib import Path


DEFAULT_INPUT = (
    "/Users/frankwang1224/Projects/rcd_sys_proj02/"
    "dataset/meta_Beauty_and_Personal_Care.jsonl"
)


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


PREPROCESSING_PLAN = """
Preprocessing plan before embedding:

1. Read the JSONL file row by row.
   - Each line is one item metadata record.
   - Use `parent_asin` as the item id.

2. Extract only `features` and `description`.
   - In this file, both fields are usually lists of strings.
   - `features` often contains product bullet points.
   - `description` is often empty, but when present can be long.

3. Normalize each text string lightly.
   - Convert HTML entities, e.g. `&amp;` -> `&`.
   - Remove HTML tags if any.
   - Collapse repeated whitespace.
   - Keep natural words, numbers, units, brands, and product terms.

4. Drop empty strings and duplicate bullets inside the same item.
   - Some product metadata repeats information.
   - Deduplication reduces repeated signal before embedding.

5. Preserve field meaning with labels.
   - Build text like:
     `Features: ... Description: ...`
   - Labels help the embedding model understand where the text came from.

6. Control length before embedding.
   - Keep the first N feature bullets.
   - Truncate very long descriptions by character length or by tokenizer later.
   - Final truncation should still be done by the embedding model tokenizer.

7. Skip items with no usable text.
   - If both `features` and `description` are empty, there is no text embedding
     input for that item from these two fields.

8. Save a clean text file for embedding.
   - Output columns:
     parent_asin, features_text, description_text, embedding_text
""".strip()


def normalize_text(value: object) -> str:
    """Light text cleanup while preserving product meaning."""
    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def field_to_list(value: object) -> list[str]:
    """Convert metadata field values into a list of clean strings."""
    if value is None:
        return []

    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    cleaned = []
    seen = set()

    for raw in raw_values:
        text = normalize_text(raw)
        if not text:
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

    return cleaned


def truncate_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def build_embedding_text(
    features: list[str],
    description: list[str],
    max_feature_bullets: int,
    max_description_chars: int,
) -> tuple[str, str, str]:
    features = features[:max_feature_bullets]
    features_text = "; ".join(features)

    description_text = " ".join(description)
    description_text = truncate_chars(description_text, max_description_chars)

    parts = []
    if features_text:
        parts.append(f"Features: {features_text}")
    if description_text:
        parts.append(f"Description: {description_text}")

    embedding_text = " ".join(parts)
    return features_text, description_text, embedding_text


def percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        default="/Users/frankwang1224/Projects/rcd_sys_proj02/"
        "feature_description_embedding_input.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick tests. Omit for full file.",
    )
    parser.add_argument("--max-feature-bullets", type=int, default=8)
    parser.add_argument("--max-description-chars", type=int, default=1500)
    parser.add_argument(
        "--write-limit",
        type=int,
        default=None,
        help="Optional cap on output rows. Omit to write all usable rows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(PREPROCESSING_PLAN)
    print("\n" + "=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    total_rows = 0
    rows_with_features = 0
    rows_with_description = 0
    rows_with_either = 0
    rows_written = 0
    bad_json_rows = 0
    feature_counts = []
    description_counts = []
    embedding_char_lengths = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "parent_asin",
                "features_text",
                "description_text",
                "embedding_text",
            ],
        )
        writer.writeheader()

        for line in fin:
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

            total_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json_rows += 1
                continue

            parent_asin = row.get("parent_asin")
            features = field_to_list(row.get("features"))
            description = field_to_list(row.get("description"))

            if features:
                rows_with_features += 1
            if description:
                rows_with_description += 1
            if features or description:
                rows_with_either += 1

            feature_counts.append(len(features))
            description_counts.append(len(description))

            features_text, description_text, embedding_text = build_embedding_text(
                features=features,
                description=description,
                max_feature_bullets=args.max_feature_bullets,
                max_description_chars=args.max_description_chars,
            )

            if not parent_asin or not embedding_text:
                continue

            if args.write_limit is not None and rows_written >= args.write_limit:
                continue

            embedding_char_lengths.append(len(embedding_text))
            writer.writerow(
                {
                    "parent_asin": parent_asin,
                    "features_text": features_text,
                    "description_text": description_text,
                    "embedding_text": embedding_text,
                }
            )
            rows_written += 1

            if total_rows % 100_000 == 0:
                print(f"Processed {total_rows:,} rows; wrote {rows_written:,} rows")

    feature_counts.sort()
    description_counts.sort()
    embedding_char_lengths.sort()

    print("\n" + "=" * 80)
    print("Profile summary")
    print("=" * 80)
    print(f"Rows scanned:                 {total_rows:,}")
    print(f"Bad JSON rows:                {bad_json_rows:,}")
    print(f"Rows with non-empty features: {rows_with_features:,}")
    print(f"Rows with description:        {rows_with_description:,}")
    print(f"Rows with either field:       {rows_with_either:,}")
    print(f"Rows written for embedding:   {rows_written:,}")

    print("\nFeature bullet count per item:")
    print(f"  p50={percentile(feature_counts, 0.50)}")
    print(f"  p90={percentile(feature_counts, 0.90)}")
    print(f"  p99={percentile(feature_counts, 0.99)}")

    print("\nDescription entry count per item:")
    print(f"  p50={percentile(description_counts, 0.50)}")
    print(f"  p90={percentile(description_counts, 0.90)}")
    print(f"  p99={percentile(description_counts, 0.99)}")

    print("\nEmbedding text character length after preprocessing:")
    print(f"  p50={percentile(embedding_char_lengths, 0.50)}")
    print(f"  p90={percentile(embedding_char_lengths, 0.90)}")
    print(f"  p99={percentile(embedding_char_lengths, 0.99)}")

    print("\nNext step:")
    print(
        "Feed `embedding_text` into your chosen text embedding model, using that "
        "model's tokenizer for final token-level truncation."
    )


if __name__ == "__main__":
    main()
