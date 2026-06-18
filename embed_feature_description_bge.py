#!/usr/bin/env python3
"""
Preprocess Beauty_and_Personal_Care feature/description text and embed with BGE.

Default input:
  dataset/meta_Beauty_and_Personal_Care.jsonl

Default output:
  embeddings/text_bge_base_en_v1_5/text_embeddings_Beauty_and_Personal_Care_*.parquet

Install dependencies first:
  python3 -m pip install torch sentence-transformers pandas pyarrow tqdm
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


PROJECT_DIR = Path("/Users/frankwang1224/Projects/rcd_sys_proj02")
DEFAULT_INPUT = PROJECT_DIR / "dataset/meta_Beauty_and_Personal_Care.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "embeddings/text_bge_base_en_v1_5"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|;\s+|\n+")

SERVICE_NOISE_RE = re.compile(
    r"\b("
    r"refund|return|replacement|warranty|guarantee|risk free|"
    r"no questions asked|customer service|after[- ]?sales|"
    r"contact us|contact seller|feel free to contact|"
    r"shipping|delivery|amazon fba|fba|prime|"
    r"satisfaction guaranteed|money back"
    r")\b",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """Normalize fancy Unicode letters and remove emoji/decorative symbols."""
    text = unicodedata.normalize("NFKC", text)
    kept = []

    for char in text:
        category = unicodedata.category(char)

        # Drop emoji/decorative symbols and control chars.
        if category in {"So", "Sk", "Cc", "Cf"}:
            kept.append(" ")
            continue

        kept.append(char)

    return "".join(kept)


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = normalize_unicode(text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def value_to_clean_list(value: object) -> list[str]:
    """Convert a JSON field into a de-duplicated list of clean strings."""
    if value is None:
        return []

    values = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    seen = set()

    for raw in values:
        text = clean_text(raw)
        if not text:
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

    return cleaned


def remove_service_noise(text: str) -> str:
    """Remove sentences/bullets about refund, shipping, warranty, contact, etc."""
    pieces = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    kept = [p for p in pieces if not SERVICE_NOISE_RE.search(p)]
    return SPACE_RE.sub(" ", "; ".join(kept)).strip()


def truncate_on_word(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def build_embedding_text(
    features: list[str],
    description: list[str],
    max_feature_bullets: int,
    max_description_chars: int,
    max_total_chars: int,
) -> str:
    clean_features = []
    for item in features[:max_feature_bullets]:
        item = remove_service_noise(item)
        if item:
            clean_features.append(item)

    description_text = remove_service_noise(" ".join(description))
    description_text = truncate_on_word(description_text, max_description_chars)

    parts = []
    if clean_features:
        parts.append("Features: " + "; ".join(clean_features))
    if description_text:
        parts.append("Description: " + description_text)

    text = SPACE_RE.sub(" ", " ".join(parts)).strip()
    return truncate_on_word(text, max_total_chars)


def iter_preprocessed_rows(
    input_path: Path,
    max_rows: int | None,
    max_feature_bullets: int,
    max_description_chars: int,
    max_total_chars: int,
) -> Iterable[dict[str, str]]:
    scanned = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if max_rows is not None and scanned >= max_rows:
                break

            scanned += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            parent_asin = row.get("parent_asin")
            if not parent_asin:
                continue

            features = value_to_clean_list(row.get("features"))
            description = value_to_clean_list(row.get("description"))

            embedding_text = build_embedding_text(
                features=features,
                description=description,
                max_feature_bullets=max_feature_bullets,
                max_description_chars=max_description_chars,
                max_total_chars=max_total_chars,
            )

            if not embedding_text:
                continue

            yield {
                "parent_asin": str(parent_asin),
                "embedding_text": embedding_text,
            }


def write_chunk(
    records: list[dict[str, object]],
    output_dir: Path,
    category: str,
    chunk_index: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"text_embeddings_{category}_{chunk_index:05d}.parquet"
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--category", default="Beauty_and_Personal_Care")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-feature-bullets", type=int, default=8)
    parser.add_argument("--max-description-chars", type=int, default=1_000)
    parser.add_argument("--max-total-chars", type=int, default=2_000)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional: cuda, mps, or cpu. If omitted, sentence-transformers chooses.",
    )
    args = parser.parse_args()

    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model:      {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max rows:   {args.max_rows}")

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = 512

    pending_texts: list[str] = []
    pending_ids: list[str] = []
    output_records: list[dict[str, object]] = []
    chunk_index = 0
    embedded_count = 0
    written_files: list[Path] = []

    rows = iter_preprocessed_rows(
        input_path=args.input,
        max_rows=args.max_rows,
        max_feature_bullets=args.max_feature_bullets,
        max_description_chars=args.max_description_chars,
        max_total_chars=args.max_total_chars,
    )

    progress_total = args.max_rows if args.max_rows is not None else None
    progress = tqdm(rows, total=progress_total, desc="preprocess/embed")

    def flush_batch() -> None:
        nonlocal pending_texts, pending_ids, output_records, chunk_index
        nonlocal embedded_count, written_files

        if not pending_texts:
            return

        embeddings = model.encode(
            pending_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        for parent_asin, text, emb in zip(pending_ids, pending_texts, embeddings):
            output_records.append(
                {
                    "parent_asin": parent_asin,
                    "text_embedding": emb.tolist(),
                    "embedding_text": text,
                    "embedding_model": args.model,
                    "embedding_dim": int(emb.shape[0]),
                }
            )

        embedded_count += len(pending_texts)
        pending_texts = []
        pending_ids = []

        if len(output_records) >= args.chunk_size:
            path = write_chunk(
                records=output_records,
                output_dir=args.output_dir,
                category=args.category,
                chunk_index=chunk_index,
            )
            written_files.append(path)
            print(f"Wrote {path} ({len(output_records):,} rows)")
            output_records = []
            chunk_index += 1

    for row in progress:
        pending_ids.append(row["parent_asin"])
        pending_texts.append(row["embedding_text"])

        if len(pending_texts) >= args.batch_size:
            flush_batch()
            progress.set_postfix(embedded=f"{embedded_count:,}")

    flush_batch()

    if output_records:
        path = write_chunk(
            records=output_records,
            output_dir=args.output_dir,
            category=args.category,
            chunk_index=chunk_index,
        )
        written_files.append(path)
        print(f"Wrote {path} ({len(output_records):,} rows)")

    manifest = {
        "input": str(args.input),
        "model": args.model,
        "category": args.category,
        "embedding_dim": 768,
        "normalized_embeddings": True,
        "max_seq_length": model.max_seq_length,
        "max_feature_bullets": args.max_feature_bullets,
        "max_description_chars": args.max_description_chars,
        "max_total_chars": args.max_total_chars,
        "rows_embedded": embedded_count,
        "num_output_files": len(written_files),
        "output_files": [str(path) for path in written_files],
    }
    manifest_path = args.output_dir / f"text_embeddings_{args.category}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Rows embedded: {embedded_count:,}")
    print(f"Output files:  {len(written_files):,}")
    print(f"Manifest:      {manifest_path}")

    if embedded_count:
        num_chunks = math.ceil(embedded_count / args.chunk_size)
        print(f"Expected chunk count from row total: about {num_chunks:,}")


if __name__ == "__main__":
    main()
