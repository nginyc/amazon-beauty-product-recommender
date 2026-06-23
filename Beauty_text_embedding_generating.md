# Beauty Text Embedding Generation Procedure

This document records the mature procedure for generating text embeddings for
`Beauty_and_Personal_Care` items, for use in a CBPR-style model with interaction
signals plus item text modality.

## Goal

Generate a reusable text embedding artifact from:

```text
/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/Beauty_and_Personal_Care.jsonl
/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/meta_Beauty_and_Personal_Care.jsonl
```

Target setup:

```text
Text encoder: BAAI/bge-base-en-v1.5
Device: mps if available, otherwise cuda/cpu
Model max sequence length: 512
Raw text embedding dim: 768
Embedding storage dtype: float32
CBPR projection: 768 -> 32
Domain: Beauty_and_Personal_Care only
```

The BGE text encoder should be used as a frozen feature extractor. CBPR
training should load saved text embeddings and learn only the recommender
parameters, including the text projection layer.

## High-Level Design

Use a three-stage workflow:

```text
Stage A: Reproduce Yun Chuan's Beauty filtering logic and derive the final item universe
Stage B: Preprocess Beauty metadata text and generate BGE embeddings
Stage C: Build a RecBole-aligned text embedding tensor for CBPR
```

Do not encode item text inside the training loop. Text encoding is expensive,
and recomputing BGE embeddings every epoch would make training slow and hard to
reproduce.

## Recommended Outputs

Suggested output directory:

```text
/Users/frankwang1224/Projects/rcd_sys_proj02/embeddings/text_bge_base_en_v1_5_4col_beauty_yc_filtered/
```

Recommended output files:

| File | Contents | Purpose |
|---|---|---|
| `beauty_final_item_universe.csv` | One row per final Beauty `parent_asin` after Yun Chuan-style review filtering | Defines the exact item universe to embed |
| `text_preprocessing_manifest.csv` | One row per final Beauty `parent_asin`, with cleaned-text availability before embedding | Pre-embedding audit file |
| `text_manifest.csv` | One row per final Beauty `parent_asin`, with text field availability and post-embedding status | Human-readable final audit file |
| `chunks/text_embeddings_*.parquet` | Chunked BGE embeddings keyed by `parent_asin` | Resumable embedding cache |
| `text_embeddings_by_parent_asin.parquet` | Combined final embedding table keyed by `parent_asin` | Reusable text feature table |
| `beauty_item_id_parent_asin_map.csv` | Mapping between model item IDs and `parent_asin` | Ensures embeddings align with RecBole items |
| `beauty_bge_base_en_v1_5_text_embeddings_by_final_item_idx_debug.pt` | Debug tensor indexed by dense `final_item_idx` over the final Beauty item universe | Intermediate/debug only; do not load directly in RecBole CBPR |
| `beauty_bge_base_en_v1_5_text_embeddings_recbole_aligned.pt` | Final PyTorch tensor aligned to RecBole internal item IDs | Direct CBPR training input |
| `summary.json` | Counts, config, model name, dtype, embedding dim, missing-text rate, QA status | QA and reproducibility |

## Step 1: Reproduce Yun Chuan's Filtering Logic

The text embeddings must be generated for the same item universe used by the
Beauty recommendation dataset.

Yun Chuan's preprocessing logic is spread across:

```text
/Users/frankwang1224/Projects/amazon-item-recommender/preprocess/process-reviews-jsonl.ipynb
/Users/frankwang1224/Projects/amazon-item-recommender/preprocess/process-items-jsonl.ipynb
/Users/frankwang1224/Projects/amazon-item-recommender/preprocess/prepare-beauty-atomic-files.ipynb
```

For the Beauty-only text pipeline, reproduce the relevant logic:

```text
category = Beauty_and_Personal_Care
START_DATE = 2021-01-01
END_DATE = 2022-12-31
MIN_RATING = None
fields from reviews = user_id, parent_asin, rating, timestamp
USER_MIN_REVIEWS = 5
TRAIN_END_CUTOFF_DATE = 2022-08-01
VALID_END_CUTOFF_DATE = 2022-10-01
WARM_USER_MIN_REVIEWS = 10
WARM_ITEM_MIN_REVIEWS = 5
```

Procedure:

1. Stream `Beauty_and_Personal_Care.jsonl`.
2. Keep reviews whose timestamp is within `2021-01-01` to `2022-12-31`.
3. Keep fields: `user_id`, `parent_asin`, `rating`, `timestamp`.
4. Sort by `timestamp`.
5. Deduplicate by `(user_id, parent_asin)`, keeping the latest review.
6. Count each user's remaining Beauty interactions.
7. Remove users with fewer than 5 Beauty interactions:

```text
USER_MIN_REVIEWS = 5
```

This filter removes extremely sparse users before the train/valid/test split.
It keeps the experiment focused on users with enough behavioral history for
BPR/CBPR-style collaborative ranking.

Important warm/cold distinction:

```text
WARM_USER_MIN_REVIEWS = 10
```

This threshold is used after the train split to label users as warm or cold. It
does not remove users with 5 to 9 total Beauty interactions. A user can also
have at least 10 total Beauty interactions but still be labeled cold if fewer
than 10 of those interactions are in the training period.

Then split by timestamp:

```text
train: timestamp <= 2022-08-01
valid: 2022-08-01 < timestamp <= 2022-10-01
test:  timestamp > 2022-10-01
```

Keep valid/test interactions only for users that appear in train.

The final item universe for text embeddings should be:

```text
final_parent_asin_set = unique parent_asin from train + valid + test
```

Do not embed all metadata rows for the final CBPR artifact. Embed only this
final item universe, so the text vectors match the actual Beauty
recommendation dataset.

## Step 2: Filter Metadata by Final Item Universe

Stream:

```text
/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/meta_Beauty_and_Personal_Care.jsonl
```

Keep rows where:

```text
metadata_row["parent_asin"] in final_parent_asin_set
```

Keep at least these fields:

```text
parent_asin
title
categories
features
description
store
price
```

The embedding text should use only:

```text
title
categories
features
description
```

`store` and `price` can be kept in manifests or downstream item files, but they
should not be mixed into the default BGE text input unless a separate ablation
intentionally tests that design.

If duplicate metadata rows exist for a `parent_asin`, prefer the row with the
most usable product text in this order:

```text
1. non-empty title
2. non-empty categories
3. non-empty features
4. non-empty description
5. longer cleaned combined text
```

The prior EDA found no duplicate Beauty `parent_asin` metadata rows, but this
check should stay in the production pipeline.

## Step 3: Build Four-Column Product Text

Use the improved four-column text logic rather than the older
feature/description-only logic.

Build the BGE input in this order:

```text
Title: ...
Categories: ...
Features: ...
Description: ...
```

The order is intentional:

1. `title` is the shortest high-signal product identity field.
2. `categories` gives stable taxonomy context.
3. `features` gives product attributes and selling points.
4. `description` can add detail but is often longer and noisier.

This order also protects the most important information if the BGE tokenizer
has to truncate long inputs.

## Step 4: Apply Text Cleaning Rules

Use light normalization. Do not over-clean product language.

For all four text fields:

1. Convert HTML entities, for example `&amp;` to `&`.
2. Remove HTML tags.
3. Remove URLs and email addresses.
4. Apply Unicode NFKC normalization.
5. Remove decorative or control Unicode categories such as emoji-like symbols,
   invisible controls, and ornamental symbols.
6. Collapse repeated whitespace.
7. Preserve product-relevant terms such as sizes, units, percentages, colors,
   materials, ingredients, model words, `SPF 50`, `UPF 50+`, `13x4`, `150%`,
   `oz`, and `ml`.

Field-specific joining:

| Field | Raw shape | Clean join rule |
|---|---|---|
| `title` | usually string | join as one text span |
| `categories` | usually list of strings | join with ` > ` |
| `features` | usually list of strings | join with `; ` |
| `description` | usually list of strings | join with spaces |

Remove seller-service noise only from `features` and `description`.

Service-noise examples:

```text
refund policy
return policy
free return
hassle-free return
free replacement
replacement guarantee
warranty service
satisfaction guaranteed
money-back guarantee
risk free
customer service
after-sales service
contact seller
contact us
free shipping
shipping policy
delivery time
Amazon FBA
fulfilled by Amazon
Amazon Prime
Prime shipping
```

This removal should happen at sentence or bullet level. Do not delete
individual words from otherwise useful product text.

Do not apply service-noise deletion to `title` or `categories` by default.
Words like `replacement` can be legitimate product identity terms, for example
replacement brush heads or refill parts.

Also avoid bare keyword removal for ambiguous words such as `prime`,
`delivery`, `shipping`, and `guarantee`. Remove clear commerce/service
phrases, not legitimate product claims such as "prime quality natural
ingredients" or "fast delivery of nutrients to skin".

After normalization and service-noise removal, deduplicate repeated pieces
first within each field and then across the whole item using exact casefolded
matching. Avoid fuzzy deduplication unless it has been manually audited.

## Step 5: Control Token Length Before Embedding

Use BGE's actual tokenizer for final length checks.

Base model setting:

```text
model.max_seq_length = 512
```

Preferred truncation priority:

```text
1. Always keep Title if present.
2. Always keep Categories if present.
3. Keep as much Features text as possible.
4. Trim Description first when over budget.
5. Trim Features only if the text is still over budget.
```

Automatic SentenceTransformers truncation is acceptable if the input order is
`Title -> Categories -> Features -> Description`, because the highest-priority
text appears first. For the mature production version, recording approximate or
actual token counts in the manifest is better for QA.

Recommended manifest columns:

| Column | Type | Meaning |
|---|---|---|
| `parent_asin` | string | Amazon item-family key |
| `has_title` | int | 1 if cleaned title is non-empty |
| `has_categories` | int | 1 if cleaned categories are non-empty |
| `has_features` | int | 1 if cleaned features are non-empty |
| `has_description` | int | 1 if cleaned description is non-empty |
| `embedding_text_chars` | int | Character length after preprocessing |
| `embedding_text_tokens` | int/null | Token length from the BGE tokenizer if measured |
| `text_embedding_ok` | int | 1 if a real BGE vector was generated |
| `text_fail_reason` | string/null | Reason for missing or failed text embedding |
| `embedding_model` | string | `BAAI/bge-base-en-v1.5` |
| `embedding_dim` | int/null | Expected to be 768 |
| `embedding_dtype` | string | `float32` |

For missing or unusable text, keep the item in the manifest and assign a zero
vector later when building the final tensor. Do not silently drop the item.

## Step 6: Encode Text with BGE

Use `BAAI/bge-base-en-v1.5` as a frozen item-text encoder.

Recommended inference settings:

```text
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
model.max_seq_length = 512
normalize_embeddings = True
batch_size = 64, adjust based on memory
output dtype = float32
```

Use plain product text as the BGE input. Do not add query instructions because
these vectors are item content features for recommendation, not query-to-passage
retrieval embeddings.

Conceptual logic:

```python
embeddings = model.encode(
    embedding_texts,
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
).astype(np.float32)
```

Expected output:

```text
text_embedding_dim = 768
L2-normalized vectors
float32 storage
```

## Step 7: Store Embeddings as Resumable Chunks

Use chunked output to make the run resumable.

Suggested chunk size:

```text
5,000 to 10,000 successfully processed items per parquet chunk
```

Each chunk should contain:

```text
parent_asin
text_embedding
text_embedding_ok
text_fail_reason
embedding_model
embedding_dim
embedding_dtype
preprocessing_version
has_title
has_categories
has_features
has_description
embedding_text_chars
embedding_text_tokens
```

Rules:

```text
completed chunks are never rewritten
temporary chunks use a .tmp suffix
on resume, skip parent_asin values already present in completed chunks
failed items are recorded, not silently ignored
```

Recommended embedding column:

```text
text_embedding: list<float32>
```

Do not rely on default pandas Parquet inference for the vector column, because
it may write `double` values and make the embedding output much larger. Use an
explicit PyArrow schema such as:

```python
pa.field("text_embedding", pa.list_(pa.float32()))
```

The older Beauty feature/description output in:

```text
/Users/frankwang1224/Projects/rcd_sys_proj02/embeddings/text_bge_base_en_v1_5/
```

should be treated as an earlier artifact, not the final mature text embedding
artifact. It embeds only `features` and `description`, and its vector column was
written as double precision.

## Step 8: Build the RecBole-Aligned Tensor

This is the most important alignment step.

CBPR needs a tensor whose row index matches the model's internal item ID:

```text
text_tensor[recbole_internal_item_id] = BGE embedding for that item
```

The tensor must include the padding row:

```text
row 0 = zero vector
```

Final tensor:

```text
beauty_bge_base_en_v1_5_text_embeddings_recbole_aligned.pt
shape = [num_recbole_items, 768]
dtype = torch.float32
```

Rows for missing or failed text:

```text
all-zero vector
```

To avoid ID mismatch, build the final tensor using the actual item-token mapping
from the loaded RecBole dataset. Do not assume that the sorted Yun Chuan-style
`iid` is the same as RecBole's internal item row index unless it is explicitly
verified.

Save:

```text
beauty_item_id_parent_asin_map.csv
```

with columns such as:

```text
recbole_internal_item_id
item_id_token
final_item_idx
parent_asin
has_text
text_embedding_ok
text_fail_reason
```

The debug tensor indexed by dense `final_item_idx` can be useful for checking
preprocessing, but CBPR should load the RecBole-aligned tensor.

## Step 9: Use the Tensor in CBPR

CBPR should load:

```text
beauty_bge_base_en_v1_5_text_embeddings_recbole_aligned.pt
```

Register it as a frozen buffer:

```python
self.register_buffer("item_text", text_embeddings)
```

Recommended text branch:

```python
self.user_text = nn.Embedding(self.n_users, 32)
self.text_proj = nn.Linear(768, 32, bias=False)
```

Score:

```text
score(u, i)
= user_collab(u) dot item_collab(i)
+ user_text(u) dot text_proj(item_text(i))
```

Training objective remains BPR pairwise ranking loss:

```text
positive item should score higher than sampled negative item
```

Only the CBPR parameters are trained:

```text
user_collab
item_collab
user_text
text_proj
```

The BGE encoder and saved text embeddings stay frozen.

## Step 10: QA Checks Before Training

Run these checks before using the tensor in CBPR:

```text
final item universe count matches the Beauty RecBole item count excluding padding
metadata rows found matches final item universe count
actual vector rows match final item universe count
tensor row 0 is all zeros
tensor dtype is float32
tensor has no NaN or inf
embedding_dim is 768 for all successful text embeddings
successful text vectors have norm close to 1.0
missing/failed text rows are all zeros
every nonzero row maps to the correct parent_asin
```

Report these counts in `summary.json`:

```text
total_final_items
metadata_rows_found
items_with_nonempty_title
items_with_nonempty_categories
items_with_nonempty_features
items_with_nonempty_description
items_with_nonempty_embedding_text
items_successfully_embedded
items_missing_or_failed_text
text_embedding_ok_rate
embedding_model
embedding_dim
embedding_dtype
model_max_seq_length
preprocessing_version
created_at
```

Also manually inspect a small random sample:

```text
20 rows from text_manifest.csv
their cleaned Title/Categories/Features/Description preview
their embedding_text_chars and token count
their text_embedding_ok flag
```

## Step 11: Recommended Experiment Naming

Use explicit model and modality names so future results are easy to compare:

```text
BPR
CBPR_text_bge_base_en_v1_5
CBPR_image_mobileclip2_s0
CBPR_text_bge_base_en_v1_5_image_mobileclip2_s0
```

For this specific procedure, the expected first text experiment is:

```text
CBPR_text_bge_base_en_v1_5
```

## Key Decisions

1. Generate text embeddings before training.
2. Use only `Beauty_and_Personal_Care` for this stage.
3. Reproduce Yun Chuan's review filtering before embedding metadata.
4. Remove users with fewer than 5 Beauty interactions before the temporal split.
5. Use warm/cold user labels after the train split; do not remove all cold users.
6. Filter metadata by the final interaction item universe.
7. Use `parent_asin` as the item key.
8. Use four metadata text fields: `title`, `categories`, `features`, `description`.
9. Put `Title` and `Categories` first in the embedding text.
10. Remove seller-service noise only from `features` and `description`.
11. Save raw 768-dim BGE vectors as normalized `float32`.
12. Learn the `768 -> 32` projection inside CBPR.
13. Align the final tensor to RecBole internal item IDs before training.
14. Keep manifest, summaries, and sample previews for debugging and reproducibility.
