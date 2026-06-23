# Beauty Image Embedding Generation Procedure

This document records the mature procedure for generating image embeddings for
`Beauty_and_Personal_Care` items, for use in a CBPR-style model with interaction
signals plus image modality.

## Goal

Generate a reusable image embedding artifact from:

```text
/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/Beauty_and_Personal_Care.jsonl
/Users/frankwang1224/Projects/rcd_sys_proj02/dataset/meta_Beauty_and_Personal_Care.jsonl
```

Target setup:

```text
Image encoder: apple/MobileCLIP2-S0
Device: mps
Embedding storage dtype: float32
CBPR projection: raw image_embedding_dim -> 32
Domain: Beauty_and_Personal_Care only
```

The image encoder should be used as a frozen feature extractor. CBPR training
should load saved image embeddings and learn only the recommender parameters,
including the image projection layer.

## High-Level Design

Use a two-stage workflow:

```text
Stage A: Prepare image embeddings
Stage B: Train CBPR from saved image embeddings
```

Do not download or encode images inside the training loop. Image downloading can
fail, image encoding is expensive, and recomputing image embeddings every epoch
would make training slow and hard to reproduce.

## Recommended Outputs

Suggested output directory:

```text
/Users/frankwang1224/Projects/rcd_sys_proj02/embeddings/image_mobileclip2_s0_beauty/
```

Recommended output files:

| File | Contents | Purpose |
|---|---|---|
| `image_manifest.csv` | One row per final Beauty `parent_asin`, with image URL choice and success/failure status | Human-readable audit file |
| `chunks/image_embeddings_*.parquet` | Chunked embeddings keyed by `parent_asin` | Resumable embedding cache |
| `image_embeddings_by_parent_asin.parquet` | Combined final embedding table keyed by `parent_asin` | Reusable image feature table |
| `beauty_item_id_parent_asin_map.csv` | Mapping between model item IDs and `parent_asin` | Ensures embeddings align with RecBole items |
| `beauty_mobileclip2_s0_image_embeddings_by_yc_iid_debug.pt` | Debug tensor indexed by Yun Chuan-style `iid` | Intermediate/debug only; do not load directly in RecBole CBPR |
| `beauty_mobileclip2_s0_image_embeddings_recbole_aligned.pt` | Final PyTorch tensor aligned to RecBole internal item IDs | Direct CBPR training input |
| `summary.json` | Counts, config, model name, dtype, embedding dim, success/failure rates | QA and reproducibility |

## Step 1: Reproduce Yun Chuan's Filtering Logic

The image embeddings must be generated for the same item universe used by the
Beauty recommendation dataset.

Yun Chuan's preprocessing logic is spread across:

```text
/Users/frankwang1224/Projects/amazon-item-recommender/preprocess/process-reviews-jsonl.ipynb
/Users/frankwang1224/Projects/amazon-item-recommender/preprocess/process-items-jsonl.ipynb
/Users/frankwang1224/Projects/amazon-item-recommender/preprocess/prepare-beauty-atomic-files.ipynb
```

For the Beauty-only image pipeline, reproduce the relevant logic:

```text
category = Beauty_and_Personal_Care
START_DATE = 2021-01-01
END_DATE = 2022-12-31
MIN_RATING = None
fields from reviews = user_id, parent_asin, rating, timestamp
```

Procedure:

1. Stream `Beauty_and_Personal_Care.jsonl`.
2. Keep reviews whose timestamp is within `2021-01-01` to `2022-12-31`.
3. Keep fields: `user_id`, `parent_asin`, `rating`, `timestamp`.
4. Sort by `timestamp`.
5. Deduplicate by `(user_id, parent_asin)`, keeping the latest review.
6. Create user and item integer maps from the filtered review table.
7. Apply the user threshold:

```text
USER_MIN_REVIEWS = 5
```

This removes users with fewer than 5 filtered Beauty interactions.

Then split by timestamp:

```text
TRAIN_END_CUTOFF_DATE = 2022-08-01
VALID_END_CUTOFF_DATE = 2022-10-01
```

So:

```text
train: timestamp <= 2022-08-01
valid: 2022-08-01 < timestamp <= 2022-10-01
test:  timestamp > 2022-10-01
```

Keep valid/test interactions only for users that appear in train.

Important note:

```text
WARM_USER_MIN_REVIEWS = 10
```

In Yun Chuan's code this is used to label warm/cold users after the train split.
It is not the same thing as filtering to warm users only. Unless the experiment
is intentionally changed, do not remove all cold users at this step.

The final item universe for image embeddings should be:

```text
final_parent_asin_set = unique parent_asin from train + valid + test
```

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
store
price
categories
images
```

Do not use review images for this CBPR image modality. Use metadata product
images. Review images are user-uploaded and often represent complaints,
condition, packaging, usage context, or failures rather than the canonical item
appearance.

If duplicate metadata rows exist for a `parent_asin`, prefer the row with the
best usable product image.

## Step 3: Select One Product Image per Item

Each metadata row has an `images` list. Choose one canonical image URL per item.

Recommended priority:

```text
1. MAIN image hi_res
2. MAIN image large
3. MAIN image thumb
4. first available hi_res
5. first available large
6. first available thumb
```

Recommended manifest columns:

| Column | Type | Meaning |
|---|---|---|
| `parent_asin` | string | Amazon item-family key |
| `selected_image_url` | string/null | Chosen product image URL |
| `selected_image_variant` | string/null | Image variant such as `MAIN` or `PT01` |
| `selected_image_source_field` | string/null | `hi_res`, `large`, or `thumb` |
| `has_image_url` | int | 1 if a non-empty URL was found, else 0 |
| `image_embedding_ok` | int | 1 if the image was downloaded/opened/encoded successfully |
| `image_fail_reason` | string/null | Reason for missing or failed image |
| `embedding_model` | string | `apple/MobileCLIP2-S0` |
| `embedding_dim` | int/null | Raw image embedding dimension after encoding |
| `embedding_dtype` | string | `float32` |

`has_image_url` is better than `image_exist` because it is precise: it means the
metadata contains a usable-looking URL. `image_embedding_ok` is the stronger
training flag because it confirms the embedding was actually generated.

For training, use:

```text
has_image = image_embedding_ok
```

## Step 4: Download and Validate Images

For each selected image URL:

1. Download with timeout and retry.
2. Open with PIL.
3. Convert to RGB:

```python
image = Image.open(image_bytes).convert("RGB")
```

4. If download/open fails, record the reason in `image_fail_reason`.
5. Do not drop the item. Keep the item and later assign a zero image vector.

Recommended failure reasons:

```text
missing_image_url
download_timeout
http_error
invalid_image_bytes
pil_open_error
encoder_error
unknown_error
```

## Step 5: Encode Images with MobileCLIP2-S0

Use MobileCLIP2-S0 as a frozen image encoder:

```text
model.eval()
torch.inference_mode()
device = mps
```

For each valid image:

1. Use the model's official preprocessing function.
2. Batch images where possible.
3. Move the batch to `mps`.
4. Encode images.
5. L2-normalize the output embedding.
6. Move embeddings back to CPU.
7. Cast to `float32`.

Conceptual logic:

```python
image_tensor = preprocess(image).unsqueeze(0).to("mps")
with torch.inference_mode():
    image_embedding = model.encode_image(image_tensor)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
image_embedding = image_embedding.cpu().float()
```

Do not reduce the image embedding to 32 dimensions here. Store the raw
MobileCLIP2-S0 image vector. The `image_embedding_dim -> 32` projection should
happen inside CBPR training as a learned layer.

## Step 6: Store Embeddings as Resumable Chunks

Use chunked output to make the run resumable.

Suggested chunk size:

```text
500 to 1000 successfully processed items per parquet chunk
```

Each chunk should contain:

```text
parent_asin
selected_image_url
has_image_url
image_embedding_ok
image_fail_reason
image_embedding
embedding_model
embedding_dim
embedding_dtype
```

Rules:

```text
completed chunks are never rewritten
temporary chunks use a .tmp suffix
on resume, skip parent_asin already present in completed chunks
failed items are recorded, not silently ignored
```

Recommended embedding column:

```text
image_embedding: list<float32>
```

For missing or failed images, either:

1. Store no vector in the parquet table and add zero vectors only in the final
   tensor-building step, or
2. Store a zero vector once `embedding_dim` is known.

The first option is cleaner during streaming; the second option is simpler for
downstream tensor creation. Either is acceptable as long as the final tensor is
complete and aligned.

## Step 7: Build the RecBole-Aligned Tensor

This is the most important alignment step.

CBPR needs a tensor whose row index matches the model's item ID:

```text
image_tensor[item_id] = image embedding for that item
```

The tensor must include the padding row:

```text
row 0 = zero vector
```

Final training tensor:

```text
beauty_mobileclip2_s0_image_embeddings_recbole_aligned.pt
shape = [num_recbole_items, raw_image_embedding_dim]
dtype = torch.float32
```

Rows for missing/failed images:

```text
all-zero vector
```

To avoid ID mismatch, build the final tensor using the same item mapping used by
the Beauty RecBole dataset. Save:

```text
beauty_item_id_parent_asin_map.csv
```

with columns such as:

```text
item_id
parent_asin
has_image
selected_image_url
image_embedding_ok
image_fail_reason
```

If building inside the RecBole training notebook, use the loaded RecBole
dataset's item token mapping rather than assuming a separate sorted
`parent_asin` list. The Yun Chuan-style `iid` tensor is not safe to index with
RecBole internal item IDs.

## Step 8: Use the Tensor in CBPR

CBPR should load the RecBole-aligned tensor:

```text
beauty_mobileclip2_s0_image_embeddings_recbole_aligned.pt
```

Register it as a frozen buffer:

```python
self.register_buffer("item_image", image_embeddings)
```

Recommended image branch:

```python
self.user_image = nn.Embedding(self.n_users, 32)
self.image_proj = nn.Linear(image_embedding_dim, 32, bias=False)
```

Score:

```text
score(u, i)
= user_collab(u) dot item_collab(i)
+ user_image(u) dot image_proj(item_image(i))
```

Training objective remains BPR pairwise ranking loss:

```text
positive item should score higher than sampled negative item
```

Only the CBPR parameters are trained:

```text
user_collab
item_collab
user_image
image_proj
```

The MobileCLIP2-S0 encoder and saved image embeddings stay frozen.

## Step 9: QA Checks Before Training

Run these checks before using the tensor in CBPR:

```text
final item count matches RecBole item count
tensor row 0 is all zeros
tensor dtype is float32
tensor has no NaN or inf
embedding_dim is constant for all successful images
missing/failed image rows are all zeros
every nonzero row maps to the correct parent_asin
```

Report these counts in `summary.json`:

```text
total_final_items
metadata_rows_found
items_with_image_url
items_successfully_embedded
items_missing_image_url
items_failed_download_or_encoding
image_embedding_ok_rate
embedding_model
embedding_dim
embedding_dtype
created_at
```

Also manually inspect a small random sample:

```text
20 rows from image_manifest.csv
their selected_image_url
their parent_asin/title/category
whether the URL opens to the expected product image
```

## Step 10: Recommended Experiment Naming

Use explicit model and modality names so future results are easy to compare:

```text
BPR
CBPR_image_mobileclip2_s0
CBPR_text_bge
CBPR_text_bge_image_mobileclip2_s0
```

For this specific procedure, the expected first experiment is:

```text
CBPR_image_mobileclip2_s0
```

## Key Decisions

1. Generate image embeddings before training.
2. Use only `Beauty_and_Personal_Care` for this stage.
3. Filter metadata by the final interaction item universe.
4. Use `parent_asin` as the item key.
5. Use metadata product images, not review images.
6. Keep items with missing images and assign zero image vectors.
7. Save raw MobileCLIP2-S0 vectors as `float32`.
8. Learn the `image_embedding_dim -> 32` projection inside CBPR.
9. Align the final tensor to RecBole item IDs before training.
10. Keep manifest and summary files for debugging and reproducibility.
