# Amazon Item Recommender

## Setup

### Prerequisites

- **[pyenv](https://github.com/pyenv/pyenv)**
- **[uv](https://github.com/astral-sh/uv)**

### Installation

Install Python dependencies and set up virtual environment:
```sh
pyenv install
pyenv exec python -m venv ./.venv
. .venv/bin/activate
uv sync
```

### Download Amazon Reviews 2023 data

Download the Amazon Reviews 2023 data from the official project site:

- https://amazon-reviews-2023.github.io/
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The site provides per-category `review` and `meta` downloads, plus links to the common data processing and loading guidance.

For this project, download the **Beauty and Personal Care** category files (`.jsonl.gz`) and place them under `data/`.

### Download CLIP image embeddings

Download the precomputed CLIP image embeddings from Google Drive: https://drive.google.com/drive/folders/1jkAkfQh-xtICmghAmU_53oRyiF3RTEFf?usp=drive_link.

Extract the multi-part zip archive into `data/embeddings/`:

```sh
mkdir -p data/embeddings
bsdtar -x -f ~/Downloads/embeddings-*-001.zip -C data/embeddings/
for f in ~/Downloads/embeddings-*-00{2,3,4,5,6,7}.zip; do
    bsdtar -x -f "$f" -C data/embeddings/
done
mv data/embeddings/embeddings/*.parquet data/embeddings/
rmdir data/embeddings/embeddings
```

**Expected format:** In `data/embeddings/`, each shard has columns `parent_asin` (str) and `embedding` (list of 512 floats, L2-normalized CLIP ViT-B/32 image embedding).

**Generating your own:** Run a CLIP model over product images for the `parent_asin` values in `data/items.csv`. You can refer to `research/clip_embedding_generation_script.ipynb` for an example of how these embeddings were generated.

## Pipeline

Run the notebooks in this order: preparation → training. The EDA notebook is optional and can be run at any time.

### 1. EDA (optional) — `eda/eda_beauty.ipynb`

Run all cells to load reviews, visualize user segments (warm / cold) and item segments (warm / cold) across temporally-ordered train/valid/test splits, and explore the data. This step creates no output files.

### 2. Process reviews — `preprocess/process-reviews-jsonl.ipynb`

Loads gzipped JSONL review files for Beauty and Personal Care and Clothing, Shoes & Jewelry, filters by date range (2021–2022), and writes a clean intermediate CSV:

- **Input:** `data/{Category}.jsonl.gz`
- **Output:** `data/reviews.csv` (columns: `user_id`, `parent_asin`, `rating`, `timestamp`, `category`)

### 3. Process item metadata — `preprocess/process-items-jsonl.ipynb`

Loads gzipped JSONL metadata files and keeps only items that appear in the reviews:

- **Input:** `data/meta_{Category}.jsonl.gz` and `data/reviews.csv`
- **Output:** `data/items.csv` (columns: `parent_asin`, `title`, `price`, `store`, `category`)

### 4. Prepare RecBole atomic files — `preprocess/prepare-beauty-atomic-files.ipynb`

Filters to Beauty and Personal Care only, drops users and items with fewer than 5 reviews, maps string user/item IDs to integers, splits temporally (80/10/10 — train by 2022-08-01, valid by 2022-10-01), labels users as warm (≥10 train reviews) / cold and items as warm (≥5 train reviews) / cold, and writes RecBole-format atomic files. Also loads CLIP image embeddings from `data/embeddings/*.parquet` and saves them as `data/beauty/clip_image_embeddings.pt`.

Output files:
- `beauty.train.inter`, `beauty.valid.inter`, `beauty.test.inter` (tab-separated with columns `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float`)
- `beauty.user` with warm (0) / cold (1) labels (`cold:float`)
- `beauty.item` with `title`, `store`, `price`, and warm (0) / cold (1) labels (`cold:float`)
- `clip_image_embeddings.pt` — precomputed CLIP ViT-B/32 image embeddings for all items

- **Input:** `data/reviews.csv`, `data/items.csv`, and `data/embeddings/*.parquet`
- **Outputs:** Atomic files under `data/beauty/`

### 5. Train models

All training notebooks load the prepared atomic files via RecBole, train a model, print the best validation score (NDCG@20), and evaluate on the test set for NDCG@20, Recall@20, and MRR@20 — both overall and per user segment (warm / cold). Model checkpoints are saved to `saved/{Model}-{timestamp}.pth`.

#### `train/train-pop.ipynb`

Trains a **Pop** (most popular) baseline. Non-learned — computes item popularity from training interactions in a single epoch.

#### `train/train-bpr.ipynb`

Trains a **BPR** (Bayesian Personalized Ranking) model — a general (non-sequential) collaborative filtering baseline.

#### `train/train-bpr-bge-init.ipynb`

Trains a **BPR** model whose item embeddings are initialized via a seeded random projection of BGE title embeddings (768 → 64) rather than random initialization.

#### `train/train-lightgcn.ipynb`

Trains a **LightGCN** model — a graph convolutional network that learns user and item embeddings by propagating them over the user–item interaction graph.

#### `train/train-mean-pool-title.ipynb`

Trains a **Mean-Pool Title** model — a custom non-learned baseline that encodes item titles with BGE sentence embeddings, represents each user as the mean of their training item embeddings, and scores via dot product. 

#### `train/train-cbpr.ipynb`

Trains a **CBPR** (Content BPR) model — a VBPR-style two-pathway recommender that combines collaborative user/item embeddings with BGE text content projected through a learned matrix. Scores are the sum of collaborative and content dot products.

#### `train/train-bpr-clip-hybrid.ipynb`

Trains a **BPR + CLIP multimodal late fusion** model — a standard BPR model whose predictions are blended at inference time with frozen CLIP text+image content scores via per-user z-score normalization. 
