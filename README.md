# Amazon Item Recommender

## Setup

### Prerequisites

- **[pyenv](https://github.com/pyenv/pyenv)**
- **[uv](https://github.com/astral-sh/uv)**

### Getting Started

Install Python dependencies and set up virtual environment:
```sh
pyenv install
pyenv exec python -m venv ./.venv
. .venv/bin/activate
uv sync
```

Download the Amazon Reviews 2023 data from the official project site:

- https://amazon-reviews-2023.github.io/
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The site provides per-category `review` and `meta` downloads, plus links to the common data processing and loading guidance. For this project, download the **Beauty and Personal Care** category files (`.jsonl.gz`) and place them under `data/`.

## Pipeline

Run the notebooks in this order: preparation → training. The EDA notebook is optional and can be run at any time.

### 1. EDA (optional) — `eda/eda_beauty.ipynb`

Run all cells to load reviews, visualize user segments (warm / cold / new) across temporally-ordered train/valid/test splits, and explore the data. This step creates no output files.

### 2. Process reviews — `preprocess/process-reviews-jsonl.ipynb`

Loads gzipped JSONL review files for Beauty and Personal Care and Clothing, Shoes & Jewelry, filters by date range (2021–2022), and writes a clean intermediate CSV:

- **Input:** `data/{Category}.jsonl.gz`
- **Output:** `data/reviews.csv` (columns: `user_id`, `parent_asin`, `rating`, `timestamp`, `category`)

### 3. Process item metadata — `preprocess/process-items-jsonl.ipynb`

Loads gzipped JSONL metadata files and keeps only items that appear in the reviews:

- **Input:** `data/meta_{Category}.jsonl.gz` and `data/reviews.csv`
- **Output:** `data/items.csv` (columns: `parent_asin`, `title`, `price`, `store`, `category`)

### 4. Prepare RecBole atomic files — `preprocess/prepare-beauty-atomic-files.ipynb`

Filters to Beauty and Personal Care only, drops users with fewer than 5 reviews, maps string user/item IDs to integers, splits temporally (80/10/10 — train by 2022-08-01, valid by 2022-10-01), labels users as warm (≥10 train reviews) / cold, and writes RecBole-format atomic files.

Output files:
- `beauty.train.inter`, `beauty.valid.inter`, `beauty.test.inter` (tab-separated with columns `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float`, `item_id_list:token_seq`)
- `beauty.user` with warm/cold category labels
- `beauty.item` with item features (`price`, `store`)

- **Input:** `data/reviews.csv` and `data/items.csv`
- **Outputs:** Atomic files under `data/beauty/`

### 5. Train models

All training notebooks load the prepared atomic files via RecBole, train a model, print the best validation score (NDCG@20), and evaluate on the test set for NDCG@20, Recall@20, and MRR@20 — both overall and per user segment (warm / cold / new). Model checkpoints are saved to `saved/{Model}-{timestamp}.pth`.

#### `train/train-pop.ipynb`

Trains a **Pop** (most popular) baseline. Non-learned — computes item popularity from training interactions in a single epoch.

#### `train/train-bpr.ipynb`

Trains a **BPR** (Bayesian Personalized Ranking) model — a general (non-sequential) collaborative filtering baseline.

#### `train/train-lightgcn.ipynb`

Trains a **LightGCN** model — a graph convolutional network that learns user and item embeddings by propagating them over the user–item interaction graph.

Trains for up to 200 epochs with 3 layers and uses NDCG@20 for early stopping.

#### `train/train-mean-pool-title.ipynb`

Trains a **Mean-Pool Title** model — a custom non-learned baseline that encodes item titles with BGE sentence embeddings, represents each user as the mean of their training item embeddings, and scores via dot product. Trains for 1 epoch (centroid computation only).

#### `train/train-dense-two-tower.ipynb`

Trains a **Dense Two-Tower** model — a learned two-tower architecture with a frozen BGE item title encoder (`BAAI/bge-base-en-v1.5`) feeding a trainable projection MLP, and a learned user embedding tower with an MLP. Uses InfoNCE loss with in-batch negatives.
