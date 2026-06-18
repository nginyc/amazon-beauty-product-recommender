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

The site provides per-category `review` and `meta` downloads, plus links to the common data processing and loading guidance. For this project, download the category files you need (`.jsonl.gz`) and place them under `data/`.

## Pipeline

Run the notebooks in this order: preparation → training. The EDA notebook is optional and can be run at any time.

### 1. EDA (optional) — `eda/eda_beauty.ipynb`

Run all cells to load reviews, visualize user segments (warm / cold / new) across temporally-ordered train/valid/test splits, and explore the data. This step creates no output files.

### 2. Process reviews — `preprocess/process-reviews-jsonl.ipynb`

Loads gzipped JSONL review files for one or more categories, filters by date range and sample size, and writes a clean intermediate CSV:

- **Input:** `data/{Category}.jsonl.gz`
- **Output:** `data/reviews.csv` (columns: `user_id`, `parent_asin`, `rating`, `timestamp`, `category`)

### 3. Process item metadata — `preprocess/process-items-jsonl.ipynb`

Loads gzipped JSONL metadata files and keeps only items that appear in the reviews:

- **Input:** `data/meta_{Category}.jsonl.gz` and `data/reviews.csv`
- **Output:** `data/items.csv` (columns: `parent_asin`, `title`, `price`, `store`, `category`)

### 4. Prepare RecBole atomic files — `preprocess/prepare-atomic-files.ipynb`

Maps string user/item IDs to integers, splits temporally (80/10/10), labels users as warm (≥5 train reviews) / cold / new, builds sequential item histories (capped at 50), and writes RecBole-format atomic files for two dataset variants:

- **`data/target/`** — target-category items only in the history sequence
- **`data/cross/`** — cross-category items in the history sequence

Output files per variant:
- `{Dataset}.train.inter`, `{Dataset}.valid.inter`, `{Dataset}.test.inter` (tab-separated with columns `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float`, `item_id_list:token_seq`)
- `{Dataset}.user` with warm/cold/new category labels
- `{Dataset}.item` with an `is_target` flag

- **Input:** `data/reviews.csv` and `data/items.csv`
- **Outputs:** Atomic files under `data/target/` and `data/cross/`

### 5. Train models

All training notebooks load the prepared atomic files via RecBole, train a model, print the best validation score, and evaluate on the test set both overall and per user segment (warm / cold / new). Model checkpoints are saved to `saved/{Model}-{timestamp}.pth`.

#### `train/train-pop.ipynb`

Trains a **Pop** (most popular) baseline. Non-learned — computes item popularity from training interactions in a single epoch.

#### `train/train-bpr.ipynb`

Trains a **BPR** (Bayesian Personalized Ranking) model — a general (non-sequential) collaborative filtering baseline.

#### `train/train-sasrec.ipynb`

Trains a **SASRec** (Self-Attentive Sequential Recommendation) model — uses sequential user item histories with self-attention.

#### `train/train-dssm.ipynb`

Trains a **DSSM** (Deep Structured Semantic Model) — a context-aware model that also incorporates item features (`price`, `store`) from the `.item` atomic file.
