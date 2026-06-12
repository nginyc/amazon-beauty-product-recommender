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

### 2. Prepare dataset — `preprocess/prepare-dataset.ipynb`

Run all cells to:
- Load and filter reviews from `data/{Category}.jsonl.gz`
- Map string user/item IDs to integers
- Write RecBole-format files to `data/{Dataset}/{Dataset}.{train,valid,test}.inter` (tab-separated with columns `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float`, `item_id_list:token_seq`)
- Write `data/{Dataset}/{Dataset}.user` with warm/cold/new category labels

Make sure `data/` contains the source `.jsonl.gz` files before running.

### 3. Train SASRec — `train/train-sasrec.ipynb`

Run all cells. The notebook:
- Loads the prepared `.inter` and `.user` files via RecBole
- Configures and trains a SASRec model, and prints the best validation score
- Evaluates on the test set, both overall and per user segment (warm / cold / new)
