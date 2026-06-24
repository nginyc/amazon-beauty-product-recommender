# Beauty Recommender Explorer (`/app`)

A small Streamlit webapp to inspect the LightGCN recommender interactively:

- **Assume any user** — type an original Amazon user hash (e.g. `AFKZENTNBQ7A7V7UXW5JJI6UGRYQ`)
  **or** the mapped integer id (e.g. `93889`). Both are supported; `🎲 Random user`
  picks one for you.
- **Interaction history** — read live from `data/Beauty_and_Personal_Care.jsonl`.
- **Item images & titles** — read live from `data/meta_Beauty_and_Personal_Care.jsonl`.
- **Recommendations** — produced by a pluggable model adaptor, preset to
  `train/saved/LightGCN-Jun-23-2026_23-36-49.pth`.

## One-time setup: build the ID maps

The app must translate between original Amazon ids (in the jsonl files) and the
integer ids the model uses. The integer ids come from `prepare-beauty-atomic-files.ipynb`,
which enumerates `sorted(set(reviews.csv ids))` as `i + 1` and **never saved that map**.
(The pre-existing `data/user_map.json` / `item_map.json` are a *different*, 0-indexed
map and do **not** match the model.) Reconstruct the correct maps once:

```bash
python app/build_id_maps.py     # reads reviews.csv, writes data/beauty/app_*_map.json
```

This reads the full `reviews.csv` (a couple of minutes, a few GB of RAM) and writes
`data/beauty/app_user_map.json` and `app_item_map.json`. The app refuses to run until
these exist.

## Run

```bash
# from the repo root, using the project venv
streamlit run app/streamlit_app.py
```

## Recommendation modes (sidebar)

Loading torch/RecBole *inside* the Streamlit process can hard-crash on macOS
(`zsh: bus error`). To avoid that, the model is, by default, **not** imported into
the Streamlit process:

| Mode | What happens |
|------|--------------|
| **Subprocess (isolated)** — default | Predictions run in a separate Python process (`predict_cli.py`). torch is never imported into Streamlit, so the UI can't bus-error. The model loads fresh per prediction (cached). |
| **In-process** | Loads the model inside Streamlit (faster repeats, but can crash on affected setups). |
| **Disabled** | No model at all — just browse a user's history + item images. Nothing torch-related is imported. |

You can also run predictions straight from the shell:

```bash
python app/predict_cli.py --user 93889 --top-k 12
python app/predict_cli.py --user AFKZENTNBQ7A7V7UXW5JJI6UGRYQ --model train/saved/LightGCN-Jun-23-2026_23-36-49.pth
```

It prints a JSON line prefixed with `__PRED_JSON__`.

## How it works

| File | Responsibility |
|------|----------------|
| `streamlit_app.py` | UI: user input, history grid, recommendations grid; runs predictions in-process or via subprocess |
| `predict_cli.py`   | Standalone prediction worker (isolated process) — prints top-k as JSON |
| `recommender.py`   | `RecommenderAdaptor` — loads the checkpoint, resolves users, returns top-k ASINs |
| `data_access.py`   | Live streaming scans of the reviews / meta jsonl files |
| `ids.py`           | Translates Amazon hash ↔ mapped integer ↔ RecBole token via `user_map.json` / `item_map.json` |
| `paths.py`         | Central path config (all relative to the repo root) |
| `_compat.py`       | NumPy 2.0 / SciPy 1.12 shims required by RecBole 1.2 |

### ID spaces

The jsonl files use original Amazon strings (`user_id` hash, `parent_asin`). Preprocessing
remapped these to integers (`user_map.json`, `item_map.json`); those integers are the tokens
in the RecBole atomic files, on top of which RecBole assigns internal contiguous ids. The
adaptor walks: **Amazon hash → mapped int (token) → RecBole internal id** for prediction, and
**item internal id → token → ASIN** for display.

### Swapping the model

`RecommenderAdaptor(model_path=...)` accepts any RecBole general-recommender checkpoint
(BPR, LightGCN, etc.). Change the path in the sidebar or pass a different default.

## Performance note

The reviews (~4.9 GB) and meta (~2.8 GB) files are **scanned live on each new lookup**
(per the chosen design). A cheap substring pre-filter skips `json.loads` on non-matching
lines, and Streamlit caches results so repeat lookups are instant. While testing, set
**“Max lines to scan”** in the sidebar to cap how deep each scan reads. A user whose reviews
sit beyond that cap will show an incomplete history; set it to `0` for a complete (slow) scan.
