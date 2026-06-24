"""Recommender model adaptor.

A thin, swappable wrapper around a saved RecBole checkpoint. It:
  - loads the model + the exact dataset/mapping it was trained with,
  - resolves an arbitrary user identifier (Amazon hash OR mapped integer) to a
    RecBole internal user id,
  - returns top-k item recommendations as original Amazon ASINs + scores.

Preset to load `train/saved/LightGCN-Jun-23-2026_23-36-49.pth`, but `model_path`
can point at any RecBole general-recommender checkpoint.
"""
from __future__ import annotations

import _compat  # noqa: F401  (must come before any recbole import)

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk

import ids
from paths import DEFAULT_MODEL_PATH

# Keep RecBole's very chatty INFO logging out of the Streamlit console.
logging.getLogger("recbole").setLevel(logging.ERROR)

# Single-threaded torch avoids a class of native SIGBUS/segfaults on macOS.
try:
    torch.set_num_threads(1)
except Exception:  # noqa: BLE001
    pass


def _stage(msg: str) -> None:
    """Flushed stderr breadcrumb so a native crash shows the last good step."""
    print(f"[adaptor] {msg}", file=sys.stderr, flush=True)


def _normalize_cold_feature(dataset, feat: pd.DataFrame, field: str = "cold") -> None:
    """Recover literal 0/1 warm/cold labels.

    The `cold` TOKEN field is remapped onto a shared token vocabulary and applied
    inconsistently across user/item feats; left as-is, `data_preparation` fails to
    cast it to a LongTensor. This mirrors the fix used in the training notebooks so
    the dataset rebuilds identically to how the checkpoint was trained.
    """
    if field not in feat:
        return
    token_of_id = {i: t for t, i in dataset.field2token_id[field].items()}

    def to_label(v: object) -> int:
        token = v if isinstance(v, str) else token_of_id.get(int(v), str(v))
        return 0 if token == "[PAD]" else int(token)

    feat[field] = feat[field].map(to_label).astype("int64")


@dataclass
class ResolvedUser:
    internal_id: int          # RecBole internal user id (for the model)
    token: str                # mapped-integer token (as stored in atomic files)
    amazon_hash: Optional[str]  # original Amazon user hash (for the reviews file)


@dataclass
class Recommendation:
    asin: str
    score: float


class RecommenderAdaptor:
    """Loads a RecBole checkpoint and serves predictions."""

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.model_path = str(model_path or DEFAULT_MODEL_PATH)
        self.device = torch.device(device)
        self._load()

    # --- loading -----------------------------------------------------------
    def _load(self) -> None:
        _stage(f"torch.load checkpoint: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        # Re-anchor data to an absolute path and force CPU for serving.
        from paths import DATA_DIR
        config["data_path"] = str(DATA_DIR)
        config["device"] = self.device

        init_seed(config["seed"], config["reproducibility"])
        _stage("create_dataset")
        dataset = create_dataset(config)
        _stage("normalize cold feature")
        _normalize_cold_feature(dataset, dataset.user_feat)
        _normalize_cold_feature(dataset, dataset.item_feat)
        _stage("data_preparation")
        train_data, valid_data, test_data = data_preparation(config, dataset)

        init_seed(config["seed"], config["reproducibility"])
        _stage(f"build model {config['model']}")
        model = get_model(config["model"])(config, train_data._dataset).to(self.device)
        _stage("load_state_dict")
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        model.eval()
        _stage("model ready")

        self.config = config
        self.dataset = dataset
        self.test_data = test_data
        self.model = model
        self.model_name = config["model"]
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

    # --- public API --------------------------------------------------------
    @property
    def n_users(self) -> int:
        return self.dataset.user_num

    @property
    def n_items(self) -> int:
        return self.dataset.item_num

    def resolve_user(self, identifier: str) -> ResolvedUser:
        """Map an Amazon hash OR mapped integer to a RecBole internal user id."""
        last_err: Exception | None = None
        for token in ids.candidate_user_tokens(identifier):
            try:
                internal = int(self.dataset.token2id(self.uid_field, token))
            except ValueError as e:  # token not known to the trained model
                last_err = e
                continue
            return ResolvedUser(
                internal_id=internal,
                token=token,
                amazon_hash=ids.amazon_hash_for(identifier, token),
            )
        raise KeyError(
            f"User '{identifier}' is not in the trained model "
            f"(unknown id, or filtered out by k-core). Original error: {last_err}"
        )

    def recommend(self, identifier: str, top_k: int = 10) -> tuple[ResolvedUser, list[Recommendation]]:
        """Top-k recommendations (already-seen items are masked to -inf)."""
        user = self.resolve_user(identifier)
        scores, indices = full_sort_topk(
            np.array([user.internal_id]), self.model, self.test_data, k=top_k, device=self.device
        )
        item_ids = indices[0].cpu().numpy()
        item_scores = scores[0].cpu().numpy()
        item_tokens = self.dataset.id2token(self.iid_field, item_ids)

        recs: list[Recommendation] = []
        for token, score in zip(item_tokens, item_scores):
            asin = ids.token_to_asin(str(token))
            if asin is None:
                continue
            recs.append(Recommendation(asin=asin, score=float(score)))
        return user, recs
