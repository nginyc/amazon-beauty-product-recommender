"""Standalone load/predict check — no Streamlit involved.

Run from the repo root:
    python app/smoke_test.py
    python app/smoke_test.py 93889         # try a specific user id / hash

If THIS crashes with a bus error too, the problem is torch/recbole/env, not
Streamlit (see the troubleshooting notes printed below). faulthandler will print
a C-level stack so you can see which library died.
"""
import faulthandler
import os
import sys

faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    from recommender import RecommenderAdaptor

    print("Loading adaptor (this rebuilds the RecBole dataset)...")
    adaptor = RecommenderAdaptor(device="cpu")
    print(f"OK: {adaptor.model_name} | {adaptor.n_users:,} users | {adaptor.n_items:,} items")

    # Pick the user id from argv, else a random internal user.
    if len(sys.argv) > 1:
        identifier = sys.argv[1]
    else:
        import random
        rid = random.randint(1, adaptor.n_users - 1)
        identifier = str(adaptor.dataset.id2token(adaptor.uid_field, rid))
    print(f"Recommending for user: {identifier}")

    user, recs = adaptor.recommend(identifier, top_k=10)
    print(f"Resolved -> internal {user.internal_id}, token {user.token}, hash {user.amazon_hash}")
    for r in recs:
        print(f"  {r.asin}   score={r.score:.4f}")


if __name__ == "__main__":
    main()
