"""Recommendation CLI — loads the model and prints top-k predictions as JSON.

Designed to be invoked as a SEPARATE process (e.g. from the Streamlit app via
subprocess, or by hand) so that torch / RecBole are never imported into the
parent process. This isolates any native crash ("bus error") to this short-lived
worker instead of taking down the UI.

Usage:
    python app/predict_cli.py --user 93889 --top-k 12
    python app/predict_cli.py --user AFKZENTNBQ7A7V7UXW5JJI6UGRYQ --model /path/to.pth

Output: a single line beginning with the marker `__PRED_JSON__` followed by a
JSON object. Everything else on stdout/stderr is logging and can be ignored.
"""
import argparse
import faulthandler
import json
import os
import sys
import traceback

# Common workarounds for native SIGBUS/segfault on macOS torch stacks. Must be
# set BEFORE torch / mkl / openmp are imported (recommender imports them).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# Persist the full fatal-signal traceback so it isn't lost to stderr truncation.
_crash_log = open(os.path.join(APP_DIR, "worker_crash.log"), "w")
faulthandler.enable(file=_crash_log, all_threads=True)
faulthandler.enable()  # also to stderr

MARKER = "__PRED_JSON__"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Amazon user hash or mapped integer id")
    ap.add_argument("--model", default=None, help="Path to the RecBole checkpoint (.pth)")
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    try:
        print("[worker] importing torch/recbole...", file=sys.stderr, flush=True)
        from recommender import RecommenderAdaptor

        print("[worker] loading model + dataset...", file=sys.stderr, flush=True)
        adaptor = RecommenderAdaptor(model_path=args.model, device="cpu")
        user, recs = adaptor.recommend(args.user, top_k=args.top_k)
        result = {
            "ok": True,
            "model": adaptor.model_name,
            "user": {
                "internal_id": user.internal_id,
                "token": user.token,
                "amazon_hash": user.amazon_hash,
            },
            "recommendations": [{"asin": r.asin, "score": r.score} for r in recs],
        }
    except Exception as e:  # noqa: BLE001
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }

    sys.stdout.write("\n" + MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
