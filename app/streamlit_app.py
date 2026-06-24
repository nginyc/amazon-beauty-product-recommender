"""Streamlit webapp: assume any user, inspect their history, get recommendations.

Run from the repo root:
    streamlit run app/streamlit_app.py

Recommendation modes (sidebar)
------------------------------
* Subprocess (default) : predictions run in a SEPARATE python process
  (`predict_cli.py`). torch / RecBole are never imported into this Streamlit
  process, which avoids the macOS "bus error" seen when loading torch under
  Streamlit. The model is loaded fresh per prediction (cached by Streamlit).
* In-process           : load the model inside Streamlit (faster repeat calls,
  but can hard-crash on some setups).
* Disabled             : don't load the model at all — just browse a user's
  history and item images. Nothing torch-related is imported.

The reviews / meta files are multi-GB and scanned live; scans are cached.
"""
from __future__ import annotations

# Dump a C-level stack trace on a fatal signal (SIGBUS/SIGSEGV) instead of a
# bare "bus error". (Only matters for In-process mode; harmless otherwise.)
import faulthandler
faulthandler.enable()

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import streamlit as st

# Make sibling modules importable when launched as `streamlit run app/streamlit_app.py`.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# NOTE: we deliberately do NOT import `recommender` (torch/recbole) at module
# load. These are light, torch-free helpers only.
import data_access
import ids
from paths import DEFAULT_MODEL_PATH, REVIEWS_JSONL, META_JSONL

st.set_page_config(page_title="Beauty Recommender", layout="wide")


# --- predictions -------------------------------------------------------------
def run_predictions_subprocess(model_path: str, identifier: str, top_k: int) -> dict:
    """Invoke predict_cli.py in a separate process; return its JSON result."""
    cli = os.path.join(APP_DIR, "predict_cli.py")
    env = {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    try:
        proc = subprocess.run(
            [sys.executable, cli, "--model", model_path, "--user", identifier,
             "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=1200, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Prediction timed out (model load too slow)."}

    for line in proc.stdout.splitlines():
        if line.startswith("__PRED_JSON__"):
            try:
                return json.loads(line[len("__PRED_JSON__"):])
            except json.JSONDecodeError:
                break

    # No result: surface the full stderr + the persisted crash log (full stack).
    crash = ""
    crash_path = os.path.join(APP_DIR, "worker_crash.log")
    if os.path.exists(crash_path):
        try:
            with open(crash_path) as f:
                crash = f.read()
        except OSError:
            pass
    sig = f" (signal/exit {proc.returncode})" if proc.returncode else ""
    detail = (proc.stderr or "")[-6000:]
    if crash:
        detail += "\n\n--- worker_crash.log ---\n" + crash[-6000:]
    return {"ok": False, "error": f"No result from worker{sig}.", "trace": detail}


@st.cache_data(show_spinner=False)
def cached_predictions_subprocess(model_path: str, identifier: str, top_k: int) -> dict:
    return run_predictions_subprocess(model_path, identifier, top_k)


@st.cache_resource(show_spinner="Loading model in-process (one-time)...")
def get_inprocess_adaptor(model_path: str):
    from recommender import RecommenderAdaptor  # lazy: imports torch/recbole
    return RecommenderAdaptor(model_path=model_path, device="cpu")


# --- cached scans ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_user_reviews(amazon_hash: str, max_lines: int | None) -> list[dict]:
    return data_access.scan_user_reviews(amazon_hash, max_lines=max_lines)


@st.cache_data(show_spinner=False)
def cached_item_meta(asins: tuple[str, ...], max_lines: int | None) -> dict[str, dict]:
    return data_access.fetch_item_meta(asins, max_lines=max_lines)


# --- helpers -----------------------------------------------------------------
def fmt_ts(ts: int) -> str:
    try:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


def stars(rating) -> str:
    try:
        n = int(round(float(rating)))
    except (TypeError, ValueError):
        return ""
    return "★" * n + "☆" * (5 - n)


def resolve_for_history(identifier: str) -> tuple[str | None, str | None]:
    """Model-free resolution: return (token, amazon_hash) for review scanning."""
    identifier = identifier.strip()
    toks = ids.candidate_user_tokens(identifier)
    token = toks[0] if toks else None
    amazon_hash = ids.amazon_hash_for(identifier, token) if token else None
    if amazon_hash is None and not identifier.isdigit():
        amazon_hash = identifier  # allow scanning a raw hash even if not in the map
    return token, amazon_hash


# --- sidebar -----------------------------------------------------------------
st.sidebar.header("Settings")
rec_mode = st.sidebar.radio(
    "Recommendations",
    ["Subprocess (isolated)", "In-process", "Disabled"],
    index=0,
    help="Subprocess keeps torch out of Streamlit and avoids the bus error.",
)
model_path = st.sidebar.text_input("Model checkpoint", value=str(DEFAULT_MODEL_PATH))
top_k = st.sidebar.slider("Recommendations (top-k)", 5, 50, 12)
max_hist = st.sidebar.number_input(
    "Max history items to display", min_value=5, max_value=500, value=40, step=5
)
scan_cap = st.sidebar.number_input(
    "Max lines to scan (0 = entire file, slow)", min_value=0, value=0, step=500_000,
    help="Caps how deep into the jsonl files each live scan reads.",
)
max_lines = None if scan_cap == 0 else int(scan_cap)
st.sidebar.caption(f"reviews: {REVIEWS_JSONL.name}")
st.sidebar.caption(f"meta: {META_JSONL.name}")

# --- header ------------------------------------------------------------------
st.title("🧴 Beauty & Personal Care — Recommender Explorer")
st.caption(f"Recommendation mode: **{rec_mode}**")

# The app needs the rebuilt id maps (original Amazon ids <-> model integer ids).
if not ids.maps_ready():
    st.error(
        "ID maps not built yet. The saved `data/user_map.json` / `item_map.json` "
        "do **not** match the model, so they must be reconstructed once from "
        "`reviews.csv`:\n\n```\npython app/build_id_maps.py\n```\n\n"
        "Then reload this page."
    )
    st.stop()

# --- user selection ----------------------------------------------------------
st.subheader("Assume a user")
col_in, col_btn = st.columns([4, 1])
with col_in:
    st.text_input(
        "User id — Amazon hash (e.g. AFKZENTNBQ7A7V7UXW5JJI6UGRYQ) or mapped integer (e.g. 93889)",
        key="user_identifier",
    )
with col_btn:
    st.write("")
    st.write("")
    if st.button("🎲 Random user (mapped int)"):
        import random
        st.session_state["user_identifier"] = str(random.choice(ids.sample_user_ints(2000)))
        st.rerun()

identifier = st.session_state.get("user_identifier", "").strip()
if not identifier:
    st.info("Enter a user id above, or hit **Random user**, to begin.")
    st.stop()

token, amazon_hash = resolve_for_history(identifier)
st.markdown(f"**User** → token `{token or 'n/a'}` · amazon hash `{amazon_hash or 'n/a'}`")

# --- recommendations ---------------------------------------------------------
recs: list[dict] = []
if rec_mode == "Disabled":
    st.info("Recommendations are disabled. Showing interaction history only.")
elif rec_mode == "Subprocess (isolated)":
    with st.spinner("Running recommender in a separate process..."):
        res = cached_predictions_subprocess(model_path, identifier, int(top_k))
    if res.get("ok"):
        recs = res["recommendations"]
        u = res.get("user", {})
        st.caption(
            f"Model: {res.get('model')} · internal id {u.get('internal_id')} · "
            f"token {u.get('token')}"
        )
        if u.get("amazon_hash"):
            amazon_hash = amazon_hash or u["amazon_hash"]
    else:
        st.error(res.get("error", "Prediction failed."))
        with st.expander("Worker traceback"):
            st.code(res.get("trace", "(none)"))
else:  # In-process
    try:
        adaptor = get_inprocess_adaptor(model_path)
        user, rec_objs = adaptor.recommend(identifier, top_k=int(top_k))
        recs = [{"asin": r.asin, "score": r.score} for r in rec_objs]
        amazon_hash = amazon_hash or user.amazon_hash
        st.caption(f"Model: {adaptor.model_name} · internal id {user.internal_id}")
    except Exception as e:  # noqa: BLE001
        st.error(f"In-process model failed: {type(e).__name__}: {e}")

# --- history scan ------------------------------------------------------------
history: list[dict] = []
if amazon_hash:
    with st.spinner(f"Scanning reviews for {amazon_hash} ..."):
        history = cached_user_reviews(amazon_hash, max_lines)
else:
    st.info("No Amazon hash for this id, so review history can't be scanned.")
history_view = history[: int(max_hist)]

# --- one meta scan for everything we display --------------------------------
asins_needed = tuple(
    sorted({r["asin"] for r in history_view if r["asin"]} | {r["asin"] for r in recs})
)
meta: dict[str, dict] = {}
if asins_needed:
    with st.spinner(f"Fetching images / titles for {len(asins_needed)} items ..."):
        meta = cached_item_meta(asins_needed, max_lines)


def render_item(asin, *, score=None, rating=None, review_title="", when=""):
    m = meta.get(asin, {})
    img = m.get("image")
    if img:
        try:
            st.image(img, use_container_width=True)
        except TypeError:
            st.image(img, use_column_width=True)
    else:
        st.markdown(
            "<div style='height:140px;display:flex;align-items:center;"
            "justify-content:center;background:#f0f0f0;border-radius:8px;'>no image</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"**{(m.get('title') or asin)[:90]}**")
    bits = [f"`{asin}`"]
    if m.get("average_rating") is not None:
        bits.append(f"avg {m['average_rating']}★")
    if score is not None:
        bits.append(f"score {score:.3f}")
    if rating is not None:
        bits.append(f"rated {stars(rating)}")
    if when:
        bits.append(when)
    st.caption(" · ".join(bits))
    if review_title:
        st.caption(f"💬 *{review_title[:80]}*")


# --- render recommendations --------------------------------------------------
if rec_mode != "Disabled":
    st.subheader(f"Recommended for this user (top {len(recs)})")
    if not recs:
        st.write("No recommendations produced.")
    else:
        cols = st.columns(4)
        for i, r in enumerate(recs):
            with cols[i % 4]:
                render_item(r["asin"], score=r.get("score"))

# --- render history ----------------------------------------------------------
st.subheader(f"Interaction history ({len(history)} reviews found)")
if not history_view:
    st.write("No reviews found for this user in the scanned range.")
else:
    cols = st.columns(4)
    for i, rv in enumerate(history_view):
        with cols[i % 4]:
            render_item(
                rv["asin"], rating=rv["rating"],
                review_title=rv["title"], when=fmt_ts(rv["timestamp"]),
            )
