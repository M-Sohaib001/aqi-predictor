"""
Load the best-performing model per forecast horizon from the Supabase-
backed model registry. Loaded model objects are cached in memory (not
just the downloaded files) -- re-downloading and re-deserializing on
every single API request would be needlessly slow for something that
only changes once a day, when the training pipeline runs.
"""

import logging
from functools import lru_cache

import joblib

from feature_pipeline.supabase_client import download_model_artifact, get_best_model

logger = logging.getLogger(__name__)

HORIZONS = (24, 48, 72)


@lru_cache
def load_champion(horizon_hours: int):
    """
    Returns a dict describing the current best model for this horizon:
    {"model": <loaded model object>, "kind": "sklearn" | "tensorflow",
     "scaler": {"mean": float, "std": float} | None}.

    "kind" comes directly from the registry row's `framework` column --
    a single source of truth, rather than a Hopsworks-backed version's
    approach of re-deriving it from which file happened to exist on disk
    after download (model.pkl vs model.keras). "scaler" comes straight
    off the same row too -- no separate scaler.json download needed.
    """
    entry = get_best_model(f"aqi_forecast_{horizon_hours}h", "rmse", "min")
    model_dir = download_model_artifact(entry)
    kind = entry["framework"]

    if kind == "sklearn":
        model = joblib.load(model_dir / "model.pkl")
    elif kind == "tensorflow":
        from tensorflow.keras.models import load_model

        model = load_model(model_dir / "model.keras")
    else:
        raise ValueError(f"Unknown framework in registry: {kind}")

    logger.info(
        "Loaded champion for %dh horizon: kind=%s (v%d)",
        horizon_hours, kind, entry["version"],
    )
    return {"model": model, "kind": kind, "scaler": entry.get("scaler_stats")}


def clear_cache() -> None:
    """Call this after retraining if you want the dashboard to pick up a
    newly-registered champion without restarting the process."""
    load_champion.cache_clear()