"""
Push a dataframe of computed features into the Hopsworks Feature Store.

Run this only after fetch_data.py and compute_features.py are both
verified working on their own -- debugging feature math is much easier
before a feature-store SDK is in the loop.
"""

import logging
from functools import lru_cache

import hopsworks
import pandas as pd

from feature_pipeline.config import get_settings
from feature_pipeline.exceptions import FeatureStoreError

logger = logging.getLogger(__name__)

# Schema guard: if compute_features.py changes and drops/renames a column,
# this catches it immediately with a clear error, instead of an opaque
# Hopsworks-side type mismatch several layers down.
REQUIRED_COLUMNS = {
    "timestamp", "collected_at", "aqi", "pm25", "pm10", "o3", "no2",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
}

# Every one of these must be a real numeric dtype before insert. If a
# column is missing a sensor reading (see _safe_float in
# compute_features.py) it holds None -- fine on its own, but if a batch
# has NO real value anywhere in that column (most likely on a brand new
# feature group's very first, single-row insert), pandas can't infer a
# concrete dtype from an all-None column and leaves it as an untyped
# "object"/"null" column. Hopsworks refuses to create a feature-store
# column with no knowable type -- explicitly casting to float64 turns a
# missing value into NaN (a real float), not an untyped None.
NUMERIC_COLUMNS = {
    "aqi", "pm25", "pm10", "o3", "no2",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
}


@lru_cache
def _get_project():
    """Cached so a single pipeline run only logs into Hopsworks once,
    even if both push_features() and read_features() are called."""
    settings = get_settings()
    return hopsworks.login(
        api_key_value=settings.hopsworks_api_key,
        project=settings.hopsworks_project_name,
    )


def get_feature_store():
    return _get_project().get_feature_store()


def get_model_registry():
    """Public accessor, reusing the same cached login as get_feature_store()
    -- training_pipeline/train.py needs the model registry, not the feature
    store, but both come from the same underlying project object."""
    return _get_project().get_model_registry()


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise FeatureStoreError(f"Dataframe is missing expected columns: {sorted(missing)}")


def push_features(
    df: pd.DataFrame,
    feature_group_name: str = "aqi_features",
    version: int = 1,
) -> None:
    """
    Insert a dataframe into (or create, on first run) a Hopsworks feature
    group. `timestamp` (hour-truncated, see compute_features.py) is both
    the primary key and the event-time column, so a re-run in the same
    hour upserts rather than duplicates.
    """
    _validate_schema(df)

    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype("float64")

    # Hopsworks requires the event-time column to be a real
    # TIMESTAMP/DATE/BIGINT type, not text that merely looks like one.
    # compute_features.py deliberately stores `timestamp` as an ISO
    # string (simple, JSON-safe) -- this is the boundary where it needs
    # to become a real datetime before Hopsworks ever sees it.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    try:
        fs = get_feature_store()
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            description="Hourly AQI, pollutant, and weather features for Karachi",
            primary_key=["timestamp"],
            event_time="timestamp",
            time_travel_format="HUDI",
        )
        feature_group.insert(df)
    except Exception as exc:
        raise FeatureStoreError(f"Failed to push features: {exc}") from exc

    logger.info("Inserted %d row(s) into '%s' v%d", len(df), feature_group_name, version)


def read_features(feature_group_name: str = "aqi_features", version: int = 1) -> pd.DataFrame:
    """Read the full feature group back out -- used by run.py to compute
    rolling/lag features, and by the training pipeline later."""
    try:
        fs = get_feature_store()
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        return feature_group.read()
    except Exception as exc:
        raise FeatureStoreError(f"Failed to read features: {exc}") from exc