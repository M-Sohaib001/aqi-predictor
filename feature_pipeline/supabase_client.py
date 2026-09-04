"""
Feature store AND model registry, both backed by Supabase (Postgres +
Storage) -- replacing Hopsworks after repeatedly hitting its serverless
free-tier quota mid-project. See Part 3 for the full rationale.

Run this only after fetch_data.py and compute_features.py are both
verified working on their own -- debugging feature math is much easier
before a feature-store client is in the loop.

Design mirrors what a Hopsworks-backed version did, one-to-one where
possible:
- aqi_features table: same primary-key/event-time upsert idempotency
  (a retried GitHub Actions run in the same hour overwrites, not
  duplicates).
- models table: same "register every candidate under its own name, then
  register the day's winner again under a shared per-horizon name"
  pattern that makes cross-algorithm comparison possible at serving time
  (see training_pipeline/train.py's run_horizon).

"""

import logging
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from supabase import Client, create_client

from feature_pipeline.config import get_settings
from feature_pipeline.exceptions import FeatureStoreError

logger = logging.getLogger(__name__)

FEATURES_TABLE = "aqi_features"
MODELS_TABLE = "models"
MODELS_BUCKET = "models"

# How many past versions of a champion's artifact FILE to keep in
# Storage. Metrics rows in the `models` table are never deleted by this
# -- old accuracy history stays fully visible in the report even after
# old files are pruned; see the module docstring above.
CHAMPION_RETENTION = 5

# Schema guard: if compute_features.py changes and drops/renames a
# column, this catches it immediately with a clear error, instead of an
# opaque Postgres-side type mismatch several layers down.
REQUIRED_COLUMNS = {
    "timestamp", "collected_at", "aqi", "pm25", "pm10", "o3", "no2",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
}

# Every one of these must be a real numeric dtype before insert -- same
# reasoning a Hopsworks-backed version needed: if a column is missing a
# sensor reading (see _safe_float in compute_features.py) it holds None,
# fine on its own, but if a batch has NO real value anywhere in that
# column (most likely on the very first, single-row insert), pandas
# can't infer a concrete dtype from an all-None column. Explicitly
# casting to float64 turns a missing value into NaN (a real float), not
# an untyped None that would serialize oddly to JSON.
NUMERIC_COLUMNS = {
    "aqi", "pm25", "pm10", "o3", "no2",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
}


@lru_cache
def _get_client() -> Client:
    """Cached so a single pipeline run only creates one client, even if
    push_features(), read_features(), and the registry helpers below are
    all called in the same process."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise FeatureStoreError(f"Dataframe is missing expected columns: {sorted(missing)}")


def push_features(df: pd.DataFrame, table: str = FEATURES_TABLE) -> None:
    """
    Upsert a dataframe into the aqi_features table. `timestamp`
    (hour-truncated, see compute_features.py) is the primary key, so a
    re-run in the same hour upserts rather than duplicates -- the same
    idempotency guarantee a Hopsworks-backed version relied on.
    """
    _validate_schema(df)

    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype("float64")

    # By the time this is called, `timestamp`/`collected_at` are usually
    # already real pandas datetimes (add_cyclical_time_features converts
    # the ISO string compute_features.py originally produced) -- but a
    # caller that read existing rows back out of Supabase and is pushing
    # them again (e.g. a migration/patch script) will have `collected_at`
    # as a raw string instead, and Postgres trims trailing zeros from its
    # fractional seconds inconsistently across rows. format="mixed" infers
    # each value's format individually rather than assuming one fixed
    # format for the whole column -- confirmed necessary in practice, not
    # just a defensive guess (a real batch crashed without it). Safe to
    # apply even when the column is already real datetimes; pandas treats
    # that as a no-op.
    for col in ("timestamp", "collected_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed").apply(
                lambda t: t.isoformat() if pd.notna(t) else None
            )

    # JSON has no NaN -- convert pandas' missing-value markers to None so
    # the client sends real JSON nulls, not the literal string "NaN".
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")

    try:
        client = _get_client()
        client.table(table).upsert(records, on_conflict="timestamp").execute()
    except Exception as exc:
        raise FeatureStoreError(f"Failed to push features: {exc}") from exc

    logger.info("Upserted %d row(s) into '%s'", len(records), table)


def read_features(table: str = FEATURES_TABLE, since: datetime | None = None) -> pd.DataFrame:
    """Read the feature table back out -- used by run.py to compute
    rolling/lag features, and by the training pipeline and dashboard.

    since: optional lower bound on timestamp. Passed through as a
    server-side .gte() filter -- callers that only need a recent window
    (e.g. the dashboard's load_recent_data) get back just that window in
    ONE paginated pass instead of downloading the entire table and
    slicing it client-side afterward. Left as None (default) for callers
    that genuinely need full history, like training_pipeline/train.py's
    read_features() call before its own TRAINING_WINDOW_DAYS filter.

    Paginated: PostgREST caps a single response at its project-level
    max-rows setting (1000 by default), so a plain select("*") silently
    truncates once the table grows past that -- no error, just fewer
    rows than actually exist. Looping with .range() until a page comes
    back short of page_size is what makes this correct past that cap.
    """
    client = _get_client()
    page_size = 1000
    all_rows: list[dict] = []
    start = 0

    try:
        while True:
            query = client.table(table).select("*")
            if since is not None:
                query = query.gte("timestamp", since.isoformat())
            response = (
                query
                .order("timestamp")
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = response.data
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
    except Exception as exc:
        raise FeatureStoreError(f"Failed to read features: {exc}") from exc

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # format="mixed": Postgres trims trailing zeros from fractional
        # seconds inconsistently across rows, which breaks pandas' fast
        # single-format parser once enough rows are read at once (seen in
        # practice on `collected_at`, not just a hypothetical) --
        # `timestamp` is hour-truncated so it never has fractional
        # seconds to begin with, but parsed the same way for consistency.
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df["collected_at"] = pd.to_datetime(df["collected_at"], format="mixed")
    return df


# --- Model registry ---------------------------------------------------

def _next_version(name: str) -> int:
    client = _get_client()
    response = (
        client.table(MODELS_TABLE)
        .select("version")
        .eq("name", name)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    return (response.data[0]["version"] + 1) if response.data else 1


def _prune_old_champion_artifacts(name: str) -> None:
    """Keep only the CHAMPION_RETENTION most recent artifact FILES in
    Storage for this name -- the `models` rows themselves (and their
    metrics) are left in place; only the Storage object is removed, and
    storage_path is cleared to reflect that."""
    client = _get_client()
    response = (
        client.table(MODELS_TABLE)
        .select("id, version, framework, storage_path")
        .eq("name", name)
        .not_.is_("storage_path", "null")
        .order("version", desc=True)
        .execute()
    )
    for row in response.data[CHAMPION_RETENTION:]:
        filename = "model.pkl" if row["framework"] == "sklearn" else "model.keras"
        client.storage.from_(MODELS_BUCKET).remove([f"{row['storage_path']}{filename}"])
        client.table(MODELS_TABLE).update({"storage_path": None}).eq("id", row["id"]).execute()


def save_model(
    model,
    framework: str,
    name: str,
    metrics: dict,
    description: str,
    scaler_stats: dict | None = None,
    upload_artifact: bool = True,
) -> None:
    """
    Register a trained model's metrics, and -- when upload_artifact=True
    -- upload its serialized artifact to Storage too. Per-algorithm
    candidates should be registered with upload_artifact=False (metrics
    only); the day's per-horizon champion should be registered again
    with upload_artifact=True, since that's the only entry serving code
    actually downloads (see dashboard/model_loader.py). See the module
    docstring for why this differs from a Hopsworks-backed version, which
    uploaded a full artifact for every candidate.

    scaler_stats (the LSTM's input mean/std, needed to reproduce its
    exact scaling at serving time) is stored directly in this table's
    jsonb column -- no separate scaler.json file in Storage is needed,
    since the metadata row already carries it.
    """
    client = _get_client()
    version = _next_version(name)
    storage_path = None

    if upload_artifact:
        prefix = f"{name}/v{version}/"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            if framework == "sklearn":
                import joblib

                local_path = tmp_dir / "model.pkl"
                joblib.dump(model, local_path)
                client.storage.from_(MODELS_BUCKET).upload(f"{prefix}model.pkl", str(local_path))
            elif framework == "tensorflow":
                local_path = tmp_dir / "model.keras"
                model.save(local_path)
                client.storage.from_(MODELS_BUCKET).upload(f"{prefix}model.keras", str(local_path))
            else:
                raise ValueError(f"Unknown framework: {framework}")
        storage_path = prefix

    client.table(MODELS_TABLE).insert({
        "name": name,
        "framework": framework,
        "version": version,
        "metrics": metrics,
        "description": description,
        "storage_path": storage_path,
        "scaler_stats": scaler_stats,
    }).execute()

    logger.info(
        "Registered '%s' v%d (framework=%s, artifact=%s) with metrics %s",
        name, version, framework, upload_artifact, metrics,
    )

    if upload_artifact:
        _prune_old_champion_artifacts(name)


def get_best_model(name: str, metric: str = "rmse", direction: str = "min") -> dict:
    """
    Best row for `name` by `metric`, across every version ever
    registered WITH a stored artifact -- the same semantics a Hopsworks-
    backed version's mr.get_best_model(name, metric, direction) had.
    Fetched and reduced client-side rather than via a database-side
    ORDER BY on a jsonb field, since PostgREST's simple query API doesn't
    support ordering by an arbitrary JSON path -- fine at this project's
    scale (at most a few hundred rows per name over its lifetime).
    """
    client = _get_client()
    response = client.table(MODELS_TABLE).select("*").eq("name", name).execute()
    rows = [r for r in response.data if r.get("storage_path")]
    if not rows:
        raise FeatureStoreError(f"No model with a stored artifact found for '{name}'")

    reducer = min if direction == "min" else max
    return reducer(rows, key=lambda r: r["metrics"][metric])


def download_model_artifact(entry: dict) -> Path:
    """Download the winning entry's serialized model file from Storage
    into a local temp directory, and return that directory -- mirrors
    what entry.download() returned in a Hopsworks-backed version."""
    client = _get_client()
    tmp_dir = Path(tempfile.mkdtemp())
    filename = "model.pkl" if entry["framework"] == "sklearn" else "model.keras"

    data = client.storage.from_(MODELS_BUCKET).download(f"{entry['storage_path']}{filename}")
    (tmp_dir / filename).write_bytes(data)
    return tmp_dir