"""
One-off migration: recompute pm25/pm10/o3/no2 for existing backfilled
rows now that backfill.py converts every pollutant to a real AQI
sub-index instead of writing OpenWeather's raw concentration.

This does NOT re-run backfill() -- it can't. backfill()'s own live-row
protection (checks temperature.notna()) would skip almost every row in
the table, including old backfilled rows, since Open-Meteo already
filled real weather into most of them. Instead this reads what's already
stored, identifies which rows were produced by the OLD (pre-fix) backfill
code, and recomputes those specific columns in place -- no re-fetching
from OpenWeather required, since the raw concentration values are
already sitting in the table (just mislabeled as if they were
sub-indices already).

A row is identified as "old-style backfilled" by a precise fingerprint
tied directly to the bug: the old code computed
`aqi = pm25_to_aqi(raw_pm25)` and `pm25 = raw_pm25` (unconverted) for
every backfilled row -- so aqi == pm25_to_aqi(pm25) holds almost exactly
for those rows (floating-point rounding aside), and essentially never
holds by coincidence for a genuinely live AQICN row, whose top-level aqi
is the max across ALL pollutant sub-indices, not derived from pm25 alone.

Run once, after deploying the updated backfill.py/compute_features.py:
    python -m feature_pipeline.migrate_pollutant_scale
"""

import logging

import pandas as pd

from feature_pipeline.compute_features import no2_to_aqi, o3_to_aqi, pm10_to_aqi, pm25_to_aqi
from feature_pipeline.logging_config import configure_logging
from feature_pipeline.supabase_client import push_features, read_features

logger = logging.getLogger(__name__)

_MATCH_TOLERANCE = 0.05  # floating-point rounding slack for the fingerprint check


def find_old_style_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where aqi == pm25_to_aqi(pm25) -- the exact computation the
    old backfill code performed. See module docstring for why this is a
    reliable fingerprint rather than a coincidence."""
    recomputed_aqi = df["pm25"].apply(pm25_to_aqi)
    is_old_style = (recomputed_aqi - df["aqi"]).abs() <= _MATCH_TOLERANCE
    return df[is_old_style.fillna(False)].copy()


def migrate() -> None:
    df = read_features()
    old_style = find_old_style_rows(df)

    if old_style.empty:
        logger.info("No old-style backfilled rows found -- nothing to migrate.")
        return

    logger.info(
        "Found %d old-style row(s) spanning %s to %s -- recomputing pm25/pm10/o3/no2.",
        len(old_style), old_style["timestamp"].min(), old_style["timestamp"].max(),
    )

    # `aqi` is already correct (that's the basis of the fingerprint above)
    # -- left untouched. Only pm25/pm10/o3/no2 need the fix; the raw
    # concentrations already sitting in these columns are recomputed
    # in place, not re-fetched.
    old_style["pm25"] = old_style["pm25"].apply(pm25_to_aqi)
    old_style["pm10"] = old_style["pm10"].apply(pm10_to_aqi)
    old_style["o3"] = old_style["o3"].apply(o3_to_aqi)
    old_style["no2"] = old_style["no2"].apply(no2_to_aqi)

    batch_size = 200
    for i in range(0, len(old_style), batch_size):
        batch = old_style.iloc[i : i + batch_size]
        push_features(batch)
        logger.info("Migrated batch %d-%d of %d", i, i + len(batch), len(old_style))

    logger.info("Migration complete.")


if __name__ == "__main__":
    configure_logging()
    migrate()