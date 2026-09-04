"""
One-off patch: fill o3/no2 for rows where OpenWeather's raw pollutant
response was missing those two fields at fetch time -- both the handful
of live rows collected before run.py's fallback existed, and any
backfilled rows where OpenWeather's own history endpoint simply didn't
have o3/no2 for that particular hour (a real, scattered source gap, not
a bug in this pipeline -- worth naming in the report alongside the other
documented source limitations).

Uses OpenWeather's Air Pollution HISTORY endpoint (fetch_openweather_
pollution_history(), same one backfill.py already uses) for the exact
hours affected -- not the live /air_pollution endpoint, which only
returns the CURRENT reading. Converts via compute_features.py's
o3_to_aqi()/no2_to_aqi() and updates ONLY those two columns.

Chunks the fetch the same way backfill.py does -- a single request
spanning years of history times out in practice, confirmed against a
real run, not a hypothetical -- but only over the specific week-sized
windows that actually contain at least one gap, rather than blindly
re-walking the entire table's history for a scattered small fraction of it.

Run once, any time -- doesn't depend on run.py/fetch_data.py/config.py
having been pushed or deployed yet:
    python -m feature_pipeline.fill_live_pollution_gaps
"""

import logging
import time
from collections.abc import Iterator

import pandas as pd

from feature_pipeline.backfill import fetch_openweather_pollution_history
from feature_pipeline.compute_features import _safe_float, no2_to_aqi, o3_to_aqi
from feature_pipeline.logging_config import configure_logging
from feature_pipeline.supabase_client import push_features, read_features

logger = logging.getLogger(__name__)


def _chunk_ranges_for_gaps(
    gap_timestamps: pd.Series, chunk_days: int = 7
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield only the (start, end) weekly windows that actually contain at
    least one gap timestamp -- fetching OpenWeather history for the full
    span of the table when only a small fraction of it needs filling
    wastes most of the requests on stretches with nothing to fix."""
    start = gap_timestamps.min().floor("D")
    end = gap_timestamps.max().floor("D") + pd.Timedelta(days=1)
    gap_set = set(gap_timestamps)

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)
        if any(chunk_start <= ts < chunk_end for ts in gap_set):
            yield chunk_start, chunk_end
        chunk_start = chunk_end


def fill_live_pollution_gaps() -> None:
    df = read_features()

    # read_features() only parses `timestamp` into a real datetime, never
    # `collected_at` -- pushing that straight back through push_features()
    # later re-parses it from Postgres' own string representation, which
    # has inconsistent fractional-second precision across rows and can
    # fail outright on a large enough batch (confirmed in practice).
    # Parsing it here, once, up front avoids that entirely.
    df["collected_at"] = pd.to_datetime(df["collected_at"], format="mixed")

    gaps = df[df["o3"].isna() | df["no2"].isna()].copy()
    if gaps.empty:
        logger.info("No rows with a null o3/no2 found -- nothing to fill.")
        return

    logger.info(
        "Found %d row(s) with null o3/no2, spanning %s to %s.",
        len(gaps), gaps["timestamp"].min(), gaps["timestamp"].max(),
    )

    by_hour: dict[pd.Timestamp, dict] = {}
    for chunk_start, chunk_end in _chunk_ranges_for_gaps(gaps["timestamp"]):
        logger.info("Fetching pollution history for %s to %s...", chunk_start.date(), chunk_end.date())
        entries = fetch_openweather_pollution_history(chunk_start.to_pydatetime(), chunk_end.to_pydatetime())
        for entry in entries:
            dt = pd.Timestamp(entry["dt"], unit="s", tz="UTC").floor("h")
            by_hour[dt] = entry.get("components", {})
        time.sleep(1)  # be polite to the API between chunks, matching backfill.py

    filled_o3 = filled_no2 = still_missing = 0
    for idx, row in gaps.iterrows():
        components = by_hour.get(row["timestamp"])
        if components is None:
            still_missing += 1
            continue
        if pd.isna(row["o3"]):
            new_o3 = o3_to_aqi(_safe_float(components.get("o3")))
            if new_o3 is not None:
                gaps.at[idx, "o3"] = new_o3
                filled_o3 += 1
        if pd.isna(row["no2"]):
            new_no2 = no2_to_aqi(_safe_float(components.get("no2")))
            if new_no2 is not None:
                gaps.at[idx, "no2"] = new_no2
                filled_no2 += 1

    logger.info(
        "Filled o3 for %d row(s), no2 for %d row(s); %d row(s) had no "
        "matching OpenWeather history entry at all (a genuine source gap, "
        "not a bug -- safe to leave null).",
        filled_o3, filled_no2, still_missing,
    )

    batch_size = 200
    for i in range(0, len(gaps), batch_size):
        batch = gaps.iloc[i : i + batch_size]
        push_features(batch)
        logger.info("Pushed batch %d-%d of %d", i, i + len(batch), len(gaps))

    logger.info("Done.")


if __name__ == "__main__":
    configure_logging()
    fill_live_pollution_gaps()