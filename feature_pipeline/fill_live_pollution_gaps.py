"""
One-off patch: fill o3/no2 for existing LIVE rows collected before the
OpenWeather pollution fallback was deployed in run.py.

Uses OpenWeather's Air Pollution HISTORY endpoint (fetch_openweather_
pollution_history(), same one backfill.py already uses) for the exact
hours affected -- not the live /air_pollution endpoint, which only
returns the CURRENT reading and can't answer "what was o3 at 2pm
yesterday". Converts via compute_features.py's o3_to_aqi()/no2_to_aqi()
and updates ONLY those two columns; aqi/pm25/pm10/temperature/etc. on
these rows are already correct AQICN-live values and are left untouched.

By the time this runs, every backfilled row should already have real
o3/no2 (see backfill.py) -- in practice this only ever affects the
handful of genuinely live rows collected before run.py's fallback
existed.

Run once, any time -- doesn't depend on run.py/fetch_data.py/config.py
having been pushed or deployed yet:
    python -m feature_pipeline.fill_live_pollution_gaps
"""

import logging

import pandas as pd

from feature_pipeline.backfill import fetch_openweather_pollution_history
from feature_pipeline.compute_features import _safe_float, no2_to_aqi, o3_to_aqi
from feature_pipeline.logging_config import configure_logging
from feature_pipeline.supabase_client import push_features, read_features

logger = logging.getLogger(__name__)


def fill_live_pollution_gaps() -> None:
    df = read_features()
    gaps = df[df["o3"].isna() | df["no2"].isna()].copy()

    if gaps.empty:
        logger.info("No rows with a null o3/no2 found -- nothing to fill.")
        return

    start = gaps["timestamp"].min().to_pydatetime()
    end = gaps["timestamp"].max().to_pydatetime() + pd.Timedelta(hours=1)
    logger.info(
        "Found %d row(s) with null o3/no2, spanning %s to %s -- fetching "
        "OpenWeather pollution history for that range.",
        len(gaps), start, end,
    )

    entries = fetch_openweather_pollution_history(start, end)
    by_hour = {
        pd.Timestamp(entry["dt"], unit="s", tz="UTC").floor("h"): entry.get("components", {})
        for entry in entries
    }

    filled_o3 = filled_no2 = 0
    for idx, row in gaps.iterrows():
        components = by_hour.get(row["timestamp"])
        if components is None:
            continue  # OpenWeather's history hasn't indexed this hour yet -- safe to re-run later
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

    logger.info("Filled o3 for %d row(s), no2 for %d row(s).", filled_o3, filled_no2)
    push_features(gaps)
    logger.info("Done.")


if __name__ == "__main__":
    configure_logging()
    fill_live_pollution_gaps()