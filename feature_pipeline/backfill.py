"""
Backfill historical training data.

Important limitation, discovered the hard way: AQICN's free token API has
NO general historical endpoint -- `/feed/` only returns the current
reading. There is no `fetch_aqicn_data(station, date=...)` call that
exists. Bulk AQICN history requires a separate institutional data-platform
request, not the standard token.

So this script backfills from OpenWeather's Air Pollution History API
instead (`/data/2.5/air_pollution/history`), which is free and covers data
back to late 2020. AQICN remains your LIVE data source going forward, via
the hourly feature_pipeline/run.py -- this script only fills in the past,
using a different (but real) source. State this explicitly in the report;
it's a genuine, worth-naming limitation, not something to hide.

A second, easier-to-miss issue: OpenWeather's raw PM2.5 concentration
(ug/m3) is NOT the same scale as AQI. An earlier version of this script
wrote that raw concentration directly into the `aqi` column, which would
have silently mixed two incompatible scales depending on whether a row
came from AQICN (live) or OpenWeather (backfilled) -- a real data
integrity defect, not just a documented limitation. pm25_to_aqi() below
converts using EPA's official breakpoints so both sources land on one
honest, comparable scale.
"""

import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from feature_pipeline.compute_features import (
    _safe_float,
    add_cyclical_time_features,
    add_derived_features,
)
from feature_pipeline.config import get_settings
from feature_pipeline.exceptions import DataFetchError
from feature_pipeline.logging_config import configure_logging
from feature_pipeline.push_to_hopsworks import push_features

logger = logging.getLogger(__name__)

# US EPA breakpoints for converting a PM2.5 concentration (ug/m3) into the
# standard 0-500 AQI scale. (bp_low, bp_high, aqi_low, aqi_high) per band.
# Source: EPA technical assistance document for the AQI (40 CFR Part 58,
# Appendix G) -- these are the official published breakpoints, not an
# approximation.
_PM25_AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25: float | None) -> float | None:
    """
    Convert a PM2.5 concentration into a real US AQI value using EPA's
    published linear-interpolation breakpoints.

    Why this function exists at all: an earlier version of this script
    wrote OpenWeather's raw PM2.5 concentration (ug/m3) directly into the
    `aqi` column, on the assumption it was "close enough" to an AQI value.
    It is not -- PM2.5 concentration and AQI are different scales
    entirely (e.g. a PM2.5 of 100 ug/m3 is a real AQI of roughly 174, not
    100). Silently blending that raw concentration into the same `aqi`
    column that live AQICN data populates would have meant the model's
    training target was inconsistent depending on which source produced
    each row -- a real data-integrity defect, not just an approximation.
    This function makes both sources land on the same, real AQI scale.

    Note this still uses a single hourly reading rather than the 24-hour
    rolling average EPA's official AQI technically specifies -- state
    this simplification explicitly in the report rather than presenting
    the backfilled AQI as identical in methodology to AQICN's live value.
    """
    if pm25 is None or pm25 < 0:
        return None
    if pm25 > 500.4:
        return 500.0

    for bp_lo, bp_hi, aqi_lo, aqi_hi in _PM25_AQI_BREAKPOINTS:
        if bp_lo <= pm25 <= bp_hi:
            return round((aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + aqi_lo, 1)
    return None


def fetch_openweather_pollution_history(start: datetime, end: datetime) -> list[dict]:
    """
    Fetch historical pollutant concentrations from OpenWeather for a date
    range. Returns a list of raw entries, each with a Unix timestamp and a
    `components` dict (pm2_5, pm10, no2, o3, ...) -- the same shape as the
    live `/air_pollution` endpoint already used in fetch_data.py, just
    covering the past instead of right now.
    """
    settings = get_settings()
    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={settings.karachi_lat}&lon={settings.karachi_lon}"
        f"&start={int(start.timestamp())}&end={int(end.timestamp())}"
        f"&appid={settings.openweather_api_key}"
    )
    response = requests.get(url, timeout=settings.request_timeout_seconds)
    if response.status_code != 200:
        raise DataFetchError(
            f"OpenWeather history request failed: {response.status_code} {response.text[:200]}"
        )
    return response.json().get("list", [])


def _openweather_entry_to_row(entry: dict) -> dict:
    """
    Convert one OpenWeather history entry into the same flat row shape
    build_feature_row() produces from AQICN, so the rest of the pipeline
    doesn't need to know which source a row came from.

    The `aqi` field is computed via pm25_to_aqi(), landing on the same
    real AQI scale AQICN's live data uses -- not OpenWeather's raw PM2.5
    concentration, and not OpenWeather's own separate 1-5 index either.
    """
    dt = datetime.fromtimestamp(entry["dt"], tz=UTC)
    event_time = dt.replace(minute=0, second=0, microsecond=0)
    components = entry.get("components", {})
    pm25 = _safe_float(components.get("pm2_5"))

    return {
        "timestamp": event_time.isoformat(),
        "collected_at": dt.isoformat(),
        "aqi": pm25_to_aqi(pm25),
        "pm25": pm25,
        "pm10": _safe_float(components.get("pm10")),
        "o3": _safe_float(components.get("o3")),
        "no2": _safe_float(components.get("no2")),
        "temperature": None,  # OpenWeather's history endpoint doesn't include weather
        "humidity": None,
        "wind_speed": None,
        "pressure": None,
    }


def backfill(days_back: int = 30) -> pd.DataFrame:
    """
    Backfill the last `days_back` days in weekly chunks (OpenWeather's
    history endpoint accepts large ranges, but chunking keeps individual
    requests small and makes a partial failure easier to retry).
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=days_back)

    all_rows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)
        logger.info("Fetching %s to %s...", chunk_start.date(), chunk_end.date())

        entries = fetch_openweather_pollution_history(chunk_start, chunk_end)
        all_rows.extend(_openweather_entry_to_row(e) for e in entries)

        chunk_start = chunk_end
        time.sleep(1)  # be polite to the API between chunks, not strictly required

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="timestamp")
    df = add_cyclical_time_features(df)
    df = add_derived_features(df)

    logger.info(
        "Backfilled %d rows spanning %s to %s",
        len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


if __name__ == "__main__":
    configure_logging()
    backfilled = backfill(days_back=30)

    # Push in batches rather than one enormous insert -- friendlier to the
    # Hopsworks API and easier to see progress on a large backfill.
    batch_size = 200
    for i in range(0, len(backfilled), batch_size):
        batch = backfilled.iloc[i : i + batch_size]
        push_features(batch)
        logger.info("Pushed batch %d-%d of %d", i, i + len(batch), len(backfilled))

    logger.info("Backfill complete.")