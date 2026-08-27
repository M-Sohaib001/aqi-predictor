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

A third issue, relevant specifically when re-running this script after
some live data already exists: backfill() always covers "the last N days
from now," so a second run would otherwise silently overwrite genuine
live AQICN+weather rows with the lower-fidelity OpenWeather-derived
version for the same hours, since the feature table upserts on the
shared hour-truncated timestamp. backfill() now checks existing data first and
skips any hour that already has real weather data (a marker only live
rows ever have), so re-running this to densify a sparse recent gap is
safe rather than a quiet quality regression.
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
    handle_outliers,
)
from feature_pipeline.config import get_settings
from feature_pipeline.exceptions import AQIPipelineError, DataFetchError
from feature_pipeline.logging_config import configure_logging

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


def fetch_historical_weather(start: datetime, end: datetime) -> dict[pd.Timestamp, dict]:
    """
    Fetch real historical weather (not just pollutants) from Open-Meteo's
    free Historical Weather API -- unlike OpenWeather, this needs no API
    key and has no paid tier gate on historical data, so it closes the
    "backfilled rows have no weather" gap entirely rather than just
    documenting it as a limitation.

    IMPORTANT unit consistency: Open-Meteo defaults wind speed to km/h,
    but the live pipeline (fetch_openweather_current, units=metric)
    reports wind speed in m/s. Mixing the two into one column would
    silently corrupt that feature -- wind_speed_unit=ms below forces a
    match. Temperature (C), humidity (%), and pressure (hPa) already
    match OpenWeather's metric units with no conversion needed.

    Returns a dict keyed by hour-truncated pandas Timestamp, so the
    caller can look up "what was the weather at this specific hour"
    directly, rather than a flat list the caller has to search.
    """
    settings = get_settings()
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={settings.karachi_lat}&longitude={settings.karachi_lon}"
        f"&start_date={start.date().isoformat()}&end_date={end.date().isoformat()}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
        "&wind_speed_unit=ms&timezone=UTC"
    )
    response = requests.get(url, timeout=settings.request_timeout_seconds)
    if response.status_code != 200:
        raise DataFetchError(
            f"Open-Meteo history request failed: {response.status_code} {response.text[:200]}"
        )

    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    pressures = hourly.get("surface_pressure", [])

    weather_by_hour = {}
    for i, t in enumerate(times):
        ts = pd.Timestamp(t, tz="UTC")
        weather_by_hour[ts] = {
            "temperature": _safe_float(temps[i]) if i < len(temps) else None,
            "humidity": _safe_float(humidities[i]) if i < len(humidities) else None,
            "wind_speed": _safe_float(winds[i]) if i < len(winds) else None,
            "pressure": _safe_float(pressures[i]) if i < len(pressures) else None,
        }
    return weather_by_hour


def _openweather_entry_to_row(entry: dict) -> dict:
    """
    Convert one OpenWeather history entry into the same flat row shape
    build_feature_row() produces from AQICN, so the rest of the pipeline
    doesn't need to know which source a row came from.

    The `aqi` field is computed via pm25_to_aqi(), landing on the same
    real AQI scale AQICN's live data uses -- not OpenWeather's raw PM2.5
    concentration, and not OpenWeather's own separate 1-5 index either.

    Weather fields default to None here -- OpenWeather's pollution
    history endpoint doesn't include them. backfill() fills them in
    afterward from Open-Meteo where available.
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
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "pressure": None,
    }


def backfill(days_back: int = 90) -> pd.DataFrame:
    """
    Backfill the last `days_back` days in weekly chunks (OpenWeather's
    history endpoint accepts large ranges, but chunking keeps individual
    requests small and makes a partial failure easier to retry).
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=days_back)

    all_rows = []
    weather_by_hour: dict[pd.Timestamp, dict] = {}
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)
        logger.info("Fetching %s to %s...", chunk_start.date(), chunk_end.date())

        entries = fetch_openweather_pollution_history(chunk_start, chunk_end)
        all_rows.extend(_openweather_entry_to_row(e) for e in entries)

        try:
            weather_by_hour.update(fetch_historical_weather(chunk_start, chunk_end))
        except DataFetchError:
            # Weather backfill is a real enhancement, not a hard
            # requirement -- if Open-Meteo is briefly unreachable, log it
            # and keep going with pollutant-only rows for this chunk
            # rather than aborting the whole backfill over it.
            logger.warning(
                "Open-Meteo weather fetch failed for %s to %s -- continuing without "
                "weather for this chunk.", chunk_start.date(), chunk_end.date(),
            )

        chunk_start = chunk_end
        time.sleep(1)  # be polite to the APIs between chunks, not strictly required

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Fill in real historical weather where Open-Meteo had it for that
    # exact hour -- this is what closes the "backfilled rows have no
    # weather" gap, rather than leaving temperature/humidity/wind_speed/
    # pressure as None for every backfilled row.
    if weather_by_hour:
        filled = 0
        for col in ("temperature", "humidity", "wind_speed", "pressure"):
            values = df["timestamp"].map(
                lambda ts, col=col: weather_by_hour.get(ts, {}).get(col)
            )
            df[col] = values
        filled = df["temperature"].notna().sum()
        logger.info("Filled real historical weather for %d of %d backfilled rows.", filled, len(df))

    # Protect existing LIVE rows from being downgraded. backfill() always
    # covers "the last N days from now," so a second run would otherwise
    # silently overwrite genuine live AQICN+weather rows with the
    # OpenWeather/Open-Meteo-derived version for the same hours, since
    # the feature table upserts on the shared hour-truncated timestamp. A
    # row is identified as "live" by having real weather data from the
    # LIVE pipeline specifically -- checked BEFORE the fill above would
    # count, so this comparison happens against the feature store, not
    # this DataFrame's own (now weather-filled) column.
    try:
        # Imported here, not at module level: keeps pm25_to_aqi,
        # fetch_historical_weather, and the rest of this file importable
        # and unit-testable without a live Supabase connection -- same
        # principle as the lazy imports in training_pipeline/train.py.
        from feature_pipeline.supabase_client import read_features

        existing = read_features()
        live_hours = set(
            pd.to_datetime(existing.loc[existing["temperature"].notna(), "timestamp"])
        )
        if live_hours:
            before = len(df)
            df = df[~df["timestamp"].isin(live_hours)]
            skipped = before - len(df)
            if skipped:
                logger.info(
                    "Skipping %d hour(s) that already have real live data -- "
                    "not overwriting them with backfilled values.",
                    skipped,
                )
    except AQIPipelineError as exc:
        # A real, worth-knowing limitation: read_features() wraps EVERY
        # failure (genuinely no feature group yet, OR a transient network/
        # credentials problem) into the same FeatureStoreError, so this
        # can't reliably tell "first run" apart from "read failed for some
        # other reason" -- both currently skip live-row protection. Logged
        # at WARNING (not INFO) specifically so an unexpected failure here
        # is visible, not just a normal first-run scenario.
        logger.warning(
            "Could not read existing features (%s) -- proceeding without "
            "live-row protection. If a feature group already exists, verify "
            "this wasn't a transient failure before trusting this backfill's "
            "output.", exc,
        )

    df = handle_outliers(df)
    df = add_cyclical_time_features(df)
    df = add_derived_features(df)

    logger.info(
        "Backfilled %d rows spanning %s to %s",
        len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


if __name__ == "__main__":
    configure_logging()
    from feature_pipeline.supabase_client import push_features

    backfilled = backfill(days_back=90)

    # Push in batches rather than one enormous insert -- friendlier to the
    # Supabase API and easier to see progress on a large backfill.
    batch_size = 200
    for i in range(0, len(backfilled), batch_size):
        batch = backfilled.iloc[i : i + batch_size]
        push_features(batch)
        logger.info("Pushed batch %d-%d of %d", i, i + len(batch), len(backfilled))

    logger.info("Backfill complete.")