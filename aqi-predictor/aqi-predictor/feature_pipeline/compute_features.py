"""
Turn raw API responses into a flat feature row, and (once you have a
history of rows) compute time-based and derived features.

Design note: everything in this file is a pure function -- no network
calls, no I/O. That's deliberate: it means these functions can be unit
tested (see tests/test_compute_features.py) without needing API keys or a
live Supabase connection, and it keeps the "what does a feature mean"
logic separate from "how do we fetch/store it".
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

# The full valid range for the US AQI scale. Anything outside this is a
# genuine data error (a sensor glitch, a unit mix-up), not a real reading
# -- the scale is defined to not go below 0 or above 500.
AQI_VALID_RANGE = (0, 500)


def _safe_float(value) -> float | None:
    """AQICN returns '-' (or omits the key entirely) for sensors that are
    temporarily offline. Coerce anything non-numeric to None rather than
    letting a stray string silently corrupt a numeric column downstream."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers deliberately, not by blindly removing anything
    statistically unusual. There are two genuinely different categories
    here, and treating them the same would be a real mistake:

    1. PHYSICALLY IMPOSSIBLE values -- a negative pollutant concentration,
       an AQI outside the defined 0-500 scale. These are real data errors
       (a sensor glitch, a unit mix-up) and are clipped to the valid
       range.
    2. STATISTICALLY extreme but physically VALID values -- e.g. a real
       AQI of 400 during an actual hazardous pollution event. These are
       NOT touched. For most sensor data, an extreme reading is noise to
       filter out; for AQI specifically, an extreme reading is usually
       the most important, legitimate signal in the whole dataset --
       exactly the kind of event this project exists to predict. Clipping
       or removing these would actively damage the model's ability to
       learn what a hazardous event looks like.
    """
    df = df.copy()

    if "aqi" in df.columns:
        out_of_range = (
            (df["aqi"] < AQI_VALID_RANGE[0]) | (df["aqi"] > AQI_VALID_RANGE[1])
        )
        n_bad = int(out_of_range.sum())
        if n_bad:
            import logging

            logging.getLogger(__name__).warning(
                "Clipping %d row(s) with an AQI outside the valid 0-%d range "
                "-- these are data errors, not real hazardous readings.",
                n_bad, AQI_VALID_RANGE[1],
            )
        df["aqi"] = df["aqi"].clip(*AQI_VALID_RANGE)

    for col in ("pm25", "pm10", "o3", "no2"):
        if col in df.columns:
            n_negative = int((df[col] < 0).sum())
            if n_negative:
                import logging

                logging.getLogger(__name__).warning(
                    "Clipping %d negative value(s) in '%s' -- pollutant "
                    "concentrations cannot be negative.", n_negative, col,
                )
            df[col] = df[col].clip(lower=0)

    return df


def build_feature_row(aqicn_data: dict, weather_data: dict) -> dict:
    """
    Combine one AQICN reading + one OpenWeather reading into a single
    flat dict, ready to be appended to your growing dataset.
    """
    collected_at = datetime.now(UTC)

    # This pipeline runs hourly, so we round the event time down to the
    # hour and use that as the feature group's primary key. This makes
    # the pipeline idempotent: if GitHub Actions retries a failed run, or
    # you trigger it manually right after the scheduled run, the second
    # run upserts the same row instead of creating a duplicate for the
    # same hour. `collected_at` keeps the exact fetch time for auditing.
    event_time = collected_at.replace(minute=0, second=0, microsecond=0)

    iaqi = aqicn_data.get("iaqi", {})

    return {
        "timestamp": event_time.isoformat(),
        "collected_at": collected_at.isoformat(),
        "aqi": _safe_float(aqicn_data.get("aqi")),
        "pm25": _safe_float(iaqi.get("pm25", {}).get("v")),
        "pm10": _safe_float(iaqi.get("pm10", {}).get("v")),
        "o3": _safe_float(iaqi.get("o3", {}).get("v")),
        "no2": _safe_float(iaqi.get("no2", {}).get("v")),
        "temperature": _safe_float(weather_data.get("main", {}).get("temp")),
        "humidity": _safe_float(weather_data.get("main", {}).get("humidity")),
        "wind_speed": _safe_float(weather_data.get("wind", {}).get("speed")),
        "pressure": _safe_float(weather_data.get("main", {}).get("pressure")),
    }


def add_cyclical_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add sin/cos encoded hour, day-of-week, and month features.

    Why cyclical encoding instead of raw integers: hour 23 and hour 0 are
    one hour apart in reality, but maximally far apart as raw numbers. A
    model trained on raw integers learns a false discontinuity at
    midnight (and at week/year boundaries). Sin/cos encoding preserves the
    true cyclical distance between any two points in time.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"'{timestamp_col}' column not found in dataframe")

    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df[timestamp_col] = ts  # replace the original (often a string, e.g.
    # from build_feature_row's .isoformat()) with the parsed real
    # datetime -- keeping it a genuine datetime type from here through
    # the rest of the pipeline avoids mixed-type columns later, e.g. when
    # concatenating this row with history read back from Supabase
    # (which read_features() already parses back into real timestamps).

    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

    return df


def add_derived_features(df: pd.DataFrame, aqi_col: str = "aqi") -> pd.DataFrame:
    """
    Add AQI change rate, rolling means, and lag features.
    Sorts by timestamp internally, so callers don't need to pre-sort --
    a silent ordering bug here (unsorted rolling/diff) is a classic,
    hard-to-notice time-series mistake.
    """
    if aqi_col not in df.columns:
        raise ValueError(f"'{aqi_col}' column not found in dataframe")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_change_rate"] = df[aqi_col].diff()
    df["aqi_rolling_mean_3h"] = df[aqi_col].rolling(window=3, min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df[aqi_col].rolling(window=24, min_periods=1).mean()
    df["aqi_lag_1h"] = df[aqi_col].shift(1)
    df["aqi_lag_24h"] = df[aqi_col].shift(24)

    return df


if __name__ == "__main__":
    # Quick sanity check with synthetic data -- run this after fetch_data.py
    # works, to confirm feature computation before wiring up Supabase.
    sample = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-01", periods=48, freq="h"),
            "aqi": np.random.randint(80, 220, size=48),
        }
    )
    sample = add_cyclical_time_features(sample)
    sample = add_derived_features(sample)
    print(sample.head(10))
    print(f"\nColumns produced: {list(sample.columns)}")