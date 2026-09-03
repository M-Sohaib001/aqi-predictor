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


# --- Pollutant concentration -> AQI sub-index conversion ---------------
#
# AQICN's live iaqi.<pollutant>.v values are already real AQI sub-indices
# (0-500 scale, EPA-breakpoint-converted) -- that's what "iaqi" means.
# OpenWeather's raw pollution concentrations are NOT: co/no/no2/o3/so2/
# pm2_5/pm10/nh3 all come back in ug/m3, an entirely different scale.
# Writing OpenWeather's raw concentration into the same pm25/pm10/o3/no2
# columns AQICN's sub-indices populate silently mixes two incompatible
# scales depending on which source produced a given row -- a real
# data-integrity defect, not just an approximation. These four functions
# put every source on the same, real AQI sub-index scale.
#
# Source: EPA technical assistance document for the AQI (40 CFR Part 58,
# Appendix G) -- official published breakpoints, not an approximation.
# All four use a single hourly reading rather than EPA's official
# averaging window (24h for PM2.5/PM10, 8h for O3) -- state this
# simplification explicitly in the report, same as PM2.5 already is.
# NO2's official window is 1h, so a single hourly reading is actually
# correct for NO2 specifically, not a simplification.
_PM25_AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

_PM10_AQI_BREAKPOINTS = [
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 504.0, 301, 400),
    (505.0, 604.0, 401, 500),
]

# O3 and NO2 breakpoints are defined in ppb, not ug/m3 -- unlike PM2.5/
# PM10, which are already in ug/m3 matching OpenWeather's units directly.
# Converted using the standard ug/m3 -> ppb formula at reference
# conditions (25C, 1013.25 hPa): ppb = ug/m3 * 24.45 / molar_mass. This
# itself is an approximation -- the true conversion depends on actual
# ambient temperature/pressure, not fixed reference conditions -- worth
# naming alongside the averaging-window simplification above.
_O3_UGM3_TO_PPB = 24.45 / 48.00  # O3 molar mass ~48.00 g/mol
_NO2_UGM3_TO_PPB = 24.45 / 46.0055  # NO2 molar mass ~46.0055 g/mol

_O3_AQI_BREAKPOINTS_PPB = [
    (0.0, 54.0, 0, 50),
    (55.0, 70.0, 51, 100),
    (71.0, 85.0, 101, 150),
    (86.0, 105.0, 151, 200),
    (106.0, 200.0, 201, 300),
    # EPA switches to 1h ozone breakpoints above 200 ppb (8h); ambient 8h
    # ozone this high is effectively never seen in practice, so this is
    # capped at 500 like every other pollutant here rather than
    # implementing the separate 1h table for a band that won't occur.
]

_NO2_AQI_BREAKPOINTS_PPB = [
    (0.0, 53.0, 0, 50),
    (54.0, 100.0, 51, 100),
    (101.0, 360.0, 101, 150),
    (361.0, 649.0, 151, 200),
    (650.0, 1249.0, 201, 300),
    (1250.0, 1649.0, 301, 400),
    (1650.0, 2049.0, 401, 500),
]


def _concentration_to_aqi(
    concentration: float | None, breakpoints: list[tuple[float, float, int, int]]
) -> float | None:
    """Shared linear-interpolation core for every pollutant->AQI conversion
    below -- one place implementing EPA's breakpoint formula, instead of
    the same interpolation math duplicated per pollutant."""
    if concentration is None or concentration < 0:
        return None
    if concentration > breakpoints[-1][1]:
        return 500.0
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            return round((aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (concentration - bp_lo) + aqi_lo, 1)
    return None


def pm25_to_aqi(pm25_ugm3: float | None) -> float | None:
    """PM2.5 concentration (ug/m3) -> real AQI sub-index. See the module
    note above for why this exists and what it's protecting against."""
    return _concentration_to_aqi(pm25_ugm3, _PM25_AQI_BREAKPOINTS)


def pm10_to_aqi(pm10_ugm3: float | None) -> float | None:
    """PM10 concentration (ug/m3) -> real AQI sub-index. See the module
    note above."""
    return _concentration_to_aqi(pm10_ugm3, _PM10_AQI_BREAKPOINTS)


def o3_to_aqi(o3_ugm3: float | None) -> float | None:
    """O3 concentration (ug/m3) -> real AQI sub-index, via a ug/m3->ppb
    conversion first (EPA's O3 breakpoints are in ppb). See the module
    note above."""
    if o3_ugm3 is None:
        return None
    return _concentration_to_aqi(o3_ugm3 * _O3_UGM3_TO_PPB, _O3_AQI_BREAKPOINTS_PPB)


def no2_to_aqi(no2_ugm3: float | None) -> float | None:
    """NO2 concentration (ug/m3) -> real AQI sub-index, via a ug/m3->ppb
    conversion first (EPA's NO2 breakpoints are in ppb). See the module
    note above."""
    if no2_ugm3 is None:
        return None
    return _concentration_to_aqi(no2_ugm3 * _NO2_UGM3_TO_PPB, _NO2_AQI_BREAKPOINTS_PPB)


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


def _pollutant_value(
        iaqi: dict, components: dict, iaqi_key: str,
        openweather_key: str, converter) -> float | None:
    """Prefer AQICN's own sub-index for this pollutant; fall back to
    converting OpenWeather's raw concentration ONLY when AQICN's station
    doesn't report this pollutant at all (e.g. a PM-only community sensor
    with no gas-phase O3/NO2 hardware). Never blends the two within the
    same reading -- one source or the other, always on the same
    sub-index scale either way."""
    value = _safe_float(iaqi.get(iaqi_key, {}).get("v"))
    if value is not None:
        return value
    return converter(_safe_float(components.get(openweather_key)))


def build_feature_row(
        aqicn_data: dict, weather_data: dict,
        pollution_fallback: dict | None = None) -> dict:
    """
    Combine one AQICN reading + one OpenWeather reading into a single
    flat dict, ready to be appended to your growing dataset.

    pollution_fallback: OpenWeather's raw /air_pollution response (see
    fetch_openweather_pollution()), used only to fill in whichever of
    pm25/pm10/o3/no2 AQICN's iaqi doesn't have for the current station.
    Each raw concentration is converted to a real AQI sub-index via the
    *_to_aqi() functions above before use, so a live row stays on the
    same scale as AQICN's own sub-indices regardless of which source
    actually supplied a given pollutant. Omit this argument (or pass
    None) to skip the fallback entirely -- every missing pollutant then
    simply stays None, as before.
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
    components = (pollution_fallback or {}).get("list", [{}])[0].get("components", {})

    return {
        "timestamp": event_time.isoformat(),
        "collected_at": collected_at.isoformat(),
        "aqi": _safe_float(aqicn_data.get("aqi")),
        "pm25": _pollutant_value(iaqi, components, "pm25", "pm2_5", pm25_to_aqi),
        "pm10": _pollutant_value(iaqi, components, "pm10", "pm10", pm10_to_aqi),
        "o3": _pollutant_value(iaqi, components, "o3", "o3", o3_to_aqi),
        "no2": _pollutant_value(iaqi, components, "no2", "no2", no2_to_aqi),
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