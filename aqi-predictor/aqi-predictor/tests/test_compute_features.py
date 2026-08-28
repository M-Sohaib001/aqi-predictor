"""
Unit tests for the pure feature-computation logic. These run without any
API keys, network access, or Supabase connection -- deliberately, since
compute_features.py has no I/O in it (see the module docstring there).
"""

import numpy as np
import pandas as pd

from feature_pipeline.compute_features import (
    _safe_float,
    add_cyclical_time_features,
    add_derived_features,
)


def test_safe_float_handles_missing_sensor_marker():
    assert _safe_float("-") is None
    assert _safe_float(None) is None
    assert _safe_float("42.5") == 42.5
    assert _safe_float(42) == 42.0


def test_cyclical_encoding_is_continuous_at_midnight():
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-07-01 23:00", "2026-07-02 00:00"])}
    )
    df = add_cyclical_time_features(df)

    # Hour 23 and hour 0 should sit close together in sin/cos space, not
    # far apart the way raw integers 23 and 0 would.
    dist = np.hypot(
        df["hour_sin"].iloc[1] - df["hour_sin"].iloc[0],
        df["hour_cos"].iloc[1] - df["hour_cos"].iloc[0],
    )
    assert dist < 0.3


def test_add_cyclical_time_features_raises_on_missing_column():
    df = pd.DataFrame({"not_timestamp": [1, 2, 3]})
    import pytest

    with pytest.raises(ValueError):
        add_cyclical_time_features(df)


def test_derived_features_produce_expected_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-01", periods=30, freq="h"),
            "aqi": np.linspace(100, 200, 30),
        }
    )
    result = add_derived_features(df)

    for col in [
        "aqi_change_rate",
        "aqi_rolling_mean_3h",
        "aqi_rolling_mean_24h",
        "aqi_lag_1h",
        "aqi_lag_24h",
    ]:
        assert col in result.columns

    # First row has no prior value to diff/lag against.
    assert pd.isna(result["aqi_change_rate"].iloc[0])
    assert pd.isna(result["aqi_lag_1h"].iloc[0])


def test_derived_features_sorts_by_timestamp_before_computing():
    # Deliberately out of order -- this must not affect the result.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-01 02:00", "2026-07-01 00:00", "2026-07-01 01:00"]
            ),
            "aqi": [120, 100, 110],
        }
    )
    result = add_derived_features(df)
    assert list(result["aqi"]) == [100, 110, 120]