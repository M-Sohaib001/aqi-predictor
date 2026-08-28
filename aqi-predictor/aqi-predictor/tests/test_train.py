"""
Unit tests for the pure, non-Supabase/non-TensorFlow logic in
training_pipeline/train.py: target construction and the chronological
train/test split. These run without any API keys, network access, or a
trained model -- same principle as test_compute_features.py.
"""

import numpy as np
import pandas as pd

from training_pipeline.train import build_targets, evaluate, time_aware_split


def test_build_targets_shifts_forward_not_backward():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="h"),
            "aqi": np.arange(30, dtype=float),
        }
    )
    result = build_targets(df)

    # target_24h at row 0 should be the aqi value 24 rows AHEAD (row 24),
    # not 24 rows behind -- the opposite direction from a lag feature.
    assert result["target_24h"].iloc[0] == result["aqi"].iloc[24]
    # the last 24 rows have no future value 24h ahead yet
    assert pd.isna(result["target_24h"].iloc[-1])


def test_time_aware_split_preserves_chronological_order():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
            "aqi": np.arange(100, dtype=float),
        }
    )
    train_df, test_df = time_aware_split(df, test_frac=0.2)

    assert len(train_df) == 80
    assert len(test_df) == 20
    # every train timestamp must be earlier than every test timestamp --
    # this is the property that prevents the model from "seeing the future"
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_evaluate_perfect_prediction_gives_zero_error():
    y_true = pd.Series([100.0, 150.0, 200.0])
    y_pred = np.array([100.0, 150.0, 200.0])

    metrics = evaluate(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0