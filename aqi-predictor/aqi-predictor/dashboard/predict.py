"""
Compute a single point forecast for one horizon, using whichever model
type (sklearn Pipeline or Keras LSTM) currently holds the "champion"
title for that horizon -- callers don't need to know or care which.
"""

import pandas as pd

from training_pipeline.train import FEATURE_COLUMNS

LSTM_WINDOW = 24


def predict_horizon(
    champion: dict, latest_features_row: pd.DataFrame, recent_aqi: pd.Series
) -> float:
    """
    latest_features_row: a single-row DataFrame with FEATURE_COLUMNS,
        used directly by sklearn models (Ridge/Random Forest Pipelines).
    recent_aqi: the last >=24 hourly AQI values, in chronological order
        (oldest first), used to build the input window for an LSTM.
    """
    if champion["kind"] == "sklearn":
        pred = champion["model"].predict(latest_features_row[FEATURE_COLUMNS])
        return float(pred[0])

    if champion["kind"] == "tensorflow":
        scaler = champion["scaler"]
        if scaler is None:
            raise ValueError(
                "LSTM champion is missing its scaler stats -- cannot predict correctly."
            )
        if len(recent_aqi) < LSTM_WINDOW:
            raise ValueError(
                f"Need at least {LSTM_WINDOW} hours of AQI history, got {len(recent_aqi)}."
            )

        recent_window = recent_aqi.to_numpy(dtype="float32")[-LSTM_WINDOW:]
        scaled = (recent_window - scaler["mean"]) / scaler["std"]
        X = scaled.reshape(1, LSTM_WINDOW, 1)

        pred_scaled = champion["model"].predict(X, verbose=0).flatten()[0]
        return float(pred_scaled * scaler["std"] + scaler["mean"])

    raise ValueError(f"Unknown champion kind: {champion['kind']}")