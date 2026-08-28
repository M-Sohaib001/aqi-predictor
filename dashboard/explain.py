"""
Compute SHAP feature-importance values for the current champion's
prediction, regardless of which algorithm currently holds that title.

Two genuinely different explanations, dispatched by model kind:
- sklearn (Ridge/Random Forest): "which pollutant/weather/time feature
  drove this prediction" -- the classic tabular SHAP use case.
- tensorflow (LSTM): there are no named features, only a sequence of
  past AQI values -- so the explanation is instead "which of the past 24
  hours mattered most," which is a legitimately different, but equally
  real, question for a sequence model.
"""

import numpy as np
import pandas as pd
import shap

from dashboard.predict import LSTM_WINDOW
from training_pipeline.train import FEATURE_COLUMNS


def explain_sklearn(champion: dict, background_df: pd.DataFrame, current_row: pd.DataFrame) -> dict:
    """
    background_df: a sample of recent historical rows (FEATURE_COLUMNS),
        used as the reference distribution SHAP compares the current row
        against -- SHAP values are always "relative to some baseline,"
        not absolute.
    current_row: the single row to explain.

    Uses shap.Explainer's model-agnostic function-based interface
    (wrapping model.predict directly) rather than a tree-specific
    explainer, since the model is wrapped in a scikit-learn Pipeline
    (imputer + scaler + model) -- SHAP can't see through that wrapper to
    the tree structure underneath automatically. This is slower than a
    dedicated TreeExplainer, but for explaining one row at a time in a
    live dashboard (not thousands of rows in bulk), that cost is
    irrelevant.
    """
    explainer = shap.Explainer(champion["model"].predict, background_df[FEATURE_COLUMNS])
    shap_values = explainer(current_row[FEATURE_COLUMNS])
    return dict(zip(FEATURE_COLUMNS, shap_values.values[0].tolist(), strict=True))


def explain_lstm(
    champion: dict, background_windows: np.ndarray, current_window: np.ndarray
) -> dict:
    """
    background_windows: shape (n_samples, LSTM_WINDOW) of past AQI
        sequences, unscaled -- the reference distribution.
    current_window: shape (LSTM_WINDOW,), the sequence to explain.

    SHAP needs a 2D input (samples x features); the LSTM needs 3D
    (samples x timesteps x 1). `predict_fn` bridges the two: it accepts
    whatever 2D array SHAP hands it, reshapes to 3D, scales, predicts,
    and returns a plain 1D array of predictions -- SHAP only ever sees a
    normal "flat features in, prediction out" function.
    """
    scaler = champion["scaler"]

    def predict_fn(X_2d: np.ndarray) -> np.ndarray:
        scaled = (X_2d - scaler["mean"]) / scaler["std"]
        X_3d = scaled.reshape(-1, LSTM_WINDOW, 1)
        preds_scaled = champion["model"].predict(X_3d, verbose=0).flatten()
        return preds_scaled * scaler["std"] + scaler["mean"]

    explainer = shap.Explainer(predict_fn, background_windows)
    shap_values = explainer(current_window.reshape(1, -1))

    # Label each contribution by how many hours before "now" it represents
    # -- e.g. "hour_-24" is the oldest value in the window, "hour_-1" the
    # most recent -- since there are no named features for a raw sequence.
    hour_labels = [f"hour_-{LSTM_WINDOW - i}" for i in range(LSTM_WINDOW)]
    return dict(zip(hour_labels, shap_values.values[0].tolist(), strict=True))