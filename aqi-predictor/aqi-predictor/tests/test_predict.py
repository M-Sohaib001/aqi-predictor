import numpy as np
import pandas as pd

from dashboard.predict import LSTM_WINDOW, predict_horizon
from training_pipeline.train import FEATURE_COLUMNS


class _FakeLSTM:
    def predict(self, X, verbose=0):
        assert X.shape == (1, LSTM_WINDOW, 1)
        return np.array([[0.5]])


def test_lstm_prediction_is_correctly_inverse_scaled():
    champion = {"model": _FakeLSTM(), "kind": "tensorflow", "scaler": {"mean": 130.0, "std": 30.0}}
    row = pd.DataFrame([{col: 1.0 for col in FEATURE_COLUMNS}])
    recent_aqi = pd.Series(np.arange(30, dtype=float))

    pred = predict_horizon(champion, row, recent_aqi)

    # 0.5 (scaled) * 30 (std) + 130 (mean) = 145
    assert abs(pred - 145.0) < 1e-6