"""
Shared data access for the dashboard: recent feature rows, used both for
the "current live reading" display the brief explicitly asks for
("real-time AND forecasted AQI data") and as model input.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd

from feature_pipeline.supabase_client import read_features


def load_recent_data(hours: int = 48) -> pd.DataFrame:
    since = datetime.now(UTC) - timedelta(hours=hours)
    df = read_features(since=since)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.tail(hours).reset_index(drop=True)


def get_current_reading(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    timestamp = latest["timestamp"]
    return {
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        "aqi": float(latest["aqi"]),
    }