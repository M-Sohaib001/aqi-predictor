"""
Entry point for one hourly feature pipeline run.
This is the script GitHub Actions calls on its cron schedule.
"""

import logging
import sys

import pandas as pd

from feature_pipeline.compute_features import (
    add_cyclical_time_features,
    add_derived_features,
    build_feature_row,
    handle_outliers,
)
from feature_pipeline.exceptions import AQIPipelineError
from feature_pipeline.fetch_data import fetch_aqicn_data, fetch_openweather_current
from feature_pipeline.logging_config import configure_logging
from feature_pipeline.supabase_client import push_features, read_features

logger = logging.getLogger(__name__)


def main() -> None:
    aqicn_data = fetch_aqicn_data()
    weather_data = fetch_openweather_current()

    new_row = build_feature_row(aqicn_data, weather_data)
    new_df = pd.DataFrame([new_row])
    new_df = handle_outliers(new_df)
    new_df = add_cyclical_time_features(new_df)

    # Derived features (rolling means, lags) need history, not just the new
    # row. Pull existing history, append, recompute, then push only the
    # freshly-updated latest row back.
    try:
        history = read_features()
        combined = pd.concat([history, new_df], ignore_index=True)
    except AQIPipelineError:
        logger.info("No existing feature group found -- treating this as the first run.")
        combined = new_df

    combined = add_derived_features(combined)
    latest_row = combined.tail(1)

    push_features(latest_row)
    logger.info("Feature pipeline run complete.")


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except AQIPipelineError:
        logger.exception("Feature pipeline run failed.")
        # Explicit non-zero exit so GitHub Actions correctly marks this
        # run as failed (and, by default, emails you) instead of a caught
        # exception silently reporting a green checkmark.
        sys.exit(1)