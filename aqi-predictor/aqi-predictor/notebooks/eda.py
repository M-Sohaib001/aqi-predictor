"""
Exploratory Data Analysis for the AQI feature store.

Run as a plain script (not a Jupyter notebook) to stay consistent with the
rest of this project's workflow -- no new tool to install. Saves each plot
as a PNG under notebooks/outputs/ instead of displaying inline, so it also
works unattended / over SSH / in CI if we ever want to regenerate these.

Run with:
    python -m notebooks.eda
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from feature_pipeline.logging_config import configure_logging
from feature_pipeline.supabase_client import read_features

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_data() -> pd.DataFrame:
    df = read_features()
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows, %s to %s", len(df), df["timestamp"].min(), df["timestamp"].max())
    return df


def plot_aqi_over_time(df: pd.DataFrame) -> None:
    """
    A plain line plot through sparse data is misleading: matplotlib draws
    a smooth straight line between any two points, however far apart in
    time they actually are, which can look like a real trend even when
    it's just two isolated readings days apart. This version explicitly
    flags that: small markers make individual data points visible, and
    any gap wider than GAP_THRESHOLD_HOURS is shaded red so a sparse,
    possibly-misleading stretch is visually distinct from a densely
    sampled, trustworthy one.
    """
    GAP_THRESHOLD_HOURS = 2

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["aqi"], linewidth=0.8, marker="o", markersize=2)

    gap_hours = df["timestamp"].diff().dt.total_seconds() / 3600
    gap_labeled = False
    for i in range(1, len(df)):
        if gap_hours.iloc[i] > GAP_THRESHOLD_HOURS:
            ax.axvspan(
                df["timestamp"].iloc[i - 1], df["timestamp"].iloc[i],
                color="red", alpha=0.15,
                label=f"gap > {GAP_THRESHOLD_HOURS}h (sparse data)" if not gap_labeled else None,
            )
            gap_labeled = True

    if gap_labeled:
        ax.legend(loc="upper left", fontsize=8)

    ax.set_title("AQI over time — Karachi (US Consulate station + OpenWeather backfill)")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "aqi_over_time.png", dpi=120)
    plt.close(fig)


def plot_aqi_by_hour_of_day(df: pd.DataFrame) -> None:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    grouped_by_hour = [df.loc[df["hour"] == h, "aqi"].dropna() for h in range(24)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(grouped_by_hour, tick_labels=list(range(24)))
    ax.set_title("AQI distribution by hour of day — checking for a traffic-driven pattern")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "aqi_by_hour.png", dpi=120)
    plt.close(fig)


def plot_aqi_vs_weather_correlation(df: pd.DataFrame) -> None:
    """
    pandas' .corr() needs overlapping non-null values in both columns to
    compute anything -- a column that's almost entirely missing (like
    weather fields, before enough live pipeline runs have accumulated)
    produces an all-NaN correlation, which is a useless, confusing plot.
    Rather than show that, this drops any column below a minimum overlap
    threshold and logs which ones were excluded and why. If NOTHING
    passes that bar, it falls back to correlating AQI against the
    pollutant columns instead, which are populated from day one, so the
    plot is still informative rather than blank.
    """
    weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
    pollutant_cols = ["pm25", "pm10", "o3", "no2"]

    min_overlap = max(10, int(0.05 * len(df)))

    valid_weather = [c for c in weather_cols if df[["aqi", c]].dropna().shape[0] >= min_overlap]
    excluded = sorted(set(weather_cols) - set(valid_weather))
    if excluded:
        logger.info(
            "Excluding %s from the correlation plot: fewer than %d rows have "
            "both AQI and that column populated. This is expected while the "
            "live pipeline is still accumulating data with real weather "
            "readings -- it will resolve on its own, not a bug to fix.",
            excluded, min_overlap,
        )

    if valid_weather:
        corr_cols = ["aqi", *valid_weather]
        title = "AQI vs. weather correlation"
    else:
        logger.warning(
            "No weather column has enough overlapping data yet (need >= %d "
            "rows) -- showing AQI vs. pollutant correlation instead.",
            min_overlap,
        )
        corr_cols = ["aqi", *pollutant_cols]
        title = "AQI vs. pollutant correlation (weather excluded -- insufficient live data so far)"

    correlations = df[corr_cols].corr()
    logger.info("Correlation with AQI:\n%s", correlations["aqi"].drop("aqi").to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "aqi_weather_correlation.png", dpi=120)
    plt.close(fig)

    return correlations


def plot_missing_data(df: pd.DataFrame) -> None:
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        logger.info("No missing values in any column.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    missing_pct.plot(kind="bar", ax=ax)
    ax.set_title("Missing data by column (%)")
    ax.set_ylabel("% missing")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_data.png", dpi=120)
    plt.close(fig)
    logger.info("Missing data:\n%s", missing_pct.to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()

    plot_aqi_over_time(df)
    plot_aqi_by_hour_of_day(df)
    plot_aqi_vs_weather_correlation(df)
    plot_missing_data(df)

    logger.info("EDA complete. Plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    configure_logging()
    main()