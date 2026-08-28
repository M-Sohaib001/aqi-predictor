"""
Training pipeline: fetch (features, targets) from the Feature Store, train
and evaluate a variety of forecasting models (statistical baseline through
deep learning, per the brief), and register results in the Model Registry.

Design decision on registry promotion: rather than have this script compare
against a previously-registered model before deciding whether to push (which
requires trusting this script's local judgment), every trained candidate is
registered with its real evaluation metrics attached, and the registry
itself is queried for the best version by metric at serving time
(`get_best_model(name, "rmse", "min")` in supabase_client.py). This keeps
promotion logic in one place (the registry) rather than duplicated between
training and serving code.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline.exceptions import AQIPipelineError
from feature_pipeline.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Structurally-NaN-at-the-start features (the first ~24 rows of any series
# can't have a 24h lag yet) -- rows missing these are dropped, not imputed,
# since imputing "what AQI was 24h ago" with a guess would fabricate a
# feature that's supposed to be a real historical fact.
LAG_COLUMNS = [
    "aqi_change_rate", "aqi_rolling_mean_3h", "aqi_rolling_mean_24h",
    "aqi_lag_1h", "aqi_lag_24h",
]

# Missing because of a real data-source limitation (OpenWeather's history
# endpoint has no weather data -- see backfill.py), not a structural gap --
# these ARE imputed (median), since dropping every backfilled row that
# lacks weather would throw away most of the training set.
WEATHER_COLUMNS = ["temperature", "humidity", "wind_speed", "pressure"]

POLLUTANT_COLUMNS = ["pm25", "pm10", "o3", "no2"]
CYCLICAL_COLUMNS = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
]

# The row's own current AQI reading -- effectively a zero-hour lag, and
# typically the single strongest predictor of a short-horizon AQI
# forecast, given how autocorrelated air quality is hour to hour. Easy to
# leave out by accident once every OTHER AQI-derived signal (the lag,
# rolling-mean, and change-rate columns above) is already explicitly a
# transform of it -- an earlier draft of this guide did exactly that,
# leaving Ridge/RF/XGBoost to forecast without seeing the most recent
# known value at all. Occasional missing readings are handled the same way
# WEATHER_COLUMNS gaps already are -- build_sklearn_pipeline's
# SimpleImputer absorbs them, no dropna change needed here.
CURRENT_READING_COLUMNS = ["aqi"]

FEATURE_COLUMNS = (
    CURRENT_READING_COLUMNS + POLLUTANT_COLUMNS + WEATHER_COLUMNS + CYCLICAL_COLUMNS + LAG_COLUMNS
)

HORIZONS = {"target_24h": 24, "target_48h": 48, "target_72h": 72}

# Chosen empirically, not assumed -- see sweep_window_sizes below. A
# six-point window-size sweep (90/180/365/730/1085/1440 days) was run
# against this same pipeline and evaluated relative to EACH window's own
# baseline (absolute RMSE isn't comparable across windows, since each
# test slice is a different, differently-volatile stretch of the
# series). Only 730 and 1085 days beat their own baseline at all three
# horizons, with real 5-9% margins; every other window (including 365,
# which looked best in an earlier four-point sweep, and 1440) beat
# baseline at 0-2 of 3 horizons, by margins indistinguishable from noise
# where they won at all. 1085 was chosen over 730 for a higher, more
# consistent margin across all three horizons. The feature store itself
# still retains the full backfilled history (see backfill.py) -- this
# constant governs how much of it a given training run uses, not what's
# kept.
TRAINING_WINDOW_DAYS = 1085


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build forward-looking training targets. This is the mirror image of
    the lag features in compute_features.py: a lag looks backward (what
    AQI *was* N hours ago), a target looks forward (what AQI *will be* N
    hours from now). `.shift(-N)` (negative) pulls a value from N rows
    *ahead* instead of behind.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    for target_col, horizon in HORIZONS.items():
        df[target_col] = df["aqi"].shift(-horizon)
    return df


def prepare_horizon_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Drop rows that can't be used for this horizon: missing lag features
    (start of series) or missing target (end of series, since there's no
    future value that far ahead yet).
    """
    return df.dropna(subset=[*LAG_COLUMNS, target_col]).reset_index(drop=True)


def time_aware_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically, NOT randomly. A random shuffle would let the
    model train on rows that come chronologically *after* some of its own
    test rows -- effectively letting it "see the future" during training,
    which would make the reported accuracy look better than it would ever
    be in real, live use. The split point is a single point in time: every
    training row is genuinely from before every test row.
    """
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Persistence baseline: predict "AQI won't change" -- i.e. the forecast
    for any horizon is just the current AQI value. Every real model must
    beat this to be worth deploying; if it can't, the model isn't adding
    information, just noise.
    """
    return df["aqi"].to_numpy()


def build_sklearn_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def tuned_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Bounded randomized search over depth/leaf-size, scored with a
    TimeSeriesSplit (not a random k-fold) so every validation fold is
    still chronologically after its training fold -- same leakage concern
    time_aware_split addresses for the outer train/test split, just
    applied inside the search too. n_iter is capped at 15 candidates x 3
    folds = 45 fits, not an exhaustive grid, to keep this a
    minutes-not-hours step.
    """
    search = RandomizedSearchCV(
        build_sklearn_pipeline(RandomForestRegressor(random_state=42)),
        param_distributions={
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_leaf": [5, 10, 20, 50],
        },
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("Random Forest best params: %s", search.best_params_)
    return search.best_estimator_


def tuned_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Same bounded-search approach as tuned_random_forest, over XGBoost's
    depth/learning-rate/subsampling parameters."""
    # Imported here, not at module level: keeps every pure function in
    # this file importable and testable without XGBoost installed.
    from xgboost import XGBRegressor

    search = RandomizedSearchCV(
        build_sklearn_pipeline(XGBRegressor(random_state=42)),
        param_distributions={
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 5, 10],
        },
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("XGBoost best params: %s", search.best_params_)
    return search.best_estimator_


def build_sequences(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a plain column of values into overlapping windows, e.g. with
    window=24: row i's input becomes values[i-24:i] (the past 24 hours),
    shaped (samples, timesteps, 1) -- the 3D shape an LSTM layer expects.
    Returns the windows plus the row-indices they correspond to, so the
    caller can align each window with the right target value.
    """
    values = series.to_numpy(dtype="float32")
    X, idx = [], []
    for i in range(window, len(values)):
        X.append(values[i - window : i])
        idx.append(i)
    return np.array(X)[..., np.newaxis], np.array(idx)


def prepare_lstm_dataset(
    df: pd.DataFrame, target_col: str, mean: float, std: float, window: int = 24
):
    """
    Same windowing as build_sequences, but scaled using the given mean/std
    BEFORE windowing -- neural nets train poorly on raw, unscaled AQI
    values (roughly 20-300) in a small number of epochs. `mean`/`std` must
    come from the training split only (see run_horizon) -- computing them
    from the full dataset would leak test-set information into training,
    the same leakage time_aware_split is designed to avoid.

    NOTE: a multivariate version of this (windowing pollutant/weather/
    cyclical features too, not just AQI) was tried and reverted -- it
    dropped every horizon's LSTM from the best-performing model to the
    worst (e.g. 24h R2 went from +0.34 to -0.68). Most likely cause:
    Keras's validation_split takes the LAST slice of the (chronologically
    ordered) training data as its validation set, so EarlyStopping was
    plausibly watching an unrepresentative chunk and stopping before the
    network had learned to use the added features. Left as a documented
    "tried, didn't work, reverted" rather than re-attempted under time
    pressure -- a real fix would need a proper held-out validation split
    or more careful tuning, not another guess.
    """
    d = df.dropna(subset=[target_col]).reset_index(drop=True)
    scaled_aqi = (d["aqi"] - mean) / std
    X, idx = build_sequences(scaled_aqi, window=window)
    y_raw = d[target_col].to_numpy(dtype="float32")[idx]
    y_scaled = (y_raw - mean) / std
    return X, y_scaled


def build_lstm_model(window: int):
    # Imported here, not at module level: TensorFlow is a heavy import,
    # and every other function in this file works fine without it -- no
    # reason to pay that cost for code paths that don't need it.
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential

    # Without this, weight initialization and training are non-deterministic
    # run to run -- confirmed in practice: identical code on identical data
    # produced 24h LSTM R2 of -0.68 on one run and +0.40 on another. Ridge/
    # RF/XGBoost don't have this problem since they're already seeded via
    # random_state=42. Seeding doesn't make the LSTM better on average; it
    # stops the reported number from being whichever random draw happened
    # to land, which matters for a report that states a specific R2/RMSE
    # as THE result rather than one sample from a distribution of results.
    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(16),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def push_to_registry(
    model, model_type: str, name: str, metrics: dict, description: str,
    scaler_stats: dict | None = None, upload_artifact: bool = True,
) -> None:
    """
    Register a trained model's metrics in the Supabase-backed registry,
    and -- when upload_artifact=True -- upload its serialized file too.
    Metrics-only registration (upload_artifact=False) is used for every
    per-algorithm candidate below, since only the day's per-horizon
    champion is ever actually downloaded at serving time -- see
    feature_pipeline/supabase_client.py's module docstring for why this
    differs from a Hopsworks-backed version, which uploaded a full
    artifact for every candidate.
    """
    # Imported here, not at module level: this keeps every pure function
    # above (build_targets, time_aware_split, evaluate, ...) importable
    # and unit-testable without a live Supabase connection -- same
    # principle as the lazy TensorFlow import in build_lstm_model.
    from feature_pipeline.supabase_client import save_model

    save_model(
        model, model_type, name, metrics, description,
        scaler_stats=scaler_stats, upload_artifact=upload_artifact,
    )
    logger.info(
        "Registered '%s' (artifact=%s) with metrics %s", name, upload_artifact, metrics
    )


def run_horizon(
    df: pd.DataFrame, target_col: str, horizon_hours: int, register: bool = True
) -> pd.DataFrame:
    """Train + evaluate every model for one horizon; return a results table.

    register=False skips every push_to_registry call -- used by
    sweep_window_sizes below, which trains this same horizon repeatedly
    across several candidate training-window sizes purely to compare
    metrics. Writing every one of those exploratory runs to the registry
    would clutter it with versions nothing should ever serve.
    """
    horizon_df = prepare_horizon_dataset(df, target_col)
    train_df, test_df = time_aware_split(horizon_df)

    results = []

    # --- Baseline ---
    baseline_pred = baseline_predict(test_df)
    baseline_metrics = evaluate(test_df[target_col], baseline_pred)
    results.append({"model": "baseline_persistence", **baseline_metrics})

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

    # --- Ridge Regression (required) ---
    ridge = build_sklearn_pipeline(RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0]))
    ridge.fit(X_train, y_train)
    ridge_metrics = evaluate(y_test, ridge.predict(X_test))
    results.append({"model": "ridge", **ridge_metrics})

    # --- Random Forest (required) -- bounded randomized search over
    # depth/leaf-size instead of one fixed guess, scored on a
    # TimeSeriesSplit ---
    rf = tuned_random_forest(X_train, y_train)
    rf_metrics = evaluate(y_test, rf.predict(X_test))
    results.append({"model": "random_forest", **rf_metrics})

    # --- XGBoost (required -- "often the strongest tabular model in
    # production") -- same bounded search approach as Random Forest ---
    xgb = tuned_xgboost(X_train, y_train)
    xgb_metrics = evaluate(y_test, xgb.predict(X_test))
    results.append({"model": "xgboost", **xgb_metrics})

    # --- LSTM (required "advanced" model) ---
    # Scale using TRAIN statistics only -- the same principle as fitting
    # SimpleImputer/StandardScaler only on train_df in the sklearn
    # pipelines above, just done manually since Keras doesn't have an
    # equivalent built-in preprocessing step wired in here.
    aqi_mean, aqi_std = train_df["aqi"].mean(), train_df["aqi"].std()
    X_train_seq, y_train_seq = prepare_lstm_dataset(train_df, target_col, aqi_mean, aqi_std)
    X_test_seq, y_test_seq = prepare_lstm_dataset(test_df, target_col, aqi_mean, aqi_std)
    lstm_metrics = None
    lstm = None
    if len(X_train_seq) > 0 and len(X_test_seq) > 0:
        lstm = build_lstm_model(window=24)
        lstm.fit(X_train_seq, y_train_seq, epochs=40, batch_size=16, verbose=0)
        lstm_pred_scaled = lstm.predict(X_test_seq, verbose=0).flatten()

        # Inverse-transform back to real AQI units before evaluating --
        # RMSE/MAE in "scaled units" would be meaningless in the report.
        lstm_pred = lstm_pred_scaled * aqi_std + aqi_mean
        y_test_real = y_test_seq * aqi_std + aqi_mean
        lstm_metrics = evaluate(y_test_real, lstm_pred)
        results.append({"model": "lstm", **lstm_metrics})
    else:
        logger.warning(
            "Not enough rows for LSTM windowing at horizon %dh -- skipped.", horizon_hours
        )

    results_df = pd.DataFrame(results)
    results_df.insert(0, "horizon_hours", horizon_hours)
    logger.info("Horizon %dh results:\n%s", horizon_hours, results_df.to_string(index=False))

    # Register every trained (non-baseline) model's metrics. upload_artifact
    # =False here -- these accuracy numbers matter for the report's
    # comparison table, but their files are never downloaded at serving
    # time, so there's no reason to spend Storage budget on them (see
    # supabase_client.py's module docstring). Skipped entirely when
    # register=False (see sweep_window_sizes).
    if register:
        push_to_registry(
            ridge, "sklearn", f"ridge_{horizon_hours}h", ridge_metrics,
            f"Ridge regression, {horizon_hours}h AQI forecast",
            upload_artifact=False,
        )
        push_to_registry(
            rf, "sklearn", f"random_forest_{horizon_hours}h", rf_metrics,
            f"Random Forest (tuned), {horizon_hours}h AQI forecast",
            upload_artifact=False,
        )
        push_to_registry(
            xgb, "sklearn", f"xgboost_{horizon_hours}h", xgb_metrics,
            f"XGBoost (tuned), {horizon_hours}h AQI forecast",
            upload_artifact=False,
        )
    lstm_scaler_stats = {"mean": float(aqi_mean), "std": float(aqi_std)}
    if lstm_metrics is not None and register:
        push_to_registry(
            lstm, "tensorflow", f"lstm_{horizon_hours}h", lstm_metrics,
            f"LSTM, {horizon_hours}h AQI forecast",
            scaler_stats=lstm_scaler_stats, upload_artifact=False,
        )

    # Also register today's best-performing model under a SHARED name per
    # horizon (not per algorithm) -- WITH its artifact uploaded this
    # time. This is what makes serving possible: get_best_model("ridge_
    # 24h", ...) can only compare different versions of Ridge against
    # each other, never against Random Forest or LSTM, since it compares
    # versions of one name, not across names. Registering the winner
    # under "aqi_forecast_24h" every run means that name accumulates one
    # version per day, and querying ITS best version picks the best
    # model across every algorithm AND every day this pipeline has ever
    # run -- not just today's winner.
    candidates = [
        ("ridge", ridge, "sklearn", ridge_metrics),
        ("random_forest", rf, "sklearn", rf_metrics),
        ("xgboost", xgb, "sklearn", xgb_metrics),
    ]
    if lstm_metrics is not None:
        candidates.append(("lstm", lstm, "tensorflow", lstm_metrics))

    champion_name, champion_model, champion_type, champion_metrics = min(
        candidates, key=lambda c: c[3]["rmse"]
    )
    logger.info(
        "Champion for %dh horizon: %s (rmse=%.2f)",
        horizon_hours, champion_name, champion_metrics["rmse"],
    )

    # The registry's champion selection only ever compares Ridge/RF/
    # XGBoost/LSTM against each other -- baseline_persistence is
    # deliberately never a candidate for deployment (persistence isn't a
    # trained artifact the registry/dashboard can version or serve the
    # same way). That means it's possible for every trained model to lose
    # to the baseline and still have one of them deployed as "champion"
    # with nothing surfacing that fact. This check doesn't change what
    # gets deployed -- it makes that outcome loud and reportable instead
    # of silent, which is the actual gap: the docstring on
    # baseline_predict says a model "must beat this to be worth
    # deploying," but nothing in the code enforced that until now.
    if champion_metrics["rmse"] >= baseline_metrics["rmse"]:
        logger.warning(
            "Champion '%s' for %dh horizon (rmse=%.2f) does NOT beat the "
            "persistence baseline (rmse=%.2f) -- deploying it anyway, since "
            "some forecast is still required, but this is worth stating "
            "plainly in the report rather than treating champion selection "
            "as proof of a working model.",
            champion_name, horizon_hours, champion_metrics["rmse"], baseline_metrics["rmse"],
        )

    if register:
        push_to_registry(
            champion_model, champion_type, f"aqi_forecast_{horizon_hours}h", champion_metrics,
            f"Best model for {horizon_hours}h AQI forecast as of this run: {champion_name}",
            scaler_stats=lstm_scaler_stats if champion_type == "tensorflow" else None,
        )

    return results_df


def sweep_window_sizes(raw_df: pd.DataFrame, window_days: list[int]) -> pd.DataFrame:
    """
    Run the exact same training + evaluation code across several candidate
    training-window sizes (e.g. 90/180/365/1440 days back from the most
    recent reading), purely to compare metrics side by side. Answers "does
    training on less/more history actually help this specific problem"
    with real numbers instead of a guess -- nothing here is deployed
    (run_horizon is called with register=False throughout), so this is
    safe to run without affecting what's actually served.

    build_targets is re-run PER WINDOW, not once on the full df beforehand
    -- target_24h/48h/72h are shift()s computed within whatever slice of
    the series is passed in, so a window's targets must be derived from
    that same window, not inherited from a shift computed over the full
    four-year series.
    """
    all_results = []
    for days in window_days:
        cutoff = raw_df["timestamp"].max() - pd.Timedelta(days=days)
        window_df = raw_df[raw_df["timestamp"] >= cutoff].copy()
        window_df = build_targets(window_df)
        logger.info(
            "--- Window sweep: last %d days (%d rows, %s -> %s) ---",
            days, len(window_df), window_df["timestamp"].min(), window_df["timestamp"].max(),
        )
        for target_col, horizon in HORIZONS.items():
            horizon_results = run_horizon(window_df, target_col, horizon, register=False)
            horizon_results.insert(0, "window_days", days)
            all_results.append(horizon_results)

    sweep_df = pd.concat(all_results, ignore_index=True)
    logger.info("\nWindow-size sweep results:\n%s", sweep_df.to_string(index=False))
    return sweep_df


def main(sweep: bool = False, sweep_days: list[int] | None = None) -> None:
    from feature_pipeline.supabase_client import read_features

    df = read_features()

    if sweep:
        # Exploratory only -- see sweep_window_sizes docstring. Nothing
        # from this path is registered/deployed. Defaults to the
        # original four-point sweep (90/180/365/1440) when no explicit
        # --days list is given.
        sweep_window_sizes(df, window_days=sweep_days or [90, 180, 365, 1440])
        logger.info("Window-size sweep complete -- nothing was registered.")
        return

    cutoff = df["timestamp"].max() - pd.Timedelta(days=TRAINING_WINDOW_DAYS)
    df = df[df["timestamp"] >= cutoff].copy()
    logger.info(
        "Training on last %d days: %d rows, %s -> %s",
        TRAINING_WINDOW_DAYS, len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    df = build_targets(df)
    all_results = [run_horizon(df, target_col, horizon) for target_col, horizon in HORIZONS.items()]
    full_results = pd.concat(all_results, ignore_index=True)

    logger.info("\n%s", full_results.to_string(index=False))
    logger.info("Training pipeline run complete.")


if __name__ == "__main__":
    import sys

    configure_logging()

    days_arg = None
    if "--days" in sys.argv:
        raw_value = sys.argv[sys.argv.index("--days") + 1]
        days_arg = [int(d) for d in raw_value.split(",")]

    try:
        main(sweep="--sweep" in sys.argv, sweep_days=days_arg)
    except AQIPipelineError:
        logger.exception("Training pipeline run failed.")
        raise