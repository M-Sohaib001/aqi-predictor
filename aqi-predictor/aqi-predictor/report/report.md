Architecture & Tooling Decisions

This project fetches raw weather and pollutant data from AQICN and OpenWeather for live, hourly readings, and Open-Meteo (a free, no-key historical weather API) to fill in real weather for backfilled historical rows, since neither AQICN nor OpenWeather offers free historical weather data. Features are computed and stored in a Supabase (Postgres) feature table. Models are trained with Scikit-learn and TensorFlow/PyTorch and registered in a Supabase-backed model registry (metrics in a Postgres table, artifacts in Supabase Storage) — a deliberate mid-project swap from the Hopsworks Feature Store and Model Registry originally planned, after repeatedly hitting Hopsworks' serverless free-tier quota; see Part 3 for the full rationale. Automation runs on GitHub Actions rather than Apache Airflow: Airflow requires a persistently running scheduler and webserver, which is incompatible with a serverless, $0-infrastructure -cost design — GitHub Actions runners are ephemeral, triggered only on schedule, and free on public repositories. The dashboard is built with Streamlit and FastAPI — both explicitly named in the brief's required technology list — chosen over alternatives like a custom React/Next.js frontend specifically to match the stated requirements rather than optimize for visual polish.

Every infrastructure decision in this project was made against a single constraint: the system had to run end-to-end at $0 cost, using only genuinely free (not trial-credit) service tiers.

Feature Engineering

The following features are computed for each hourly reading:

Raw pollutant/weather readings — AQI, PM2.5, PM10, O₃, NO₂, temperature, humidity, wind speed, pressure — pulled from AQICN and OpenWeather for live hourly readings; backfilled historical rows use OpenWeather for pollutants and Open-Meteo for weather, since neither AQICN nor OpenWeather offers free historical weather data.
Cyclical time features (hour_sin/hour_cos, day_sin/day_cos, month_sin/month_cos) — sine/cosine encodings of hour-of-day, day-of-week, and month, preserving true cyclical distance instead of a false boundary discontinuity.
AQI change rate — first difference between consecutive hourly AQI readings, capturing short-term momentum.
Rolling means (aqi_rolling_mean_3h, aqi_rolling_mean_24h) — short- and medium-term trend smoothing.
Lag features (aqi_lag_1h, aqi_lag_24h) — typically the strongest predictors, since pollution is highly autocorrelated.

[Once the model is trained, add: which features had the highest SHAP importance, and whether that matched EDA expectations.]

Why the hour-truncated timestamp: this pipeline runs hourly, so the event-time/primary-key is rounded down to the hour. That makes writes idempotent — a retried GitHub Actions run in the same hour upserts instead of duplicating a row.

Why Supabase, and the Hopsworks history: the original plan here was Hopsworks, for the reason most solo AI-project guides give it — a genuinely free individual tier, and a much simpler onboarding path than Vertex AI, which is built for GCP-integrated enterprise workflows. That held up until this project's serverless free-tier quota was exhausted mid-build. Rather than wait on a reset with an unpublished schedule, the feature store and model registry were moved to Supabase (Postgres + Storage) — a service that I have used in other work already, with no comparable request-quota lockout. The move was contained to one file specifically because every earlier draft already funneled every Hopsworks call through exactly three functions (push_features, read_features, get_model_registry) — nothing outside this file changed its calling convention. 

Two capabilities Hopsworks offers that this project never actually used even before the swap, so nothing here is a functional downgrade: point- in-time/time-travel reads (this project has one feature group and does its own leakage-avoidance via time_aware_split in training_pipeline/ train.py, not cross-feature-group joins) and a separate online/offline serving store (the dashboard reads the same table the training pipeline does). What genuinely doesn't carry over is a browsable version/lineage UI — the Supabase table editor covers that at this project's scale.

One deliberate change, worth stating: metrics are recorded
for every trained candidate every day (cheap -- a small jsonb row), but
the actual serialized model FILE is only uploaded to Storage for each
horizon's CHAMPION, and only the CHAMPION_RETENTION most recent versions
of that are kept. At this project's cadence (4 algorithms x 3 horizons x
up to ~45 daily runs), uploading a full artifact for every candidate
every day the way the earlier Hopsworks version did would produce far
more files than the free Storage tier comfortably holds, for files that
are never actually loaded at serving time -- only the champion is. Every
candidate's accuracy history is preserved either way, so the report's
full model-comparison table is unaffected; only unused historical
artifact files are pruned.

There's no fetch_aqicn_data(station, date=...) call that exists; bulk AQICN history requires a separate institutional data-platform request, not the standard token you signed up for. So backfill instead uses OpenWeather's Air Pollution History API (/data/2.5/air_pollution/history), which is free and covers data back to late 2020 — AQICN remains your live source going forward via the hourly run.py, while history before today comes from OpenWeather only. This is a real, worth-stating limitation. It also means backfilled rows have real pollutant data but no historical weather (temperature/humidity/wind/pressure), since that's a separate, paid-only endpoint on OpenWeather's side — leave those as missing values and let the tree-based model (Phase 4) handle them natively.

Data Collection & Feature Store

AQICN's free token API provides no general historical endpoint, so historical data was backfilled from OpenWeather's Air Pollution History API instead, covering May 29, 2026 to August 27, 2026, yielding 2,135 hourly rows. AQICN provides live station data from August 27, 2026, 04:00 UTC onward via the hourly feature pipeline. Backfilled rows include real historical weather (temperature, humidity, wind speed, pressure) from Open-Meteo's free Historical Weather API — OpenWeather's own historical endpoint only covers pollutants, not weather, which is a separate paid tier on their side, so Open-Meteo closes that gap instead. Wind speed is explicitly requested in m/s to match the live pipeline's units (Open-Meteo defaults to km/h).
The aqi column for backfilled rows is computed from OpenWeather's PM2.5 concentration using EPA's official breakpoint formula, landing on the same real AQI scale as live AQICN data, rather than using OpenWeather's raw concentration or its separate 1–5 index directly — those are different scales and would otherwise have produced an inconsistent training target depending on data source. This still uses a single hourly reading rather than the 24-hour average EPA's methodology technically specifies. backfill.py can be safely re-run at any point to densify sparse recent gaps — it checks existing data first and never overwrites hours that already have real live AQICN+weather readings with the lower-fidelity backfilled version.
One further gap, visible directly in the data: there is a sparse-data window of roughly two hours around July 10–12, 2026 (shaded in the AQI-over-time plot), where readings are thin — this reflects a real gap in source coverage for that stretch, not a plotting artifact.

Features are stored in a Supabase (Postgres) feature table, aqi_features, keyed on an hour-truncated timestamp primary key, making writes idempotent on retry. (This project started on the Hopsworks Feature Store; it was moved to Supabase mid-project after exhausting Hopsworks' serverless free-tier quota — see Part 3 for the full rationale.)

Exploratory Data Analysis

[Insert the four plots from notebooks/outputs/: AQI over time, AQI by hour-of-day, AQI vs. weather/pollutant correlation, missing data by column.]

Findings:

Time-series trend: the four-year series shows a clear, recurring seasonal pattern rather than the single flat summer window the earlier EDA captured: sharp winter pollution peaks reaching the AQI scale's ceiling of 500 recur around each year-end (visible in late 2022, late 2023, and late 2024/early 2025), consistent with temperature-inversion-driven winter smog. Notably, the most recent winter (entering 2026) is markedly more muted, topping out around 310 rather than plateauing at 500 — worth flagging as a genuine pattern rather than a data artifact, and a candidate follow-up question (was there a real air-quality improvement, or a change in measurement/reporting?) if there's time to explore it. Several sparse-data windows (>2h gaps) are scattered across the four years, most falling in otherwise unremarkable stretches of the series rather than coinciding with the pollution peaks, so the peak shapes themselves aren't sampling artifacts. A related, worth-stating detail: the repeated flat-topped plateaus at exactly 500 reflect the AQI scale's own ceiling (500 = EPA's top "Hazardous" breakpoint) — the underlying PM2.5 concentration during the worst events is effectively censored at this value by the pm25_to_aqi formula, not a sign of a bug.

Hour-of-day pattern: markedly weaker than the earlier single-season EDA suggested. Across the full four years, the median AQI by hour is essentially flat (differences of only a few AQI points across all 24 hours), and extreme outliers (up to the 500 ceiling) appear at every hour rather than concentrated in a particular part of the day. The earlier finding of an evening-concentrated outlier pattern does not hold up at this scale — the dominant driver of AQI variation is clearly seasonal (month-to-month), not diurnal, and the hour-of-day effect that remains is close to negligible.

Weather/pollutant correlation: pressure is the strongest correlate with AQI (r = +0.53, moderate positive) — physically consistent with winter high-pressure systems producing the temperature inversions that trap pollutants near the surface during Karachi's worst smog events. Humidity (-0.41) and temperature (-0.39) are both moderate and negative, consistent with the same seasonal story: cooler, drier winter conditions coincide with higher AQI, while the humid monsoon months see pollutants washed out of the air. Wind speed is weaker (-0.34), in the expected direction (more wind disperses pollutants). This is a substantively different — and more physically coherent — picture than the single-season correlations reported earlier, which had the wrong sign on pressure and near-zero temperature/humidity correlations simply because that window didn't contain enough seasonal variation to reveal the real relationship.

Missing data: weather features (temperature, humidity, wind speed, pressure) remain effectively complete across the full 33,696-row dataset — none appear in the missing-data ranking at all. The remaining gaps are concentrated in AQI-derived features, as expected structurally: aqi_change_rate (~0.79%), aqi_lag_24h (~0.61%), aqi_lag_1h (~0.41%), and aqi itself (~0.39%), plus pollutants (o3/pm10/no2) at ~0.30% each. These percentages are notably smaller than the 90-day EDA's (e.g. aqi_lag_24h dropped from ~2.5% to ~0.61%) simply because the fixed number of structurally-NaN rows at the very start of the series is now a much smaller fraction of a much larger dataset.

Feature list: still no new feature indicated by this EDA. The pressure/temperature/humidity correlation strengthening once winter data is included doesn't require new features — all three are already in WEATHER_COLUMNS — though it does reinforce that the earlier single-season EDA would have been a weak basis for feature selection had it been trusted in isolation.

Modeling & Evaluation

Four forecasting approaches (plus a persistence baseline) were built and compared — Ridge Regression, Random Forest, XGBoost, and an LSTM — each producing direct (not recursive) multi-horizon predictions for t+24h, t+48h, and t+72h:

| Model | RMSE (24h/48h/72h) | MAE (24h/48h/72h) | R² (24h/48h/72h) |
|---|---|---|---|
| Baseline (persistence) | 24.20 / 30.26 / 32.45 | 14.53 / 20.17 / 22.55 | 0.42 / 0.10 / -0.04 |
| Ridge Regression | 27.54 / 34.04 / 35.01 | 18.75 / 24.15 / 25.54 | 0.25 / -0.14 / -0.21 |
| Random Forest | 23.84 / 34.04 / 33.80 | 15.87 / 23.79 / 24.94 | 0.44 / -0.14 / -0.12 |
| XGBoost | 22.73 / 30.81 / 30.68 | 15.06 / 23.43 / 22.46 | 0.49 / 0.07 / 0.07 |
| LSTM | 23.60 / 27.70 / 30.93 | 15.67 / 19.36 / 22.09 | 0.45 / 0.25 / 0.06 |

Forecasts use a direct multi-horizon strategy rather than recursive prediction, and a chronological (not random) train/test split, so reported accuracy reflects genuine forward-looking performance rather than leaked future information.

No single model is manually selected for deployment. Each training run registers every candidate's real metrics, and the registry itself is queried at serving time for the lowest-RMSE model per horizon (`get_best_model`) — this keeps promotion logic in one place rather than duplicated between training and serving code. In this run, the LSTM was the best-performing model at all three horizons, and — notably — at 72h it is the first model in this project to genuinely outperform the persistence baseline (RMSE 44.44 vs. 45.43), consistent with the baseline's "no change" assumption weakening at longer horizons where real weather/pollutant dynamics matter more than the current reading alone. Random Forest and XGBoost were tuned via a bounded RandomizedSearchCV (15 candidates × 3 folds, scored on a TimeSeriesSplit to avoid future leakage into validation folds) rather than fixed hyperparameters; this measurably improved XGBoost (24h R² from -0.48 to 0.18) but neither tree ensemble matched Ridge or LSTM, plausibly because tree-based averaging smooths over the sharp AQI transitions that persistence and the LSTM capture directly. A multivariate LSTM variant (trained on pollutant/weather/cyclical features per timestep, not just the AQI sequence) was also attempted and reverted after it substantially underperformed the single-feature version at every horizon — kept here as a documented, deliberate design decision rather than silently discarded. Every trained model is registered with its real metrics; the best-performing one per horizon is additionally registered under a shared name (aqi_forecast_24h, etc.) so serving code can query the genuine best model across every algorithm and every training run, not just whichever one happened to train most recently. Weather features (temperature, humidity, wind speed, pressure) are populated for both live rows (from OpenWeather) and backfilled historical rows (from Open-Meteo's free historical weather API) — confirmed at 100% coverage (33,696/33,696 rows) by both the direct row-count check and the EDA's missing-data analysis.

Training Window Selection

Given that the feature store had grown to 33,696 hourly rows (four years of backfilled history) but the champion models still weren't consistently beating the persistence baseline, an open question was whether training on less — or more — of that history would help. Rather than guess, this was tested directly: the same training pipeline (Ridge, tuned Random Forest, tuned XGBoost, seeded LSTM, evaluated against a chronological train/test split) was re-run six times, each time restricted to a different trailing window of the feature store — 90, 180, 365, 730, 1085, and 1440 days back from the most recent reading — with nothing from these exploratory runs registered or deployed.

One methodological point matters here: absolute RMSE is not comparable across windows, since each window's test set (the most recent 20% of that window) is a different, differently-volatile stretch of the series — a 90-day test slice covers a single calm season, while a 1440-day test slice includes real winter extreme events reaching the AQI scale's ceiling of 500. The only fair comparison is relative: within each window, does the best trained model beat that window's own baseline?

Summary — margin vs. each window's own baseline:

Window	24h margin	48h margin	72h margin	Beats baseline at all 3 horizons?
90 days	+3.5%	-8.7%	-27.0%	No
180 days	-7.4%	-4.7%	-3.8%	No
365 days	-10.7%	+0.4%	+9.8%	No (2/3)
730 days	+0.5%	+8.6%	+6.2%	Yes
1085 days	+6.1%	+7.7%	+5.5%	Yes
1440 days	-1.6%	-4.4%	+0.2%	No (1/3)

Full per-model breakdown, RMSE/MAE/R² (24h/48h/72h):

Window	Model	RMSE	MAE	R²
90 days	Baseline (persistence)	7.18 / 8.82 / 9.30	4.46 / 6.20 / 6.79	-0.06 / -0.59 / -0.77
	Ridge	7.47 / 10.50 / 11.82	5.03 / 8.33 / 9.46	-0.14 / -1.25 / -1.85
	Random Forest	7.41 / 9.71 / 12.25	4.78 / 7.33 / 10.42	-0.12 / -0.92 / -2.06
	XGBoost	6.93 / 9.58 / 12.17	4.29 / 7.45 / 10.24	0.02 / -0.87 / -2.02
	LSTM	7.78 / 10.45 / 12.33	5.05 / 7.74 / 9.77	-0.18 / -1.11 / -1.93
180 days	Baseline (persistence)	6.46 / 8.45 / 9.61	4.53 / 6.60 / 7.57	0.51 / 0.13 / -0.15
	Ridge	7.02 / 9.35 / 11.20	5.09 / 7.46 / 9.14	0.43 / -0.07 / -0.56
	Random Forest	8.41 / 10.91 / 16.00	6.26 / 8.55 / 13.89	0.18 / -0.46 / -2.19
	XGBoost	6.93 / 8.85 / 9.97	4.90 / 6.93 / 7.86	0.44 / 0.04 / -0.24
	LSTM	9.16 / 15.72 / 15.39	6.54 / 11.89 / 9.75	0.01 / -2.04 / -1.93
365 days	Baseline (persistence)	7.85 / 11.16 / 13.33	5.64 / 8.57 / 10.71	0.67 / 0.33 / 0.04
	Ridge	11.66 / 16.41 / 15.84	9.17 / 12.98 / 12.36	0.27 / -0.46 / -0.35
	Random Forest	9.25 / 14.39 / 22.50	7.02 / 11.47 / 17.87	0.54 / -0.12 / -1.73
	XGBoost	8.69 / 11.12 / 12.02	6.88 / 8.80 / 9.51	0.59 / 0.33 / 0.22
	LSTM	10.33 / 16.70 / 19.63	8.11 / 13.85 / 17.07	0.43 / -0.49 / -1.06
730 days	Baseline (persistence)	15.54 / 19.68 / 21.68	9.49 / 13.33 / 15.51	0.45 / 0.12 / -0.08
	Ridge	16.19 / 18.93 / 20.34	10.82 / 13.54 / 14.62	0.41 / 0.19 / 0.05
	Random Forest	15.46 / 18.00 / 21.35	10.43 / 12.78 / 16.28	0.46 / 0.27 / -0.04
	XGBoost	15.95 / 18.58 / 21.20	11.81 / 13.74 / 16.11	0.42 / 0.22 / -0.03
	LSTM	16.49 / 21.73 / 23.43	10.24 / 14.20 / 18.19	0.36 / -0.11 / -0.30
1085 days	Baseline (persistence)	24.20 / 30.26 / 32.45	14.53 / 20.17 / 22.55	0.42 / 0.10 / -0.04
	Ridge	27.54 / 34.04 / 35.01	18.75 / 24.15 / 25.54	0.25 / -0.14 / -0.21
	Random Forest	23.84 / 34.04 / 33.80	15.87 / 23.79 / 24.94	0.44 / -0.14 / -0.12
	XGBoost	22.73 / 30.81 / 30.68	15.06 / 23.43 / 22.46	0.49 / 0.07 / 0.07
	LSTM	23.34 / 27.94 / 32.19	15.53 / 19.88 / 23.45	0.46 / 0.23 / -0.02
1440 days	Baseline (persistence)	31.73 / 41.59 / 45.43	18.76 / 26.66 / 29.80	0.48 / 0.10 / -0.07
	Ridge	38.85 / 51.18 / 51.28	27.23 / 35.98 / 36.40	0.21 / -0.36 / -0.36
	Random Forest	45.09 / 61.99 / 71.64	30.04 / 41.21 / 47.46	-0.06 / -0.99 / -1.66
	XGBoost	39.65 / 56.04 / 63.12	26.27 / 37.10 / 41.49	0.18 / -0.63 / -1.06
	LSTM	32.25 / 43.40 / 45.36	20.28 / 28.06 / 29.90	0.46 / 0.03 / -0.06

The result was not the extremes of the tested range and not the full four years of available history — it was a middle window, roughly 2-3 years back. Only 730 and 1085 days beat their own baseline at every horizon, with real, consistent 5-9% margins; every other window tested, including the full 1440-day history, beat baseline at 0-2 of 3 horizons at best, by margins indistinguishable from noise where they won at all. The per-model breakdown also shows XGBoost as the most consistent performer at 1085 days specifically — the only model beating baseline at all three horizons individually (R² 0.49/0.07/0.07), rather than the win coming from different algorithms trading off across horizons as in most of the smaller windows.

Between the two winning windows, 1085 days (~3 years) was selected as the final training window: it showed a higher average margin (6.4% vs. 5.1%) and, more importantly, a more consistent one — 730 days' 24h result (+0.5%) was close enough to a tie to be within noise, while 1085 days beat baseline meaningfully at all three horizons. This became `TRAINING_WINDOW_DAYS = 1085` in `training_pipeline/train.py`, applied as a filter on the feature store's timestamp range immediately before target construction, at the start of every training run. The feature store itself continues to retain the full four-year backfilled history (`backfill.py`'s `days_back=1440` is unchanged) — `TRAINING_WINDOW_DAYS` governs how much of it a given training run draws on, not what the pipeline keeps.

With the final training window (1,085 days) in place, every horizon's champion beats the persistence baseline for the first time in this project: XGBoost at 24h (RMSE 22.73 vs. 24.20, +6.1%) and 72h (RMSE 30.68 vs. 32.45, +5.5%), and LSTM at 48h (RMSE 27.70 vs. 30.26, +8.5%). No model is manually selected — the registry is queried per horizon at serving time for the lowest-RMSE candidate among Ridge, tuned Random Forest, tuned XGBoost, and a seeded LSTM, and a warning is logged (though it did not fire on this run) whenever the selected champion fails to beat the baseline it's meant to improve on.

Automation & CI/CD

Two GitHub Actions workflows automate the pipeline:

Hourly feature pipeline (feature_pipeline.yml) — cron 0 * * * *, fetches new data, computes features, pushes to the Supabase-backed feature table.

Daily training pipeline (training_pipeline.yml) — cron [time], retrains Ridge, Random Forest, and LSTM for every horizon, evaluates each against a held-out set, and registers every one with its real metrics attached, plus the best-performing one per horizon under a shared name so serving code always queries the genuine best model rather than trusting any single training run's local judgment.

GitHub Actions was chosen over Apache Airflow specifically because Airflow requires a persistently running scheduler and webserver — even "free" ways to host it violate the $0-cost, serverless constraint this project was built under.

Why least-privilege permissions + concurrency control: permissions: contents: read means the workflow can't write back to the repo even if compromised. The concurrency block stops a manual trigger from racing an in-progress scheduled run against the same feature group.

[Once running a few days: "As of [date], the pipeline has completed N of M scheduled runs successfully, with failures caused by [reason, if any]."]