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

Hour-of-day pattern: weak, not absent. Median AQI ranges only from about 64 (mid-morning, hours 6–13 UTC) to about 69 (late evening, hours 18–23 UTC) — a roughly 5-point swing across the full day. There's no sharp double-peak rush-hour signature. The clearer effect is in the outliers, not the median: extreme spikes (AQI 140–180) appear concentrated in the afternoon-to-late-evening hours (14:00–23:00 UTC), while early-morning hours show almost none. Stated plainly: a modest diurnal effect exists, concentrated in extreme events rather than the typical case.

Time-series trend: two clear pollution events stand out — a sharp peak in early-to-mid June reaching ~175–180 AQI, and a smaller one in early July reaching ~110–115. The one shaded gap region (~July 10–12) sits in a relatively flat, unremarkable part of the series, not near either peak, so neither event's shape is an artifact of sparse sampling — both are backed by dense data. One point genuinely worth naming rather than glossing over: a sharp single-point spike to ~160 appears right at the end of the series (late August) with no shading around it, meaning it's technically dense-sampled data by the plot's own criterion — but a single-point spike of that size is also exactly the kind of thing worth a manual sanity check against the raw AQICN reading for that hour before trusting it outright in the report.

Weather/pollutant correlation: pressure correlated most strongly with AQI (r = -0.46, moderate negative) — physically sensible, since low-pressure systems are often associated with the stagnant air that traps pollutants. Humidity was weaker (+0.20), wind speed weaker still (-0.23), and temperature negligible (+0.06). No fallback to pollutants occurred — the plot title and the underlying log both confirm all four weather variables had sufficient overlapping data to compute directly.

Missing data: contrary to what was expected going in, weather fields are effectively complete (not present in the missing-data chart at all). The real gaps are concentrated in the AQI-derived features: aqi_lag_24h at ~2.5% missing, aqi_change_rate at ~0.5%, aqi_lag_1h at ~0.3%, aqi itself at ~0.2%, and pollutants (pm10/o3/no2) at ~0.05% each. This is mostly structural — the first 24 hours of the series can't have a 24-hour lag value yet, by definition — with a small additional contribution from the one sparse-data window noted above. This is a materially better result than the report template assumed, and worth stating as such rather than forcing the pre-written "weather is missing" narrative onto data that shows otherwise.

Feature list: no new feature was indicated by this EDA. The one feature-list gap this project actually had — the model never seeing the row's own current aqi reading, only its lagged/derived versions — was already caught and fixed before this stage (see Part 5). Pressure's comparatively stronger correlation doesn't require a new feature, since pressure is already in WEATHER_COLUMNS.