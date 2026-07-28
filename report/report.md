Architecture & Tooling Decisions

This project fetches raw weather and pollutant data from AQICN and OpenWeather, computes features, and stores them in the Hopsworks Feature Store. Models are trained with Scikit-learn and TensorFlow/PyTorch and registered in the Hopsworks Model Registry. Automation runs on GitHub Actions rather than Apache Airflow: Airflow requires a persistently running scheduler and webserver, which is incompatible with a serverless, $0-infrastructure-cost design — GitHub Actions runners are ephemeral, triggered only on schedule, and free on public repositories. The dashboard is built with Streamlit and FastAPI — both explicitly named in the brief's required technology list — chosen over alternatives like a custom React/Next.js frontend specifically to match the stated requirements rather than optimize for visual polish.

Every infrastructure decision in this project was made against a single constraint: the system had to run end-to-end at $0 cost, using only genuinely free (not trial-credit) service tiers.

Feature Engineering

The following features are computed for each hourly reading:

Raw pollutant/weather readings — AQI, PM2.5, PM10, O₃, NO₂, temperature, humidity, wind speed, pressure — pulled directly from AQICN and OpenWeather.
Cyclical time features (hour_sin/hour_cos, day_sin/day_cos, month_sin/month_cos) — sine/cosine encodings of hour-of-day, day-of-week, and month, preserving true cyclical distance instead of a false boundary discontinuity.
AQI change rate — first difference between consecutive hourly AQI readings, capturing short-term momentum.
Rolling means (aqi_rolling_mean_3h, aqi_rolling_mean_24h) — short- and medium-term trend smoothing.
Lag features (aqi_lag_1h, aqi_lag_24h) — typically the strongest predictors, since pollution is highly autocorrelated.

[Once the model is trained, add: which features had the highest SHAP importance, and whether that matched EDA expectations.]