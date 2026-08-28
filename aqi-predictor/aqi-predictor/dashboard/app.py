"""
Streamlit dashboard for the Pearls AQI Predictor. Talks to the FastAPI
backend (dashboard/api.py) over HTTP rather than importing its functions
directly -- this matches how the two are actually meant to be deployed
(FastAPI as its own service, e.g. on Cloud Run; Streamlit as a separate
frontend calling it), and means the dashboard shows a clear error message
if the backend is down, rather than crashing outright.

Run the backend first:
    uvicorn dashboard.api:app --reload
Then the dashboard, in a second terminal:
    streamlit run dashboard/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("AQI_API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="centered")


def api_get(path: str) -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}{path}: {exc}")
        return None


st.title("🌫️ Pearls AQI Predictor")
st.caption("Karachi, Pakistan — live AQI and 3-day forecast")

current = api_get("/current")
forecast = api_get("/forecast")
alert = api_get("/alert")

if alert and alert.get("alert", True) and "message" in alert:
    st.error(f"⚠️ {alert['message']}")

col1, col2 = st.columns(2)
with col1:
    if current:
        st.metric("Current AQI (live)", f"{current['aqi']:.0f}", help=current["category"])
        st.caption(f"As of {current['timestamp']}")
with col2:
    if forecast:
        st.metric("24h forecast", forecast["forecast"]["24h"])

if forecast:
    st.subheader("3-day forecast")
    horizons = list(forecast["forecast"].keys())
    values = list(forecast["forecast"].values())
    chart_df = pd.DataFrame({"Horizon": horizons, "Predicted AQI": values})
    st.bar_chart(chart_df.set_index("Horizon"))

st.subheader("Why this forecast? (SHAP explanation)")
horizon_choice = st.selectbox("Horizon", [24, 48, 72], format_func=lambda h: f"{h}h")
explanation = api_get(f"/explain/{horizon_choice}")
if explanation:
    st.caption(f"Model type currently in use: {explanation['model_kind']}")
    exp_df = pd.DataFrame(explanation["top_features"])
    st.bar_chart(exp_df.set_index("feature"))