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

import logging
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("AQI_API_BASE_URL", "http://localhost:8000")

# Set, not required -- when unset, the "System status" section at the
# bottom of the page simply doesn't render at all. This is a lightweight
# viewer-facing gate (keeps casual visitors from seeing internal API
# state), not real authentication -- see the accompanying chat message
# for why full RBAC isn't the right call here.
ADMIN_PASSCODE = os.getenv("AQI_ADMIN_PASSCODE")

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

# EPA AQI category -> color, matching the same US EPA scale already used
# throughout the pipeline (pm25_to_aqi, dashboard.alerts.categorize).
# Matched by substring, case-insensitively, so this doesn't break if
# categorize()'s exact string casing/wording differs slightly.
CATEGORY_COLORS = [
    ("hazardous", "#7E0023"),
    ("very unhealthy", "#8F3F97"),
    ("unhealthy for sensitive", "#FF7E00"),
    ("unhealthy", "#FF0000"),
    ("moderate", "#F2C230"),
    ("good", "#009B4D"),
]
DEFAULT_COLOR = "#6B7280"  # neutral gray fallback if category text is unrecognized

# Plain-language names + one-line explanations for the raw feature-store
# column names that show up in the SHAP chart. A general visitor checking
# whether it's safe to go outside has no reason to know what "no2" or
# "aqi_rolling_mean_3h" means -- these translate the technical name
# without hiding it (the raw name still shows as a caption).
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "aqi": ("Current AQI", "The most recent measured air quality reading."),
    "pm25": ("Fine particles (PM2.5)",
             "Tiny airborne particles small enough to reach deep into the lungs",
              " -- usually the biggest driver of AQI."),
    "pm10": ("Coarse particles (PM10)",
             "Larger airborne particles, e.g. dust and pollen."),
    "o3": ("Ozone (O₃)", "Ground-level ozone, often higher on hot, sunny days."),
    "no2": ("Nitrogen dioxide (NO₂)", "A gas mainly from vehicle and industrial emissions."),
    "temperature": ("Temperature", "Current air temperature."),
    "humidity": ("Humidity", "How much moisture is in the air."),
    "wind_speed": ("Wind speed", "Faster wind generally disperses pollution."),
    "pressure": ("Air pressure",
                 "Low pressure can trap pollution near the ground; high pressure",
                 " (e.g. winter smog) can too."),
    "aqi_lag_1h": ("AQI 1 hour ago",
                   "What the AQI reading was one hour before now."),
    "aqi_lag_24h": ("AQI 24 hours ago",
                    "What the AQI reading was the same time yesterday."),
    "aqi_rolling_mean_3h": ("3-hour average AQI",
                            "The average AQI over the last 3 hours, smoothing out short spikes."),
    "aqi_rolling_mean_24h": ("24-hour average AQI", "The average AQI over the last full day."),
    "aqi_change_rate": ("Recent AQI trend", "How quickly AQI has been rising or falling."),
    "hour_sin": ("Time of day", "Captures typical patterns at this hour."),
    "hour_cos": ("Time of day", "Captures typical patterns at this hour."),
    "day_sin": ("Day of week", "Captures typical patterns on this day."),
    "day_cos": ("Day of week", "Captures typical patterns on this day."),
    "month_sin": ("Time of year", "Captures typical seasonal patterns."),
    "month_cos": ("Time of year", "Captures typical seasonal patterns."),
}


def friendly_feature_label(raw_name: str) -> str:
    label, _ = FEATURE_LABELS.get(raw_name, (raw_name, ""))
    return label


# Plain-language guidance per EPA category -- shown under the live AQI
# card so "Unhealthy" means something concrete, not just a colored badge.
CATEGORY_GUIDANCE = [
    ("hazardous",
     "Health warning of emergency conditions. Everyone should avoid outdoor activity."),
    ("very unhealthy",
     "Health alert. Everyone may experience serious effects -- avoid outdoor exertion."),
    ("unhealthy for sensitive",
     "Sensitive groups (children, elderly, those with respiratory issues)",
     " should limit outdoor exertion."),
    ("unhealthy",
     "Everyone may begin to experience health effects;",
     "sensitive groups more seriously."),
    ("moderate",
     "Air quality is acceptable, though there may be a risk for unusually sensitive people."),
    ("good",
     "Air quality is satisfactory, and air pollution poses little or no risk."),
]


def guidance_for_category(category: str | None) -> str:
    if not category:
        return ""
    lowered = category.lower()
    for needle, guidance in CATEGORY_GUIDANCE:
        if needle in lowered:
            return guidance
    return ""


def color_for_category(category: str | None) -> str:
    if not category:
        return DEFAULT_COLOR
    lowered = category.lower()
    for needle, color in CATEGORY_COLORS:
        if needle in lowered:
            return color
    return DEFAULT_COLOR


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }

        .aqi-hero {
            border-radius: 20px;
            padding: 2rem 2.25rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.5rem;
        }
        .aqi-hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .aqi-hero-caption {
            color: rgba(255,255,255,0.55);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .aqi-card {
            border-radius: 16px;
            padding: 1.5rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            height: 100%;
        }
        .aqi-card-label {
            color: rgba(255,255,255,0.55);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .aqi-card-value {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1;
        }
        .aqi-card-sub {
            color: rgba(255,255,255,0.45);
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }

        .aqi-badge {
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            color: #0b0b0d;
            margin-top: 0.6rem;
        }

        .aqi-alert-banner {
            border-radius: 14px;
            padding: 1rem 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(255, 87, 87, 0.35);
            background: rgba(255, 87, 87, 0.10);
            color: #ffb3b3;
        }

        .aqi-section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 0.75rem;
        }

        .aqi-unavailable {
            color: rgba(255,255,255,0.45);
            font-size: 0.9rem;
            font-style: italic;
            padding: 0.75rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_get(path: str) -> dict | None:
    """GET from the backend. On failure, logs the real exception (with
    the URL) for developers, but shows the user a clean, non-technical
    message -- the raw exception/URL used to render directly in the UI,
    which leaked internal infrastructure (hostnames, ports) to anyone
    viewing the dashboard. Returns None on failure either way, so every
    call site's existing `if current:` / `if forecast:` guard still works
    unchanged."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.warning("API call to %s%s failed: %s", API_BASE_URL, path, exc)
        return None


def render_unavailable(label: str) -> None:
    st.markdown(
        f'<div class="aqi-unavailable">{label} is still warming up -- '
        f"check back shortly once more recent data has accumulated.</div>",
        unsafe_allow_html=True,
    )


def bar_figure(x: list[str], y: list[float], colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker=dict(color=colors or "#4F8EF7", line=dict(width=0)),
            text=[f"{v:.1f}" for v in y],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        font=dict(family="Inter, sans-serif", size=13),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
    )
    return fig


inject_custom_css()

st.markdown(
    """
    <div class="aqi-hero">
        <p class="aqi-hero-title">🌫️ Pearls AQI Predictor</p>
        <p class="aqi-hero-caption">Karachi, Pakistan — live air quality and a 3-day forecast</p>
    </div>
    """,
    unsafe_allow_html=True,
)

current = api_get("/current")
forecast = api_get("/forecast")
alert = api_get("/alert")

if alert and alert.get("alert", True) and "message" in alert:
    st.markdown(
        f'<div class="aqi-alert-banner">⚠️ {alert["message"]}</div>',
        unsafe_allow_html=True,
    )

if current:
    color = color_for_category(current.get("category"))
    guidance = guidance_for_category(current.get("category"))
    hero_inner = f"""
        <div class="aqi-card-label">Current AQI (live)</div>
        <div
            class="aqi-card-value"
            style="font-size:4rem; color:{color};"
        >{current['aqi']:.0f}</div>
        <div class="aqi-badge" style="background:{color};">{current.get('category', '')}</div>
        <div class="aqi-card-sub">As of {current['timestamp']}</div>
        <div class="aqi-card-sub" style="margin-top:0.75rem; max-width: 480px;">{guidance}</div>
    """
else:
    hero_inner = (
        '<div class="aqi-card-label">Current AQI (live)</div>'
        '<div class="aqi-unavailable">Live AQI is still warming up -- '
        "check back shortly once more recent data has accumulated.</div>"
    )
st.markdown(
    f'<div class="aqi-card" style="margin-bottom: 1.5rem;">'
    f"{hero_inner}</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="aqi-section-title" style="margin-top:0;">'
    "3-day forecast</div>",
    unsafe_allow_html=True,
)

col_24, col_48, col_72 = st.columns(3)

for col, label, key in [
    (col_24, "24h forecast", "24h"),
    (col_48, "48h forecast", "48h"),
    (col_72, "72h forecast", "72h"),
]:
    with col:
        if forecast:
            delta = forecast["forecast"][key] - forecast["current_aqi"]
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "―"
            card_inner = f"""
                <div class="aqi-card-label">{label}</div>
                <div class="aqi-card-value">{forecast['forecast'][key]:.0f}</div>
                <div class="aqi-card-sub">{arrow} {abs(delta):.1f} vs. current</div>
            """
        else:
            card_inner = (
                f'<div class="aqi-card-label">{label}</div>'
                '<div class="aqi-unavailable">Still warming up -- '
                "check back shortly once more recent data has accumulated.</div>"
            )
        st.markdown(f'<div class="aqi-card">{card_inner}</div>', unsafe_allow_html=True)

if forecast:
    horizons = list(forecast["forecast"].keys())
    values = list(forecast["forecast"].values())
    st.plotly_chart(bar_figure(horizons, values), use_container_width=True)
else:
    render_unavailable("The 3-day forecast chart")

st.markdown(
    '<div class="aqi-section-title">'
    "Why this forecast? (SHAP explanation)</div>",
    unsafe_allow_html=True,
)
horizon_choice = st.selectbox("Horizon", [24, 48, 72], format_func=lambda h: f"{h}h")
explanation = api_get(f"/explain/{horizon_choice}")
if explanation:
    st.caption(f"Model type currently in use: {explanation['model_kind']}")
    exp_df = pd.DataFrame(explanation["top_features"])
    friendly_labels = [friendly_feature_label(f) for f in exp_df["feature"]]
    colors = ["#FF7E00" if v > 0 else "#4F8EF7" for v in exp_df["shap_value"]]
    st.plotly_chart(
        bar_figure(friendly_labels, exp_df["shap_value"].tolist(), colors),
        use_container_width=True,
    )
    st.caption("🟧 pushes AQI higher for this forecast &nbsp;&nbsp; 🟦 pushes AQI lower")
    with st.expander("What do these mean?"):
        for raw_name in exp_df["feature"]:
            label, description = FEATURE_LABELS.get(raw_name, (raw_name, ""))
            if description:
                st.markdown(f"**{label}** (`{raw_name}`) — {description}")
else:
    render_unavailable("The SHAP explanation")

# --- Optional, passcode-gated system status -- see ADMIN_PASSCODE comment above ---
if ADMIN_PASSCODE:
    with st.expander("System status"):
        entered = st.text_input("Passcode", type="password", key="admin_passcode_input")
        if entered == ADMIN_PASSCODE:
            health = api_get("/health")
            st.write("**Backend health:**", "🟢 reachable" if health else "🔴 unreachable")
            st.write("**API base URL:**", API_BASE_URL)
            if current:
                st.write("**Latest feature row timestamp:**", current["timestamp"])
            if forecast:
                st.write("**Raw /forecast response:**")
                st.json(forecast)
            if alert:
                st.write("**Raw /alert response:**")
                st.json(alert)
        elif entered:
            st.error("Incorrect passcode.")