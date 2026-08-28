"""
Hazardous-AQI alert logic. This is the core-MVP version: a threshold
check against a point forecast. A probability-based upgrade (using
prediction intervals) is a documented future enhancement, not built
here -- it would need uncertainty estimates this project's current
models don't produce.
"""

# US EPA AQI category breakpoints -- 151+ ("Unhealthy" and above) is
# widely used as the "take action" threshold for sensitive groups.
HAZARDOUS_THRESHOLD = 151

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def categorize(aqi: float) -> str:
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Hazardous" if aqi > 500 else "Unknown"


def check_alert(forecasts: dict[int, float]) -> dict | None:
    """
    forecasts: {24: value, 48: value, 72: value}
    Returns an alert dict if ANY horizon crosses the hazardous threshold,
    naming the EARLIEST such horizon (the most actionable one -- knowing
    "hazardous within 24h" matters more urgently than "within 72h"), or
    None if nothing crosses it.
    """
    breaching = {h: v for h, v in forecasts.items() if v >= HAZARDOUS_THRESHOLD}
    if not breaching:
        return None

    earliest_horizon = min(breaching)
    predicted = breaching[earliest_horizon]
    return {
        "horizon_hours": earliest_horizon,
        "predicted_aqi": predicted,
        "category": categorize(predicted),
        "message": (
            f"AQI is forecast to reach {predicted:.0f} ({categorize(predicted)}) "
            f"within {earliest_horizon}h."
        ),
    }