from dashboard.alerts import categorize, check_alert


def test_categorize_boundaries():
    assert categorize(50) == "Good"
    assert categorize(151) == "Unhealthy"


def test_check_alert_picks_earliest_breaching_horizon():
    result = check_alert({24: 100, 48: 180, 72: 220})
    assert result["horizon_hours"] == 48