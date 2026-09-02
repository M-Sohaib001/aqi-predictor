"""
Fetch raw AQI + weather data from AQICN and OpenWeather.

Security & reliability notes:
- API keys are loaded through `config.get_settings()` and are never
  hardcoded or logged -- only status codes and non-secret fields are logged.
- Every request goes through a shared `requests.Session` with a bounded
  timeout -- an unbounded request is a common way for a "serverless" cron
  job to silently hang and burn Actions minutes.
- Transient failures (network blips, 5xx, 429 rate limiting) are retried
  with exponential backoff, rather than failing the entire hourly run on a
  single bad request.

Run standalone to sanity-check your API keys:
    python -m feature_pipeline.fetch_data
"""

import logging

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from feature_pipeline.config import get_settings
from feature_pipeline.exceptions import DataFetchError

logger = logging.getLogger(__name__)

_session = requests.Session()
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class _RetryableHTTPError(Exception):
    """Internal marker so retries only trigger on transient statuses,
    not on e.g. a 401 from a bad API key, which retrying won't fix."""


def _raise_for_retry(response: requests.Response) -> None:
    if response.status_code in _RETRYABLE_STATUS:
        raise _RetryableHTTPError(f"Retryable status {response.status_code} from {response.url}")
    response.raise_for_status()


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RetryableHTTPError),
)
def _get(url: str, timeout: int) -> requests.Response:
    response = _session.get(url, timeout=timeout)
    _raise_for_retry(response)
    return response


def classify_aqicn_response(payload: dict) -> tuple[bool, dict | None, str | None]:
    """Returns (is_success, station_data, error_message).

    AQICN's error responses aren't always where you'd expect: a genuine
    error (e.g. "Unknown ID") can come back as top-level status="ok" with
    the real error nested one level down in `data`:
        {"status": "ok", "data": {"status": "error", "msg": "Unknown ID"}}
    confirmed directly against a real station. The previous version of
    this function only checked `data.get("status") != "ok"` at the top
    level -- which means a nested error like the one above was read as
    success, and `data["data"]` (itself `{"status": "error", ...}`) would
    get returned and written downstream as if it were a real reading.

    Success therefore requires top-level status=="ok" AND `data` actually
    looking like a station reading (has an "aqi" key), not just a
    top-level "ok". Shared with vet_stations.py so there's one place that
    knows how to read an AQICN response.
    """
    top_status = payload.get("status")
    data = payload.get("data")

    if isinstance(data, dict) and data.get("status") == "error":
        return False, None, str(data.get("msg", "unknown error"))

    if top_status == "ok" and isinstance(data, dict) and "aqi" in data:
        return True, data, None

    if top_status == "error":
        return False, None, str(data)

    return False, None, f"unrecognized response shape: {payload!r}"


def fetch_aqicn_data(station: str | None = None) -> dict:
    """Fetch a single AQICN station's current reading.

    If `station` is given explicitly, only that station is tried (for
    callers/tests that want one specific station, no fallback). Otherwise
    this walks the configured chain -- settings.aqicn_station first, then
    each entry in settings.aqicn_fallback_stations in order -- moving to
    the next candidate only when a station is genuinely unreachable or
    returns an error, never merely because a reading looks stale (staleness
    is a feature-pipeline concern, not a fetch concern).

    Raises DataFetchError only if every candidate in the chain fails.
    """
    settings = get_settings()
    candidates = (
        [station]
        if station
        else [settings.aqicn_station, *settings.aqicn_fallback_stations]
)

    last_error: DataFetchError | None = None
    for candidate in candidates:
        url = f"https://api.waqi.info/feed/{candidate}/?token={settings.aqicn_token}"
        try:
            response = _get(url, timeout=settings.request_timeout_seconds)
        except (requests.RequestException, _RetryableHTTPError) as exc:
            last_error = DataFetchError(f"AQICN request to {candidate} failed: {exc}")
            continue

        ok, data, err = classify_aqicn_response(response.json())
        if ok:
            if candidate != candidates[0]:
                logger.warning("Primary AQICN station unreachable; used fallback %s", candidate)
            return data

        # Deliberately don't log the full payload -- AQICN error responses
        # can echo request parameters back, and logging the raw body is an
        # easy way to accidentally leak the token into CI logs.
        last_error = DataFetchError(f"AQICN station {candidate} returned: {err}")

    raise last_error or DataFetchError("No AQICN stations configured")


def fetch_openweather_current(lat: float | None = None, lon: float | None = None) -> dict:
    settings = get_settings()
    lat = lat if lat is not None else settings.karachi_lat
    lon = lon if lon is not None else settings.karachi_lon

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={settings.openweather_api_key}&units=metric"
    )
    try:
        response = _get(url, timeout=settings.request_timeout_seconds)
    except (requests.RequestException, _RetryableHTTPError) as exc:
        raise DataFetchError(f"OpenWeather request failed: {exc}") from exc

    return response.json()


def fetch_openweather_pollution(lat: float | None = None, lon: float | None = None) -> dict:
    settings = get_settings()
    lat = lat if lat is not None else settings.karachi_lat
    lon = lon if lon is not None else settings.karachi_lon

    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={settings.openweather_api_key}"
    )
    try:
        response = _get(url, timeout=settings.request_timeout_seconds)
    except (requests.RequestException, _RetryableHTTPError) as exc:
        raise DataFetchError(f"OpenWeather pollution request failed: {exc}") from exc

    return response.json()


if __name__ == "__main__":
    from feature_pipeline.logging_config import configure_logging

    configure_logging()

    logger.info("Testing AQICN...")
    aqicn = fetch_aqicn_data()
    logger.info("AQI: %s", aqicn.get("aqi"))

    logger.info("Testing OpenWeather (current weather)...")
    weather = fetch_openweather_current()
    logger.info("Temp: %s°C", weather["main"]["temp"])

    logger.info("Testing OpenWeather (air pollution)...")
    pollution = fetch_openweather_pollution()
    logger.info("Components: %s", pollution["list"][0]["components"])

    logger.info("All three sources responded successfully.")