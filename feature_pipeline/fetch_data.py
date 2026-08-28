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


def fetch_aqicn_data(station: str | None = None) -> dict:
    settings = get_settings()
    station = station or settings.aqicn_station
    url = f"https://api.waqi.info/feed/{station}/?token={settings.aqicn_token}"

    try:
        response = _get(url, timeout=settings.request_timeout_seconds)
    except (requests.RequestException, _RetryableHTTPError) as exc:
        raise DataFetchError(f"AQICN request failed: {exc}") from exc

    data = response.json()
    if data.get("status") != "ok":
        # Deliberately don't log the full `data` payload -- AQICN error
        # responses can echo request parameters back, and logging the raw
        # body is an easy way to accidentally leak the token into CI logs.
        raise DataFetchError(f"AQICN API returned status={data.get('status')!r}")

    return data["data"]


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