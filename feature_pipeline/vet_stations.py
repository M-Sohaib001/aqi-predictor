"""
vet_stations.py -- vet a list of candidate AQICN station UIDs and print one
comparison table: which are actually reachable, under which ID prefix, how
fresh their last reading is, and whether their numbers look internally
consistent.

Two things this fixes, confirmed against real responses rather than guessed:

1. ID prefix: AQICN's feed endpoint takes an ID of the form @<UID> or
   A<UID> depending on the station's underlying data-provider type.
   Station 401143 needs "A", not "@" -- so every candidate here is tried
   both ways instead of assumed.

2. The nested-status bug: a genuine "Unknown ID" does NOT come back as a
   top-level error. It comes back as top-level status="ok" with the real
   error one level down:
       {"status": "ok", "data": {"status": "error", "msg": "Unknown ID"}}
   This is the exact raw body seen from @401143. Treating top-level
   status=="ok" as success (as an earlier version of this script did)
   silently treats that as a stale-but-valid reading instead of a clear
   failure. classify_aqicn_response() (now shared with fetch_data.py, so
   there's one place that knows how to read an AQICN response) checks the
   *shape* of `data`, not just the top-level status, to catch this.

Run:
    python -m feature_pipeline.vet_stations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from feature_pipeline.config import get_settings
from feature_pipeline.fetch_data import classify_aqicn_response
from feature_pipeline.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Fill in the rest of the ~10 candidate UIDs you pulled off the AQICN
# interactive map. 401143 is the one already confirmed to need "A".
CANDIDATE_STATIONS: list[int] = [
    401143,
]

_PREFIXES = ("@", "A")
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_REQUEST_GAP_SECONDS = 0.2  # polite pacing across ~20 worst-case requests
_session = requests.Session()


class _RetryableHTTPError(Exception):
    """Internal marker so retries only trigger on transient statuses."""


def _raise_for_retry(response: requests.Response) -> None:
    if response.status_code in _RETRYABLE_STATUS:
        raise _RetryableHTTPError(f"Retryable status {response.status_code}")
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


@dataclass
class StationResult:
    uid: int
    working_id: str | None = None  # e.g. "A401143"; None if nothing worked
    ok: bool = False
    error: str | None = None
    name: str | None = None
    aqi: float | None = None
    dominant_pollutant: str | None = None
    age_hours: float | None = None
    max_sub_index: float | None = None
    consistent: bool | None = None  # None = couldn't check
    notes: list[str] = field(default_factory=list)


def _parse_reading(uid: int, prefix: str, data: dict) -> StationResult:
    result = StationResult(uid=uid, working_id=f"{prefix}{uid}", ok=True)

    result.name = (data.get("city") or {}).get("name")
    result.dominant_pollutant = data.get("dominentpol")

    aqi_raw = data.get("aqi")
    try:
        result.aqi = float(aqi_raw)
    except (TypeError, ValueError):
        result.notes.append(f"non-numeric top-level aqi: {aqi_raw!r}")

    time_block = data.get("time") or {}
    epoch = time_block.get("v")
    if epoch:
        reading_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        age = datetime.now(tz=timezone.utc) - reading_dt
        result.age_hours = round(age.total_seconds() / 3600, 1)
    else:
        result.notes.append("no timestamp in response")

    # AQICN's overall aqi is meant to be the max of the individual
    # pollutant sub-indices (iaqi). A sub-index far above both the
    # reported aqi and the 0-500 AQI scale is the same shape of problem
    # @545320 showed (1938 vs ~9-45 across the rest) -- almost always
    # means that field holds a raw concentration, not an AQI-converted
    # value, and the station's reading shouldn't be trusted as-is.
    iaqi = data.get("iaqi") or {}
    sub_values = [
        entry["v"]
        for entry in iaqi.values()
        if isinstance(entry, dict) and isinstance(entry.get("v"), (int, float))
    ]

    if sub_values:
        result.max_sub_index = max(sub_values)
        if result.max_sub_index > 500:
            result.consistent = False
            result.notes.append(f"sub-index {result.max_sub_index} exceeds the 0-500 AQI scale")
        elif result.aqi is not None and result.max_sub_index - result.aqi > 50:
            result.consistent = False
            result.notes.append(
                f"max sub-index ({result.max_sub_index}) far above reported aqi ({result.aqi})"
            )
        else:
            result.consistent = True

    return result


def vet_station(uid: int, token: str, timeout: int) -> StationResult:
    last_error = None
    for prefix in _PREFIXES:
        url = f"https://api.waqi.info/feed/{prefix}{uid}/?token={token}"
        try:
            response = _get(url, timeout=timeout)
            payload = response.json()
        except (requests.RequestException, _RetryableHTTPError) as exc:
            last_error = f"{prefix}{uid}: request failed ({exc})"
            time.sleep(_REQUEST_GAP_SECONDS)
            continue

        ok, data, err = classify_aqicn_response(payload)
        time.sleep(_REQUEST_GAP_SECONDS)

        if ok:
            return _parse_reading(uid, prefix, data)

        last_error = f"{prefix}{uid}: {err}"
        if err != "Unknown ID":
            # A different kind of error (bad token, rate limit, etc.)
            # won't be fixed by trying the other prefix.
            break

    return StationResult(uid=uid, ok=False, error=last_error)


def _format_row(r: StationResult) -> str:
    if not r.ok:
        return f"{r.uid:<8} FAILED   {r.error}"

    flag = "OK" if r.consistent else ("SUSPECT" if r.consistent is False else "?")
    name = (r.name or "?")[:28]
    return (
        f"{r.working_id:<10} {name:<28} "
        f"aqi={str(r.aqi):<6} dom={str(r.dominant_pollutant):<6} "
        f"age={str(r.age_hours):>5}h  {flag:<7} {'; '.join(r.notes)}"
    )


def main() -> None:
    configure_logging()
    settings = get_settings()

    results = [
        vet_station(uid, settings.aqicn_token, settings.request_timeout_seconds)
        for uid in CANDIDATE_STATIONS
    ]

    working = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    # Freshest + internally-consistent first, so the top of the table is
    # your primary + fallback-chain candidates in priority order.
    working.sort(
        key=lambda r: ((r.consistent is False),
                       r.age_hours if r.age_hours is not None else float("inf"))
    )

    print("\nWorking stations (best candidates first):")
    print("-" * 100)
    for r in working:
        print(_format_row(r))

    if failed:
        print("\nFailed (neither @ nor A prefix worked, or a real API error):")
        print("-" * 100)
        for r in failed:
            print(_format_row(r))

    logger.info("%d/%d candidate stations usable", len(working), len(results))


if __name__ == "__main__":
    main()