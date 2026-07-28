"""
Centralized, validated configuration.

Loading every required environment variable through a single Pydantic
settings object means the pipeline fails fast, with one clear error
message naming exactly which variable is missing -- instead of failing
halfway through a run with a confusing `NoneType has no attribute` error
three function calls deep.

`get_settings()` is lazy (only instantiated when actually called), so
importing this module -- e.g. when pytest collects test files -- never
fails just because `.env` isn't present in that context.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aqicn_token: str
    openweather_api_key: str
    hopsworks_api_key: str
    hopsworks_project_name: str

    # "karachi" (no @ prefix) queries AQICN's generic city-level lookup,
    # which resolves to whichever station AQICN currently considers
    # nearest -- not a fixed sensor. For a time-series project, that's a
    # real risk: if AQICN's routing ever changes which station answers
    # this query, the data source silently switches mid-dataset, and the
    # model would wrongly read that as a real AQI jump. Pinning to a
    # specific, known station ID (the US Consulate station, found during
    # setup research) avoids that -- @ prefix means "exact station", not
    # "nearest to this query".
    aqicn_station: str = "@11790"
    karachi_lat: float = 24.8607
    karachi_lon: float = 67.0011

    request_timeout_seconds: int = Field(default=10, ge=1, le=60)
    max_retries: int = Field(default=3, ge=0, le=10)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()