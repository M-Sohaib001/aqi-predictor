"""
Centralized, validated settings, loaded from environment variables (or a
local .env file via python-dotenv). Every other module reads secrets
through get_settings() rather than os.environ directly, so there's
exactly one place that knows where a key comes from -- and one place
that fails loudly and immediately (a clear pydantic validation error) if
a required key is missing, rather than a confusing KeyError deep inside
fetch_data.py or supabase_client.py.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aqicn_token: str
    openweather_api_key: str
    supabase_url: str
    supabase_key: str

    aqicn_station: str = "karachi"
    karachi_lat: float = 24.8607
    karachi_lon: float = 67.0011
    request_timeout_seconds: int = 10


@lru_cache
def get_settings() -> Settings:
    """Cached so environment variables are only parsed/validated once per
    process, not on every call site that needs a setting."""
    return Settings()