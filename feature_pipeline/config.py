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

    # Vetted against real live readings via vet_stations.py (2026-09-02):
    # A401143 (University of Karachi) was freshest + internally consistent.
    # Was "karachi" (a city-level alias, not a real station ID) -- kept the
    # field name as-is so nothing else referencing settings.aqicn_station
    # breaks, just updated the value.
    aqicn_station: str = "A401143"

    # Fallback chain fetch_aqicn_data() walks through, in order, when the
    # primary station is unreachable or returns an error. Also vetted via
    # vet_stations.py -- all seven returned fresh, consistent readings.
    # Excludes A554545 / A545140 / A544708, which were 17-26h stale with
    # no current AQI at vetting time.
    aqicn_fallback_stations: list[str] = [
        "A545320",
        "A544681",
        "A545422",
        "A558319",
        "A547342",
        "A544699",
    ]

    karachi_lat: float = 24.8607
    karachi_lon: float = 67.0011
    request_timeout_seconds: int = 10

    # --- extension modules: all optional, only required if you use them ---
    gemini_api_key: str | None = None                    # A, D, E
    upstash_vector_rest_url: str | None = None            # A
    upstash_vector_rest_token: str | None = None          # A
    telegram_bot_token: str | None = None                 # F
    telegram_chat_id: str | None = None                   # F
    langfuse_public_key: str | None = None                # G, I
    langfuse_secret_key: str | None = None                # G, I
    langfuse_host: str = "https://cloud.langfuse.com"      # G, I


@lru_cache
def get_settings() -> Settings:
    """Cached so environment variables are only parsed/validated once per
    process, not on every call site that needs a setting."""
    return Settings()