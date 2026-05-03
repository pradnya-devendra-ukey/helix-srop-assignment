import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Force .env to win for GOOGLE_API_KEY ─────────────────────────────────────
# pydantic-settings gives system env vars higher priority than .env by default.
# If a stale/leaked GOOGLE_API_KEY is set at the OS level it silently overrides
# the key in .env.  We work around this by reading .env directly with
# dotenv_values (which never touches os.environ) and explicitly writing
# GOOGLE_API_KEY into os.environ so that both pydantic-settings and the ADK
# SDK see the correct key.
_dot_env_path = Path(__file__).parent.parent / ".env"
_dot_env_values = dotenv_values(_dot_env_path)
if "GOOGLE_API_KEY" in _dot_env_values and _dot_env_values["GOOGLE_API_KEY"]:
    os.environ["GOOGLE_API_KEY"] = _dot_env_values["GOOGLE_API_KEY"].strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = ""
    adk_model: str = "gemini-2.5-flash"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10


settings = Settings()
