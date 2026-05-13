from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="SYS_DYNA_",
        extra="ignore",
    )

    db_path: Path = Field(default=PROJECT_ROOT / "data" / "sys_dyna.db")

    user_id: str = "sample.user"
    user_display_name: str = "Sample User"
    user_department: str = "Analytics"

    model_name: str = "gemini-3.1-pro-preview-customtools"

    max_tool_calls: int = 10
    per_tool_timeout_sec: float = 10.0
    turn_timeout_sec: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
