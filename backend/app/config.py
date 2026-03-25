"""Application settings, read from environment variables and an optional `.env` file."""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"


class Settings(BaseSettings):
    # The repo-root .env is used when running from backend/ during development; backend/.env, if
    # present, overrides it. In Docker, settings come from the container environment.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")

    # Portkey gateway (OpenAI-compatible). Authenticated with x-portkey-api-key, never a bearer token.
    portkey_base_url: str = DEFAULT_PORTKEY_BASE_URL
    portkey_api_key: SecretStr
    portkey_virtual_key: SecretStr | None = None
    # Embeddings may use their own Portkey credentials; each falls back to the chat value.
    embedding_portkey_api_key: SecretStr | None = None
    embedding_portkey_virtual_key: SecretStr | None = None

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(256, ge=1, le=2048)
    llm_stream_usage: bool = True
    # Blank = don't send a temperature (reasoning models reject the parameter).
    llm_temperature: float | None = Field(0.2, ge=0, le=2)

    qdrant_url: str = "http://qdrant:6333"
    database_url: str = "sqlite+aiosqlite:////data/app.db"
    upload_dir: Path = Path("/data/uploads")
    max_upload_mb: int = Field(25, ge=1, le=200)

    agent_max_steps: int = Field(8, ge=1, le=20)
    tool_timeout_seconds: float = Field(20, gt=0, le=300)

    log_level: str = "INFO"
    cors_origins: Annotated[list[str], NoDecode] = []

    @field_validator("portkey_api_key")
    @classmethod
    def _require_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("PORTKEY_API_KEY must not be empty")
        return value

    @field_validator(
        "portkey_virtual_key",
        "embedding_portkey_api_key",
        "embedding_portkey_virtual_key",
        "llm_temperature",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("portkey_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def chat_api_key(self) -> str:
        return self.portkey_api_key.get_secret_value()

    @property
    def chat_virtual_key(self) -> str | None:
        return self.portkey_virtual_key.get_secret_value() if self.portkey_virtual_key else None

    @property
    def embedding_api_key(self) -> str:
        if self.embedding_portkey_api_key:
            return self.embedding_portkey_api_key.get_secret_value()
        return self.chat_api_key

    @property
    def embedding_virtual_key(self) -> str | None:
        if self.embedding_portkey_virtual_key:
            return self.embedding_portkey_virtual_key.get_secret_value()
        return self.chat_virtual_key

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings_or_exit() -> Settings:
    """Settings, or a readable configuration error (exit code 2) instead of a traceback."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}" for err in exc.errors()
        )
        print(
            f"Configuration error:\n{problems}\nSet these in .env (copy .env.example) or the environment.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


@lru_cache
def get_settings() -> Settings:
    return load_settings_or_exit()
