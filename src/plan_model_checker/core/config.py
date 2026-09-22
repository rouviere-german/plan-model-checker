"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from plan_model_checker.core.security import parse_key_entry

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_KEY_ENTRY = re.compile(r"^[0-9a-f]{64}(:[a-z]+(\+[a-z]+)*)?$")
_PUBLIC_KEY_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(default="plan-model-checker", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- agentic kit ---
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    llm_model: str = "claude-opus-5"
    # The spec asks for two roles: a cheaper planner and the default (Opus) model for repair.
    llm_planner_model: str = "claude-sonnet-5"
    llm_repair_model: str = "claude-opus-5"
    repairer: Literal["scripted", "anthropic"] = "scripted"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # The SDK default. Non-streaming requests up to 16K output tokens can legitimately take minutes;
    # a shorter total timeout cuts them off, and each retry can be billed again.
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    llm_refusal_fallback: bool = True
    budget_max_steps: int = Field(default=50, ge=0)
    budget_max_tokens: int = Field(default=200_000, ge=0)
    budget_max_usd: float = Field(default=1.0, ge=0)

    # --- plan model checker ---
    registry_path: Path | None = None
    registry_public_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    registry_sha256: str | None = None
    specs_path: Path | None = None
    repair_max_attempts: int = Field(default=3, ge=0, le=10)
    check_max_states: int = Field(default=250_000, ge=100)
    plan_max_nodes: int = Field(default=200, ge=1, le=2_000)
    plan_max_depth: int = Field(default=8, ge=1, le=32)
    plan_max_loop_bound: int = Field(default=8, ge=1, le=64)

    # --- storage ---
    data_dir: Path = Path("var")

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for entry in value:
            if not _KEY_ENTRY.fullmatch(entry):
                raise ValueError(
                    "api_key_sha256 entries must be lowercase hex SHA-256 digests, optionally "
                    "followed by :scope+scope"
                )
            parse_key_entry(entry)
        return value

    @field_validator("registry_path", "specs_path", "registry_sha256", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("registry_public_keys", mode="before")
    @classmethod
    def _parse_public_keys(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("registry_public_keys")
    @classmethod
    def _check_public_keys(cls, value: list[str]) -> list[str]:
        for key in value:
            if not _PUBLIC_KEY_HEX.fullmatch(key):
                raise ValueError("registry_public_keys must be hex-encoded raw Ed25519 keys")
        return value

    @field_validator("registry_sha256")
    @classmethod
    def _check_registry_pin(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_HEX.fullmatch(value):
            raise ValueError("registry_sha256 must be a lowercase hex SHA-256 digest")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
