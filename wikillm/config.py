"""Typed application configuration loaded from environment / ``.env``.

This module centralises every tunable knob of the service into a single
:class:`Settings` object built on top of ``pydantic-settings``. It is imported
by virtually every other module, so it deliberately has no heavy dependencies.

The settings cover four areas:

* HTTP server + optional authentication.
* Logging (kept off ``stdout`` for the MCP stdio transport).
* Storage: which vector backend to use and where it persists.
* Embeddings: a single provider *or* a rotation across several providers/keys.

Use :func:`get_settings` to obtain a cached singleton everywhere in the app.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider identifier type. ``openai_compatible`` is the generic escape hatch
# for any service exposing an OpenAI-style ``/embeddings`` endpoint.
ProviderName = Literal[
    "local",
    "openai_compatible",
    "openrouter",
    "groq",
    "gemini",
    "cloudflare",
    "cerebras",
    "mistral",
    "nvidia",
    "puter",
]

VectorBackend = Literal["chroma", "qdrant"]
RotationStrategy = Literal["round_robin", "random"]

# Built-in default base URLs for the remote, OpenAI-compatible providers.
# ``cloudflare`` and ``puter`` are intentionally absent: they require a
# user-specific base URL (account id / custom gateway) and therefore must be
# provided explicitly via ``EMBEDDING_BASE_URL`` (or per-member in rotation).
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai_compatible": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}


class RotationMember(BaseSettings):
    """A single provider/key entry participating in embedding rotation.

    Attributes:
        provider: The provider identifier (e.g. ``"mistral"``). ``"local"`` is
            allowed but unusual inside a rotation.
        model: The embedding model name for this member.
        api_key: Bearer token for the provider, if it requires one.
        base_url: Optional override for the provider's default base URL.
    """

    provider: ProviderName = "openai_compatible"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class Settings(BaseSettings):
    """Strongly-typed runtime configuration for the whole service.

    Values are read (in order of precedence) from explicit constructor
    arguments, real environment variables, then a ``.env`` file in the current
    working directory. Field names map to upper-case environment variables
    (e.g. ``app_port`` ← ``APP_PORT``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- HTTP server -------------------------------------------------------
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --- Authentication (optional) ----------------------------------------
    # When ``None`` / empty, authentication is disabled and every endpoint is
    # open. When set, both REST and MCP HTTP endpoints require the key.
    api_key: Optional[str] = None

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # --- Storage -----------------------------------------------------------
    data_dir: str = "data"
    collection_name: str = "wikillm"
    vector_backend: VectorBackend = "chroma"

    chroma_path: str = "data/chroma"

    qdrant_path: str = "data/qdrant"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Embeddings: single provider --------------------------------------
    embedding_provider: ProviderName = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

    # --- Embeddings: rotation ---------------------------------------------
    # JSON list of member dicts; takes precedence over the parallel lists.
    embedding_rotation: Optional[list[dict[str, Any]]] = None
    embedding_providers: Optional[list[str]] = None
    embedding_api_keys: Optional[list[str]] = None
    embedding_models: Optional[list[str]] = None
    embedding_base_urls: Optional[list[str]] = None
    embedding_rotation_strategy: RotationStrategy = "round_robin"
    embedding_rotation_cooldown: float = 30.0

    # --- MCP ---------------------------------------------------------------
    mcp_sse_path: str = "/mcp"

    # --- Startup ingestion -------------------------------------------------
    # When set, the stdio server ingests this file/directory once at startup,
    # before serving requests. Useful for pre-loading a mounted corpus.
    ingest_on_start: Optional[str] = None
    ingest_on_start_recursive: bool = True

    # ----------------------------------------------------------------------
    # Validators / normalisers
    # ----------------------------------------------------------------------
    @field_validator(
        "embedding_rotation",
        "embedding_providers",
        "embedding_api_keys",
        "embedding_models",
        "embedding_base_urls",
        mode="before",
    )
    @classmethod
    def _parse_listish(cls, value: Any) -> Any:
        """Accept JSON arrays or comma-separated strings for list fields.

        ``pydantic-settings`` would otherwise try to JSON-decode every list
        field, which makes simple ``A,B,C`` env values fail. This validator
        first tries JSON, then falls back to comma splitting.

        Args:
            value: The raw environment value (or already-parsed object).

        Returns:
            A Python list, ``None``, or the original value untouched.
        """
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
                return parsed
            except json.JSONDecodeError:
                return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator(
        "api_key", "embedding_api_key", "embedding_base_url", "qdrant_url", "log_file", "ingest_on_start"
    )
    @classmethod
    def _empty_to_none(cls, value: Optional[str]) -> Optional[str]:
        """Treat empty strings (common in ``.env`` files) as ``None``."""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        """Ensure chunking parameters are internally consistent."""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be a positive integer")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must satisfy 0 <= overlap < CHUNK_SIZE")
        return self

    # ----------------------------------------------------------------------
    # Derived helpers
    # ----------------------------------------------------------------------
    @property
    def auth_enabled(self) -> bool:
        """Whether request authentication is active (i.e. an API key is set)."""
        return bool(self.api_key)

    def default_base_url(self, provider: str) -> Optional[str]:
        """Return the built-in default base URL for a remote provider.

        Args:
            provider: A provider identifier.

        Returns:
            The default base URL, or ``None`` if the provider has none
            (``local``, ``cloudflare``, ``puter``).
        """
        return DEFAULT_BASE_URLS.get(provider)

    def rotation_members(self) -> list[RotationMember]:
        """Build the list of rotation members from configuration.

        Resolution order:

        1. ``embedding_rotation`` (JSON list of dicts) if present.
        2. Otherwise the parallel lists ``embedding_providers`` /
           ``embedding_api_keys`` / ``embedding_models`` /
           ``embedding_base_urls``, aligned by index.

        For the parallel-list form, ``models`` / ``base_urls`` shorter than the
        provider list reuse the single-provider defaults
        (:attr:`embedding_model` / :attr:`embedding_base_url`).

        Returns:
            A possibly empty list of :class:`RotationMember`. An empty list
            means "no rotation configured" and the caller should fall back to
            the single-provider configuration.
        """
        if self.embedding_rotation:
            return [RotationMember(**member) for member in self.embedding_rotation]

        if not self.embedding_providers:
            return []

        members: list[RotationMember] = []
        for index, provider in enumerate(self.embedding_providers):
            members.append(
                RotationMember(
                    provider=provider,  # type: ignore[arg-type]
                    api_key=_pick(self.embedding_api_keys, index, self.embedding_api_key),
                    model=_pick(self.embedding_models, index, self.embedding_model),
                    base_url=_pick(self.embedding_base_urls, index, self.embedding_base_url),
                )
            )
        return members


def _pick(values: Optional[list[str]], index: int, fallback: Optional[str]) -> Optional[str]:
    """Return ``values[index]`` when available, else a fallback value.

    Args:
        values: An optional list of strings (aligned with providers).
        index: The provider index being resolved.
        fallback: Value to use when the list is missing or too short.

    Returns:
        The selected string, or ``fallback``.
    """
    if values and index < len(values):
        return values[index]
    return fallback


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Using an ``lru_cache`` ensures the ``.env`` file is parsed only once and
    that the REST app and the MCP server observe identical configuration.

    Returns:
        The shared :class:`Settings` singleton.
    """
    return Settings()
