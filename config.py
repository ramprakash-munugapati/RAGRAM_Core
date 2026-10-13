"""Environment-driven configuration for RAGRAM Core. No hardcoded values."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Optional: load a local `.env` file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a dev convenience, not a hard dep
    pass


def _env_str(key: str, default: str) -> str:
    """Read a string env var, falling back to `default` when unset/empty."""
    value = os.getenv(key)
    return value if value and value.strip() else default


def _env_int(key: str, default: int) -> int:
    """Read an int env var; raises a clear error on malformed values."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key}={raw!r} is not a valid integer") from exc


def _env_float_or_none(key: str, default: float | None) -> float | None:
    """Read an optional float env var. Empty string disables the threshold."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    if raw.strip().lower() in {"none", "off", "disabled"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key}={raw!r} is not a valid float") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration resolved from environment variables."""

    # --- PostgreSQL connection ---
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_dbname: str

    # --- Embedding generation ---
    embedding_provider: str  # "openai" | "huggingface"
    embedding_model: str
    embedding_dim: int  # must match the model's output dimensionality

    # --- LLM synthesis ---
    llm_model: str
    openai_api_key: str | None

    # --- Chunking / retrieval ---
    chunk_size: int  # max characters per chunk
    chunk_overlap: int  # characters of shared context between adjacent chunks
    top_k: int  # number of chunks retrieved per query
    max_distance: float | None  # cosine-distance cutoff (lower = stricter relevance)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a Settings instance from `RAGRAM_*` environment variables."""
        settings = cls(
            pg_host=_env_str("RAGRAM_PG_HOST", "localhost"),
            pg_port=_env_int("RAGRAM_PG_PORT", 5432),
            pg_user=_env_str("RAGRAM_PG_USER", "postgres"),
            pg_password=_env_str("RAGRAM_PG_PASSWORD", ""),
            pg_dbname=_env_str("RAGRAM_PG_DBNAME", "ragram"),
            embedding_provider=_env_str("RAGRAM_EMBEDDING_PROVIDER", "openai").lower(),
            embedding_model=_env_str("RAGRAM_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=_env_int("RAGRAM_EMBEDDING_DIM", 1536),
            llm_model=_env_str("RAGRAM_LLM_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("RAGRAM_OPENAI_API_KEY"),
            chunk_size=_env_int("RAGRAM_CHUNK_SIZE", 900),
            chunk_overlap=_env_int("RAGRAM_CHUNK_OVERLAP", 150),
            top_k=_env_int("RAGRAM_TOP_K", 5),
            max_distance=_env_float_or_none("RAGRAM_MAX_DISTANCE", 0.5),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail fast on structurally invalid configuration."""
        if self.embedding_provider not in {"openai", "huggingface"}:
            raise ValueError(
                f"Unsupported embedding provider {self.embedding_provider!r}. "
                "Expected 'openai' or 'huggingface'."
            )
        if self.embedding_dim <= 0:
            raise ValueError("RAGRAM_EMBEDDING_DIM must be a positive integer.")
        if self.chunk_size <= 0:
            raise ValueError("RAGRAM_CHUNK_SIZE must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAGRAM_CHUNK_OVERLAP must satisfy 0 <= overlap < chunk_size.")
        if self.top_k <= 0:
            raise ValueError("RAGRAM_TOP_K must be positive.")

    @property
    def sqlalchemy_url(self) -> str:
        """Connection URL for SQLAlchemy using the psycopg3 driver."""
        # postgresql+psycopg:// selects psycopg3 under SQLAlchemy 2.x.
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_dbname}"
        )
