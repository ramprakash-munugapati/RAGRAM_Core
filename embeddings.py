"""Embedding generation: pluggable providers with a common interface.

Supported providers (selected via RAGRAM_EMBEDDING_PROVIDER):
    - "openai"       : OpenAI Embeddings API (e.g. text-embedding-3-small)
    - "huggingface"  : local SentenceTransformers (e.g. all-MiniLM-L6-v2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from config import Settings


class EmbeddingProvider(ABC):
    """Contract for turning text into fixed-dimension float vectors."""

    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks (order-preserving)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI Embeddings API provider (openai>=1.x SDK)."""

    def __init__(self, model: str, dimension: int, api_key: str | None) -> None:
        from openai import OpenAI  # deferred import keeps module import cheap

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # One API call for the whole batch; `data` is returned in input order.
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in response.data]
        self._check_dimensions(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _check_dimensions(self, vectors: list[list[float]]) -> None:
        """Surface a config error immediately instead of failing at INSERT time."""
        if vectors and len(vectors[0]) != self.dimension:
            raise ValueError(
                f"Model {self._model!r} returned {len(vectors[0])}-dim vectors, "
                f"but RAGRAM_EMBEDDING_DIM={self.dimension}. Update the setting."
            )


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Local SentenceTransformers provider (no API key, runs on CPU/GPU)."""

    def __init__(self, model: str, dimension: int) -> None:
        from sentence_transformers import SentenceTransformer  # deferred import

        self._model = SentenceTransformer(model)
        self._model_name = model
        self.dimension = dimension
        # Cross-check the declared dim against the model's real output size.
        model_dim = self._model.get_sentence_embedding_dimension()
        if model_dim is not None and model_dim != dimension:
            raise ValueError(
                f"Model {model!r} outputs {model_dim}-dim vectors, but "
                f"RAGRAM_EMBEDDING_DIM={dimension}. Update the setting."
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory mapping the configured provider name to a concrete instance."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            api_key=settings.openai_api_key,
        )
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddingProvider(
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
