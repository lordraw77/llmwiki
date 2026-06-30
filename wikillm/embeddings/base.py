"""Embedding provider abstraction.

Every provider implements the same minimal contract so the rest of the system
(knowledge base, vector store) never depends on a concrete embedding backend.
"""

from __future__ import annotations

import abc
from typing import List


class EmbeddingError(Exception):
    """Raised when an embedding backend fails to produce vectors.

    Carries an optional ``retryable`` flag so that wrappers such as the
    rotating provider can decide whether failing over to another member is
    worthwhile (e.g. rate limits / transient network errors are retryable,
    a malformed request is not).
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description.
            retryable: Whether the failure is transient and worth retrying on a
                different provider/key.
        """
        super().__init__(message)
        self.retryable = retryable


class EmbeddingProvider(abc.ABC):
    """Abstract base class for all embedding providers.

    Concrete subclasses must implement :meth:`embed_texts` and expose the
    embedding :attr:`dimension`. A convenience :meth:`embed_query` is provided
    for single-string inputs.
    """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the vectors produced by this provider."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A short human-readable identifier (used in logs)."""

    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: The input strings to embed. Must be non-empty.

        Returns:
            A list of vectors, one per input text, each of length
            :attr:`dimension`.

        Raises:
            EmbeddingError: If the backend fails to produce embeddings.
        """

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: The query text.

        Returns:
            A single embedding vector.

        Raises:
            EmbeddingError: If the backend fails.
        """
        return self.embed_texts([text])[0]
