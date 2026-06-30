"""On-device embeddings backed by ``sentence-transformers``.

This provider runs entirely locally — no API key, no network at inference time
(the model is downloaded once on first use). It is the zero-configuration
default. ``sentence-transformers`` (and its ``torch`` dependency) is imported
lazily so the rest of the project works without the heavy ``local`` extra
installed when only remote providers are used.
"""

from __future__ import annotations

from typing import List, Optional

from wikillm.embeddings.base import EmbeddingError, EmbeddingProvider
from wikillm.logging_setup import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embed text using a local ``sentence-transformers`` model.

    The model is loaded lazily on the first embedding call to keep import time
    and memory low for processes that never embed (e.g. a help command).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialise the provider.

        Args:
            model_name: A ``sentence-transformers`` model identifier.
        """
        self._model_name = model_name
        self._model = None  # type: ignore[var-annotated]
        self._dimension: Optional[int] = None

    def _ensure_model(self) -> None:
        """Load the underlying model on first use.

        Raises:
            EmbeddingError: If ``sentence-transformers`` is not installed or the
                model cannot be loaded.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise EmbeddingError(
                "The 'local' embedding provider requires 'sentence-transformers'. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading local embedding model '%s'...", self._model_name)
        try:
            self._model = SentenceTransformer(self._model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:  # pragma: no cover - model/network issues
            raise EmbeddingError(f"Failed to load model '{self._model_name}': {exc}") from exc
        logger.info("Local embedding model ready (dimension=%d).", self._dimension)

    @property
    def dimension(self) -> int:
        """Return the model's embedding dimension, loading it if necessary."""
        if self._dimension is None:
            self._ensure_model()
        assert self._dimension is not None  # for type checkers
        return self._dimension

    @property
    def name(self) -> str:
        """Return a human-readable provider name."""
        return f"local:{self._model_name}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts locally.

        Args:
            texts: Input strings.

        Returns:
            One vector per input text.

        Raises:
            EmbeddingError: If the model cannot encode the inputs.
        """
        if not texts:
            return []
        self._ensure_model()
        assert self._model is not None
        try:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - runtime encode failure
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc
        return [vector.tolist() for vector in vectors]
