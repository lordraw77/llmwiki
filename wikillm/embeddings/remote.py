"""Remote embeddings via OpenAI-compatible HTTP endpoints.

Most modern inference providers (OpenRouter, Groq, Mistral, NVIDIA, Cerebras,
Cloudflare Workers AI, Google Gemini's OpenAI-compat surface, ...) expose an
``POST {base_url}/embeddings`` endpoint that accepts and returns the OpenAI
schema. This single provider covers all of them; the only differences are the
base URL (resolved from :data:`wikillm.config.DEFAULT_BASE_URLS` or overridden)
and the bearer API key.

``puter`` is treated as a generic OpenAI-compatible endpoint as well: because it
has no fixed server-side embeddings URL, the caller must supply ``base_url``
explicitly.
"""

from __future__ import annotations

from typing import List, Optional

import httpx

from wikillm.config import DEFAULT_BASE_URLS
from wikillm.embeddings.base import EmbeddingError, EmbeddingProvider
from wikillm.logging_setup import get_logger

logger = get_logger(__name__)

# HTTP status codes that indicate a transient condition worth failing over.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RemoteEmbeddingProvider(EmbeddingProvider):
    """Call an OpenAI-compatible ``/embeddings`` endpoint over HTTP.

    The embedding dimension is discovered lazily from the first successful
    response and cached.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialise the provider.

        Args:
            provider: Provider identifier (used for default base URL + logs).
            model: The embedding model name to request.
            api_key: Bearer token. Optional for keyless/self-hosted endpoints.
            base_url: Base URL of the OpenAI-compatible API. When omitted, the
                provider's built-in default is used.
            timeout: Per-request timeout in seconds.

        Raises:
            EmbeddingError: If no base URL can be resolved for the provider.
        """
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._base_url = (base_url or DEFAULT_BASE_URLS.get(provider) or "").rstrip("/")
        if not self._base_url:
            raise EmbeddingError(
                f"No base URL configured for provider '{provider}'. "
                "Set EMBEDDING_BASE_URL (or the member's base_url)."
            )
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, probing the API once if needed."""
        if self._dimension is None:
            # Probe with a tiny input to discover the vector size.
            self._dimension = len(self.embed_texts(["dimension probe"])[0])
        return self._dimension

    @property
    def name(self) -> str:
        """Return a human-readable provider name."""
        return f"{self._provider}:{self._model}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts via the remote endpoint.

        Args:
            texts: Input strings.

        Returns:
            One vector per input text, ordered to match the request.

        Raises:
            EmbeddingError: On HTTP errors, timeouts, or malformed responses.
                The ``retryable`` flag is set for transient failures so the
                rotating provider can fail over.
        """
        if not texts:
            return []

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model, "input": texts}
        url = f"{self._base_url}/embeddings"

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise EmbeddingError(f"{self.name}: request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"{self.name}: HTTP error: {exc}", retryable=True) from exc

        if response.status_code != 200:
            retryable = response.status_code in _RETRYABLE_STATUS
            body = response.text[:500]
            raise EmbeddingError(
                f"{self.name}: embeddings endpoint returned {response.status_code}: {body}",
                retryable=retryable,
            )

        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> List[List[float]]:
        """Extract and order the embedding vectors from an API response.

        Args:
            response: A successful (200) HTTP response.

        Returns:
            Vectors ordered by the response's ``index`` field.

        Raises:
            EmbeddingError: If the JSON body does not match the expected schema.
        """
        try:
            data = response.json()
            items = data["data"]
            ordered = sorted(items, key=lambda item: item.get("index", 0))
            vectors = [list(map(float, item["embedding"])) for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"{self.name}: malformed embeddings response: {exc}") from exc

        if not vectors:
            raise EmbeddingError(f"{self.name}: empty embeddings response")
        return vectors
