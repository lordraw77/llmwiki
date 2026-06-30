"""Embedding providers and the factory that builds them.

An *embedding provider* turns text into dense float vectors. The package offers
three concrete kinds, all implementing :class:`~wikillm.embeddings.base.EmbeddingProvider`:

* :class:`~wikillm.embeddings.local.LocalEmbeddingProvider` — on-device
  ``sentence-transformers`` models (no network, no API key).
* :class:`~wikillm.embeddings.remote.RemoteEmbeddingProvider` — any
  OpenAI-compatible ``/embeddings`` HTTP endpoint (OpenRouter, Groq, Gemini,
  Mistral, NVIDIA, Cerebras, Cloudflare, Puter, ...).
* :class:`~wikillm.embeddings.rotating.RotatingEmbeddingProvider` — wraps a list
  of providers and rotates across them with automatic failover.

Use :func:`~wikillm.embeddings.factory.build_embedding_provider` to construct
the right one from :class:`~wikillm.config.Settings`.
"""

from __future__ import annotations

from wikillm.embeddings.base import EmbeddingError, EmbeddingProvider
from wikillm.embeddings.factory import build_embedding_provider

__all__ = ["EmbeddingError", "EmbeddingProvider", "build_embedding_provider"]
