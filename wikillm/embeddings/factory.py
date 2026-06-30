"""Factory that builds the right :class:`EmbeddingProvider` from settings.

Resolution logic:

1. If a rotation is configured (``EMBEDDING_ROTATION`` JSON or the parallel
   ``EMBEDDING_*`` lists), build one provider per member and wrap them in a
   :class:`~wikillm.embeddings.rotating.RotatingEmbeddingProvider`.
2. Otherwise build a single provider from ``EMBEDDING_PROVIDER`` — a
   :class:`~wikillm.embeddings.local.LocalEmbeddingProvider` when it is
   ``local``, else a :class:`~wikillm.embeddings.remote.RemoteEmbeddingProvider`.
"""

from __future__ import annotations

from wikillm.config import RotationMember, Settings
from wikillm.embeddings.base import EmbeddingProvider
from wikillm.embeddings.local import LocalEmbeddingProvider
from wikillm.embeddings.remote import RemoteEmbeddingProvider
from wikillm.embeddings.rotating import RotatingEmbeddingProvider
from wikillm.logging_setup import get_logger

logger = get_logger(__name__)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the embedding provider described by ``settings``.

    Args:
        settings: The loaded application settings.

    Returns:
        A ready-to-use :class:`EmbeddingProvider`.

    Raises:
        ValueError: If a rotation is configured but contains no members.
    """
    members = settings.rotation_members()
    if members:
        providers = [_build_member(member, settings) for member in members]
        logger.info(
            "Embedding rotation enabled: %d member(s), strategy=%s",
            len(providers),
            settings.embedding_rotation_strategy,
        )
        return RotatingEmbeddingProvider(
            members=providers,
            strategy=settings.embedding_rotation_strategy,
            cooldown_seconds=settings.embedding_rotation_cooldown,
        )

    return _build_single(settings)


def _build_single(settings: Settings) -> EmbeddingProvider:
    """Build a single (non-rotating) provider from the top-level settings.

    Args:
        settings: The loaded application settings.

    Returns:
        A local or remote embedding provider.
    """
    provider = settings.embedding_provider
    if provider == "local":
        logger.info("Using local embedding provider: %s", settings.embedding_model)
        return LocalEmbeddingProvider(model_name=settings.embedding_model)

    logger.info("Using remote embedding provider: %s (%s)", provider, settings.embedding_model)
    return RemoteEmbeddingProvider(
        provider=provider,
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def _build_member(member: RotationMember, settings: Settings) -> EmbeddingProvider:
    """Build one provider for a rotation member.

    Members inherit the top-level model as a default when they don't specify
    one, so the parallel-list configuration form stays terse.

    Args:
        member: The rotation member specification.
        settings: The application settings (used for fallback defaults).

    Returns:
        A local or remote embedding provider for this member.
    """
    model = member.model or settings.embedding_model
    if member.provider == "local":
        return LocalEmbeddingProvider(model_name=model)
    return RemoteEmbeddingProvider(
        provider=member.provider,
        model=model,
        api_key=member.api_key,
        base_url=member.base_url,
    )
