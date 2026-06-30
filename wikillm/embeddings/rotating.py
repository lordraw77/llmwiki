"""Rotating embedding provider with round-robin scheduling and failover.

This provider wraps a list of concrete :class:`EmbeddingProvider` members and
spreads calls across them. Two scheduling strategies are supported:

* ``round_robin`` — each call starts at the next member (load balancing).
* ``random`` — each call starts at a random member.

On a *retryable* failure (rate limit, timeout, 5xx) the current member is put on
a short cooldown and the call transparently fails over to the next available
member. The call only fails once every member has been exhausted.

All members must produce vectors of the same dimensionality; this is validated
the first time :attr:`dimension` is resolved.
"""

from __future__ import annotations

import random
import threading
import time
from typing import List, Optional

from wikillm.embeddings.base import EmbeddingError, EmbeddingProvider
from wikillm.logging_setup import get_logger

logger = get_logger(__name__)


class RotatingEmbeddingProvider(EmbeddingProvider):
    """Round-robin / random load balancer with failover over many providers."""

    def __init__(
        self,
        members: List[EmbeddingProvider],
        strategy: str = "round_robin",
        cooldown_seconds: float = 30.0,
    ) -> None:
        """Initialise the rotating provider.

        Args:
            members: Concrete providers to rotate across. Must be non-empty.
            strategy: ``"round_robin"`` or ``"random"``.
            cooldown_seconds: How long a member that raised a retryable error is
                skipped before being eligible again.

        Raises:
            ValueError: If ``members`` is empty or ``strategy`` is unknown.
        """
        if not members:
            raise ValueError("RotatingEmbeddingProvider requires at least one member")
        if strategy not in {"round_robin", "random"}:
            raise ValueError(f"Unknown rotation strategy: {strategy}")

        self._members = members
        self._strategy = strategy
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._cursor = 0
        # Monotonic timestamp until which a member (by index) is on cooldown.
        self._cooldown_until: list[float] = [0.0] * len(members)
        self._dimension: Optional[int] = None

    @property
    def name(self) -> str:
        """Return a human-readable provider name listing the members."""
        return "rotating[" + ", ".join(m.name for m in self._members) + "]"

    @property
    def dimension(self) -> int:
        """Return the common embedding dimension across all members.

        Returns:
            The shared vector dimension.

        Raises:
            EmbeddingError: If members disagree on dimensionality or none can be
                probed successfully.
        """
        if self._dimension is not None:
            return self._dimension

        dims: list[int] = []
        last_error: Optional[Exception] = None
        for member in self._members:
            try:
                dims.append(member.dimension)
            except Exception as exc:  # probing may hit the network
                last_error = exc
                logger.warning("Could not probe dimension of %s: %s", member.name, exc)

        if not dims:
            raise EmbeddingError(
                f"Could not determine embedding dimension from any member: {last_error}"
            )
        if len(set(dims)) != 1:
            raise EmbeddingError(
                "Rotation members disagree on embedding dimension: "
                + ", ".join(f"{m.name}={d}" for m, d in zip(self._members, dims))
            )
        self._dimension = dims[0]
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the next member(s), failing over on errors.

        Args:
            texts: Input strings.

        Returns:
            One vector per input text.

        Raises:
            EmbeddingError: If every member fails for this call.
        """
        if not texts:
            return []

        order = self._call_order()
        errors: list[str] = []

        for index in order:
            if self._on_cooldown(index):
                continue
            member = self._members[index]
            try:
                vectors = member.embed_texts(texts)
                logger.debug("Embedded %d text(s) via %s", len(texts), member.name)
                return vectors
            except EmbeddingError as exc:
                errors.append(f"{member.name}: {exc}")
                if exc.retryable:
                    self._mark_cooldown(index)
                    logger.warning("Member %s failed (retryable), failing over: %s", member.name, exc)
                    continue
                logger.warning("Member %s failed (non-retryable): %s", member.name, exc)
            except Exception as exc:  # defensive: treat unknown errors as retryable
                errors.append(f"{member.name}: {exc}")
                self._mark_cooldown(index)
                logger.warning("Member %s raised unexpectedly, failing over: %s", member.name, exc)

        raise EmbeddingError(
            "All embedding rotation members failed: " + " | ".join(errors)
        )

    # ------------------------------------------------------------------
    # Internal scheduling helpers
    # ------------------------------------------------------------------
    def _call_order(self) -> List[int]:
        """Compute the order in which members are tried for one call.

        Returns:
            A list of member indices. The first index reflects the chosen
            strategy; the remainder follow in round-robin order so failover
            always has somewhere to go.
        """
        count = len(self._members)
        with self._lock:
            if self._strategy == "random":
                start = random.randrange(count)
            else:
                start = self._cursor
                self._cursor = (self._cursor + 1) % count
        return [(start + offset) % count for offset in range(count)]

    def _on_cooldown(self, index: int) -> bool:
        """Return whether member ``index`` is currently cooling down."""
        with self._lock:
            return time.monotonic() < self._cooldown_until[index]

    def _mark_cooldown(self, index: int) -> None:
        """Put member ``index`` on cooldown for ``cooldown_seconds``."""
        with self._lock:
            self._cooldown_until[index] = time.monotonic() + self._cooldown
