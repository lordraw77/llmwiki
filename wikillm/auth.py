"""Optional API-key authentication shared by the REST API and MCP HTTP transport.

Authentication is **disabled by default**: when ``API_KEY`` is unset, every
endpoint is open. When ``API_KEY`` is configured, requests must present the key
via either header:

* ``Authorization: Bearer <API_KEY>``
* ``X-API-Key: <API_KEY>``

The same logic powers both the FastAPI dependency (:func:`require_auth`) and the
raw-ASGI check used to guard the MCP SSE mount (:func:`is_request_authorized`).
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, status

from wikillm.config import get_settings


def _check_key(provided: Optional[str]) -> bool:
    """Return whether ``provided`` matches the configured API key.

    Uses a constant-time comparison to avoid timing side channels. When no key
    is configured, authentication is considered to pass (open mode).

    Args:
        provided: The candidate key extracted from a request header.

    Returns:
        ``True`` if access is allowed, ``False`` otherwise.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided, settings.api_key or "")


def _extract_key(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    """Pull the API key out of the supported headers.

    Args:
        authorization: Value of the ``Authorization`` header, if any.
        x_api_key: Value of the ``X-API-Key`` header, if any.

    Returns:
        The extracted key, or ``None`` if neither header carried one.
    """
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
        # Tolerate a bare token without the "Bearer" scheme.
        if not token:
            return authorization.strip()
    if x_api_key:
        return x_api_key.strip()
    return None


async def require_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency that enforces authentication when enabled.

    Args:
        authorization: The ``Authorization`` request header (injected).
        x_api_key: The ``X-API-Key`` request header (injected).

    Raises:
        HTTPException: ``401 Unauthorized`` if the key is missing or invalid
            while authentication is enabled.
    """
    if _check_key(_extract_key(authorization, x_api_key)):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def is_request_authorized(headers: dict[str, str]) -> bool:
    """Authorise a request from its raw (lower-cased) header mapping.

    Used by the MCP SSE ASGI guard, which does not have FastAPI's dependency
    injection available.

    Args:
        headers: A case-insensitive-keyed mapping of request headers. Keys are
            expected to be lower-cased (as Starlette provides them).

    Returns:
        ``True`` if the request may proceed, ``False`` otherwise.
    """
    authorization = headers.get("authorization")
    x_api_key = headers.get("x-api-key")
    return _check_key(_extract_key(authorization, x_api_key))
