"""OIDC JWT validation for Clerk-issued tokens."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import httpx
from jose import JWTError, jwt  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class OIDCValidator:
    """Validates JWTs against a Clerk OIDC issuer's JWKS endpoint.

    Caches JWKS keys with configurable TTL to avoid per-request network calls.
    """

    def __init__(
        self,
        issuer: str,  # e.g. "https://clerk.your-app.com"
        audience: str,  # e.g. "engram-api"
        group_claim: str = "org_id",  # JWT claim that maps to group_id
        jwks_cache_ttl: float = 300.0,  # 5 minutes
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._group_claim = group_claim
        self._jwks_cache_ttl = jwks_cache_ttl
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: float = 0.0
        self._client = httpx.AsyncClient(timeout=10.0)

    async def _get_jwks(self) -> dict[str, Any]:
        """Fetch JWKS from issuer, with TTL cache."""
        now = time.monotonic()
        if self._jwks and (now - self._jwks_fetched_at) < self._jwks_cache_ttl:
            return self._jwks

        jwks_url = f"{self._issuer}/.well-known/jwks.json"
        resp = await self._client.get(jwks_url)
        resp.raise_for_status()
        self._jwks = cast(dict[str, Any], resp.json())
        self._jwks_fetched_at = now
        logger.debug("Refreshed JWKS from %s", jwks_url)
        return self._jwks

    async def validate_token(self, token: str) -> dict[str, Any]:
        """Validate a JWT and return claims.

        Returns dict with at least:
          - sub: user ID
          - group_id: resolved from group_claim (or "default")

        Raises ValueError on invalid/expired tokens.
        """
        try:
            jwks = await self._get_jwks()
            claims = cast(
                dict[str, Any],
                jwt.decode(
                    token,
                    jwks,
                    algorithms=["RS256"],
                    audience=self._audience,
                    issuer=self._issuer,
                ),
            )
            # Extract group_id from configured claim
            group_id = claims.get(self._group_claim, "default")
            if not group_id:
                group_id = "default"
            claims["group_id"] = group_id
            return claims
        except JWTError as e:
            raise ValueError(f"Invalid JWT: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
