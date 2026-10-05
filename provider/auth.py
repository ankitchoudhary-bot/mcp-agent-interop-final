"""
Token -> identity/scope resolution for the provider.

This is intentionally simple (an in-memory token table) because the point of
this project is the interoperability and failure-handling behavior around
auth, not building a real identity provider. Swapping this module for a real
OAuth/JWT verifier would not require changing anything in server.py beyond
the call to `authenticate`.

Every tool call must carry a `token` argument (see shared/schemas.py). This
keeps auth transport-agnostic: it works identically whether the MCP session
is stdio or a networked transport like SSE/streamable HTTP, where we can't
always rely on transport-level headers.
"""
from __future__ import annotations

from dataclasses import dataclass


class AuthError(Exception):
    """Raised for any auth failure. Carries a machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Identity:
    name: str
    scopes: frozenset[str]

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


# --------------------------------------------------------------------------
# Toy token table. In a real system this would be a database row or a call
# out to an identity provider, not a hardcoded dict.
# --------------------------------------------------------------------------
_TOKENS: dict[str, Identity] = {
    "tok-consumer-full": Identity(
        name="consumer-agent",
        scopes=frozenset({"calendar:read", "calendar:write", "db:read", "search:read"}),
    ),
    "tok-consumer-readonly": Identity(
        name="consumer-agent-readonly",
        scopes=frozenset({"calendar:read", "db:read", "search:read"}),
    ),
    "tok-search-only": Identity(
        name="limited-agent",
        scopes=frozenset({"search:read"}),
    ),
}


def authenticate(token: str | None) -> Identity:
    """
    Resolve a bearer token to an Identity.

    Raises:
        AuthError(code="AUTH_MISSING")  if token is None/empty.
        AuthError(code="AUTH_INVALID")  if token doesn't match a known identity.
    """
    if not token:
        raise AuthError("AUTH_MISSING", "No auth token was provided with this call.")
    identity = _TOKENS.get(token)
    if identity is None:
        raise AuthError("AUTH_INVALID", "The provided auth token is not recognized.")
    return identity


def authorize(identity: Identity, required_scope: str) -> None:
    """
    Raises:
        AuthError(code="AUTH_SCOPE_DENIED") if identity lacks required_scope.
    """
    if not identity.has_scope(required_scope):
        raise AuthError(
            "AUTH_SCOPE_DENIED",
            f"Identity '{identity.name}' does not have required scope '{required_scope}'.",
        )


#npx @modelcontextprotocol/inspector@latest

#cloudflared tunnel --url http://127.0.0.1:8000



# $env:MCP_TRANSPORT="streamable-http"
# $env:MCP_HOST="127.0.0.1"
# $env:MCP_PORT="8000"
# $env:MCP_PUBLIC_HOST="binding-grams-cities-legitimate.trycloudflare.com"

# python -m provider.server