"""
Thin async wrapper around mcp's v2 `Client` -- the single object that
replaced v1's three-layer transport/ClientSession/initialize() dance (see
https://py.sdk.modelcontextprotocol.io/v2/whats-new/). `Client` accepts a
URL (streamable HTTP), a StdioServerParameters (a real stdio subprocess), or
an in-process server object, and manages the underlying transport's task
group internally -- which is also why this rewrite replaced an earlier,
more manual version of this file built on the legacy
ClientSession+stdio_client/streamablehttp_client pattern: nesting two
independently-managed anyio task groups by hand, inside a pytest fixture
that yields across the boundary, triggered a genuine
"cancel scope in a different task" bug. Letting Client own that lifecycle
removes the manual nesting entirely.

Two jobs, both load-bearing for Phase 2's requirements:

1. Discovery. `connect()` calls the server's discovery endpoint
   (`list_tools`) and its `server://info` resource at runtime. Nothing
   about which tools exist or what arguments they take is hardcoded here
   or in consumer/agent.py -- `list_discovered_tools()` is the only source
   of truth agent.py is allowed to use when building the tool list it
   hands to Claude.

2. Failure translation. Every way a tool call can go wrong -- bad auth, a
   tool that's vanished, a timeout, a malformed response, an incompatible
   server -- is normalized into one of five typed exceptions below so
   consumer/agent.py can give each one its own explicit handling branch
   instead of one catch-all `except Exception`.

One explicit, deliberate exception to "never hardcode tool names": this
module keeps `_OUTPUT_MODELS`, a small registry mapping a known tool name to
the Pydantic model the consumer validates that tool's *response* against.
This isn't the same thing as hardcoding which tools exist or what
arguments they take (that's still 100% runtime-discovered) -- it's the
consumer's own typed expectation of response shape for the tools it knows
how to interpret, the same way any strongly-typed client of a versioned API
has response models for the endpoints it understands. A tool discovered at
runtime that ISN'T in this registry is not an error -- see
`_validate_output` -- it's simply passed through unvalidated, which is the
correct behavior for a genuinely dynamic client.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from mcp import Client, StdioServerParameters

from shared import schemas as schemas_module
from shared.schemas import (
    schema_major_version,
    CalendarListEventsOutput,
    CalendarCreateEventOutput,
    DatabaseQueryOutput,
    SearchWebOutput,
)

DEFAULT_TIMEOUT_SECONDS = 5.0
CLOSE_TIMEOUT_SECONDS = 5.0

_logger = logging.getLogger(__name__)

_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "calendar_list_events": CalendarListEventsOutput,
    "calendar_create_event": CalendarCreateEventOutput,
    "database_query": DatabaseQueryOutput,
    "search_web": SearchWebOutput,
}


# --------------------------------------------------------------------------
# Typed failure modes -- one per case named in the Phase 2 brief, plus a
# base class for anything else well-formed-but-rejected.
# --------------------------------------------------------------------------

class ConsumerError(Exception):
    """Base class for every typed failure the consumer can raise."""


class AuthFailure(ConsumerError):
    """Auth token missing/invalid, or the caller lacks the required scope."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ToolNotFoundError(ConsumerError):
    """The requested tool is not, or is no longer, offered by the server."""


class ToolTimeoutError(ConsumerError):
    """The tool call did not complete within the configured timeout."""


class ResponseValidationError(ConsumerError):
    """The tool's response does not match the consumer's expected schema."""


class SchemaVersionMismatch(ConsumerError):
    """The server's schema major version is newer than this client understands."""


@dataclass
class DiscoveredTool:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClientWrapper:
    """
    Transport-agnostic on purpose: everything above this class (agent.py,
    the failure-mode logic, the schema-version check) is identical whether
    the provider is a stdio subprocess or a genuinely separate process
    reachable over HTTP. Only `connect()` branches on transport -- that is
    the entire footprint Phase 3 (moving from stdio to streamable HTTP) has
    on this codebase.
    """

    def __init__(
        self,
        *,
        transport: str = "stdio",
        server_params: StdioServerParameters | None = None,
        url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if transport == "stdio" and server_params is None:
            raise ValueError("transport='stdio' requires server_params")
        if transport == "streamable-http" and url is None:
            raise ValueError("transport='streamable-http' requires url")
        if transport not in ("stdio", "streamable-http"):
            raise ValueError(f"Unsupported transport: {transport!r}")

        self._transport = transport
        self._server_params = server_params
        self._url = url
        self._timeout = timeout_seconds
        self._client: Client | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools_by_name: dict[str, DiscoveredTool] = {}

    async def connect(self) -> None:
        """
        Connect, discover tools, and check schema compatibility -- in that
        order, failing fast on a version mismatch before anything else is
        attempted. There is no point discovering tools we can't safely
        interpret the responses of.

        connect() owns the transport until it successfully returns: if
        anything after opening it fails (the version check in particular),
        it cleans up immediately, in this same call, rather than leaving a
        half-open connection for the caller to close later via a separate
        close() call. That distinction turned out to matter in practice,
        not just in principle: a real, reliably-reproduced hang occurred
        when cleanup was instead deferred to a later close() -- something
        about anyio's CancelScope nesting doesn't tolerate the exit stack
        being closed from a resumed frame after the exception has already
        propagated out and been caught elsewhere, even within the same
        asyncio Task. Closing it here, before re-raising, avoided the hang
        entirely and is also just the structurally correct place for it.
        """
        self._exit_stack = AsyncExitStack()
        try:
            target = self._server_params if self._transport == "stdio" else self._url
            self._client = await self._exit_stack.enter_async_context(Client(target))

            info = await self._read_server_info()
            server_version = info.get("schema_version", "v0")
            server_major = schema_major_version(server_version)
            # Read via the module, not a name imported at module-load time, so
            # that SUPPORTED_MAJOR_VERSION reflects its value *at connect time*
            # (this also makes the check straightforwardly testable/monkeypatchable
            # -- see consumer/demo.py's schema-version-mismatch demo).
            supported = schemas_module.SUPPORTED_MAJOR_VERSION
            if server_major > supported:
                raise SchemaVersionMismatch(
                    f"Server reports schema {server_version} (major {server_major}), "
                    f"but this client only understands up to major {supported}. "
                    f"Refusing to proceed rather than risk silently misinterpreting responses."
                )

            list_result = await self._client.list_tools()
            self._tools_by_name = {
                t.name: DiscoveredTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {},
                )
                for t in list_result.tools
            }
        except BaseException:
            await self._close_exit_stack()
            raise

    async def close(self) -> None:
        await self._close_exit_stack()

    async def _close_exit_stack(self) -> None:
        """
        Shared cleanup path for both connect()'s failure branch and a
        normal close(). Bounded by a timeout so a hung underlying
        transport shutdown (observed with a stdio-backed Client on
        Windows -- a known category of issue with asyncio subprocess
        transports on the ProactorEventLoop, not something fixable from
        this side of the SDK) can't block the caller forever; logs a
        warning and moves on instead.
        """
        if self._exit_stack is None:
            return
        stack, self._exit_stack = self._exit_stack, None
        self._client = None
        self._tools_by_name = {}
        try:
            await asyncio.wait_for(stack.aclose(), timeout=CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            _logger.warning(
                "MCPClientWrapper cleanup timed out after %.1fs waiting for the "
                "underlying connection to shut down; abandoning it rather than "
                "hanging forever.",
                CLOSE_TIMEOUT_SECONDS,
            )

    async def _read_server_info(self) -> dict[str, Any]:
        result = await self._client.read_resource("server://info")
        for content in result.contents:
            text = getattr(content, "text", None)
            if text is not None:
                return json.loads(text)
        raise ConsumerError("server://info resource returned no readable content.")

    def list_discovered_tools(self) -> list[DiscoveredTool]:
        return list(self._tools_by_name.values())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Executes one tool call end to end. Returns the validated output
        dict on success. Raises one of the five typed failures above on any
        problem -- never lets a raw MCPError / asyncio.TimeoutError /
        pydantic.ValidationError escape to the caller.
        """
        if name not in self._tools_by_name:
            raise ToolNotFoundError(
                f"Tool '{name}' is not offered by the connected server. "
                f"Available tools: {sorted(self._tools_by_name)}"
            )

        try:
            result = await asyncio.wait_for(self._client.call_tool(name, arguments), timeout=self._timeout)
            
        except asyncio.TimeoutError as e:
            raise ToolTimeoutError(f"Tool '{name}' did not respond within {self._timeout}s.") from e
        
        except Exception as e:  # noqa: BLE001 -- translated below, not swallowed
            message = str(e)
            if "not found" in message.lower() or "unknown" in message.lower():
                raise ToolNotFoundError(f"Tool '{name}' was removed from the server after discovery: {message}") from e
            
            raise ConsumerError(f"Unexpected protocol error calling '{name}': {message}") from e

        payload = self._parse_tool_result(result, tool_name=name)

        if payload.get("ok") is False:
            error = payload.get("error", {})
            code = error.get("code", "UNKNOWN")
            message = error.get("message", "Tool call was rejected.")
            if code.startswith("AUTH_"):
                raise AuthFailure(code, message)
            raise ConsumerError(f"[{code}] {message}")

        return self._validate_output(name, payload)

    @staticmethod
    def _parse_tool_result(result: Any, *, tool_name: str) -> dict[str, Any]:
        # v2's Client populates .structured_content directly from a tool's
        # typed return value when available -- prefer it, since it skips a
        # redundant JSON round-trip. Fall back to parsing the text content
        # block (what provider/server.py's tools actually return today, and
        # what any older/lower-level path would still produce) so this
        # keeps working regardless of exactly which of those two the
        # installed SDK version populates for a given tool.
        structured = getattr(result, "structured_content", None)
        if structured:
            return structured

        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    raise ResponseValidationError(f"Response from '{tool_name}' was not valid JSON: {e}") from e
                
        raise ResponseValidationError(f"Response from '{tool_name}' had no text content to parse.")

    @staticmethod
    def _validate_output(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        model = _OUTPUT_MODELS.get(tool_name)
        if model is None:
            return payload  # unknown-but-successful tool: pass through, don't crash
        try:
            validated = model(**{k: v for k, v in payload.items() if k != "ok"})
        except ValidationError as e:
            raise ResponseValidationError(f"Response from '{tool_name}' failed schema validation: {e.errors()}") from e
        return validated.model_dump()
