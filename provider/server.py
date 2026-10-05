"""
MCP server exposing calendar, database, and search tools.

Every tool call goes through the same pipeline, implemented once in
`_dispatch` and reused by each thin @mcp.tool() wrapper:

    authenticate(token) -> authorize(scope) -> validate input schema
        -> call real tool implementation -> validate output schema
        -> audit log -> return a typed result

On any failure in that pipeline, the tool returns a plain dict shaped like
{"ok": False, "error": {...ToolError fields}} rather than letting an
exception propagate as a raw MCP protocol error. This is a deliberate
design choice (see ARCHITECTURE.md, design decision #2): different MCP
client SDKs and transports surface `isError` / raised-exception results
inconsistently, but a JSON payload with a stable "ok" discriminator is
identical everywhere, so the consumer's failure-handling code has exactly
one shape to check rather than one per transport quirk.

Run with:
    python -m provider.server            # stdio transport (default)
    MCP_TRANSPORT=sse python -m provider.server
    MCP_TRANSPORT=streamable-http python -m provider.server
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

from pydantic import ValidationError

from mcp.server.mcpserver import MCPServer

from provider.auth import authenticate, authorize, AuthError
from provider.audit_log import AuditLogger
from provider.tools import calendar, database, search
from shared.schemas import (
    SCHEMA_VERSION,
    CalendarListEventsInput,
    CalendarListEventsOutput,
    CalendarCreateEventInput,
    CalendarCreateEventOutput,
    DatabaseQueryInput,
    DatabaseQueryOutput,
    SearchWebInput,
    SearchWebOutput,
)

audit_logger = AuditLogger()

mcp = MCPServer("tool-provider",
    instructions=(
        f"Exposes calendar, read-only database, and web search tools. "
        f"Schema version {SCHEMA_VERSION}. Every tool requires a 'token' "
        f"argument. See the 'server://info' resource for full discovery "
        f"metadata including schema version and required scopes per tool."
    ),
)


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "schema_version": SCHEMA_VERSION}}


def _dispatch(*, tool_name: str, required_scope: str, raw_args: dict[str, Any], input_model: type, output_model: type, call_impl: Callable[[Any], dict],) -> dict[str, Any]:
    """
    Shared pipeline for every tool. `call_impl` receives the validated input
    model instance and must return a plain dict matching `output_model`'s
    fields (schema_version is filled in automatically).
    """
    started = time.monotonic()
    
    # If the caller omitted 'token' (it's an optional argument on every
    # tool -- see calendar_list_events etc. below) and this server was
    # started with PROVIDER_STATIC_TOKEN set, substitute it here, before
    # anything else -- including Pydantic input validation -- ever touches
    # raw_args. This is what lets a real MCP client like Claude Desktop use
    # this server with a single fixed identity, without the caller (a
    # human, or the model itself) ever needing to know a token exists at
    # all: a real caller in claude_desktop_config.json's "env" field is a
    # legitimate way to configure who "this server instance" is, distinct
    # from this project's other, deliberately more elaborate multi-token
    # demonstration (tests/demo.py/eval_harness.py/cli.py all still pass an
    # explicit token per call and are completely unaffected by this).
    # If neither an explicit token nor a static one is available, raw_args
    # is left with token=None, which fails Pydantic validation exactly like
    # a missing required field always has -- no new failure path needed.
    if not raw_args.get("token"):
        static_token = os.environ.get("PROVIDER_STATIC_TOKEN")
        if static_token:
            raw_args = {**raw_args, "token": static_token}
    token = raw_args.get("token")
    caller = "unknown"

    try:
        identity = authenticate(token)
        caller = identity.name
        authorize(identity, required_scope)

        validated_input = input_model(**raw_args)

        result_dict = call_impl(validated_input)
        validated_output = output_model(**result_dict)

        latency_ms = (time.monotonic() - started) * 1000
        audit_logger.log_call(caller=caller, tool_name=tool_name, arguments=raw_args, outcome="success", latency_ms=latency_ms,)
        return {"ok": True, **validated_output.model_dump()}

    except AuthError as e:
        latency_ms = (time.monotonic() - started) * 1000
        audit_logger.log_call(
            caller=caller,
            tool_name=tool_name,
            arguments=raw_args,
            outcome="rejected",
            error_code=e.code,
            error_message=e.message,
            latency_ms=latency_ms,
        )
        return _error_payload(e.code, e.message)

    except ValidationError as e:
        latency_ms = (time.monotonic() - started) * 1000
        message = f"Input/output failed schema validation: {e.errors()}"
        audit_logger.log_call(
            caller=caller,
            tool_name=tool_name,
            arguments=raw_args,
            outcome="rejected",
            error_code="VALIDATION_ERROR",
            error_message=message,
            latency_ms=latency_ms,
        )
        return _error_payload("VALIDATION_ERROR", message)

    except database.UnsafeQueryError as e:
        latency_ms = (time.monotonic() - started) * 1000
        audit_logger.log_call(
            caller=caller,
            tool_name=tool_name,
            arguments=raw_args,
            outcome="rejected",
            error_code="UNSAFE_QUERY",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        return _error_payload("UNSAFE_QUERY", str(e))

    except Exception as e:  # noqa: BLE001 -- last-resort catch, logged distinctly
        latency_ms = (time.monotonic() - started) * 1000
        audit_logger.log_call(
            caller=caller,
            tool_name=tool_name,
            arguments=raw_args,
            outcome="error",
            error_code="INTERNAL_ERROR",
            error_message=str(e),
            latency_ms=latency_ms,
        )
        return _error_payload("INTERNAL_ERROR", "An internal error occurred processing this tool call.")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@mcp.tool()
def calendar_list_events(start_date: str, end_date: str, token: str | None = None) -> dict:
    """List calendar events whose start falls within [start_date, end_date] (inclusive, YYYY-MM-DD).
       token: usually not needed -- omit it. Only required if this server
       wasn't started with a default identity configured."""
    return _dispatch(
        tool_name="calendar.list_events",
        required_scope="calendar:read",
        raw_args={"token": token, "start_date": start_date, "end_date": end_date},
        input_model=CalendarListEventsInput,
        output_model=CalendarListEventsOutput,
        call_impl=lambda inp: {"events": calendar.list_events(inp.start_date, inp.end_date)},
    )


@mcp.tool()
def calendar_create_event(title: str, start: str, end: str, token :str | None = None) -> dict:
    """Create a calendar event. start/end are ISO 8601 datetimes; end must be after start.
        token: usually not needed -- omit it. Only required if this server
        wasn't started with a default identity configured."""
    return _dispatch(
        tool_name="calendar.create_event",
        required_scope="calendar:write",
        raw_args={"token": token, "title": title, "start": start, "end": end},
        input_model=CalendarCreateEventInput,
        output_model=CalendarCreateEventOutput,
        call_impl=lambda inp: {"event": calendar.create_event(inp.title, inp.start, inp.end)},
    )


@mcp.tool()
def database_query(sql: str, token: str | None = None) -> dict:
    """Run a read-only SELECT query against the product catalog. Writes are rejected.
    token: usually not needed -- omit it. Only required if this server
    wasn't started with a default identity configured."""
    return _dispatch(
        tool_name="database.query",
        required_scope="db:read",
        raw_args={"token": token, "sql": sql},
        input_model=DatabaseQueryInput,
        output_model=DatabaseQueryOutput,
        call_impl=lambda inp: database.query(inp.sql),
    )


@mcp.tool()
def search_web(token: str, query: str, max_results: int = 5) -> dict:
    """Search the web (or an offline fixture if no SEARCH_API_KEY is configured)."""
    return _dispatch(
        tool_name="search.web",
        required_scope="search:read",
        raw_args={"token": token, "query": query, "max_results": max_results},
        input_model=SearchWebInput,
        output_model=SearchWebOutput,
        call_impl=lambda inp: {"results": search.web(inp.query, inp.max_results)},
    )


# --------------------------------------------------------------------------
# Discovery resource -- reports schema version explicitly, since a tool's
# JSON-schema alone (from list_tools) doesn't say which contract *version*
# it belongs to.
# --------------------------------------------------------------------------

@mcp.resource("server://info")
def server_info() -> str:
    import json

    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "tools": {
                "calendar.list_events": {"required_scope": "calendar:read"},
                "calendar.create_event": {"required_scope": "calendar:write"},
                "database.query": {"required_scope": "db:read"},
                "search.web": {"required_scope": "search:read"},
            },
        }
    )


def _bootstrap() -> None:
    calendar.init_db()
    database.init_db()

from mcp.server.transport_security import TransportSecuritySettings
def main() -> None:
    _bootstrap()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_PORT", "8000"))
        # streamable_http_path defaults to "/mcp" -- kept implicit here since
        # every client below (tests, demo.py) builds its URL as f"{base}/mcp".
        public_host = os.environ.get(
            "MCP_PUBLIC_HOST",
            "binding-grams-cities-legitimate.trycloudflare.com",
        )

        security = TransportSecuritySettings(
            allowed_hosts=[
                public_host,
                f"{public_host}:443",
            ],
        )

        mcp.run(transport="streamable-http", host=host, port=port, transport_security=security)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()


#npx @modelcontextprotocol/inspector@latest