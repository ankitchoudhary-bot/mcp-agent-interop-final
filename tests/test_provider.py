"""
Unit tests for each provider tool in isolation, plus integration tests that
start the real MCP server as a subprocess and exercise it over stdio.

Run with:
    pytest tests/test_provider.py -v

NOTE: the integration tests at the bottom depend on the exact shape of
CallToolResult / ReadResourceResult from the installed `mcp` package
version. If these fail to even parse a result (as opposed to failing on
content), check `python -c "import mcp; print(mcp.__version__)"` and adjust
the `_tool_json` / `_resource_json` helpers below -- the pipeline logic
itself (in provider/server.py's _dispatch) does not change per SDK version.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from provider import auth
from provider.audit_log import AuditLogger
from provider.tools import calendar, database, search


# --------------------------------------------------------------------------
# Unit tests: provider/auth.py
# --------------------------------------------------------------------------

def test_authenticate_missing_token():
    with pytest.raises(auth.AuthError) as exc_info:
        auth.authenticate(None)
    assert exc_info.value.code == "AUTH_MISSING"


def test_authenticate_invalid_token():
    with pytest.raises(auth.AuthError) as exc_info:
        auth.authenticate("not-a-real-token")
    assert exc_info.value.code == "AUTH_INVALID"


def test_authenticate_valid_token():
    identity = auth.authenticate("tok-consumer-full")
    assert identity.name == "consumer-agent"
    assert "calendar:read" in identity.scopes


def test_authorize_denies_missing_scope():
    identity = auth.authenticate("tok-search-only")
    with pytest.raises(auth.AuthError) as exc_info:
        auth.authorize(identity, "calendar:write")
    assert exc_info.value.code == "AUTH_SCOPE_DENIED"


def test_authorize_allows_granted_scope():
    identity = auth.authenticate("tok-consumer-full")
    auth.authorize(identity, "calendar:write")  # should not raise


# --------------------------------------------------------------------------
# Unit tests: provider/audit_log.py
# --------------------------------------------------------------------------

def test_audit_log_redacts_token(tmp_path: Path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.log_call(
        caller="tester",
        tool_name="calendar.list_events",
        arguments={"token": "tok-consumer-full", "start_date": "2026-09-01"},
        outcome="success",
        latency_ms=1.0,
    )
    entries = logger.read_all()
    assert len(entries) == 1
    assert entries[0]["arguments"]["token"].startswith("<redacted:")
    assert "tok-consumer-full" not in json.dumps(entries[0])


def test_audit_log_records_rejected_call(tmp_path: Path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.log_call(
        caller="unknown",
        tool_name="database.query",
        arguments={"token": "bad", "sql": "DROP TABLE products"},
        outcome="rejected",
        error_code="UNSAFE_QUERY",
        error_message="Only SELECT statements are allowed.",
    )
    entries = logger.read_all()
    assert entries[0]["outcome"] == "rejected"
    assert entries[0]["error_code"] == "UNSAFE_QUERY"


# --------------------------------------------------------------------------
# Unit tests: provider/tools/calendar.py
# --------------------------------------------------------------------------

@pytest.fixture()
def calendar_db(tmp_path: Path):
    db_path = tmp_path / "calendar.db"
    calendar.init_db(db_path)
    return db_path


def test_calendar_create_and_list(calendar_db):
    created = calendar.create_event(
        "Sync", "2026-09-26T14:00:00", "2026-09-26T15:00:00", db_path=calendar_db
    )
    assert created["title"] == "Sync"
    assert created["id"] is not None

    events = calendar.list_events("2026-09-26", "2026-09-26", db_path=calendar_db)
    assert len(events) == 1
    assert events[0]["title"] == "Sync"


def test_calendar_list_excludes_out_of_range(calendar_db):
    calendar.create_event("Sync", "2026-09-26T14:00:00", "2026-09-26T15:00:00", db_path=calendar_db)
    events = calendar.list_events("2026-10-01", "2026-10-02", db_path=calendar_db)
    assert events == []


def test_calendar_create_rejects_end_before_start(calendar_db):
    with pytest.raises(ValueError):
        calendar.create_event(
            "Bad", "2026-09-26T15:00:00", "2026-09-26T14:00:00", db_path=calendar_db
        )


# --------------------------------------------------------------------------
# Unit tests: provider/tools/database.py
# --------------------------------------------------------------------------

@pytest.fixture()
def catalog_db(tmp_path: Path):
    db_path = tmp_path / "catalog.db"
    database.init_db(db_path)
    return db_path


def test_database_select_works(catalog_db):
    result = database.query("SELECT name FROM products WHERE category = 'electronics'", db_path=catalog_db)
    assert result["row_count"] == 2
    assert result["columns"] == ["name"]


@pytest.mark.parametrize(
    "bad_sql",
    [
        "INSERT INTO products (id, name, category, price) VALUES (99,'x','y',1)",
        "DROP TABLE products",
        "DELETE FROM products",
        "SELECT * FROM products; DROP TABLE products",
        "UPDATE products SET price = 0",
    ],
)
def test_database_rejects_writes(catalog_db, bad_sql):
    with pytest.raises(database.UnsafeQueryError):
        database.query(bad_sql, db_path=catalog_db)


def test_database_readonly_at_connection_level(catalog_db):
    """Even bypassing _validate_readonly_sql, the connection itself can't write."""
    import sqlite3

    conn = sqlite3.connect(f"file:{catalog_db}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO products (id, name, category, price) VALUES (999,'x','y',1)")
        conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Unit tests: provider/tools/search.py
# --------------------------------------------------------------------------

def test_search_fixture_returns_relevant_result():
    results = search.web("what is MCP?")
    assert any("Model Context Protocol" in r["title"] for r in results)


def test_search_fixture_respects_max_results():
    results = search.web("mcp", max_results=1)
    assert len(results) == 1


def test_search_fixture_default_for_unknown_query():
    results = search.web("completely unrelated gibberish query")
    assert len(results) >= 1


# ==========================================================================
# Integration tests: real server process, both transports
# ==========================================================================
# These require `mcp` to be installed and are skipped with a clear reason
# if it isn't, so unit tests above still run in a minimal environment.

mcp_sdk = pytest.importorskip("mcp", reason="mcp SDK not installed; skipping integration tests")

import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from mcp import Client, StdioServerParameters  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRANSPORTS = ["stdio", "streamable-http"]


def _first_text(content_list) -> str:
    for item in content_list:
        text = getattr(item, "text", None)
        if text is not None:
            return text
    raise AssertionError(f"No text content found in {content_list!r}")


async def _call_tool(client: Client, name: str, arguments: dict) -> dict:
    result = await client.call_tool(name, arguments)
    # v2's Client populates .structured_content directly from a tool's typed
    # return value when available; fall back to parsing the text content
    # block otherwise (matches consumer/mcp_client.py's _parse_tool_result).
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    return json.loads(_first_text(result.content))


async def _read_resource(client: Client, uri: str) -> dict:
    result = await client.read_resource(uri)
    return json.loads(_first_text(result.contents))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Provider server process exited early with code {proc.returncode}.")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Provider server did not start listening on {host}:{port} within {timeout_seconds}s.")


@asynccontextmanager
async def _connected_client(transport: str):
    """
    Opens a real connection to a fresh instance of the real provider server,
    entirely within the calling test's own coroutine -- deliberately NOT a
    shared `@pytest_asyncio.fixture` that yields across the test boundary.
    An earlier version used exactly that pattern and every test failed at
    teardown with "Attempted to exit cancel scope in a different task than
    it was entered in": `mcp`'s Client (and the lower-level transports it
    wraps) owns an anyio task group internally, which strictly requires
    __aenter__/__aexit__ to run in the same asyncio Task, and pytest-asyncio
    ran this fixture's teardown in a different Task than its setup. Opening
    and closing the client inline, in one `async with` per test, keeps
    setup and teardown in the same Task and avoids the bug entirely -- the
    same fix already applied in consumer/mcp_client.py's MCPClientWrapper.
    """
    if transport == "stdio":
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "provider.server"], cwd=str(PROJECT_ROOT)
        )
        async with Client(params) as client:
            yield client
        return

    # streamable-http: a real separate process, reachable only by URL.
    port = _find_free_port()
    env = os.environ.copy()
    env["MCP_TRANSPORT"] = "streamable-http"
    env["MCP_HOST"] = "127.0.0.1"
    env["MCP_PORT"] = str(port)
    proc = subprocess.Popen([sys.executable, "-m", "provider.server"], cwd=str(PROJECT_ROOT), env=env)
    try:
        _wait_for_port("127.0.0.1", port, proc)
        url = f"http://127.0.0.1:{port}/mcp"
        async with Client(url) as client:
            yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_discovery_reports_schema_version(transport):
    async with _connected_client(transport) as client:
        info = await _read_resource(client, "server://info")
        assert info["schema_version"] == "v1"
        assert "calendar.list_events" in info["tools"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_successful_calendar_round_trip(transport):
    async with _connected_client(transport) as client:
        created = await _call_tool(
            client,
            "calendar_create_event",
            {
                "token": "tok-consumer-full",
                "title": "Integration Test Sync",
                "start": "2026-09-26T14:00:00",
                "end": "2026-09-26T15:00:00",
            },
        )
        assert created["ok"] is True
        assert created["event"]["title"] == "Integration Test Sync"

        listed = await _call_tool(
            client,
            "calendar_list_events",
            {"token": "tok-consumer-full", "start_date": "2026-09-26", "end_date": "2026-09-26"},
        )
        assert listed["ok"] is True
        assert any(e["title"] == "Integration Test Sync" for e in listed["events"])


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_rejected_call_bad_token(transport):
    async with _connected_client(transport) as client:
        result = await _call_tool(
            client,
            "calendar_list_events",
            {"token": "not-a-real-token", "start_date": "2026-09-01", "end_date": "2026-09-30"},
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "AUTH_INVALID"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_rejected_call_scope_denied(transport):
    # tok-search-only has no calendar:write scope
    async with _connected_client(transport) as client:
        result = await _call_tool(
            client,
            "calendar_create_event",
            {
                "token": "tok-search-only",
                "title": "Should Fail",
                "start": "2026-09-26T14:00:00",
                "end": "2026-09-26T15:00:00",
            },
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "AUTH_SCOPE_DENIED"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_rejected_unsafe_query(transport):
    async with _connected_client(transport) as client:
        result = await _call_tool(
            client,
            "database_query",
            {"token": "tok-consumer-full", "sql": "DROP TABLE products"},
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "UNSAFE_QUERY"
