"""
Standing regression suite for the five failure modes named in the Phase 2
brief (Phase 4: "automate every failure case ... so it runs on every
change, not a one-off manual check").

Unlike tests/test_consumer.py's offline versions (a mocked MCP client, for
speed and to run without any server), every test here spawns the REAL
provider server and drives it through the REAL ConsumerAgent/MCPClientWrapper
stack. These are the tests that would actually catch a regression in how a
failure propagates through the genuine protocol layer -- test_consumer.py
only proves _execute_tool_call's own branching logic is correct in
isolation, which is necessary but not sufficient.

Parametrized over both transports (same pattern as tests/test_provider.py)
since nothing in ConsumerAgent's failure handling is meant to be
transport-specific, and this is cheap insurance that it stays that way.

Run with:
    pytest tests/test_failure_modes.py -v
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

mcp_sdk = pytest.importorskip("mcp", reason="mcp SDK not installed; skipping failure-mode regression suite")

from mcp import StdioServerParameters

from consumer.agent import ConsumerAgent
from consumer.mcp_client import MCPClientWrapper
import consumer.mcp_client as mcp_client_module
import shared.schemas as schemas_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSPORTS = ["stdio", "streamable-http"]


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
async def _mcp_client(transport: str, timeout_seconds: float = 5.0):
    """
    Yields an unconnected MCPClientWrapper, plus manages the real server
    process's lifecycle for the streamable-http case. Deliberately not a
    shared pytest yield-fixture -- see tests/test_provider.py's
    _connected_client docstring for why that triggers a real anyio/
    pytest-asyncio "cancel scope in a different task" bug.
    """
    if transport == "stdio":
        yield MCPClientWrapper(
            transport="stdio",
            server_params=StdioServerParameters(
                command=sys.executable, args=["-m", "provider.server"], cwd=str(PROJECT_ROOT)
            ),
            timeout_seconds=timeout_seconds,
        )
        return

    port = _find_free_port()
    env = os.environ.copy()
    env["MCP_TRANSPORT"] = "streamable-http"
    env["MCP_HOST"] = "127.0.0.1"
    env["MCP_PORT"] = str(port)
    proc = subprocess.Popen([sys.executable, "-m", "provider.server"], cwd=str(PROJECT_ROOT), env=env)
    try:
        _wait_for_port("127.0.0.1", port, proc)
        yield MCPClientWrapper(
            transport="streamable-http", url=f"http://127.0.0.1:{port}/mcp", timeout_seconds=timeout_seconds
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@asynccontextmanager
async def _connected_agent(transport: str, auth_token: str = "tok-consumer-full", timeout_seconds: float = 5.0):
    async with _mcp_client(transport, timeout_seconds=timeout_seconds) as client:
        agent = ConsumerAgent(mcp_client=client, auth_token=auth_token)
        await agent.start()
        try:
            yield agent
        finally:
            await agent.close()


# --------------------------------------------------------------------------
# Canary: proves the harness can tell success from failure at all. If this
# ever fails, don't trust the failure-mode tests below -- something more
# fundamental (server startup, auth, discovery) broke first.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_canary_successful_call_does_not_degrade_or_escalate(transport):
    async with _connected_agent(transport) as agent:
        outcome = await agent._execute_tool_call(
            "calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"}
        )
        assert outcome["log_outcome"] == "success"
        assert outcome["degrade"] is False
        assert outcome["escalate"] is False


# --------------------------------------------------------------------------
# Failure mode 1/5: auth missing/invalid -> degrade
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_auth_invalid_degrades(transport):
    async with _connected_agent(transport, auth_token="this-token-does-not-exist") as agent:
        outcome = await agent._execute_tool_call(
            "calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"}
        )
        assert outcome["degrade"] is True
        assert outcome["escalate"] is False
        assert outcome["log_outcome"] == "degraded:AUTH_INVALID"


# --------------------------------------------------------------------------
# Failure mode 2/5: tool no longer exists -> degrade
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_tool_not_found_degrades(transport):
    async with _connected_agent(transport) as agent:
        outcome = await agent._execute_tool_call("calendar_delete_event", {"id": 1})
        assert outcome["degrade"] is True
        assert outcome["escalate"] is False
        assert outcome["log_outcome"] == "degraded:tool_not_found"


# --------------------------------------------------------------------------
# Failure mode 3/5: timeout -> retry with backoff, then degrade
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_timeout_retries_then_degrades(transport, monkeypatch):
    import consumer.agent as agent_module

    monkeypatch.setattr(agent_module, "RETRY_BACKOFF_BASE_SECONDS", 0.01)  # keep the test fast
    async with _connected_agent(transport, timeout_seconds=0.00001) as agent:
        outcome = await agent._execute_tool_call(
            "calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"}
        )
        assert outcome["degrade"] is True
        assert outcome["escalate"] is False
        assert outcome["log_outcome"] == "degraded:timeout_exhausted"


# --------------------------------------------------------------------------
# Failure mode 4/5: response fails schema validation -> degrade
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_response_validation_failure_degrades(transport):
    from pydantic import BaseModel

    class ImpossibleOutput(BaseModel):
        this_field_will_never_be_in_a_real_response: str

    original_model = mcp_client_module._OUTPUT_MODELS["search_web"]
    mcp_client_module._OUTPUT_MODELS["search_web"] = ImpossibleOutput
    try:
        async with _connected_agent(transport) as agent:
            outcome = await agent._execute_tool_call("search_web", {"query": "mcp"})
            assert outcome["degrade"] is True
            assert outcome["escalate"] is False
            assert outcome["log_outcome"] == "degraded:validation_error"
    finally:
        mcp_client_module._OUTPUT_MODELS["search_web"] = original_model


# --------------------------------------------------------------------------
# Failure mode 5/5: server schema newer than client -> escalate at connect()
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_schema_version_mismatch_escalates_at_connect(transport):
    # connect() now cleans up its own exit stack internally on any failure
    # (including this one) before re-raising -- see the docstring on
    # MCPClientWrapper.connect() in consumer/mcp_client.py for why that
    # ownership boundary, not just "someone eventually calls close()",
    # turned out to matter. No manual close() needed here as a result.
    original_supported = schemas_module.SUPPORTED_MAJOR_VERSION
    schemas_module.SUPPORTED_MAJOR_VERSION = 0
    try:
        async with _mcp_client(transport) as client:
            with pytest.raises(mcp_client_module.SchemaVersionMismatch):
                await client.connect()
    finally:
        schemas_module.SUPPORTED_MAJOR_VERSION = original_supported
