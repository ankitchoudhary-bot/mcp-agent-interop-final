"""
Unit tests for the consumer agent's tool-translation logic and its five
failure-mode branches, using a fake MCPClientWrapper so these run fast and
offline -- no live provider server, no ANTHROPIC_API_KEY required (the
Anthropic client in ConsumerAgent is constructed lazily and is never
touched by _execute_tool_call).

Run with:
    pytest tests/test_consumer.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from consumer.agent import (
    ConsumerAgent,
    _strip_token_from_schema,
    _translate_tools_for_claude,
    MAX_RETRIES_ON_TIMEOUT,
)
from consumer import agent as agent_module
from consumer.mcp_client import (
    DiscoveredTool,
    AuthFailure,
    ToolNotFoundError,
    ToolTimeoutError,
    ResponseValidationError,
    SchemaVersionMismatch,
    ConsumerError,
)


# --------------------------------------------------------------------------
# Schema translation: token must never reach Claude
# --------------------------------------------------------------------------

def test_strip_token_from_schema_removes_property_and_required():
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string"}, "start_date": {"type": "string"}},
        "required": ["token", "start_date"],
    }
    stripped = _strip_token_from_schema(schema)
    assert "token" not in stripped["properties"]
    assert "token" not in stripped["required"]
    assert "start_date" in stripped["properties"]


def test_strip_token_from_schema_handles_empty_schema():
    assert _strip_token_from_schema({}) == {"type": "object", "properties": {}}


def test_translate_tools_for_claude_shape():
    tools = [
        DiscoveredTool(
            name="calendar_list_events",
            description="List events",
            input_schema={
                "type": "object",
                "properties": {"token": {"type": "string"}, "start_date": {"type": "string"}},
                "required": ["token", "start_date"],
            },
        )
    ]
    translated = _translate_tools_for_claude(tools)
    assert translated == [
        {
            "name": "calendar_list_events",
            "description": "List events",
            "input_schema": {
                "type": "object",
                "properties": {"start_date": {"type": "string"}},
                "required": ["start_date"],
            },
        }
    ]


# --------------------------------------------------------------------------
# _execute_tool_call: one branch per failure mode, driven by a fake client
# --------------------------------------------------------------------------

class FakeMCPClient:
    """Stands in for MCPClientWrapper. call_tool is an AsyncMock the test
    configures per-case with side_effect (a return value, an exception, or
    a list consumed one call at a time)."""

    def __init__(self):
        self.call_tool = AsyncMock()


@pytest.fixture()
def agent() -> ConsumerAgent:
    return ConsumerAgent(mcp_client=FakeMCPClient(), auth_token="tok-test")


@pytest.mark.asyncio
async def test_execute_tool_call_success(agent):
    agent._mcp.call_tool.return_value = {"events": [], "schema_version": "v1"}
    outcome = await agent._execute_tool_call("calendar_list_events", {"start_date": "2026-09-26"})
    assert outcome["log_outcome"] == "success"
    assert outcome["degrade"] is False
    assert outcome["escalate"] is False
    assert outcome["is_error"] is False


@pytest.mark.asyncio
async def test_execute_tool_call_degrades_on_auth_failure(agent):
    agent._mcp.call_tool.side_effect = AuthFailure("AUTH_INVALID", "bad token")
    outcome = await agent._execute_tool_call("calendar_list_events", {})
    assert outcome["degrade"] is True
    assert outcome["escalate"] is False
    assert "AUTH_INVALID" in outcome["message"]


@pytest.mark.asyncio
async def test_execute_tool_call_degrades_on_tool_not_found(agent):
    agent._mcp.call_tool.side_effect = ToolNotFoundError("gone")
    outcome = await agent._execute_tool_call("some_removed_tool", {})
    assert outcome["degrade"] is True
    assert outcome["escalate"] is False
    assert outcome["log_outcome"] == "degraded:tool_not_found"


@pytest.mark.asyncio
async def test_execute_tool_call_degrades_on_validation_error(agent):
    agent._mcp.call_tool.side_effect = ResponseValidationError("bad shape")
    outcome = await agent._execute_tool_call("search_web", {"query": "x"})
    assert outcome["degrade"] is True
    assert outcome["escalate"] is False
    assert outcome["log_outcome"] == "degraded:validation_error"


@pytest.mark.asyncio
async def test_execute_tool_call_escalates_on_schema_version_mismatch(agent):
    agent._mcp.call_tool.side_effect = SchemaVersionMismatch("server too new")
    outcome = await agent._execute_tool_call("calendar_list_events", {})
    assert outcome["escalate"] is True
    assert outcome["degrade"] is False
    assert outcome["log_outcome"] == "escalated:schema_version_mismatch"


@pytest.mark.asyncio
async def test_execute_tool_call_passes_other_rejections_to_claude(agent):
    agent._mcp.call_tool.side_effect = ConsumerError("[UNSAFE_QUERY] no writes allowed")
    outcome = await agent._execute_tool_call("database_query", {"sql": "DROP TABLE x"})
    assert outcome["degrade"] is False
    assert outcome["escalate"] is False
    assert outcome["is_error"] is True
    assert "UNSAFE_QUERY" in outcome["content_for_claude"]


@pytest.mark.asyncio
async def test_execute_tool_call_retries_then_succeeds_on_timeout(agent, monkeypatch):
    monkeypatch.setattr(agent_module, "RETRY_BACKOFF_BASE_SECONDS", 0.001)
    agent._mcp.call_tool.side_effect = [
        ToolTimeoutError("slow"),
        {"events": [], "schema_version": "v1"},
    ]
    outcome = await agent._execute_tool_call("calendar_list_events", {})
    assert outcome["log_outcome"] == "success"
    assert agent._mcp.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_execute_tool_call_degrades_after_timeout_retries_exhausted(agent, monkeypatch):
    monkeypatch.setattr(agent_module, "RETRY_BACKOFF_BASE_SECONDS", 0.001)
    agent._mcp.call_tool.side_effect = ToolTimeoutError("always slow")
    outcome = await agent._execute_tool_call("calendar_list_events", {})
    assert outcome["degrade"] is True
    assert outcome["log_outcome"] == "degraded:timeout_exhausted"
    assert agent._mcp.call_tool.await_count == MAX_RETRIES_ON_TIMEOUT + 1


@pytest.mark.asyncio
async def test_execute_tool_call_injects_token_not_visible_to_claude(agent):
    agent._mcp.call_tool.return_value = {"events": [], "schema_version": "v1"}
    await agent._execute_tool_call("calendar_list_events", {"start_date": "2026-09-26"})
    called_name, called_args = agent._mcp.call_tool.await_args.args
    assert called_args["token"] == "tok-test"
    assert called_args["start_date"] == "2026-09-26"


# --------------------------------------------------------------------------
# Full run() loop, end to end, entirely offline: a fake MCP client plus the
# scripted FakeAnthropicClient from consumer/fake_planner.py stand in for
# both external dependencies. This proves the multi-step orchestration
# (assistant tool_use -> user tool_result -> assistant tool_use -> ... ->
# final text) is wired correctly, independent of whether the real Claude
# API or the real provider server is reachable.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_run_end_to_end_with_fake_planner():
    from consumer.fake_planner import FakeAnthropicClient

    fake_mcp = FakeMCPClient()
    fake_mcp.call_tool.side_effect = [
        {"events": [], "schema_version": "v1"},
        {
            "event": {"id": 1, "title": "Sync", "start": "2026-09-26T14:00:00", "end": "2026-09-26T15:00:00"},
            "schema_version": "v1",
        },
    ]

    agent = ConsumerAgent(
        mcp_client=fake_mcp, auth_token="tok-test", anthropic_client=FakeAnthropicClient()
    )
    result = await agent.run("find a free hour tomorrow afternoon and book a meeting called Sync")

    assert result.degraded is False
    assert result.escalated_error is None
    assert [c["tool"] for c in result.tool_calls] == ["calendar_list_events", "calendar_create_event"]
    assert all(c["outcome"] == "success" for c in result.tool_calls)
    assert "Sync" in result.final_text
    assert fake_mcp.call_tool.await_count == 2
