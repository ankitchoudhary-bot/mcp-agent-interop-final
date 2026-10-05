"""
LLM-driven planning loop. Discovers the provider's tools at runtime via
MCPClientWrapper and lets Claude decide which ones to call and in what
order -- the `tools=[...]` list handed to the Anthropic API is built
entirely from what was discovered at connect() time, never hand-written.

Failure-mode strategy (mirrors the Phase 2 brief's own three buckets):

  1. Auth missing/invalid        -> degrade (tell the user what couldn't
  2. Tool no longer exists       -> degrade  be done; a retry or a
  4. Response fails validation   -> degrade  differently-worded prompt
                                              won't fix any of these)
  3. Timeout                     -> retry with backoff, THEN degrade if
                                     retries are exhausted
  5. Server schema newer than    -> escalate (checked once at connect(),
     client understands             before any task runs -- this is a
                                     whole-session problem, not a
                                     per-call one, and nothing about
                                     continuing is safe)

Every other well-formed rejection (e.g. an unsafe SQL query) is neither
degraded nor escalated -- it's handed back to Claude as a tool_result with
is_error=True so the planner itself can see the rejection and decide
whether to try a different approach.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass, field

import anthropic

from consumer.mcp_client import (
    MCPClientWrapper,
    DiscoveredTool,
    AuthFailure,
    ToolNotFoundError,
    ToolTimeoutError,
    ResponseValidationError,
    SchemaVersionMismatch,
    ConsumerError,
)

MODEL = os.environ.get("CONSUMER_MODEL", "claude-sonnet-5")
MAX_TURNS = 8
MAX_RETRIES_ON_TIMEOUT = 2
RETRY_BACKOFF_BASE_SECONDS = 0.5


@dataclass
class AgentRunResult:
    final_text: str
    turns_used: int
    tool_calls: list[dict] = field(default_factory=list)
    degraded: bool = False
    escalated_error: str | None = None


def _strip_token_from_schema(schema: dict) -> dict:
    """
    Every provider tool requires a 'token' argument (see
    shared/schemas.py) so auth works uniformly across transports. The
    planner should never see or invent a credential -- we show Claude a
    copy of the schema with 'token' removed, and inject the real token
    ourselves right before dispatching the call (_execute_tool_call).
    """
    schema = copy.deepcopy(schema) if schema else {"type": "object", "properties": {}}
    properties = schema.get("properties", {})
    properties.pop("token", None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r != "token"]
    return schema


def _translate_tools_for_claude(tools: list[DiscoveredTool]) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": _strip_token_from_schema(t.input_schema),
        }
        for t in tools
    ]


class ConsumerAgent:
    def __init__(
        self,
        mcp_client: MCPClientWrapper,
        auth_token: str,
        model: str = MODEL,
        anthropic_client: anthropic.Anthropic | None = None,
    ):
        self._mcp = mcp_client
        self._token = auth_token
        self._model = model
        # If a client is injected (e.g. consumer.fake_planner.FakeAnthropicClient
        # for a free/offline demo), use it as-is. Otherwise the real
        # anthropic.Anthropic() is constructed lazily on first use in
        # _client(), so tests that never call run() never need
        # ANTHROPIC_API_KEY set at all.
        self._anthropic: anthropic.Anthropic | None = anthropic_client
        self._claude_tools: list[dict] = []

    # def _client(self) -> anthropic.Anthropic:
    #     if self._anthropic is None:
    #         self._anthropic = anthropic.Anthropic()
    #     return self._anthropic
    
    def _client(self) -> anthropic.Anthropic:
        if self._anthropic is None:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

            if not openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set. Set it in the environment before running the consumer.")

            self._anthropic = anthropic.Anthropic(api_key=openrouter_api_key, base_url="https://openrouter.ai/api",)

        return self._anthropic

    async def start(self) -> None:
        """Connect to the provider and discover its tools. Call once before run()."""
        await self._mcp.connect()
        self._claude_tools = _translate_tools_for_claude(self._mcp.list_discovered_tools())

    async def close(self) -> None:
        await self._mcp.close()

    async def run(self, task: str) -> AgentRunResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        tool_calls_log: list[dict] = []

        for turn in range(MAX_TURNS):
            response = self._client().messages.create(
                model=self._model,
                max_tokens=1024,
                tools=self._claude_tools,
                messages=messages,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                final_text = "".join(b.text for b in response.content if b.type == "text")
                return AgentRunResult(final_text=final_text, turns_used=turn + 1, tool_calls=tool_calls_log)

            messages.append({"role": "assistant", "content": response.content})

            tool_result_blocks = []
            for block in tool_use_blocks:
                started = time.monotonic()
                outcome = await self._execute_tool_call(block.name, block.input)
                latency_ms = (time.monotonic() - started) * 1000
                tool_calls_log.append(
                    {
                        "tool": block.name,
                        "input": block.input,
                        "outcome": outcome["log_outcome"],
                        "latency_ms": latency_ms,
                    }
                )

                if outcome["degrade"]:
                    return AgentRunResult(
                        final_text=outcome["message"],
                        turns_used=turn + 1,
                        tool_calls=tool_calls_log,
                        degraded=True,
                    )
                if outcome["escalate"]:
                    return AgentRunResult(
                        final_text="",
                        turns_used=turn + 1,
                        tool_calls=tool_calls_log,
                        escalated_error=outcome["message"],
                    )

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": outcome["content_for_claude"],
                        "is_error": outcome["is_error"],
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        return AgentRunResult(
            final_text="Reached max planning turns without a final answer.",
            turns_used=MAX_TURNS,
            tool_calls=tool_calls_log,
            escalated_error="max_turns_exceeded",
        )

    async def _execute_tool_call(self, name: str, claude_args: dict) -> dict:
        """
        One tool call, with each of the five brief-mandated failure modes
        on its own explicit branch. Returns a dict the caller (run())
        interprets uniformly: {content_for_claude, is_error, degrade,
        escalate, message, log_outcome}.
        """
        args = dict(claude_args)
        args["token"] = self._token  # injected here, never seen by Claude

        last_timeout_error: ToolTimeoutError | None = None
        for attempt in range(MAX_RETRIES_ON_TIMEOUT + 1):
            try:
                result = await self._mcp.call_tool(name, args)
                return {
                    "content_for_claude": json.dumps(result),
                    "is_error": False,
                    "degrade": False,
                    "escalate": False,
                    "message": "",
                    "log_outcome": "success",
                }

            # --- failure mode 3: timeout -> retry with backoff ---
            except ToolTimeoutError as e:
                last_timeout_error = e
                if attempt < MAX_RETRIES_ON_TIMEOUT:
                    await asyncio.sleep(RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt))
                    continue
                break  # retries exhausted -> degrade, below the loop

            # --- failure mode 1: auth missing/invalid -> degrade ---
            except AuthFailure as e:
                return {
                    "content_for_claude": f"Auth error: {e}",
                    "is_error": True,
                    "degrade": True,
                    "escalate": False,
                    "message": (
                        f"I couldn't do this because I don't have valid access "
                        f"({e.code}): {e}"
                    ),
                    "log_outcome": f"degraded:{e.code}",
                }

            # --- failure mode 2: tool no longer exists -> degrade ---
            except ToolNotFoundError as e:
                return {
                    "content_for_claude": f"Tool unavailable: {e}",
                    "is_error": True,
                    "degrade": True,
                    "escalate": False,
                    "message": f"I couldn't do this because the '{name}' capability is no longer available: {e}",
                    "log_outcome": "degraded:tool_not_found",
                }

            # --- failure mode 4: response fails schema validation -> degrade ---
            except ResponseValidationError as e:
                return {
                    "content_for_claude": f"Invalid response: {e}",
                    "is_error": True,
                    "degrade": True,
                    "escalate": False,
                    "message": (
                        f"I couldn't do this because '{name}' returned a response "
                        f"I couldn't safely trust: {e}"
                    ),
                    "log_outcome": "degraded:validation_error",
                }

            # --- failure mode 5: server schema newer than client -> escalate ---
            except SchemaVersionMismatch as e:
                return {
                    "content_for_claude": f"Schema mismatch: {e}",
                    "is_error": True,
                    "degrade": False,
                    "escalate": True,
                    "message": f"Escalating -- this session cannot safely continue: {e}",
                    "log_outcome": "escalated:schema_version_mismatch",
                }

            # --- any other well-formed rejection: let Claude see and react ---
            except ConsumerError as e:
                return {
                    "content_for_claude": str(e),
                    "is_error": True,
                    "degrade": False,
                    "escalate": False,
                    "message": "",
                    "log_outcome": "rejected",
                }

        return {
            "content_for_claude": f"Timed out: {last_timeout_error}",
            "is_error": True,
            "degrade": True,
            "escalate": False,
            "message": (
                f"I couldn't do this because '{name}' kept timing out after "
                f"{MAX_RETRIES_ON_TIMEOUT + 1} attempts: {last_timeout_error}"
            ),
            "log_outcome": "degraded:timeout_exhausted",
        }



# $env:OPENROUTER_API_KEY = ""                                          
# (.venv) PS C:\Users\91966\Downloads\mcp-agent-interop> $env:CONSUMER_MODEL = "anthropic/claude-haiku-4.5"