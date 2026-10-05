"""
Phase 2/3 checkpoint script.

    python -m consumer.demo                                   # stdio, needs ANTHROPIC_API_KEY
    python -m consumer.demo --failures-only                    # just the 5 failure modes
    python -m consumer.demo --transport streamable-http         # Phase 3: real separate process over HTTP
    python -m consumer.demo --transport streamable-http --port 8765

Part 1 runs one real end-to-end multi-step task through the full planning
loop (Claude decides to call calendar_list_events, then calendar_create_event).

Part 2 deliberately triggers each of the five failure modes from the brief.
None of these are printed/faked: each one is a real exception raised by the
real code path.

--transport streamable-http (Phase 3) spawns `python -m provider.server` as
a genuinely separate OS process listening on a TCP port, and the consumer
only ever talks to it via a URL -- no shared stdio pipe, no parent/child
relationship beyond "one process happened to start another." This is the
same MCPClientWrapper and ConsumerAgent code as the stdio path; only the
transport argument passed to MCPClientWrapper differs.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from mcp import StdioServerParameters

from consumer.agent import ConsumerAgent
from consumer.mcp_client import MCPClientWrapper
import consumer.mcp_client as mcp_client_module
import shared.schemas as schemas_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Transport plumbing
# --------------------------------------------------------------------------

class HttpServerProcess:
    """
    Starts `python -m provider.server` in streamable-http mode as a real,
    separate OS process -- not just a different code path -- and blocks
    until it's actually accepting TCP connections before returning control.
    This is what makes --transport streamable-http a genuine interoperability
    test: the provider and consumer no longer share a stdio pipe or a
    parent/child relationship beyond "one process happened to start
    another"; they only know about each other via a URL, exactly as they
    would across a real network boundary.
    """

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/mcp"
        self._proc: subprocess.Popen | None = None

    def start(self, timeout_seconds: float = 10.0) -> None:
        env = os.environ.copy()
        env["MCP_TRANSPORT"] = "streamable-http"
        env["MCP_HOST"] = self.host
        env["MCP_PORT"] = str(self.port)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "provider.server"],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"Provider server process exited early with code {self._proc.returncode}.")
            try:
                with socket.create_connection((self.host, self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError(
            f"Provider server did not start listening on {self.host}:{self.port} within {timeout_seconds}s.")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None


def _client_kwargs(transport: str, http_server: HttpServerProcess | None) -> dict:
    if transport == "stdio":
        return {
            "transport": "stdio",
            "server_params": StdioServerParameters(
                command=sys.executable,
                args=["-m", "provider.server"],
                cwd=str(PROJECT_ROOT),
            ),
        }
    assert http_server is not None
    return {"transport": "streamable-http", "url": http_server.url}


# --------------------------------------------------------------------------
# Part 1: one real successful multi-step run
# --------------------------------------------------------------------------

async def run_successful_multi_step_task(client_kwargs: dict, free_planner: bool = False) -> None:
    _print_header("PART 1: successful multi-step run (calendar_list_events -> calendar_create_event)")

    client = MCPClientWrapper(**client_kwargs)

    if free_planner:
        from consumer.fake_planner import FakeAnthropicClient

        print(
            "(--free-planner: using a scripted stand-in for Claude, not the real "
            "Anthropic API -- see consumer/fake_planner.py. This proves the "
            "orchestration/message-threading logic, not real LLM reasoning.)\n"
        )
        agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full", anthropic_client=FakeAnthropicClient())
    else:
        agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")

    await agent.start()

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    task = (
        f"Today's date is {today}. Find a free hour tomorrow ({tomorrow}) afternoon "
        f"and book a meeting called 'Sync'. Afternoon means between 12:00 and 18:00. "
        f"Check the calendar first before booking."
    )

    result = await agent.run(task)

    print(f"\nTask: {task}\n")
    for call in result.tool_calls:
        print(f"  tool call -> {call['tool']}({call['input']})  outcome={call['outcome']}")
    print(f"\nFinal answer ({result.turns_used} turn(s)):\n{result.final_text}")

    await agent.close()


# --------------------------------------------------------------------------
# Part 2: five deliberately-triggered failure modes
# --------------------------------------------------------------------------

async def demo_bad_auth(client_kwargs: dict) -> None:
    _print_header("FAILURE MODE 1/5: auth token invalid -> degrade")
    client = MCPClientWrapper(**client_kwargs)
    agent = ConsumerAgent(mcp_client=client, auth_token="this-token-does-not-exist")
    await agent.start()
    outcome = await agent._execute_tool_call("calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"})
    print(f"degrade={outcome['degrade']}  escalate={outcome['escalate']}")
    print(f"message shown to user: {outcome['message']}")
    await agent.close()


async def demo_tool_not_found(client_kwargs: dict) -> None:
    _print_header("FAILURE MODE 2/5: requested tool no longer exists -> degrade")
    client = MCPClientWrapper(**client_kwargs)
    agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")
    await agent.start()
    outcome = await agent._execute_tool_call("calendar_delete_event", {"id": 1})
    print(f"degrade={outcome['degrade']}  escalate={outcome['escalate']}")
    print(f"message shown to user: {outcome['message']}")
    await agent.close()


async def demo_timeout(client_kwargs: dict) -> None:
    _print_header("FAILURE MODE 3/5: tool call times out -> retry with backoff, then degrade")
    # A real, absurdly small timeout against the real server -- this is a
    # genuine asyncio.wait_for timeout, not a simulated one, over whichever
    # transport was requested.
    client = MCPClientWrapper(**client_kwargs, timeout_seconds=0.00001)
    agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")
    await agent.start()
    outcome = await agent._execute_tool_call("calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"})
    print(f"degrade={outcome['degrade']}  escalate={outcome['escalate']}")
    print(f"message shown to user: {outcome['message']}")
    await agent.close()


async def demo_response_validation_failure(client_kwargs: dict) -> None:
    _print_header("FAILURE MODE 4/5: response fails schema validation -> degrade")
    from pydantic import BaseModel

    class ImpossibleOutput(BaseModel):
        this_field_will_never_be_in_a_real_response: str

    original_model = mcp_client_module._OUTPUT_MODELS["search_web"]
    mcp_client_module._OUTPUT_MODELS["search_web"] = ImpossibleOutput
    try:
        client = MCPClientWrapper(**client_kwargs)
        agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")
        await agent.start()
        outcome = await agent._execute_tool_call("search_web", {"query": "mcp"})
        print(f"degrade={outcome['degrade']}  escalate={outcome['escalate']}")
        print(f"message shown to user: {outcome['message']}")
        await agent.close()
    finally:
        mcp_client_module._OUTPUT_MODELS["search_web"] = original_model


async def demo_schema_version_mismatch(client_kwargs: dict) -> None:
    _print_header("FAILURE MODE 5/5: server schema newer than client -> escalate")
    original_supported = schemas_module.SUPPORTED_MAJOR_VERSION
    schemas_module.SUPPORTED_MAJOR_VERSION = 0
    try:
        client = MCPClientWrapper(**client_kwargs)
        try:
            await client.connect()
            print("!!! did not raise -- this should not happen")
        except Exception as e:
            print(f"escalated at connect() time, before any task ran: {e}")
        finally:
            await client.close()
    finally:
        schemas_module.SUPPORTED_MAJOR_VERSION = original_supported


async def main(failures_only: bool, free_planner: bool, transport: str, port: int, ) -> None:
    http_server = None
    try:
        if transport == "streamable-http":
            http_server = HttpServerProcess(port=port)
            http_server.start()
            print(
                f"\n(spawned the provider as a separate OS process (pid {http_server._proc.pid}), "
                f"listening at {http_server.url} -- the consumer only knows this URL, "
                f"nothing else about the provider)"
            )

        client_kwargs = _client_kwargs(transport, http_server)

        if not failures_only:
            await run_successful_multi_step_task(client_kwargs, free_planner=free_planner)
        await demo_bad_auth(client_kwargs)
        await demo_tool_not_found(client_kwargs)
        await demo_timeout(client_kwargs)
        await demo_response_validation_failure(client_kwargs)
        await demo_schema_version_mismatch(client_kwargs)
    finally:
        if http_server is not None:
            http_server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument(
        "--free-planner",
        action="store_true",
        help="Run Part 1 against a scripted stand-in instead of the real Anthropic API (no API key, no reasoning).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport between consumer and provider. streamable-http spawns the provider as a separate OS process (Phase 3).",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for --transport streamable-http.")
    args = parser.parse_args()
    asyncio.run(
        main(
            failures_only=args.failures_only,
            free_planner=args.free_planner,
            transport=args.transport,
            port=args.port,
        )
    )
