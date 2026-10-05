"""
Interactive front door for the consumer agent: type a request in plain
English, the Claude planner decides which provider tools to call, and you
see each tool call plus the final answer.

    python -m consumer.cli                                  # interactive prompt
    python -m consumer.cli "book a meeting called Sync tomorrow at 3pm"
    python -m consumer.cli "which products cost under 30 dollars?"
    python -m consumer.cli --transport streamable-http      # provider as a separate process
    python -m consumer.cli --token tok-consumer-readonly    # try a token without calendar:write

Each request is independent -- the agent has no memory of earlier prompts
in the same session. Today's date is prepended automatically, because the
planner has no other way to know what "tomorrow" means.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date
from pathlib import Path

from mcp import StdioServerParameters

from consumer.agent import ConsumerAgent
from consumer.demo import HttpServerProcess
from consumer.mcp_client import ConsumerError, MCPClientWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _with_date_context(task: str) -> str:
    today = date.today()
    return f"Today's date is {today.isoformat()} ({today.strftime('%A')}). {task}"


async def _run_one(agent: ConsumerAgent, task: str) -> None:
    result = await agent.run(_with_date_context(task))

    for call in result.tool_calls:
        print(f"  [tool] {call['tool']}({call['input']})  -> {call['outcome']}")

    if result.escalated_error:
        print(f"\nCould not continue: {result.escalated_error}\n")
    elif result.degraded:
        print(f"\n{result.final_text}\n")
    else:
        print(f"\n{result.final_text}\n")


async def main(args: argparse.Namespace) -> int:
    planner_kwargs: dict = {}
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY is not set. In PowerShell:\n"
            '    $env:OPENROUTER_API_KEY = "sk-or-..."\n'
        )
        return 1

    http_server = None
    try:
        if args.transport == "streamable-http":
            http_server = HttpServerProcess(port=args.port)
            http_server.start()
            client = MCPClientWrapper(transport="streamable-http", url=http_server.url)
        else:
            client = MCPClientWrapper(
                transport="stdio",
                server_params=StdioServerParameters(
                    command=sys.executable, args=["-m", "provider.server"], cwd=str(PROJECT_ROOT)
                ),
            )

        agent = ConsumerAgent(mcp_client=client, auth_token=args.token, **planner_kwargs)
        try:
            await agent.start()
        except ConsumerError as e:
            print(f"Could not connect to the provider: {e}")
            return 1

        try:
            tools = ", ".join(t.name for t in client.list_discovered_tools())
            print(f"Connected. Tools discovered at runtime: {tools}\n")

            if args.task:
                await _run_one(agent, args.task)
            else:
                print("Type a request (or 'exit'). Examples:")
                print("  book a 30 minute meeting called Standup tomorrow at 10am")
                print("  what's on my calendar tomorrow?")
                print("  what is the most expensive product in the catalog?\n")
                while True:
                    try:
                        task = (await asyncio.to_thread(input, "> ")).strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if task.lower() in {"exit", "quit"}:
                        break
                    if task:
                        await _run_one(agent, task)
        finally:
            await agent.close()
        return 0
    finally:
        if http_server is not None:
            http_server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk to the MCP consumer agent in plain English.")
    parser.add_argument("task", nargs="?", help="Run this one request and exit (omit for an interactive prompt).")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765, help="Port for --transport streamable-http.")
    parser.add_argument(
        "--token",
        default="tok-consumer-full",
        help="Auth token to present (tok-consumer-full, tok-consumer-readonly, tok-search-only).",
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
