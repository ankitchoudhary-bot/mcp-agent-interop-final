"""
Phase 4 evaluation harness.

Runs a small set of sample tasks N times against the real provider server
and the real planner (unless --free-planner/--ollama-model is given), then
reports:

  - tool-selection accuracy: did the planner call the expected tool(s), in
    the expected order, for each sample task?
  - failure-recovery rate: for each of the five failure modes, did the
    agent produce the correct degrade/escalate outcome, every run?
  - average latency per tool call (wall-clock; includes the real Claude
    API round trip when using the real planner)

    python -m consumer.eval_harness --runs 3
    python -m consumer.eval_harness --runs 5 --free-planner
    python -m consumer.eval_harness --runs 3 --transport streamable-http

This is meant to be genuinely re-run on every change (the brief's own
"treat it as a first-class deliverable, not an afterthought"), not executed
once for a report and forgotten. Tool-selection accuracy is only a
meaningful signal with a real reasoning planner -- --free-planner's scripted
stand-in will trivially score 100% on the one task it's hardcoded for and
0%/undefined on the others, since it doesn't look at the task text at all.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from mcp import StdioServerParameters

from consumer.agent import ConsumerAgent
from consumer.mcp_client import MCPClientWrapper
from consumer.demo import HttpServerProcess
import consumer.mcp_client as mcp_client_module
import shared.schemas as schemas_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _reset_calendar_db() -> None:
    """
    Deletes the calendar's SQLite file so each run of a calendar-touching
    task starts from a clean slate. Without this, run 2 of N would find the
    afternoon partially booked by run 1 and legitimately behave differently
    -- exactly the kind of test-polluting-test bug this project already hit
    once for real (see the Phase 2 checkpoint transcript). Safe to call any
    time: provider/tools/calendar.py recreates the table on next use via
    init_db(), and no connection is held open between calls.
    """
    db_path = PROJECT_ROOT / "provider" / "data" / "calendar.db"
    db_path.unlink(missing_ok=True)


def _instrument_latency(client: MCPClientWrapper, latencies: list[float]) -> None:
    """Wraps client.call_tool to record wall-clock time per call, without
    touching agent.py's or mcp_client.py's public contract at all."""
    original_call_tool = client.call_tool

    async def timed_call_tool(name, arguments):
        started = time.perf_counter()
        try:
            return await original_call_tool(name, arguments)
        finally:
            latencies.append(time.perf_counter() - started)

    client.call_tool = timed_call_tool  # type: ignore[method-assign]


# --------------------------------------------------------------------------
# Sample tasks + expected tool sequences
# --------------------------------------------------------------------------

@dataclass
class SampleTask:
    name: str
    prompt_template: str  # may reference {today}/{tomorrow}
    expected_tools: list[str]  # exact expected call sequence


SAMPLE_TASKS = [
    SampleTask(
        name="book_free_afternoon_slot",
        prompt_template=(
            "Today's date is {today}. Find a free hour tomorrow ({tomorrow}) afternoon "
            "and book a meeting called 'Sync'. Afternoon means between 12:00 and 18:00. "
            "Check the calendar first before booking."
        ),
        expected_tools=["calendar_list_events", "calendar_create_event"],
    ),
    SampleTask(
        name="cheapest_electronics_product",
        prompt_template=(
            "Using the product catalog, what is the cheapest item in the 'electronics' "
            "category? Answer with just the product name and price."
        ),
        expected_tools=["database_query"],
    ),
    SampleTask(
        name="what_is_mcp",
        prompt_template="Search the web and tell me in one sentence what the Model Context Protocol (MCP) is.",
        expected_tools=["search_web"],
    ),
]

FAILURE_CASES = ["bad_auth", "tool_not_found", "timeout", "validation_error", "version_mismatch"]


@dataclass
class TaskRunResult:
    task_name: str
    run_index: int
    correct_tool_selection: bool
    actual_tools: list[str]
    degraded: bool
    escalated: str | None
    tool_latencies: list[float]


@dataclass
class FailureCaseRunResult:
    case_name: str
    run_index: int
    recovered_correctly: bool
    detail: str


async def _run_one_task(task: SampleTask, client_kwargs: dict, run_index: int, planner_kwargs: dict) -> TaskRunResult:
    _reset_calendar_db()
    client = MCPClientWrapper(**client_kwargs)
    latencies: list[float] = []
    agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full", **planner_kwargs)
    await agent.start()
    _instrument_latency(client, latencies)

    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    prompt = task.prompt_template.format(today=today, tomorrow=tomorrow)

    result = await agent.run(prompt)
    await agent.close()

    actual_tools = [c["tool"] for c in result.tool_calls]
    return TaskRunResult(
        task_name=task.name,
        run_index=run_index,
        correct_tool_selection=(actual_tools == task.expected_tools),
        actual_tools=actual_tools,
        degraded=result.degraded,
        escalated=result.escalated_error,
        tool_latencies=latencies,
    )


async def _run_one_failure_case(case: str, client_kwargs: dict, run_index: int) -> tuple[FailureCaseRunResult, list[float]]:
    if case == "bad_auth":
        client = MCPClientWrapper(**client_kwargs)
        agent = ConsumerAgent(mcp_client=client, auth_token="this-token-does-not-exist")
        await agent.start()
        latencies: list[float] = []
        _instrument_latency(client, latencies)
        outcome = await agent._execute_tool_call("calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"})
        await agent.close()
        correct = outcome["degrade"] is True and outcome["escalate"] is False
        return FailureCaseRunResult(case, run_index, correct, outcome["log_outcome"]), latencies

    if case == "tool_not_found":
        client = MCPClientWrapper(**client_kwargs)
        agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")
        await agent.start()
        latencies = []
        _instrument_latency(client, latencies)
        outcome = await agent._execute_tool_call("calendar_delete_event", {"id": 1})
        await agent.close()
        correct = outcome["degrade"] is True and outcome["escalate"] is False
        return FailureCaseRunResult(case, run_index, correct, outcome["log_outcome"]), latencies

    if case == "timeout":
        client = MCPClientWrapper(**client_kwargs, timeout_seconds=0.00001)
        agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")
        await agent.start()
        latencies = []
        _instrument_latency(client, latencies)
        outcome = await agent._execute_tool_call("calendar_list_events", {"start_date": "2026-09-26", "end_date": "2026-09-26"})
        await agent.close()
        correct = outcome["degrade"] is True and outcome["log_outcome"] == "degraded:timeout_exhausted"
        return FailureCaseRunResult(case, run_index, correct, outcome["log_outcome"]), latencies

    if case == "validation_error":
        from pydantic import BaseModel

        class ImpossibleOutput(BaseModel):
            this_field_will_never_be_in_a_real_response: str

        original = mcp_client_module._OUTPUT_MODELS["search_web"]
        mcp_client_module._OUTPUT_MODELS["search_web"] = ImpossibleOutput
        try:
            client = MCPClientWrapper(**client_kwargs)
            agent = ConsumerAgent(mcp_client=client, auth_token="tok-consumer-full")
            await agent.start()
            latencies = []
            _instrument_latency(client, latencies)
            outcome = await agent._execute_tool_call("search_web", {"query": "mcp"})
            await agent.close()
            correct = outcome["degrade"] is True and outcome["log_outcome"] == "degraded:validation_error"
            return FailureCaseRunResult(case, run_index, correct, outcome["log_outcome"]), latencies
        finally:
            mcp_client_module._OUTPUT_MODELS["search_web"] = original

    if case == "version_mismatch":
        original_supported = schemas_module.SUPPORTED_MAJOR_VERSION
        schemas_module.SUPPORTED_MAJOR_VERSION = 0
        try:
            client = MCPClientWrapper(**client_kwargs)
            started = time.perf_counter()
            try:
                await client.connect()
                detail = "did not raise -- BUG"
                correct = False
            except mcp_client_module.SchemaVersionMismatch as e:
                detail = f"escalated: {e}"
                correct = True
            latency = time.perf_counter() - started
            await client.close()
            return FailureCaseRunResult(case, run_index, correct, detail), [latency]
        finally:
            schemas_module.SUPPORTED_MAJOR_VERSION = original_supported

    raise ValueError(f"Unknown failure case: {case}")


def _print_report(task_results: list[TaskRunResult], failure_results: list[FailureCaseRunResult], all_latencies: list[float], runs: int) -> None:
    print("\n" + "=" * 78)
    print(f"EVALUATION REPORT ({runs} run(s) per task/case)")
    print("=" * 78)

    print("\nTool-selection accuracy (exact expected tool sequence called):")
    by_task: dict[str, list[TaskRunResult]] = {}
    for r in task_results:
        by_task.setdefault(r.task_name, []).append(r)
    for name, results in by_task.items():
        correct = sum(1 for r in results if r.correct_tool_selection)
        print(f"  {name}: {correct}/{len(results)} correct")
        for r in results:
            mark = "OK  " if r.correct_tool_selection else "MISS"
            note = f"  (degraded)" if r.degraded else (f"  (escalated: {r.escalated})" if r.escalated else "")
            print(f"    [{mark}] run {r.run_index}: called {r.actual_tools}{note}")

    print("\nFailure-recovery rate (matched expected degrade/escalate outcome):")
    by_case: dict[str, list[FailureCaseRunResult]] = {}
    for r in failure_results:
        by_case.setdefault(r.case_name, []).append(r)
    for name, results in by_case.items():
        correct = sum(1 for r in results if r.recovered_correctly)
        print(f"  {name}: {correct}/{len(results)} recovered correctly")
        for r in results:
            mark = "OK  " if r.recovered_correctly else "FAIL"
            print(f"    [{mark}] run {r.run_index}: {r.detail}")

    if all_latencies:
        mean_ms = statistics.mean(all_latencies) * 1000
        print(
            f"\nAverage latency per tool call: {mean_ms:.1f} ms "
            f"(n={len(all_latencies)}, min={min(all_latencies) * 1000:.1f}ms, "
            f"max={max(all_latencies) * 1000:.1f}ms)"
        )
    else:
        print("\nNo tool-call latency samples recorded.")


async def run_harness(runs: int, transport: str, port: int, free_planner: bool) -> None:
    http_server = None
    try:
        if transport == "streamable-http":
            http_server = HttpServerProcess(port=port)
            http_server.start()
            print(f"\n(provider running as a separate process at {http_server.url})")

        client_kwargs = (
            {
                "transport": "stdio",
                "server_params": StdioServerParameters(
                    command=sys.executable, args=["-m", "provider.server"], cwd=str(PROJECT_ROOT)
                ),
            }
            if transport == "stdio"
            else {"transport": "streamable-http", "url": http_server.url}
        )

        planner_kwargs: dict = {}
        if free_planner:
            from consumer.fake_planner import FakeAnthropicClient

            planner_kwargs = {"anthropic_client": FakeAnthropicClient()}
            print(
                "(planner: scripted stand-in -- only validates harness plumbing, "
                "not real tool-selection accuracy; it ignores task text entirely)"
            )

        task_results: list[TaskRunResult] = []
        for task in SAMPLE_TASKS:
            for i in range(runs):
                task_results.append(await _run_one_task(task, client_kwargs, i, planner_kwargs))

        all_latencies: list[float] = [lat for r in task_results for lat in r.tool_latencies]

        failure_results: list[FailureCaseRunResult] = []
        for case in FAILURE_CASES:
            for i in range(runs):
                result, lats = await _run_one_failure_case(case, client_kwargs, i)
                failure_results.append(result)
                all_latencies.extend(lats)

        _print_report(task_results, failure_results, all_latencies, runs)
    finally:
        if http_server is not None:
            http_server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--free-planner", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_harness(args.runs, args.transport, args.port, args.free_planner))
