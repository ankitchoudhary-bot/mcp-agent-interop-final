"""
A deterministic stand-in for anthropic.Anthropic(), used ONLY so
consumer/demo.py's Part 1 (the multi-step run) can be exercised for free,
offline, with no API key.

This is a test double, not a real planner: it hardcodes the exact plan a
real Claude call would very plausibly make for the specific "find a free
hour tomorrow afternoon and book a meeting called Sync" task -- list events,
then create one. It does not reason, does not handle any other task, and is
NOT a substitute for actually exercising the real planning loop against the
real Anthropic API at some point before you consider Phase 2 fully proven
out end to end. What it DOES prove, for free: that ConsumerAgent.run()'s
message-threading, tool dispatch, and result round-tripping are wired up
correctly -- the parts that have nothing to do with which LLM is on the
other end of client.messages.create().

Drop-in usage:
    agent = ConsumerAgent(mcp_client, auth_token="...", anthropic_client=FakeAnthropicClient())
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace


class _FakeMessagesAPI:
    def __init__(self, outer: "FakeAnthropicClient"):
        self._outer = outer

    def create(self, *, model, max_tokens, tools, messages):  # noqa: ARG002 -- match real signature
        return self._outer._next_response()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessagesAPI(self)
        self._step = 0
        tomorrow = date.today() + timedelta(days=1)
        self._tomorrow = tomorrow.isoformat()

    def _next_response(self):
        self._step += 1

        if self._step == 1:
            block = SimpleNamespace(
                type="tool_use",
                id="fake_call_1",
                name="calendar_list_events",
                input={"start_date": self._tomorrow, "end_date": self._tomorrow},
            )
            return SimpleNamespace(content=[block])

        if self._step == 2:
            block = SimpleNamespace(
                type="tool_use",
                id="fake_call_2",
                name="calendar_create_event",
                input={
                    "title": "Sync",
                    "start": f"{self._tomorrow}T14:00:00",
                    "end": f"{self._tomorrow}T15:00:00",
                },
            )
            return SimpleNamespace(content=[block])

        text = SimpleNamespace(
            type="text",
            text=(
                "Booked 'Sync' for tomorrow, 2:00-3:00 PM -- the afternoon was "
                "completely free. [scripted stand-in planner: this answer did not "
                "come from a real Claude call]"
            ),
        )
        return SimpleNamespace(content=[text])
