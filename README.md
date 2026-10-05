# mcp-agent-interop

Two independent agents that talk only over the Model Context Protocol (MCP):
a **tool-provider** agent (calendar, read-only database, web search behind
an MCP server with auth + audit logging) and a **consumer** agent (an
LLM-driven planner that discovers those tools at runtime and calls them).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # or: . .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # only needed once the consumer agent (Phase 2) is wired up

python -c "import mcp, anthropic, pydantic; print('imports OK')"
```

## Running the provider server standalone

```bash
python -m provider.server                       # stdio transport (default)
MCP_TRANSPORT=streamable-http python -m provider.server   # Phase 3: HTTP, defaults to 127.0.0.1:8000/mcp
MCP_TRANSPORT=streamable-http MCP_PORT=9000 python -m provider.server
```

On first run it creates `provider/data/calendar.db` (empty) and
`provider/data/catalog.db` (seeded with 5 sample products), and appends to
`audit.log.jsonl` at the repo root on every tool call.

## Auth tokens (toy, in-memory -- see provider/auth.py)

| token                    | scopes                                              |
|--------------------------|------------------------------------------------------|
| `tok-consumer-full`      | calendar:read, calendar:write, db:read, search:read  |
| `tok-consumer-readonly`  | calendar:read, db:read, search:read                  |
| `tok-search-only`        | search:read                                          |

## Running tests

```bash
pytest tests/test_provider.py -v
```

Unit tests (auth, audit log, calendar, database, search) have no external
dependency beyond `pydantic`/`pytest` and should pass immediately. The
integration tests at the bottom of `test_provider.py` spin up the real
server over stdio and require `mcp` to be installed; they're auto-skipped
otherwise via `pytest.importorskip`.

## Inspecting the audit log

```bash
tail -f audit.log.jsonl | python -m json.tool --json-lines 2>/dev/null || tail -f audit.log.jsonl
```

## Running the Phase 2 consumer agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# full demo: one real successful multi-step run, then all 5 failure modes
python -m consumer.demo

# just the 5 failure modes -- no ANTHROPIC_API_KEY needed, since these call
# ConsumerAgent._execute_tool_call() directly rather than going through the
# Claude planning loop
python -m consumer.demo --failures-only
```

### Running Part 1 for free (no Anthropic API key)

The brief specifies the Anthropic SDK for the planner, so treat these as
development conveniences for iterating without spending money -- not a
substitute for eventually running the real thing at least once.

**Option A -- scripted stand-in, zero setup, no reasoning at all:**
```bash
python -m consumer.demo --free-planner
```
Uses `consumer/fake_planner.py`, a hardcoded two-step script. It proves
`ConsumerAgent.run()`'s message-threading and tool-dispatch logic works,
but it cannot handle a different task, a full calendar, or react to
anything -- it always does the same two tool calls regardless of input.

**Option B -- a real local LLM via Ollama, actually reasons, free:**
```bash
pip install ollama
ollama pull llama3.1      # or another tool-calling-capable model
ollama serve               # if not already running

python -m consumer.demo --ollama-model llama3.1
```
Uses `consumer/ollama_planner.py`, which translates between Anthropic's
and Ollama's tool-call formats so it drops into `ConsumerAgent` the same
way the real client does. Local open models are noticeably less reliable
at multi-step tool-use planning than Claude -- expect occasional wrong
tool choices or a model that doesn't recognize the task is finished.

```bash
pytest tests/test_consumer.py -v   # offline, no server/API key needed
pytest tests/ -v                   # everything (both transports, ~2x the integration tests)
```

## Phase 3: streamable HTTP transport

```bash
# spawns the provider as a real separate OS process, consumer only knows a URL
python -m consumer.demo --transport streamable-http
python -m consumer.demo --transport streamable-http --failures-only
python -m consumer.demo --transport streamable-http --port 9000
```

`tests/test_provider.py`'s integration tests are parametrized over both
`stdio` and `streamable-http` (`pytest tests/test_provider.py -v` runs each
one twice). See `ARCHITECTURE.md` for what does and doesn't change in
behavior between the two transports, particularly around timeouts.

## Phase 4: evaluation harness

```bash
pytest tests/test_failure_modes.py -v   # standing regression suite, all 5 failure modes x 2 transports

python -m consumer.eval_harness --runs 3                                    # real Claude planner
python -m consumer.eval_harness --runs 5 --free-planner                     # free, harness-plumbing only
python -m consumer.eval_harness --runs 3 --transport streamable-http --port 9000
```

`eval_harness.py` runs three sample tasks and all five failure cases N
times each, then reports tool-selection accuracy (did the planner call the
exact expected tool sequence for each task), failure-recovery rate (did
each failure mode produce the correct degrade/escalate outcome every time),
and average latency per tool call. Tool-selection accuracy is only a
meaningful number with the real Claude planner -- `--free-planner`'s
scripted stand-in ignores the task text entirely, so it isn't a real
accuracy measurement, only a check that the harness itself runs end to end.

The calendar-booking task resets `provider/data/calendar.db` before every
run so repeated runs don't accumulate bookings and change what "a free
afternoon" means run to run -- this bit us for real during Phase 2 (see
the transcript / ARCHITECTURE.md).

## Connecting to a real MCP client (e.g. Claude Desktop)

This project's own `consumer/` was built to demonstrate multiple tokens
and scopes, so every tool call passes an explicit `token`. A real MCP
host like Claude Desktop has no equivalent of that -- there's no code
there to inject a credential the way `consumer/agent.py` does. So `token`
is optional on every tool: if it's omitted and the server was started
with `PROVIDER_STATIC_TOKEN` set, that value is used automatically,
before validation ever happens. If it's omitted and no static token is
configured, it fails validation exactly like a missing required field
always has -- no behavior change for anything already in this project
(tests, `demo.py`, `eval_harness.py`, `cli.py` all still pass an explicit
token per call and are unaffected).

**Claude Desktop setup:**

1. Find `claude_desktop_config.json`:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

   (Claude Desktop's tray/menu-bar icon -- not the chat window --
   -> Settings -> Developer -> Edit Config opens it directly.)

2. Add this server, using the **absolute path to your venv's own
   Python** (Claude Desktop launches this directly, not through an
   activated shell):

```json
{
  "mcpServers": {
    "mcp-agent-interop": {
      "command": "C:\\path\\to\\mcp-agent-interop\\.venv\\Scripts\\python.exe",
      "args": ["-m", "provider.server"],
      "cwd": "C:\\path\\to\\mcp-agent-interop",
      "env": {
        "PROVIDER_STATIC_TOKEN": "tok-consumer-full"
      }
    }
  }
}
```

3. Fully quit Claude Desktop (not just close the window) and reopen it.
   Look for the hammer/tools icon in the chat input box.

Without the `PROVIDER_STATIC_TOKEN` line, this still works, but Claude
will see `token` as an available (optional) argument on every tool with
no way to know what value to use -- you'd need to tell it directly in
conversation (e.g. "use tok-consumer-full as the token").

## Using it interactively

Everything above runs fixed, pre-scripted tasks. To type your own
requests and see the agent handle them:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m consumer.cli                                              # interactive prompt
python -m consumer.cli "book a meeting called Sync tomorrow at 3pm"  # one-shot, no prompt
```

In the interactive prompt, just type things like:

```
> book a 30 minute meeting called Standup tomorrow at 10am
> what's on my calendar tomorrow?
> what's the cheapest item in the electronics category?
> search the web for what MCP is
> exit
```

The agent decides which tool(s) to call for whatever you type -- it isn't
limited to the exact sample tasks used elsewhere in this project. Each
tool call it makes is printed before the final answer, so you see the
plan, not just the result. Today's date is prepended to every request
automatically, since the planner has no other way to know what "tomorrow"
means.

Other flags:

```bash
python -m consumer.cli --transport streamable-http    # real separate process
python -m consumer.cli --ollama-model llama3.1          # free, local model, real reasoning
python -m consumer.cli --token tok-consumer-readonly    # try a restricted token, watch it degrade
```

## Continuous integration

`.github/workflows/tests.yml` runs the full `pytest tests/` suite
automatically on every push and pull request, on a 2x2 matrix of
Ubuntu/Windows x Python 3.11/3.12 -- no API key needed, since nothing in
`tests/` touches the real Anthropic API (see the workflow file's own
comments for why). This is Phase 4's "runs on every change" made literal:
push a change that breaks any of the five failure modes, on either
transport, and this fails automatically rather than waiting for someone
to remember to run it by hand.

`.github/workflows/eval-harness.yml` runs `consumer/eval_harness.py`
against the real Claude API -- tool-selection accuracy, failure-recovery
rate, latency -- but only when triggered manually from the Actions tab, or
automatically once a week (Mondays 06:00 UTC). It costs a small amount of
real money per run, which is why it isn't wired to every push the way
`tests.yml` is.

**One-time setup on GitHub's side, if you push this to a repo:** add your
key as a repository secret named `ANTHROPIC_API_KEY` (Settings -> Secrets
and variables -> Actions -> New repository secret) before the eval-harness
workflow will run successfully; `tests.yml` needs no secrets at all.

## Status

- [x] Phase 0: scaffold
- [x] Phase 1: tool-provider agent (server, tools, auth, audit log, tests)
- [x] Phase 2: consumer agent (planner + mcp_client, all 5 failure modes)
- [x] Phase 3: cross-process transport (streamable HTTP)
- [x] Phase 4: evaluation harness (standing regression suite + N-run report)
- [x] Phase 5: ARCHITECTURE.md write-up
