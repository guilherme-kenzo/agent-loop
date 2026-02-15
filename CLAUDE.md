# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

agent-loop is a lightweight Python framework for building multi-provider AI agents with tool support. It uses an async-first architecture targeting Python 3.11+.

## Build & Development Commands

Development uses Docker via `just` commands:

```bash
just build                      # Build Docker image
just test                       # Run all tests in Docker container (pytest)
just example mock_agent         # Run a specific example (no API key needed)
just example minimal_agent      # Run example requiring OPENAI_API_KEY
```

Running tests locally (outside Docker):

```bash
uv run pytest                   # All 118 tests
uv run pytest tests/test_agent.py             # Single test file
uv run pytest tests/test_agent.py::test_name  # Single test
```

pytest is configured with `asyncio_mode = "auto"` in pyproject.toml, so async tests don't need explicit marks.

## Architecture

### Core Loop (`agent_loop/agent.py`)

The `Agent` class orchestrates the agentic loop:
1. Takes user input, sends messages to an LLM via a `ModelAdaptor`
2. If the LLM returns a `tool_call`, validates input via Pydantic, executes the tool, appends result as a message, and loops
3. If the LLM returns a `final_response`, the loop ends
4. Loops up to `max_iterations` (default 20). Tool execution retries up to 3 times with hook-controlled behavior.

### Adapter Pattern (`agent_loop/adaptors/`)

Each LLM provider has a `ModelAdaptor` subclass that converts between agent-loop's internal `Message`/`Tool` formats and the provider's API format:
- `OpenAIAdaptor` — uses raw httpx (default, supports OpenAI-compatible endpoints)
- `AnthropicAdaptor` — uses `anthropic` SDK (optional extra)
- `GeminiAdaptor` — uses `google-genai` SDK (optional extra)
- `OllamaAdaptor` — uses `ollama` SDK (optional extra, local models)

SDK-based adaptors are conditionally imported; missing dependencies won't break the package.

### Tool System (`agent_loop/tools.py`)

Tools are **instances** (not classes) passed to the Agent. Each tool defines:
- `name`, `description` — for LLM tool selection
- `input_model` — a Pydantic `ToolInput` subclass for validation and JSON schema generation
- `async execute(**kwargs) -> str` — always async

### Hook System (`agent_loop/hooks.py`)

9 lifecycle hooks (before_run, after_run, before/after_iteration, before/after_model_call, before/after_tool_call, on_tool_error) allow non-invasive extensibility. Hooks can influence execution via `HookResponse` actions: skip, retry, abort, cache, modify arguments. The `Middleware` base class provides a stateful alternative to function-based hooks.

### Data Structures (`agent_loop/execution.py`)

`Execution` captures the full trace: all messages, tool calls, iteration count, and final state. `Message` uses roles: "user", "assistant", "tool". `ModelResponse.type` is either `"final_response"` or `"tool_call"`.

### Exceptions (`agent_loop/exceptions.py`)

Hierarchy under `AgentLoopError`: `ContextSizeExceeded`, `MaxIterationsReached`, `ToolValidationError`, `ToolNotFound`, `ToolExecutionError`.

## Key Design Decisions

- **Async-first**: All I/O is async; the sync `Agent.run()` wraps `asyncio.run()` around `run_async()`
- **Character-based context limits** (not token-based) via `max_context_size`
- **One tool call per iteration** (parallel tool execution planned for v0.2)
- **Optional dependencies**: Provider SDKs installed via extras (`pip install agent-loop[anthropic]`, `[all]`, etc.)
