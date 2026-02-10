# agent-loop

A minimal Python library for building multi-provider AI agents with tools.

## Install

```bash
pip install git+https://github.com/guilherme-kenzo/agent-loop.git
```

With a specific provider:

```bash
pip install "agent-loop[anthropic] @ git+https://github.com/guilherme-kenzo/agent-loop.git"
pip install "agent-loop[gemini] @ git+https://github.com/guilherme-kenzo/agent-loop.git"
pip install "agent-loop[ollama] @ git+https://github.com/guilherme-kenzo/agent-loop.git"
pip install "agent-loop[all] @ git+https://github.com/guilherme-kenzo/agent-loop.git"
```

## Quick start

```python
from agent_loop import Agent, Tool, ToolInput, OpenAIAdaptor
from pydantic import Field


class CalculatorInput(ToolInput):
    expression: str = Field(description="A math expression")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate math expressions"
    input_model = CalculatorInput

    async def execute(self, expression: str) -> str:
        return str(eval(expression))


model = OpenAIAdaptor(api_key="sk-...", model="gpt-5-mini")
agent = Agent(model=model, tools=[CalculatorTool()])

execution = agent.run("What is 25 * 17?")
print(execution.response)
```

## Providers

| Provider | Adaptor | Extra | Env var |
|----------|---------|-------|---------|
| OpenAI (and compatible) | `OpenAIAdaptor` | _(none)_ | `OPENAI_API_KEY` |
| Anthropic | `AnthropicAdaptor` | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `GeminiAdaptor` | `gemini` | `GOOGLE_API_KEY` |
| Ollama | `OllamaAdaptor` | `ollama` | _(none)_ |

## Tools

Tools are instances with Pydantic input validation:

```python
class SearchInput(ToolInput):
    query: str = Field(description="Search query")


class SearchTool(Tool):
    name = "search"
    description = "Search the web"
    input_model = SearchInput

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def execute(self, query: str) -> str:
        # your implementation
        return "results"
```

## Hooks

Hooks let you tap into the agent lifecycle for logging, caching, retries, monitoring or anything else you might need. Register them as decorators on the agent or through a `HookRegistry`:

```python
@agent.hook("after_tool_call")
async def on_tool_call(event):
    print(f"Tool '{event.tool_name}' returned in {event.execution_time_ms:.0f}ms")

@agent.hook("on_tool_error")
async def on_error(event):
    print(f"Error in '{event.tool_name}': {event.error_message}")
```

For "Stateful hook logic", use the `Middleware` base class:

```python
from agent_loop import Middleware

class MetricsMiddleware(Middleware):
    def __init__(self):
        self.tool_calls = 0
        self.total_tool_time_ms = 0.0

    async def after_tool_call(self, event):
        self.tool_calls += 1
        self.total_tool_time_ms += event.execution_time_ms

    async def after_run(self, event):
        print(f"{self.tool_calls} tool calls, {self.total_tool_time_ms:.0f}ms total")

metrics = MetricsMiddleware()
agent = Agent(model=model, tools=tools, middlewares=[metrics])
```

Available hooks:

| Hook | Fires when |
|------|------------|
| `before_run` | Agent starts processing input |
| `after_run` | Agent finishes (includes total time) |
| `before_iteration` | Each loop iteration begins |
| `after_iteration` | Each loop iteration ends |
| `before_model_call` | About to call the LLM |
| `after_model_call` | LLM response received |
| `before_tool_call` | About to execute a tool |
| `after_tool_call` | Tool execution completed |
| `on_tool_error` | Tool execution failed |

## Development

Requires [just](https://github.com/casey/just) and Docker.

```bash
just build               # build the container
just test                # run tests
just example mock_agent  # run an example
```
