# Agent-Loop Examples

This directory contains minimal working examples demonstrating agent-loop in action.

## Quick Start

### Option 1: Mock Example (No API Key Required) ⭐

Perfect for learning and testing without costs:

```bash
python examples/mock_agent.py
```

This example:
- ✅ Works without OpenAI API key
- ✅ Shows how the agent loop executes
- ✅ Uses predefined responses to demonstrate flow
- ✅ Displays tool calls and results

**Output includes:**
- Execution trace (input, iterations, messages)
- Tool calls made (calculator, web search)
- Final response
- Message history

### Option 2: Real OpenAI API Example

If you have an OpenAI API key:

```bash
export OPENAI_API_KEY='sk-...'
python examples/minimal_agent.py
```

This example:
- ✅ Uses real OpenAI API (gpt-4-mini)
- ✅ Demonstrates actual multi-tool agent loop
- ✅ Shows dynamic model responses
- ✅ Handles tool validation and execution

**Requirements:**
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- Internet connection

## Example Query

Both examples run the same query:

```
"What is 25 + 17? And what's the weather in São Paulo? Then calculate 100 * 2."
```

This query demonstrates:
1. **Multi-turn reasoning** - Complex query requiring multiple steps
2. **Tool usage** - Calculator and web search tools
3. **Sequential execution** - Tools called in order with results fed back to model
4. **Response generation** - Final answer synthesized from all results

## File Structure

```
examples/
├── README.md              # This file
├── tools.py              # Shared tool definitions
├── minimal_agent.py      # Real OpenAI API example
└── mock_agent.py         # Mocked example (no API key)
```

## Tools Included

### Calculator Tool

Evaluates mathematical expressions.

```python
# Usage by the agent
Tool: calculator
Input: {"expression": "25 + 17"}
Output: {"expression": "25 + 17", "result": 42}
```

Features:
- Safe expression evaluation (no access to dangerous functions)
- Supports: `+`, `-`, `*`, `/`, `**` (power), `//` (floor divide), `%` (modulo)
- Built-in functions: `abs()`, `round()`, `max()`, `min()`
- Error handling for invalid expressions

### Web Search Tool

Performs simulated web searches (returns mock results).

```python
# Usage by the agent
Tool: web_search
Input: {"query": "weather in São Paulo"}
Output: {
  "query": "weather in São Paulo",
  "results": [
    {
      "title": "Weather in São Paulo",
      "url": "https://weather.example.com/sp",
      "snippet": "São Paulo weather: Partly cloudy, 28°C, humid"
    }
  ],
  "count": 1
}
```

Features:
- Simulated search results (returns mocks for demonstration)
- Easy to replace with real API (e.g., Google Search, Brave Search)
- Returns structured JSON results
- In production, would call actual search API

## Understanding the Output

### Execution Trace Components

1. **Input** - The user's original query
2. **Summary** - State, iterations count, message count, tool call count
3. **Tool Calls** - Each tool invoked with arguments and results
4. **Final Response** - Model's synthesized answer
5. **Message History** - Full conversation flow

Example trace:

```
📌 Tool Calls
────────────────────────────────────────────────────────────────

  1. calculator
     ID: call-001
     Arguments: {'expression': '25 + 17'}
     Result: {"expression": "25 + 17", "result": 42}

  2. web_search
     ID: call-002
     Arguments: {'query': 'weather in São Paulo'}
     Result: {"query": "weather in São Paulo", "results": [...], "count": 1}

  3. calculator
     ID: call-003
     Arguments: {'expression': '100 * 2'}
     Result: {"expression": "100 * 2", "result": 200}
```

## Modifying the Examples

### Change the Query

Edit the query string in either example:

```python
# In minimal_agent.py or mock_agent.py
query = "Your custom query here"
```

### Add New Tools

1. Create a new Tool class in `tools.py`:

```python
class MyToolInput(ToolInput):
    param: str = Field(..., description="Parameter description")

class MyTool(Tool):
    name = "my_tool"
    description = "What this tool does"
    input_model = MyToolInput
    
    async def execute(self, param: str) -> str:
        # Your implementation
        return json.dumps({"result": "..."})
```

2. Add to the tools list:

```python
tools = [
    CalculatorTool(),
    WebSearchTool(),
    MyTool(),  # Add here
]
```

### Use Different Model

In `minimal_agent.py`, change:

```python
# Current
model = OpenAIAdaptor(api_key=api_key, model="gpt-4-mini")

# Try other models
model = OpenAIAdaptor(api_key=api_key, model="gpt-4")        # More powerful
model = OpenAIAdaptor(api_key=api_key, model="gpt-3.5-turbo") # Faster, cheaper
```

### Use Custom API Endpoint

For local models or other providers:

```python
model = OpenAIAdaptor(
    api_key="sk-local",  # Can be any value for local models
    model="llama2",
    base_url="http://localhost:8000/v1"  # Your local server URL
)
```

Supported:
- OpenAI API (default)
- Groq API (OpenAI-compatible)
- Local models (ollama, vLLM, etc.)
- Any OpenAI-compatible endpoint

## Common Issues

### "OPENAI_API_KEY not set"

Use the mock example instead:

```bash
python examples/mock_agent.py
```

Or set your API key:

```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### "API rate limited"

The mock example won't hit rate limits:

```bash
python examples/mock_agent.py
```

### Model returns empty response

Some models may not recognize tool definitions. Try:
- Using `gpt-4-mini` or `gpt-4` (recommended)
- Adding more context to tool descriptions
- Checking API key validity

## Next Steps

After running these examples:

1. **Explore the code** - Check `agent_loop/` to understand the architecture
2. **Run tests** - `pytest tests/ -v` to see comprehensive test coverage
3. **Build your agent** - Create a custom agent with your own tools
4. **Add persistence** - Save/load execution results
5. **Add streaming** - Implement token-by-token response streaming

## Learn More

- **README.md** - Main project documentation
- **SPEC.md** - Detailed specification of architecture and components
- **agent_loop/** - Core library implementation
- **tests/** - Comprehensive test examples

## Questions?

The code is documented with docstrings. Check:
- Tool classes for how to implement custom tools
- `Agent.run()` for synchronous execution
- `Agent.run_async()` for asynchronous execution
- Exception handling in the core library

Happy agent-building! 🐢
