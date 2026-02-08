#!/usr/bin/env python3
"""Minimal working example of agent-loop with mocked model responses.

This example demonstrates agent-loop WITHOUT requiring an OpenAI API key.
It uses a mock ModelAdaptor to show how the agent loop works and what
the execution trace looks like.

This is useful for:
- Testing without API costs
- Demonstrating agent-loop architecture
- Understanding the execution flow
- Running in offline environments

Requirements:
- agent-loop installed (no API key needed!)

Run:
    python examples/mock_agent.py
"""

import os
import sys
from typing import Optional

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import Agent, ModelAdaptor, ModelResponse, Tool, ToolCall
from tools import CalculatorTool, WebSearchTool


class MockModelAdaptor(ModelAdaptor):
    """Mock model adaptor that simulates responses without calling an API.

    This demonstrates how to implement a custom ModelAdaptor.
    Each call cycles through predefined responses.
    """

    def __init__(self):
        self.call_count = 0
        # Predefined responses simulating a multi-turn conversation
        self.responses = [
            # First response: calculator tool call
            ModelResponse(
                type="tool_call",
                content="I'll help you with those calculations and search.",
                tool_call=ToolCall(
                    id="call-001",
                    tool_name="calculator",
                    arguments={"expression": "25 + 17"},
                ),
            ),
            # Second response: web search tool call
            ModelResponse(
                type="tool_call",
                content="Now let me search for the weather information.",
                tool_call=ToolCall(
                    id="call-002",
                    tool_name="web_search",
                    arguments={"query": "weather in São Paulo"},
                ),
            ),
            # Third response: another calculator call
            ModelResponse(
                type="tool_call",
                content="And finally, let me calculate the multiplication.",
                tool_call=ToolCall(
                    id="call-003",
                    tool_name="calculator",
                    arguments={"expression": "100 * 2"},
                ),
            ),
            # Final response: summarize
            ModelResponse(
                type="final_response",
                content=(
                    "Based on my calculations:\n"
                    "- 25 + 17 = 42\n"
                    "- The weather in São Paulo is partly cloudy at 28°C\n"
                    "- 100 * 2 = 200\n\n"
                    "All calculations completed successfully!"
                ),
            ),
        ]

    async def call(self, messages: list, tools: list[Tool], **kwargs) -> ModelResponse:
        """Return the next predefined response.

        Args:
            messages: List of messages (ignored in mock)
            tools: List of available tools (ignored in mock)
            **kwargs: Additional arguments (ignored in mock)

        Returns:
            Next response in the sequence
        """
        if self.call_count >= len(self.responses):
            # Safety: return final response if we exceed predefined responses
            return ModelResponse(
                type="final_response",
                content="(Mock adaptor ran out of predefined responses)",
            )

        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def print_header(text: str, width: int = 70) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n📌 {title}")
    print("-" * 70)


def print_execution_trace(execution) -> None:
    """Pretty-print the execution trace."""
    print_header("Execution Trace")

    print_section("Input")
    print(f"  User Query: {execution.input}")

    print_section("Execution Summary")
    print(f"  State: {execution.state}")
    print(f"  Iterations: {execution.iterations}")
    print(f"  Messages: {len(execution.messages)}")
    print(f"  Tool Calls: {len(execution.tool_calls)}")

    if execution.tool_calls:
        print_section("Tool Calls")
        for i, tool_call in enumerate(execution.tool_calls, 1):
            print(f"\n  {i}. {tool_call.tool_name}")
            print(f"     ID: {tool_call.id}")
            print(f"     Arguments: {tool_call.arguments}")
            if tool_call.result:
                result_preview = tool_call.result[:100]
                if len(tool_call.result) > 100:
                    result_preview += "..."
                print(f"     Result: {result_preview}")
            if tool_call.error:
                print(f"     Error: {tool_call.error}")

    print_section("Final Response")
    if execution.response:
        print(execution.response)
    else:
        print("  (No response available)")

    print_section("Message History")
    for i, msg in enumerate(execution.messages, 1):
        role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}.get(
            msg.role, "❓"
        )
        content_preview = msg.content[:80]
        if len(msg.content) > 80:
            content_preview += "..."
        print(f"  {i}. {role_emoji} {msg.role.upper()}: {content_preview}")

    print("\n" + "=" * 70 + "\n")


def main() -> int:
    """Run the mock agent example.

    Returns:
        0 on success
    """
    print_header("🐢 Agent-Loop Minimal Example (Mocked)")
    print("This example uses mocked responses and requires NO API key!")

    try:
        # Create mock model adaptor
        print("\nInitializing mock model adaptor...")
        model = MockModelAdaptor()
        print("✓ Mock adaptor ready")

        # Create tools
        print("\nCreating tools...")
        tools = [
            CalculatorTool(),
            WebSearchTool(),
        ]
        print(f"✓ Created {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        # Create agent
        print("\nInitializing agent...")
        agent = Agent(model=model, tools=tools, max_iterations=10)
        print("✓ Agent ready (max 10 iterations)")

        # Run a query that requires tool use
        query = (
            "What is 25 + 17? And what's the weather in São Paulo? "
            "Then calculate 100 * 2."
        )

        print_header("Running Agent")
        print(f"Query: {query}\n")
        print("Processing (using mocked responses)...\n")

        # Execute the agent
        execution = agent.run(query)

        # Print the trace
        print_execution_trace(execution)

        if execution.state == "completed":
            print("✅ Mocked execution completed successfully!")
            print("\n💡 To run with real OpenAI API, use minimal_agent.py:")
            print("   export OPENAI_API_KEY='sk-...'")
            print("   python examples/minimal_agent.py")
            return 0
        else:
            print(f"⚠️  Execution ended with state: {execution.state}")
            return 1

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
