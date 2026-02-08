#!/usr/bin/env python3
"""Minimal working example of agent-loop with OpenAI API.

This example demonstrates a complete agentic loop with:
- Calculator tool (math expressions)
- Web search tool (simulated)
- OpenAI API integration
- Execution tracking and result printing

Requirements:
- OPENAI_API_KEY environment variable set
- agent-loop installed

Run:
    python examples/minimal_agent.py
"""

import os
import sys
from typing import Optional

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import Agent, OpenAIAdaptor
from tools import CalculatorTool, WebSearchTool


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
            print(f"     ID: {tool_call.id[:8]}...")
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


def main() -> Optional[int]:
    """Run the minimal agent example.

    Returns:
        0 on success, 1 on error
    """
    print_header("🐢 Agent-Loop Minimal Example")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print(
            "\nTo run this example, set your OpenAI API key:\n"
            "    export OPENAI_API_KEY='sk-...'\n"
            "    python examples/minimal_agent.py\n"
        )
        print("Or use examples/mock_agent.py to see a demonstration without an API key.")
        return 1

    print("Found OPENAI_API_KEY")

    try:
        # Initialize the model adaptor
        model = OpenAIAdaptor(api_key=api_key, model="gpt-5-mini")

        print("\nCreating tools...")
        tools = [
            CalculatorTool(),
            WebSearchTool(),
        ]
        print(f"Created {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        # Create agent
        print("\nInitializing agent...")
        agent = Agent(model=model, tools=tools, max_iterations=10)
        print("Agent ready (max 10 iterations)")

        # Run a query that requires tool use
        query = (
            "What is 25 + 17? And what's the weather in São Paulo? "
            "Then calculate 100 * 2."
        )

        print_header("Running Agent")
        print(f"Query: {query}\n")
        print("Processing (this may take a moment)...\n")

        # Execute the agent
        execution = agent.run(query)

        # Print the trace
        print_execution_trace(execution)

        if execution.state == "completed":
            print("✅ Execution completed successfully!")
            return 0
        else:
            print(f"⚠️  Execution ended with state: {execution.state}")
            return 1

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
