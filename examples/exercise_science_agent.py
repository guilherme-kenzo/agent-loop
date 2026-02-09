#!/usr/bin/env python3
"""Exercise science research agent with real Brave Search and logging hooks.

This example demonstrates:
- A real API tool (Brave Web Search) using httpx
- Decorator-style hooks for lifecycle logging
- A system prompt via the messages parameter
- A domain-specific research agent persona

Requirements:
- OPENAI_API_KEY environment variable set
- BRAVE_API_KEY environment variable set (https://brave.com/search/api/)
- agent-loop installed

Run:
    export OPENAI_API_KEY="sk-..."
    export BRAVE_API_KEY="BSA..."
    python examples/exercise_science_agent.py
"""

import json
import os
import sys
import time

import httpx
from pydantic import Field

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import Agent, Message, OpenAIAdaptor, Tool, ToolInput


# ── Brave Search Tool ────────────────────────────────────────────────────────


class BraveSearchInput(ToolInput):
    """Input model for the Brave web search tool."""

    query: str = Field(
        ...,
        description="Search query to find information about",
    )


class BraveSearchTool(Tool):
    """Web search tool that calls the real Brave Search API."""

    name = "brave_search"
    description = (
        "Searches the web using Brave Search and returns relevant results "
        "with titles, URLs, and descriptions."
    )
    input_model = BraveSearchInput

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def execute(self, query: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": self.api_key},
                params={"q": query, "count": 5},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                }
            )

        return json.dumps({"query": query, "results": results})


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an exercise science research assistant. Your role is to help "
    "users find and synthesize evidence-based information about exercise "
    "physiology, training methodology, sports nutrition, and recovery. "
    "When answering questions, search for recent research and authoritative "
    "sources. Cite the URLs you find so the user can verify your claims. "
    "Be precise, cite your sources, and distinguish between well-established "
    "findings and emerging or preliminary research."
)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    # Check for required API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = os.environ.get("OPENAI_MODEL") or "gpt-5-mini"
    brave_key = os.environ.get("BRAVE_API_KEY") 

    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return 1
    if not brave_key:
        print("ERROR: BRAVE_API_KEY environment variable not set")
        return 1

    # Set up model and tools
    model = OpenAIAdaptor(api_key=openai_key, model=model, base_url=base_url)
    tools = [BraveSearchTool(api_key=brave_key)]
    agent = Agent(model=model, tools=tools, max_iterations=10)

    # ── Logging Hooks ─────────────────────────────────────────────────────

    run_start: float = 0.0

    @agent.hook("before_run")
    async def on_before_run(event):
        nonlocal run_start
        run_start = time.time()
        print(f"\n[hook] Starting agent with input: {event.input[:80]}...")

    @agent.hook("after_run")
    async def on_after_run(event):
        elapsed = (time.time() - run_start) * 1000
        print(f"[hook] Agent completed in {elapsed:.0f}ms")

    @agent.hook("before_model_call")
    async def on_before_model_call(event):
        print("[hook] Calling model...")

    @agent.hook("after_model_call")
    async def on_after_model_call(event):
        resp_type = event.model_response.type
        print(f"[hook] Model responded ({resp_type}) in {event.response_time_ms:.0f}ms")

    @agent.hook("before_tool_call")
    async def on_before_tool_call(event):
        print(f"[hook] Calling tool '{event.tool_name}' with {event.arguments}")

    @agent.hook("after_tool_call")
    async def on_after_tool_call(event):
        preview = event.result[:120] + "..." if len(event.result) > 120 else event.result
        print(
            f"[hook] Tool '{event.tool_name}' returned in "
            f"{event.execution_time_ms:.0f}ms: {preview}"
        )

    # ── Run ───────────────────────────────────────────────────────────────

    query = (
        "What does current research say about the optimal rest period "
        "between sets for muscle hypertrophy?"
    )

    print(f"Query: {query}")

    execution = agent.run(
        query,
        messages=[Message(role="system", content=SYSTEM_PROMPT)],
    )

    # Print results
    print(f"\n{'=' * 70}")
    print("RESPONSE")
    print(f"{'=' * 70}")
    print(execution.response)
    print(f"\nState: {execution.state} | Iterations: {execution.iterations} | "
          f"Tool calls: {len(execution.tool_calls)}")

    return 0 if execution.state == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
