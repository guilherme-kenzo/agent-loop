#!/usr/bin/env python3
"""Example: Agent using tools from an MCP server.

Connects to the minimal MCP server (mcp_server.py) over stdio,
discovers its tools, and runs an agent that calls the 'hello' tool.

Uses a mock model so no API key is needed.

Requirements:
    pip install agent-loop[mcp]

Run:
    python examples/mcp_agent.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.client.stdio import StdioServerParameters

from agent_loop import Agent, MCPConnection, ModelAdaptor, ModelResponse
from agent_loop.execution import ToolCall


class MockModel(ModelAdaptor):
    """Model that calls the 'hello' tool once, then gives a final response."""

    def __init__(self):
        self.call_count = 0

    async def call(self, messages, tools, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return ModelResponse(
                type="tool_call",
                content="I'll greet the user.",
                tool_call=ToolCall(
                    id="call_1",
                    tool_name="hello",
                    arguments={"name": "World"},
                ),
            )
        # Use the tool result from the previous message
        tool_result = messages[-1].content
        return ModelResponse(
            type="final_response",
            content=f"The server said: {tool_result}",
        )


async def main():
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    async with MCPConnection(server_params) as mcp_tools:
        print(f"Discovered {len(mcp_tools)} MCP tool(s): {[t.name for t in mcp_tools]}")

        agent = Agent(model=MockModel(), tools=mcp_tools)
        execution = await agent.run_async("Say hello to World")

        print(f"State: {execution.state}")
        print(f"Tool calls: {len(execution.tool_calls)}")
        for tc in execution.tool_calls:
            print(f"  {tc.tool_name}({tc.arguments}) -> {tc.result}")
        print(f"Response: {execution.response}")


if __name__ == "__main__":
    asyncio.run(main())
