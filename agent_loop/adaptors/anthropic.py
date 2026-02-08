"""Anthropic API adaptor for agent-loop."""

import os
from typing import Optional

from anthropic import AsyncAnthropic

from agent_loop.execution import Message, ToolCall
from agent_loop.model import ModelAdaptor, ModelResponse
from agent_loop.tools import Tool


class AnthropicAdaptor(ModelAdaptor):
    """Anthropic model adaptor using the official SDK.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY environment variable.
        model: Model name (default: claude-sonnet-4-5-20250929).
        max_tokens: Maximum tokens in the response (default: 1024).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 1024,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Pass api_key argument or set ANTHROPIC_API_KEY environment variable."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def call(
        self,
        messages: list[Message],
        tools: list[Tool],
        **kwargs,
    ) -> ModelResponse:
        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = [self._convert_tool(tool) for tool in tools] if tools else []

        create_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": anthropic_messages,
        }
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = await self.client.messages.create(**create_kwargs)
        return self._parse_response(response)

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        anthropic_messages = []
        for msg in messages:
            if msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.tool_name,
                            "input": tc.arguments,
                        })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks or msg.content,
                })
            elif msg.role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })
        return anthropic_messages

    def _convert_tool(self, tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema(),
        }

    def _parse_response(self, response) -> ModelResponse:
        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use":
                    tool_call = ToolCall(
                        id=block.id,
                        tool_name=block.name,
                        arguments=block.input,
                    )
                    text = ""
                    for b in response.content:
                        if b.type == "text":
                            text = b.text
                            break
                    return ModelResponse(
                        type="tool_call",
                        content=text or f"Calling {tool_call.tool_name}",
                        tool_call=tool_call,
                    )

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        return ModelResponse(type="final_response", content=text)
