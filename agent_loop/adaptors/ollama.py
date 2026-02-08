"""Ollama adaptor for agent-loop."""

import json
from typing import Optional

from ollama import AsyncClient

from agent_loop.execution import Message, ToolCall
from agent_loop.model import ModelAdaptor, ModelResponse
from agent_loop.tools import Tool


class OllamaAdaptor(ModelAdaptor):
    """Ollama model adaptor using the official SDK.

    Args:
        model: Model name (default: llama3.1).
        host: Ollama server URL (default: None, SDK defaults to localhost:11434).
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: Optional[str] = None,
    ):
        self.model = model
        self.client = AsyncClient(host=host)

    async def call(
        self,
        messages: list[Message],
        tools: list[Tool],
        **kwargs,
    ) -> ModelResponse:
        ollama_messages = self._convert_messages(messages)
        ollama_tools = [self._convert_tool(tool) for tool in tools] if tools else None

        chat_kwargs = {
            "model": self.model,
            "messages": ollama_messages,
        }
        if ollama_tools:
            chat_kwargs["tools"] = ollama_tools

        response = await self.client.chat(**chat_kwargs)
        return self._parse_response(response)

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        ollama_messages = []
        for msg in messages:
            if msg.role == "user":
                ollama_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                ollama_msg = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    ollama_msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.tool_name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                ollama_messages.append(ollama_msg)
            elif msg.role == "tool":
                # Find tool name from tool_call_id by looking back through messages
                tool_name = self._find_tool_name(messages, msg.tool_call_id)
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "name": tool_name,
                })
        return ollama_messages

    def _find_tool_name(self, messages: list[Message], tool_call_id: str) -> str:
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id == tool_call_id:
                        return tc.tool_name
        return "unknown"

    def _convert_tool(self, tool: Tool) -> dict:
        schema = tool.schema()
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }

    def _parse_response(self, response) -> ModelResponse:
        message = response.message
        content = message.content or ""

        if message.tool_calls:
            tc = message.tool_calls[0]
            tool_call = ToolCall(
                id=f"call_{tc.function.name}",
                tool_name=tc.function.name,
                arguments=tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments),
            )
            return ModelResponse(
                type="tool_call",
                content=content or f"Calling {tool_call.tool_name}",
                tool_call=tool_call,
            )

        return ModelResponse(
            type="final_response",
            content=content,
        )
