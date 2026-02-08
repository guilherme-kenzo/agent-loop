"""OpenAI API adaptor for agent-loop."""

import json
import os
from typing import Optional

import httpx

from agent_loop.execution import ToolCall
from agent_loop.model import ModelAdaptor, ModelResponse
from agent_loop.tools import Tool


class OpenAIAdaptor(ModelAdaptor):
    """OpenAI-compatible model adaptor.

    Supports OpenAI API and compatible endpoints (local models, proxies, etc.).

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
        model: Model name (default: gpt-5-mini).
        base_url: Base URL for the API (default: https://api.openai.com/v1).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Pass api_key argument or set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"

    async def call(
        self,
        messages: list,
        tools: list[Tool],
        **kwargs,
    ) -> ModelResponse:
        """Call the OpenAI API with messages and available tools.

        Args:
            messages: List of Message objects from agent-loop.
            tools: List of Tool objects from agent-loop.
            **kwargs: Additional arguments passed to the API.

        Returns:
            ModelResponse with type, content, and optional tool_call.

        Raises:
            ValueError: If API response is malformed or unexpected.
            httpx.HTTPError: If the API request fails.
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Convert tools to OpenAI format
        openai_tools = [self._convert_tool(tool) for tool in tools] if tools else None

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": openai_messages,
        }

        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Call OpenAI API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=kwargs.get("timeout", 60.0),
            )

        # Handle errors
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            raise ValueError(f"OpenAI API error: {error_msg}")

        # Parse response
        data = response.json()
        return self._parse_response(data)

    def _convert_messages(self, messages: list) -> list[dict]:
        """Convert agent-loop Message objects to OpenAI format.

        Args:
            messages: List of agent-loop Message objects.

        Returns:
            List of OpenAI-format message dicts.
        """
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            
            # Handle tool messages - include tool_call_id
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            # Handle assistant messages with tool calls
            if msg.role == "assistant" and msg.tool_calls:
                openai_msg["tool_calls"] = self._format_tool_calls(msg.tool_calls)
            
            openai_messages.append(openai_msg)
        return openai_messages

    def _format_tool_calls(self, tool_calls: list) -> list[dict]:
        """Convert agent-loop ToolCall objects to OpenAI tool_calls format.

        Args:
            tool_calls: List of ToolCall objects.

        Returns:
            List of OpenAI-format tool_calls dicts.
        """
        formatted = []
        for tool_call in tool_calls:
            formatted.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            })
        return formatted

    def _convert_tool(self, tool: Tool) -> dict:
        """Convert an agent-loop Tool to OpenAI tool format.

        Args:
            tool: An agent-loop Tool instance.

        Returns:
            OpenAI tool definition dict.
        """
        schema = tool.schema()
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }

    def _parse_response(self, data: dict) -> ModelResponse:
        """Parse OpenAI API response into ModelResponse.

        Args:
            data: Raw OpenAI API response.

        Returns:
            ModelResponse with type, content, and optional tool_call.

        Raises:
            ValueError: If response format is unexpected.
        """
        # Extract the first choice (OpenAI returns a list of choices)
        if not data.get("choices"):
            raise ValueError("OpenAI response missing 'choices' field")

        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""

        # Check if there's a tool call
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            # Tool call case
            tool_call_data = tool_calls[0]  # Handle first tool call
            tool_call = ToolCall(
                id=tool_call_data["id"],
                tool_name=tool_call_data["function"]["name"],
                arguments=json.loads(tool_call_data["function"]["arguments"]),
            )
            return ModelResponse(
                type="tool_call",
                content=content or f"Calling {tool_call.tool_name}",
                tool_call=tool_call,
            )
        else:
            # Final response case
            return ModelResponse(
                type="final_response",
                content=content,
            )
