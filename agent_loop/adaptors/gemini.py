"""Google Gemini API adaptor for agent-loop."""

import json
import os
from typing import Optional

from google import genai
from google.genai import types

from agent_loop.execution import Message, ToolCall
from agent_loop.model import ModelAdaptor, ModelResponse
from agent_loop.tools import Tool


class GeminiAdaptor(ModelAdaptor):
    """Google Gemini model adaptor using the official google-genai SDK.

    Args:
        api_key: Google AI API key. Falls back to GOOGLE_API_KEY environment variable.
        model: Model name (default: gemini-2.5-flash).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. "
                "Pass api_key argument or set GOOGLE_API_KEY environment variable."
            )

        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    async def call(
        self,
        messages: list[Message],
        tools: list[Tool],
        **kwargs,
    ) -> ModelResponse:
        contents = self._convert_messages(messages)
        gemini_tools = [self._convert_tools(tools)] if tools else []

        config = types.GenerateContentConfig(
            tools=gemini_tools or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return self._parse_response(response)

    def _convert_messages(self, messages: list[Message]) -> list[types.Content]:
        contents = []
        for msg in messages:
            if msg.role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.content)],
                ))
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append(types.Part(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        part_kwargs = {
                            "function_call": types.FunctionCall(
                                name=tc.tool_name,
                                args=tc.arguments,
                            ),
                        }
                        if getattr(tc, "thought_signature", None) is not None:
                            part_kwargs["thought_signature"] = tc.thought_signature
                        parts.append(types.Part(**part_kwargs))
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool":
                # Find tool name from tool_call_id by looking back through messages
                tool_name = self._find_tool_name(messages, msg.tool_call_id)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": msg.content},
                        ),
                    )],
                ))
        return contents

    def _find_tool_name(self, messages: list[Message], tool_call_id: str) -> str:
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id == tool_call_id:
                        return tc.tool_name
        return "unknown"

    def _convert_tools(self, tools: list[Tool]) -> types.Tool:
        declarations = []
        for tool in tools:
            schema = self._clean_schema(tool.schema())
            declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=schema,
            ))
        return types.Tool(function_declarations=declarations)

    def _clean_schema(self, schema: dict, is_properties: bool = False) -> dict:
        """Remove JSON Schema fields that Gemini doesn't support.

        Pydantic generates fields like additionalProperties, anyOf, $defs,
        title, default, etc. that the Gemini API rejects. Only a subset of
        OpenAPI 3.0 is supported: type, description, enum, items, properties,
        required, nullable.
        """
        if not isinstance(schema, dict):
            return schema

        SUPPORTED_KEYS = {
            "type", "description", "enum", "items",
            "properties", "required", "nullable",
        }

        cleaned = {}
        for key, value in schema.items():
            # Inside "properties", keys are user-defined property names — keep all
            if is_properties:
                cleaned[key] = self._clean_schema(value) if isinstance(value, dict) else value
                continue

            # anyOf: [{"type": "X"}, {"type": "null"}] → type X + nullable
            if key == "anyOf":
                non_null = [s for s in value if s.get("type") != "null"]
                has_null = any(s.get("type") == "null" for s in value)
                if len(non_null) == 1:
                    resolved = self._clean_schema(non_null[0])
                    if has_null:
                        resolved["nullable"] = True
                    cleaned.update(resolved)
                continue

            if key not in SUPPORTED_KEYS:
                continue

            if key == "properties" and isinstance(value, dict):
                cleaned[key] = self._clean_schema(value, is_properties=True)
            elif isinstance(value, dict):
                cleaned[key] = self._clean_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._clean_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value

        return cleaned

    def _extract_thought_signature(self, response) -> bytes | None:
        """Extract thought_signature from the first function call part."""
        try:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call and getattr(part, "thought_signature", None):
                        return part.thought_signature
        except (AttributeError, TypeError):
            pass
        return None

    def _parse_response(self, response) -> ModelResponse:
        if response.function_calls:
            fc = response.function_calls[0]
            thought_sig = self._extract_thought_signature(response)
            tool_call = ToolCall(
                id=fc.id if hasattr(fc, "id") and fc.id else f"call_{fc.name}",
                tool_name=fc.name,
                arguments=dict(fc.args) if fc.args else {},
                thought_signature=thought_sig,
            )
            return ModelResponse(
                type="tool_call",
                content=response.text or f"Calling {tool_call.tool_name}" if response.text else f"Calling {tool_call.tool_name}",
                tool_call=tool_call,
            )

        return ModelResponse(
            type="final_response",
            content=response.text or "",
        )
