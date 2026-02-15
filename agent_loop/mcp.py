"""MCP (Model Context Protocol) integration for agent-loop.

Wraps MCP server tools as native agent-loop Tool instances so they
integrate seamlessly with the existing architecture.

Requires: pip install agent-loop[mcp]
"""

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, create_model

from agent_loop.exceptions import ToolExecutionError
from agent_loop.tools import Tool

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client


_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

DEFAULT_TIMEOUT = 30.0


def _json_schema_to_python_type(prop_schema: dict, prop_name: str, parent_name: str) -> type:
    """Convert a single JSON Schema property to a Python type annotation.

    Handles: primitive types, typed arrays, nested objects, enums, and
    falls back to Any for unrecognized schemas ($ref, anyOf, oneOf, etc.).
    """
    # Enum: use Literal
    if "enum" in prop_schema:
        values = tuple(prop_schema["enum"])
        return Literal[values]  # type: ignore[valid-type]

    schema_type = prop_schema.get("type")

    if schema_type is None:
        # No type key — likely $ref, anyOf, oneOf, etc.
        return Any

    # Primitives
    if schema_type in _JSON_TYPE_MAP:
        return _JSON_TYPE_MAP[schema_type]

    # Typed arrays
    if schema_type == "array":
        items = prop_schema.get("items")
        if items and "type" in items and items["type"] in _JSON_TYPE_MAP:
            return list[_JSON_TYPE_MAP[items["type"]]]
        return list

    # Nested objects with properties → recursive Pydantic model
    if schema_type == "object":
        if "properties" in prop_schema:
            nested_name = f"{parent_name}_{prop_name}"
            return _schema_to_pydantic(nested_name, prop_schema)
        return dict

    return Any


def _schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
    """Convert a JSON Schema dict (from MCP inputSchema) to a Pydantic model.

    Handles primitive types, typed arrays (list[str] etc.), nested objects
    (recursive Pydantic models), enums (Literal), Optional typing for
    non-required fields, descriptions, and defaults. Falls back to Any for
    unrecognized schema constructs ($ref, anyOf, oneOf, etc.).
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        python_type = _json_schema_to_python_type(prop_schema, prop_name, tool_name)
        is_required = prop_name in required
        default = ... if is_required else prop_schema.get("default", None)
        description = prop_schema.get("description", "")

        if not is_required and default is None:
            python_type = Optional[python_type]

        fields[prop_name] = (python_type, Field(default=default, description=description))

    return create_model(f"{tool_name}_Input", **fields)


class MCPTool(Tool):
    """Wraps a single MCP server tool as a native agent-loop Tool."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        session: ClientSession,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.name = name
        self.description = description
        self.input_model = _schema_to_pydantic(name, input_schema)
        self._session = session
        self._timeout = timeout

    async def execute(self, **kwargs) -> str:
        """Call the MCP tool via the session and return the result as a string."""
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self.name, arguments=kwargs),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            raise ToolExecutionError(
                f"MCP tool '{self.name}' timed out after {self._timeout}s"
            )
        except Exception as e:
            raise ToolExecutionError(
                f"MCP tool '{self.name}' call failed: {e}"
            ) from e

        if result.isError:
            texts = [c.text for c in result.content if hasattr(c, "text")]
            raise ToolExecutionError(
                f"MCP tool '{self.name}' returned error: {' '.join(texts)}"
            )

        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        non_text_count = sum(1 for c in result.content if not hasattr(c, "text"))

        if non_text_count > 0:
            text_parts.append(f"[{non_text_count} non-text content block(s) omitted]")

        if len(text_parts) == 1:
            return text_parts[0]
        if text_parts:
            return json.dumps(text_parts)
        return ""


class MCPConnection:
    """Manages the lifecycle of one MCP server connection.

    Supports two transports:
    - stdio: MCPConnection(StdioServerParameters(command="python", args=["server.py"]))
    - streamable HTTP: MCPConnection("http://localhost:8000/mcp")

    Args:
        server_params: StdioServerParameters for stdio, or a URL string for HTTP.
        timeout: Timeout in seconds for MCP operations (initialize, list_tools,
            and individual tool calls). Defaults to 30s.

    Usage as async context manager:
        async with MCPConnection(server_params) as tools:
            agent = Agent(model=model, tools=tools)
            result = await agent.run_async("query")
    """

    def __init__(
        self,
        server_params: Union[StdioServerParameters, str],
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._server_params = server_params
        self._timeout = timeout
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> list[Tool]:
        """Open transport, initialize session, and return wrapped tools."""
        if self._exit_stack is not None:
            await self.disconnect()

        self._exit_stack = AsyncExitStack()
        try:
            if isinstance(self._server_params, str):
                transport = await self._exit_stack.enter_async_context(
                    streamablehttp_client(self._server_params)
                )
            else:
                transport = await self._exit_stack.enter_async_context(
                    stdio_client(self._server_params)
                )

            read_stream, write_stream, *_ = transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await asyncio.wait_for(
                self._session.initialize(), timeout=self._timeout
            )

            tools_result = await asyncio.wait_for(
                self._session.list_tools(), timeout=self._timeout
            )
            return [
                MCPTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema,
                    session=self._session,
                    timeout=self._timeout,
                )
                for t in tools_result.tools
            ]
        except BaseException:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Clean up transport and session resources."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    async def __aenter__(self) -> list[Tool]:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()
