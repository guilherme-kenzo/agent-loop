import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from agent_loop.mcp import (
    DEFAULT_TIMEOUT,
    MCPConnection,
    MCPTool,
    _json_schema_to_python_type,
    _schema_to_pydantic,
)
from agent_loop.exceptions import ToolExecutionError


# --- _schema_to_pydantic tests ---


class TestSchemaToPydantic:
    def test_simple_string_field(self):
        schema = {
            "properties": {"name": {"type": "string", "description": "A name"}},
            "required": ["name"],
        }
        model = _schema_to_pydantic("test", schema)
        assert issubclass(model, BaseModel)
        instance = model(name="alice")
        assert instance.name == "alice"

    def test_multiple_types(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "age"],
        }
        model = _schema_to_pydantic("multi", schema)
        instance = model(name="bob", age=30)
        assert instance.name == "bob"
        assert instance.age == 30
        assert instance.score is None
        assert instance.active is None

    def test_array_and_object_types(self):
        schema = {
            "properties": {
                "tags": {"type": "array"},
                "metadata": {"type": "object"},
            },
            "required": ["tags"],
        }
        model = _schema_to_pydantic("complex", schema)
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]
        assert instance.metadata is None

    def test_optional_fields_with_defaults(self):
        schema = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }
        model = _schema_to_pydantic("defaults", schema)
        instance = model(query="test")
        assert instance.limit == 10

    def test_empty_schema(self):
        schema = {}
        model = _schema_to_pydantic("empty", schema)
        instance = model()
        assert isinstance(instance, BaseModel)

    def test_all_optional(self):
        schema = {
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "integer"},
            },
        }
        model = _schema_to_pydantic("allopt", schema)
        instance = model()
        assert instance.x is None
        assert instance.y is None

    def test_optional_fields_use_optional_type(self):
        """Optional fields with default=None should have Optional[T] annotation."""
        schema = {
            "properties": {
                "name": {"type": "string"},
            },
        }
        model = _schema_to_pydantic("opttype", schema)
        field = model.model_fields["name"]
        # Should accept None without validation error
        instance = model(name=None)
        assert instance.name is None

    def test_description_preserved(self):
        schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        model = _schema_to_pydantic("desc", schema)
        field_info = model.model_fields["query"]
        assert field_info.description == "Search query"

    def test_model_name(self):
        model = _schema_to_pydantic("my_tool", {"properties": {}})
        assert model.__name__ == "my_tool_Input"

    def test_typed_array(self):
        schema = {
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["names"],
        }
        model = _schema_to_pydantic("typed_arr", schema)
        instance = model(names=["a", "b"])
        assert instance.names == ["a", "b"]

    def test_untyped_array_fallback(self):
        schema = {
            "properties": {
                "data": {"type": "array"},
            },
            "required": ["data"],
        }
        model = _schema_to_pydantic("untyped_arr", schema)
        instance = model(data=[1, "two", 3.0])
        assert instance.data == [1, "two", 3.0]

    def test_nested_object(self):
        schema = {
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
            "required": ["address"],
        }
        model = _schema_to_pydantic("nested", schema)
        instance = model(address={"city": "Berlin", "zip": "10115"})
        assert instance.address.city == "Berlin"
        assert instance.address.zip == "10115"

    def test_plain_object_without_properties(self):
        schema = {
            "properties": {
                "meta": {"type": "object"},
            },
            "required": ["meta"],
        }
        model = _schema_to_pydantic("plain_obj", schema)
        instance = model(meta={"key": "val"})
        assert instance.meta == {"key": "val"}

    def test_enum_field(self):
        schema = {
            "properties": {
                "color": {"enum": ["red", "green", "blue"]},
            },
            "required": ["color"],
        }
        model = _schema_to_pydantic("enum_test", schema)
        instance = model(color="red")
        assert instance.color == "red"

    def test_unknown_type_falls_back_to_any(self):
        schema = {
            "properties": {
                "x": {"type": "null"},
            },
            "required": ["x"],
        }
        model = _schema_to_pydantic("unknown", schema)
        instance = model(x=None)
        assert instance.x is None

    def test_missing_type_key_falls_back_to_any(self):
        """Properties with no 'type' (e.g. $ref, anyOf) fall back to Any."""
        schema = {
            "properties": {
                "x": {"description": "some ref field"},
            },
            "required": ["x"],
        }
        model = _schema_to_pydantic("notype", schema)
        instance = model(x="anything")
        assert instance.x == "anything"


# --- MCPTool tests ---


def _make_text_content(text: str):
    """Create a mock TextContent object."""
    content = MagicMock()
    content.text = text
    return content


def _make_image_content():
    """Create a mock non-text content object (e.g. ImageContent)."""
    content = MagicMock(spec=[])  # no .text attribute
    return content


class TestMCPTool:
    def test_schema_roundtrip(self):
        schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
        session = MagicMock()
        tool = MCPTool(
            name="search",
            description="Search for things",
            input_schema=schema,
            session=session,
        )
        assert tool.name == "search"
        assert tool.description == "Search for things"
        json_schema = tool.schema()
        assert "query" in json_schema["properties"]
        assert "limit" in json_schema["properties"]

    async def test_execute_single_text(self):
        session = AsyncMock()
        result = MagicMock()
        result.isError = False
        result.content = [_make_text_content("hello world")]
        session.call_tool.return_value = result

        tool = MCPTool(
            name="greet",
            description="Greet",
            input_schema={"properties": {"name": {"type": "string"}}, "required": ["name"]},
            session=session,
        )
        output = await tool.execute(name="alice")
        assert output == "hello world"
        session.call_tool.assert_called_once_with("greet", arguments={"name": "alice"})

    async def test_execute_multiple_text_blocks(self):
        session = AsyncMock()
        result = MagicMock()
        result.isError = False
        result.content = [_make_text_content("one"), _make_text_content("two")]
        session.call_tool.return_value = result

        tool = MCPTool(
            name="multi",
            description="Multi",
            input_schema={"properties": {}},
            session=session,
        )
        output = await tool.execute()
        assert json.loads(output) == ["one", "two"]

    async def test_execute_empty_content(self):
        session = AsyncMock()
        result = MagicMock()
        result.isError = False
        result.content = []
        session.call_tool.return_value = result

        tool = MCPTool(
            name="empty",
            description="Empty",
            input_schema={"properties": {}},
            session=session,
        )
        output = await tool.execute()
        assert output == ""

    async def test_execute_error_raises(self):
        session = AsyncMock()
        result = MagicMock()
        result.isError = True
        result.content = [_make_text_content("something went wrong")]
        session.call_tool.return_value = result

        tool = MCPTool(
            name="failing",
            description="Fails",
            input_schema={"properties": {}},
            session=session,
        )
        with pytest.raises(ToolExecutionError, match="something went wrong"):
            await tool.execute()

    async def test_execute_non_text_content_noted(self):
        """Non-text content blocks are noted in output, not silently dropped."""
        session = AsyncMock()
        result = MagicMock()
        result.isError = False
        result.content = [_make_text_content("hello"), _make_image_content()]
        session.call_tool.return_value = result

        tool = MCPTool(
            name="mixed",
            description="Mixed",
            input_schema={"properties": {}},
            session=session,
        )
        output = await tool.execute()
        parsed = json.loads(output)
        assert parsed[0] == "hello"
        assert "non-text" in parsed[1]

    async def test_execute_only_non_text_content(self):
        session = AsyncMock()
        result = MagicMock()
        result.isError = False
        result.content = [_make_image_content()]
        session.call_tool.return_value = result

        tool = MCPTool(
            name="img",
            description="Image",
            input_schema={"properties": {}},
            session=session,
        )
        output = await tool.execute()
        assert "non-text" in output

    async def test_execute_transport_error_wrapped(self):
        """Transport-level exceptions are wrapped in ToolExecutionError."""
        session = AsyncMock()
        session.call_tool.side_effect = ConnectionError("pipe broken")

        tool = MCPTool(
            name="broken",
            description="Broken",
            input_schema={"properties": {}},
            session=session,
        )
        with pytest.raises(ToolExecutionError, match="pipe broken"):
            await tool.execute()

    async def test_execute_timeout(self):
        """Tool calls that exceed the timeout raise ToolExecutionError."""
        session = AsyncMock()

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)

        session.call_tool.side_effect = slow_call

        tool = MCPTool(
            name="slow",
            description="Slow",
            input_schema={"properties": {}},
            session=session,
            timeout=0.01,
        )
        with pytest.raises(ToolExecutionError, match="timed out"):
            await tool.execute()

    def test_custom_timeout(self):
        session = MagicMock()
        tool = MCPTool(
            name="t",
            description="t",
            input_schema={"properties": {}},
            session=session,
            timeout=60.0,
        )
        assert tool._timeout == 60.0

    def test_default_timeout(self):
        session = MagicMock()
        tool = MCPTool(
            name="t",
            description="t",
            input_schema={"properties": {}},
            session=session,
        )
        assert tool._timeout == DEFAULT_TIMEOUT


# --- MCPConnection tests ---


def _mock_mcp_infra(tools=None):
    """Set up mock patches for stdio_client and ClientSession."""
    if tools is None:
        tools = []

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = MagicMock(tools=tools)
    mock_session.initialize = AsyncMock()

    mock_read = MagicMock()
    mock_write = MagicMock()

    return mock_session, mock_read, mock_write


class TestMCPConnection:
    async def test_connect_stdio(self):
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        mock_session, mock_read, mock_write = _mock_mcp_infra([mock_tool])

        with patch("agent_loop.mcp.stdio_client") as mock_stdio:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = (mock_read, mock_write)
            mock_cm.__aexit__.return_value = False
            mock_stdio.return_value = mock_cm

            with patch("agent_loop.mcp.ClientSession") as mock_session_cls:
                mock_session_cm = AsyncMock()
                mock_session_cm.__aenter__.return_value = mock_session
                mock_session_cm.__aexit__.return_value = False
                mock_session_cls.return_value = mock_session_cm

                from mcp.client.stdio import StdioServerParameters

                conn = MCPConnection(
                    StdioServerParameters(command="python", args=["server.py"])
                )
                tools = await conn.connect()

                assert len(tools) == 1
                assert tools[0].name == "test_tool"
                assert tools[0].description == "A test tool"
                assert isinstance(tools[0], MCPTool)

                await conn.disconnect()

    async def test_connect_http(self):
        mock_session, mock_read, mock_write = _mock_mcp_infra()

        with patch("agent_loop.mcp.streamablehttp_client") as mock_http:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = (mock_read, mock_write)
            mock_cm.__aexit__.return_value = False
            mock_http.return_value = mock_cm

            with patch("agent_loop.mcp.ClientSession") as mock_session_cls:
                mock_session_cm = AsyncMock()
                mock_session_cm.__aenter__.return_value = mock_session
                mock_session_cm.__aexit__.return_value = False
                mock_session_cls.return_value = mock_session_cm

                conn = MCPConnection("http://localhost:8000/mcp")
                tools = await conn.connect()

                assert tools == []
                mock_http.assert_called_once_with("http://localhost:8000/mcp")

                await conn.disconnect()

    async def test_context_manager(self):
        mock_session, mock_read, mock_write = _mock_mcp_infra()

        with patch("agent_loop.mcp.stdio_client") as mock_stdio:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = (mock_read, mock_write)
            mock_cm.__aexit__.return_value = False
            mock_stdio.return_value = mock_cm

            with patch("agent_loop.mcp.ClientSession") as mock_session_cls:
                mock_session_cm = AsyncMock()
                mock_session_cm.__aenter__.return_value = mock_session
                mock_session_cm.__aexit__.return_value = False
                mock_session_cls.return_value = mock_session_cm

                from mcp.client.stdio import StdioServerParameters

                async with MCPConnection(
                    StdioServerParameters(command="echo", args=["hi"])
                ) as tools:
                    assert isinstance(tools, list)

    async def test_disconnect_without_connect(self):
        """Calling disconnect before connect is a safe no-op."""
        from mcp.client.stdio import StdioServerParameters

        conn = MCPConnection(
            StdioServerParameters(command="echo", args=["hi"])
        )
        await conn.disconnect()
        assert conn._exit_stack is None

    async def test_double_connect_cleans_up_first(self):
        """Calling connect() twice disconnects the first session."""
        mock_session, mock_read, mock_write = _mock_mcp_infra()

        with patch("agent_loop.mcp.stdio_client") as mock_stdio:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = (mock_read, mock_write)
            mock_cm.__aexit__.return_value = False
            mock_stdio.return_value = mock_cm

            with patch("agent_loop.mcp.ClientSession") as mock_session_cls:
                mock_session_cm = AsyncMock()
                mock_session_cm.__aenter__.return_value = mock_session
                mock_session_cm.__aexit__.return_value = False
                mock_session_cls.return_value = mock_session_cm

                from mcp.client.stdio import StdioServerParameters

                conn = MCPConnection(
                    StdioServerParameters(command="echo", args=["hi"])
                )
                await conn.connect()
                first_stack = conn._exit_stack

                await conn.connect()
                # First stack should have been closed, new one created
                assert conn._exit_stack is not first_stack

                await conn.disconnect()

    async def test_connect_failure_cleans_up(self):
        """If connect() fails partway through, resources are cleaned up."""
        mock_session, mock_read, mock_write = _mock_mcp_infra()
        mock_session.initialize.side_effect = RuntimeError("init failed")

        with patch("agent_loop.mcp.stdio_client") as mock_stdio:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = (mock_read, mock_write)
            mock_cm.__aexit__.return_value = False
            mock_stdio.return_value = mock_cm

            with patch("agent_loop.mcp.ClientSession") as mock_session_cls:
                mock_session_cm = AsyncMock()
                mock_session_cm.__aenter__.return_value = mock_session
                mock_session_cm.__aexit__.return_value = False
                mock_session_cls.return_value = mock_session_cm

                from mcp.client.stdio import StdioServerParameters

                conn = MCPConnection(
                    StdioServerParameters(command="echo", args=["hi"])
                )
                with pytest.raises(RuntimeError, match="init failed"):
                    await conn.connect()

                # Resources should be cleaned up
                assert conn._exit_stack is None
                assert conn._session is None

    async def test_custom_timeout_propagated(self):
        """Timeout parameter is propagated to MCPTool instances."""
        mock_tool = MagicMock()
        mock_tool.name = "t"
        mock_tool.description = "t"
        mock_tool.inputSchema = {"properties": {}}
        mock_session, mock_read, mock_write = _mock_mcp_infra([mock_tool])

        with patch("agent_loop.mcp.stdio_client") as mock_stdio:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = (mock_read, mock_write)
            mock_cm.__aexit__.return_value = False
            mock_stdio.return_value = mock_cm

            with patch("agent_loop.mcp.ClientSession") as mock_session_cls:
                mock_session_cm = AsyncMock()
                mock_session_cm.__aenter__.return_value = mock_session
                mock_session_cm.__aexit__.return_value = False
                mock_session_cls.return_value = mock_session_cm

                from mcp.client.stdio import StdioServerParameters

                conn = MCPConnection(
                    StdioServerParameters(command="echo", args=["hi"]),
                    timeout=120.0,
                )
                tools = await conn.connect()
                assert tools[0]._timeout == 120.0
                await conn.disconnect()


# --- Integration-style test: MCPTool works with Agent ---


class TestMCPToolWithAgent:
    async def test_agent_uses_mcp_tool(self):
        """MCPTool instances work with the Agent just like native tools."""
        from agent_loop.agent import Agent
        from agent_loop.execution import ToolCall
        from agent_loop.model import ModelAdaptor, ModelResponse

        session = AsyncMock()
        mcp_result = MagicMock()
        mcp_result.isError = False
        mcp_result.content = [_make_text_content("42")]
        session.call_tool.return_value = mcp_result

        tool = MCPTool(
            name="compute",
            description="Compute something",
            input_schema={
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
            session=session,
        )

        class MockModel(ModelAdaptor):
            def __init__(self):
                self.call_count = 0

            async def call(self, messages, tools, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return ModelResponse(
                        type="tool_call",
                        content="Let me compute that.",
                        tool_call=ToolCall(
                            id="call_1",
                            tool_name="compute",
                            arguments={"expr": "6*7"},
                        ),
                    )
                return ModelResponse(type="final_response", content="The answer is 42.")

        agent = Agent(model=MockModel(), tools=[tool])
        execution = await agent.run_async("What is 6*7?")

        assert execution.state == "completed"
        assert len(execution.tool_calls) == 1
        assert execution.tool_calls[0].result == "42"
        session.call_tool.assert_called_once_with("compute", arguments={"expr": "6*7"})
