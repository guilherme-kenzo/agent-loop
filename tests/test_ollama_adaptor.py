"""Tests for the Ollama adaptor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_loop.execution import Message, ToolCall
from agent_loop.model import ModelResponse
from agent_loop.tools import Tool

from pydantic import BaseModel


# --- Test fixtures ---


class SearchInput(BaseModel):
    query: str


class SearchTool(Tool):
    name = "search"
    description = "Search the web"
    input_model = SearchInput

    async def execute(self, query: str) -> str:
        return f"Results for: {query}"


def make_tool_call(name, arguments):
    tc = MagicMock()
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def make_response(content="", tool_calls=None):
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = content
    response.message.tool_calls = tool_calls
    return response


@pytest.fixture
def adaptor():
    with patch("agent_loop.adaptors.ollama.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock()
        mock_cls.return_value = mock_client

        from agent_loop.adaptors.ollama import OllamaAdaptor

        a = OllamaAdaptor(model="llama3.1")
        yield a, mock_client


# --- Constructor tests ---


class TestOllamaAdaptorInit:
    def test_defaults(self):
        with patch("agent_loop.adaptors.ollama.AsyncClient"):
            from agent_loop.adaptors.ollama import OllamaAdaptor

            a = OllamaAdaptor()
            assert a.model == "llama3.1"

    def test_custom_params(self):
        with patch("agent_loop.adaptors.ollama.AsyncClient") as mock_cls:
            from agent_loop.adaptors.ollama import OllamaAdaptor

            a = OllamaAdaptor(model="mistral", host="http://remote:11434")
            assert a.model == "mistral"
            mock_cls.assert_called_with(host="http://remote:11434")

    def test_no_api_key_required(self):
        """Ollama doesn't require an API key."""
        with patch("agent_loop.adaptors.ollama.AsyncClient"):
            from agent_loop.adaptors.ollama import OllamaAdaptor

            a = OllamaAdaptor()
            assert a.model == "llama3.1"


# --- Message conversion tests ---


class TestOllamaMessageConversion:
    def test_user_message(self, adaptor):
        a, _ = adaptor
        messages = [Message(role="user", content="Hello")]
        result = a._convert_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_text_only(self, adaptor):
        a, _ = adaptor
        messages = [Message(role="assistant", content="Hi there")]
        result = a._convert_messages(messages)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_message_with_tool_calls(self, adaptor):
        a, _ = adaptor
        tc = ToolCall(id="tc_1", tool_name="search", arguments={"query": "test"})
        messages = [Message(role="assistant", content="Searching", tool_calls=[tc])]
        result = a._convert_messages(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Searching"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"query": "test"}

    def test_tool_message(self, adaptor):
        a, _ = adaptor
        tc = ToolCall(id="tc_1", tool_name="search", arguments={"query": "test"})
        messages = [
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content="search results", tool_call_id="tc_1"),
        ]
        result = a._convert_messages(messages)
        assert result[1]["role"] == "tool"
        assert result[1]["content"] == "search results"
        assert result[1]["name"] == "search"

    def test_find_tool_name(self, adaptor):
        a, _ = adaptor
        tc = ToolCall(id="tc_1", tool_name="search", arguments={})
        messages = [
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content="result", tool_call_id="tc_1"),
        ]
        assert a._find_tool_name(messages, "tc_1") == "search"
        assert a._find_tool_name(messages, "nonexistent") == "unknown"


# --- Tool conversion tests ---


class TestOllamaToolConversion:
    def test_convert_tool(self, adaptor):
        a, _ = adaptor
        tool = SearchTool()
        result = a._convert_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search the web"
        assert "parameters" in result["function"]
        assert result["function"]["parameters"]["properties"]["query"]["type"] == "string"


# --- Response parsing tests ---


class TestOllamaResponseParsing:
    def test_final_response(self, adaptor):
        a, _ = adaptor
        response = make_response(content="The answer is 42.")
        result = a._parse_response(response)
        assert result.type == "final_response"
        assert result.content == "The answer is 42."
        assert result.tool_call is None

    def test_tool_call_response(self, adaptor):
        a, _ = adaptor
        tc = make_tool_call("search", {"query": "test"})
        response = make_response(content="Searching...", tool_calls=[tc])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.content == "Searching..."
        assert result.tool_call.tool_name == "search"
        assert result.tool_call.arguments == {"query": "test"}

    def test_tool_call_without_content(self, adaptor):
        a, _ = adaptor
        tc = make_tool_call("search", {"query": "test"})
        response = make_response(content="", tool_calls=[tc])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.content == "Calling search"

    def test_tool_call_with_string_arguments(self, adaptor):
        """Ollama may return arguments as a JSON string."""
        a, _ = adaptor
        tc = make_tool_call("search", '{"query": "test"}')
        response = make_response(content="", tool_calls=[tc])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.tool_call.arguments == {"query": "test"}


# --- Full call tests ---


class TestOllamaCall:
    @pytest.mark.asyncio
    async def test_call_final_response(self, adaptor):
        a, mock_client = adaptor
        mock_client.chat.return_value = make_response(content="Hello!")
        result = await a.call(
            [Message(role="user", content="Hi")],
            [],
        )
        assert result.type == "final_response"
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_call_tool_response(self, adaptor):
        a, mock_client = adaptor
        tc = make_tool_call("search", {"query": "python"})
        mock_client.chat.return_value = make_response(
            content="Searching...", tool_calls=[tc]
        )
        result = await a.call(
            [Message(role="user", content="Search for python")],
            [SearchTool()],
        )
        assert result.type == "tool_call"
        assert result.tool_call.tool_name == "search"

    @pytest.mark.asyncio
    async def test_call_passes_tools(self, adaptor):
        a, mock_client = adaptor
        mock_client.chat.return_value = make_response(content="ok")
        await a.call(
            [Message(role="user", content="test")],
            [SearchTool()],
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_call_without_tools(self, adaptor):
        a, mock_client = adaptor
        mock_client.chat.return_value = make_response(content="ok")
        await a.call(
            [Message(role="user", content="test")],
            [],
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert "tools" not in call_kwargs
