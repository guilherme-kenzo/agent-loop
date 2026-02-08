"""Tests for the Gemini adaptor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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


def make_function_call(name, args, id=None):
    fc = MagicMock()
    fc.name = name
    fc.args = args
    fc.id = id
    return fc


def make_response(text=None, function_calls=None):
    response = MagicMock()
    response.text = text
    response.function_calls = function_calls
    return response


@pytest.fixture
def adaptor():
    with patch("agent_loop.adaptors.gemini.genai") as mock_genai, \
         patch("agent_loop.adaptors.gemini.types") as mock_types:
        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock()
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        mock_genai.Client.return_value = mock_client

        # Make types constructors return MagicMocks that pass through
        mock_types.Content = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.Part = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.FunctionCall = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.FunctionResponse = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.FunctionDeclaration = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.Tool = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))
        mock_types.AutomaticFunctionCallingConfig = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))

        from agent_loop.adaptors.gemini import GeminiAdaptor

        a = GeminiAdaptor(api_key="test-key")
        yield a, mock_client, mock_types


# --- Constructor tests ---


class TestGeminiAdaptorInit:
    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("agent_loop.adaptors.gemini.genai"), \
                 patch("agent_loop.adaptors.gemini.types"):
                from agent_loop.adaptors.gemini import GeminiAdaptor

                with pytest.raises(ValueError, match="Google API key"):
                    GeminiAdaptor()

    def test_defaults(self):
        with patch("agent_loop.adaptors.gemini.genai"), \
             patch("agent_loop.adaptors.gemini.types"):
            from agent_loop.adaptors.gemini import GeminiAdaptor

            a = GeminiAdaptor(api_key="key")
            assert a.model == "gemini-2.5-flash"

    def test_custom_model(self):
        with patch("agent_loop.adaptors.gemini.genai"), \
             patch("agent_loop.adaptors.gemini.types"):
            from agent_loop.adaptors.gemini import GeminiAdaptor

            a = GeminiAdaptor(api_key="key", model="gemini-2.5-pro")
            assert a.model == "gemini-2.5-pro"

    def test_env_var_fallback(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}):
            with patch("agent_loop.adaptors.gemini.genai"), \
                 patch("agent_loop.adaptors.gemini.types"):
                from agent_loop.adaptors.gemini import GeminiAdaptor

                a = GeminiAdaptor()
                assert a.api_key == "env-key"


# --- Message conversion tests ---


class TestGeminiMessageConversion:
    def test_user_message(self, adaptor):
        a, _, mock_types = adaptor
        messages = [Message(role="user", content="Hello")]
        result = a._convert_messages(messages)
        assert len(result) == 1
        mock_types.Content.assert_called()

    def test_assistant_message(self, adaptor):
        a, _, mock_types = adaptor
        messages = [Message(role="assistant", content="Hi there")]
        result = a._convert_messages(messages)
        assert len(result) == 1

    def test_tool_message(self, adaptor):
        a, _, mock_types = adaptor
        tc = ToolCall(id="tc_1", tool_name="search", arguments={"query": "test"})
        messages = [
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content="results", tool_call_id="tc_1"),
        ]
        result = a._convert_messages(messages)
        assert len(result) == 2

    def test_find_tool_name(self, adaptor):
        a, _, _ = adaptor
        tc = ToolCall(id="tc_1", tool_name="search", arguments={})
        messages = [
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content="result", tool_call_id="tc_1"),
        ]
        assert a._find_tool_name(messages, "tc_1") == "search"
        assert a._find_tool_name(messages, "nonexistent") == "unknown"


# --- Tool conversion tests ---


class TestGeminiToolConversion:
    def test_convert_tools(self, adaptor):
        a, _, mock_types = adaptor
        tools = [SearchTool()]
        result = a._convert_tools(tools)
        mock_types.FunctionDeclaration.assert_called_once()
        call_kwargs = mock_types.FunctionDeclaration.call_args[1]
        assert call_kwargs["name"] == "search"
        assert call_kwargs["description"] == "Search the web"


# --- Response parsing tests ---


class TestGeminiResponseParsing:
    def test_final_response(self, adaptor):
        a, _, _ = adaptor
        response = make_response(text="The answer is 42.", function_calls=None)
        result = a._parse_response(response)
        assert result.type == "final_response"
        assert result.content == "The answer is 42."
        assert result.tool_call is None

    def test_tool_call_response(self, adaptor):
        a, _, _ = adaptor
        fc = make_function_call("search", {"query": "test"}, id="tc_1")
        response = make_response(text="Searching...", function_calls=[fc])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.content == "Searching..."
        assert result.tool_call.tool_name == "search"
        assert result.tool_call.arguments == {"query": "test"}

    def test_tool_call_without_text(self, adaptor):
        a, _, _ = adaptor
        fc = make_function_call("search", {"query": "test"})
        response = make_response(text=None, function_calls=[fc])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.content == "Calling search"

    def test_empty_response(self, adaptor):
        a, _, _ = adaptor
        response = make_response(text=None, function_calls=None)
        result = a._parse_response(response)
        assert result.type == "final_response"
        assert result.content == ""


# --- Full call tests ---


class TestGeminiCall:
    @pytest.mark.asyncio
    async def test_call_final_response(self, adaptor):
        a, mock_client, _ = adaptor
        mock_client.aio.models.generate_content.return_value = make_response(
            text="Hello!", function_calls=None
        )
        result = await a.call(
            [Message(role="user", content="Hi")],
            [],
        )
        assert result.type == "final_response"
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_call_tool_response(self, adaptor):
        a, mock_client, _ = adaptor
        fc = make_function_call("search", {"query": "python"}, id="tc_1")
        mock_client.aio.models.generate_content.return_value = make_response(
            text="Searching...", function_calls=[fc]
        )
        result = await a.call(
            [Message(role="user", content="Search for python")],
            [SearchTool()],
        )
        assert result.type == "tool_call"
        assert result.tool_call.tool_name == "search"
