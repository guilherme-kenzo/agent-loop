"""Tests for the Anthropic adaptor."""

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


def make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(id, name, input_data):
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input_data
    return block


def make_response(stop_reason, content_blocks):
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content_blocks
    return response


@pytest.fixture
def adaptor():
    with patch("agent_loop.adaptors.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock()
        mock_cls.return_value = mock_client

        from agent_loop.adaptors.anthropic import AnthropicAdaptor

        a = AnthropicAdaptor(api_key="test-key")
        yield a, mock_client


# --- Constructor tests ---


class TestAnthropicAdaptorInit:
    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("agent_loop.adaptors.anthropic.AsyncAnthropic"):
                from agent_loop.adaptors.anthropic import AnthropicAdaptor

                with pytest.raises(ValueError, match="Anthropic API key"):
                    AnthropicAdaptor()

    def test_defaults(self):
        with patch("agent_loop.adaptors.anthropic.AsyncAnthropic"):
            from agent_loop.adaptors.anthropic import AnthropicAdaptor

            a = AnthropicAdaptor(api_key="key")
            assert a.model == "claude-sonnet-4-5-20250929"
            assert a.max_tokens == 1024

    def test_custom_params(self):
        with patch("agent_loop.adaptors.anthropic.AsyncAnthropic"):
            from agent_loop.adaptors.anthropic import AnthropicAdaptor

            a = AnthropicAdaptor(api_key="key", model="claude-haiku-3", max_tokens=2048)
            assert a.model == "claude-haiku-3"
            assert a.max_tokens == 2048

    def test_env_var_fallback(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("agent_loop.adaptors.anthropic.AsyncAnthropic"):
                from agent_loop.adaptors.anthropic import AnthropicAdaptor

                a = AnthropicAdaptor()
                assert a.api_key == "env-key"


# --- Message conversion tests ---


class TestAnthropicMessageConversion:
    def test_user_message(self, adaptor):
        a, _ = adaptor
        messages = [Message(role="user", content="Hello")]
        result = a._convert_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_text_only(self, adaptor):
        a, _ = adaptor
        messages = [Message(role="assistant", content="Hi there")]
        result = a._convert_messages(messages)
        assert result == [{
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there"}],
        }]

    def test_assistant_message_with_tool_calls(self, adaptor):
        a, _ = adaptor
        tc = ToolCall(id="tc_1", tool_name="search", arguments={"query": "test"})
        messages = [Message(role="assistant", content="Let me search", tool_calls=[tc])]
        result = a._convert_messages(messages)
        assert result[0]["role"] == "assistant"
        blocks = result[0]["content"]
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Let me search"}
        assert blocks[1] == {
            "type": "tool_use",
            "id": "tc_1",
            "name": "search",
            "input": {"query": "test"},
        }

    def test_tool_message(self, adaptor):
        a, _ = adaptor
        messages = [Message(role="tool", content="search results", tool_call_id="tc_1")]
        result = a._convert_messages(messages)
        assert result == [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "tc_1",
                "content": "search results",
            }],
        }]


# --- Tool conversion tests ---


class TestAnthropicToolConversion:
    def test_convert_tool(self, adaptor):
        a, _ = adaptor
        tool = SearchTool()
        result = a._convert_tool(tool)
        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert "input_schema" in result
        assert result["input_schema"]["properties"]["query"]["type"] == "string"


# --- Response parsing tests ---


class TestAnthropicResponseParsing:
    def test_final_response(self, adaptor):
        a, _ = adaptor
        response = make_response("end_turn", [make_text_block("The answer is 42.")])
        result = a._parse_response(response)
        assert result.type == "final_response"
        assert result.content == "The answer is 42."
        assert result.tool_call is None

    def test_tool_call_response(self, adaptor):
        a, _ = adaptor
        response = make_response("tool_use", [
            make_text_block("Let me search"),
            make_tool_use_block("tc_1", "search", {"query": "test"}),
        ])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.content == "Let me search"
        assert result.tool_call.id == "tc_1"
        assert result.tool_call.tool_name == "search"
        assert result.tool_call.arguments == {"query": "test"}

    def test_tool_call_without_text(self, adaptor):
        a, _ = adaptor
        response = make_response("tool_use", [
            make_tool_use_block("tc_1", "search", {"query": "test"}),
        ])
        result = a._parse_response(response)
        assert result.type == "tool_call"
        assert result.content == "Calling search"


# --- Full call tests ---


class TestAnthropicCall:
    @pytest.mark.asyncio
    async def test_call_final_response(self, adaptor):
        a, mock_client = adaptor
        mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("Hello!")]
        )
        result = await a.call(
            [Message(role="user", content="Hi")],
            [],
        )
        assert result.type == "final_response"
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_call_tool_response(self, adaptor):
        a, mock_client = adaptor
        mock_client.messages.create.return_value = make_response(
            "tool_use",
            [
                make_text_block("Searching..."),
                make_tool_use_block("tc_1", "search", {"query": "python"}),
            ],
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
        mock_client.messages.create.return_value = make_response(
            "end_turn", [make_text_block("ok")]
        )
        await a.call(
            [Message(role="user", content="test")],
            [SearchTool()],
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "search"
