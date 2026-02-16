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


def make_part(function_call=None, text=None, thought_signature=None):
    part = MagicMock()
    part.function_call = function_call
    part.text = text
    part.thought_signature = thought_signature
    return part


def make_response(text=None, function_calls=None, parts=None):
    response = MagicMock()
    response.text = text
    response.function_calls = function_calls
    if parts is not None:
        candidate = MagicMock()
        candidate.content.parts = parts
        response.candidates = [candidate]
    else:
        response.candidates = []
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


# --- Thought signature tests ---


class TestThoughtSignature:
    def test_parse_response_extracts_thought_signature(self, adaptor):
        a, _, _ = adaptor
        fc = make_function_call("search", {"query": "test"}, id="tc_1")
        sig = b"encrypted-thought-sig"
        parts = [make_part(function_call=fc, thought_signature=sig)]
        response = make_response(
            text="Searching...", function_calls=[fc], parts=parts,
        )
        result = a._parse_response(response)
        assert result.tool_call.thought_signature == sig

    def test_parse_response_no_signature_older_models(self, adaptor):
        a, _, _ = adaptor
        fc = make_function_call("search", {"query": "test"}, id="tc_1")
        # Parts without thought_signature attribute (older SDK / model)
        part = MagicMock(spec=["function_call", "text"])
        part.function_call = fc
        part.text = None
        response = make_response(
            text="Searching...", function_calls=[fc], parts=[part],
        )
        result = a._parse_response(response)
        assert result.tool_call.thought_signature is None

    def test_parse_response_no_candidates(self, adaptor):
        """Older models may not populate candidates in our mock structure."""
        a, _, _ = adaptor
        fc = make_function_call("search", {"query": "test"}, id="tc_1")
        response = make_response(text="Searching...", function_calls=[fc])
        result = a._parse_response(response)
        assert result.tool_call.thought_signature is None

    def test_convert_messages_includes_thought_signature(self, adaptor):
        a, _, mock_types = adaptor
        sig = b"encrypted-thought-sig"
        tc = ToolCall(
            id="tc_1", tool_name="search",
            arguments={"query": "test"}, thought_signature=sig,
        )
        messages = [Message(role="assistant", content="", tool_calls=[tc])]
        a._convert_messages(messages)

        # Verify types.Part was called with thought_signature
        part_calls = mock_types.Part.call_args_list
        fc_part_call = [
            c for c in part_calls
            if "function_call" in c.kwargs
        ]
        assert len(fc_part_call) == 1
        assert fc_part_call[0].kwargs["thought_signature"] == sig

    def test_convert_messages_omits_signature_when_none(self, adaptor):
        a, _, mock_types = adaptor
        tc = ToolCall(
            id="tc_1", tool_name="search",
            arguments={"query": "test"},
        )
        messages = [Message(role="assistant", content="", tool_calls=[tc])]
        a._convert_messages(messages)

        part_calls = mock_types.Part.call_args_list
        fc_part_call = [
            c for c in part_calls
            if "function_call" in c.kwargs
        ]
        assert len(fc_part_call) == 1
        assert "thought_signature" not in fc_part_call[0].kwargs

    @pytest.mark.asyncio
    async def test_full_roundtrip_preserves_signature(self, adaptor):
        """Simulate a multi-turn call where the signature must survive."""
        a, mock_client, mock_types = adaptor
        sig = b"roundtrip-sig"

        # First call returns a tool call with a thought signature
        fc = make_function_call("search", {"query": "test"}, id="tc_1")
        parts = [make_part(function_call=fc, thought_signature=sig)]
        mock_client.aio.models.generate_content.return_value = make_response(
            text="Searching...", function_calls=[fc], parts=parts,
        )
        result = await a.call(
            [Message(role="user", content="Search")],
            [SearchTool()],
        )
        assert result.tool_call.thought_signature == sig

        # Build the follow-up message history (as the agent loop would)
        history = [
            Message(role="user", content="Search"),
            Message(role="assistant", content="Searching...", tool_calls=[result.tool_call]),
            Message(role="tool", content="Results for: test", tool_call_id="tc_1"),
        ]

        # Second call returns final response
        mock_client.aio.models.generate_content.return_value = make_response(
            text="Here are the results.",
        )
        await a.call(history, [SearchTool()])

        # Check the second generate_content call's contents argument
        second_call_args = mock_client.aio.models.generate_content.call_args
        contents = second_call_args.kwargs["contents"]

        # The assistant message (index 1) should have a Part with the signature
        fc_part_calls = [
            c for c in mock_types.Part.call_args_list
            if "function_call" in c.kwargs and "thought_signature" in c.kwargs
        ]
        assert len(fc_part_calls) >= 1
        assert fc_part_calls[0].kwargs["thought_signature"] == sig
