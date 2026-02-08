"""Tests for OpenAI ModelAdaptor."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from agent_loop.adaptors.openai import OpenAIAdaptor
from agent_loop.execution import Message
from agent_loop.tools import Tool


# --- Test Fixtures ---


class DummyToolInput(BaseModel):
    query: str


class DummyTool(Tool):
    name = "dummy"
    description = "A dummy tool for testing"
    input_model = DummyToolInput

    async def execute(self, query: str) -> str:
        return f"Result for {query}"


class AnotherToolInput(BaseModel):
    value: int


class AnotherTool(Tool):
    name = "another"
    description = "Another dummy tool"
    input_model = AnotherToolInput

    async def execute(self, value: int) -> str:
        return f"Value is {value}"


# --- Tests for Initialization ---


class TestOpenAIAdaptorInit:
    def test_init_with_explicit_api_key(self):
        adaptor = OpenAIAdaptor(api_key="sk-test123", model="gpt-4")
        assert adaptor.api_key == "sk-test123"
        assert adaptor.model == "gpt-4"
        assert adaptor.base_url == "https://api.openai.com/v1"

    def test_init_with_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env123")
        adaptor = OpenAIAdaptor()
        assert adaptor.api_key == "sk-env123"
        assert adaptor.model == "gpt-5-mini"  # Default

    def test_init_custom_base_url(self):
        adaptor = OpenAIAdaptor(
            api_key="sk-test", base_url="http://localhost:8000/v1"
        )
        assert adaptor.base_url == "http://localhost:8000/v1"

    def test_init_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            OpenAIAdaptor()

    def test_init_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        adaptor = OpenAIAdaptor(api_key="sk-explicit")
        assert adaptor.api_key == "sk-explicit"


# --- Tests for Message Conversion ---


class TestConvertMessages:
    def test_convert_single_user_message(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        messages = [Message(role="user", content="Hello")]

        result = adaptor._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_convert_multiple_messages(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello there"),
            Message(role="user", content="How are you?"),
        ]

        result = adaptor._convert_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_convert_tool_message(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        messages = [
            Message(role="tool", content="Tool result", tool_call_id="call_123")
        ]

        result = adaptor._convert_messages(messages)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"

    def test_convert_assistant_message_with_tool_calls(self):
        """Test that assistant messages with tool_calls are properly formatted for OpenAI API."""
        from agent_loop.execution import ToolCall

        adaptor = OpenAIAdaptor(api_key="sk-test")
        tool_call = ToolCall(
            id="call_123",
            tool_name="dummy",
            arguments={"query": "test"},
        )
        messages = [
            Message(
                role="assistant",
                content="Let me search for that",
                tool_calls=[tool_call],
            )
        ]

        result = adaptor._convert_messages(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me search for that"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

        tool_call_msg = result[0]["tool_calls"][0]
        assert tool_call_msg["id"] == "call_123"
        assert tool_call_msg["type"] == "function"
        assert tool_call_msg["function"]["name"] == "dummy"
        assert json.loads(tool_call_msg["function"]["arguments"]) == {"query": "test"}

    def test_convert_messages_with_tool_flow(self):
        """Test complete flow: assistant with tool_call -> tool message."""
        from agent_loop.execution import ToolCall

        adaptor = OpenAIAdaptor(api_key="sk-test")
        tool_call = ToolCall(
            id="call_456",
            tool_name="another",
            arguments={"value": 42},
        )
        messages = [
            Message(role="user", content="Do something"),
            Message(
                role="assistant",
                content="Executing",
                tool_calls=[tool_call],
            ),
            Message(
                role="tool",
                content="Result: success",
                tool_call_id="call_456",
            ),
        ]

        result = adaptor._convert_messages(messages)

        # Check user message
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Do something"

        # Check assistant message with tool_calls
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Executing"
        assert result[1]["tool_calls"][0]["id"] == "call_456"
        assert result[1]["tool_calls"][0]["function"]["name"] == "another"

        # Check tool message with tool_call_id
        assert result[2]["role"] == "tool"
        assert result[2]["content"] == "Result: success"
        assert result[2]["tool_call_id"] == "call_456"


# --- Tests for Tool Conversion ---


class TestConvertTool:
    def test_convert_tool_basic(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        tool = DummyTool()

        result = adaptor._convert_tool(tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "dummy"
        assert result["function"]["description"] == "A dummy tool for testing"
        assert "parameters" in result["function"]

    def test_convert_tool_parameters(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        tool = DummyTool()

        result = adaptor._convert_tool(tool)
        params = result["function"]["parameters"]

        assert "properties" in params
        assert "query" in params["properties"]

    def test_convert_multiple_tools(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        tools = [DummyTool(), AnotherTool()]

        results = [adaptor._convert_tool(tool) for tool in tools]

        assert len(results) == 2
        assert results[0]["function"]["name"] == "dummy"
        assert results[1]["function"]["name"] == "another"


# --- Tests for Response Parsing ---


class TestParseResponse:
    def test_parse_final_response(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is the final answer.",
                        "tool_calls": None,
                    }
                }
            ]
        }

        result = adaptor._parse_response(response_data)

        assert result.type == "final_response"
        assert result.content == "This is the final answer."
        assert result.tool_call is None

    def test_parse_tool_call_response(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me search for that.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "dummy",
                                    "arguments": json.dumps({"query": "test"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        result = adaptor._parse_response(response_data)

        assert result.type == "tool_call"
        assert result.content == "Let me search for that."
        assert result.tool_call is not None
        assert result.tool_call.id == "call_123"
        assert result.tool_call.tool_name == "dummy"
        assert result.tool_call.arguments == {"query": "test"}

    def test_parse_tool_call_multiple_calls_selects_first(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Multiple calls",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "dummy",
                                    "arguments": json.dumps({"query": "first"}),
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "another",
                                    "arguments": json.dumps({"value": 42}),
                                },
                            },
                        ],
                    }
                }
            ]
        }

        result = adaptor._parse_response(response_data)

        assert result.tool_call.id == "call_1"
        assert result.tool_call.tool_name == "dummy"

    def test_parse_response_missing_choices(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        response_data = {}

        with pytest.raises(ValueError, match="missing 'choices'"):
            adaptor._parse_response(response_data)

    def test_parse_response_empty_content(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": None,
                    }
                }
            ]
        }

        result = adaptor._parse_response(response_data)

        assert result.type == "final_response"
        assert result.content == ""


# --- Tests for API Calls ---


class TestOpenAIAdaptorCall:
    @pytest.mark.asyncio
    async def test_call_final_response(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        messages = [Message(role="user", content="Hello")]
        tools = []

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hi there!",
                        "tool_calls": None,
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await adaptor.call(messages, tools)

            assert result.type == "final_response"
            assert result.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        messages = [Message(role="user", content="Search for something")]
        tools = [DummyTool()]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Searching",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "dummy",
                                    "arguments": json.dumps({"query": "example"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await adaptor.call(messages, tools)

            assert result.type == "tool_call"
            assert result.tool_call.tool_name == "dummy"

    @pytest.mark.asyncio
    async def test_call_api_error(self):
        adaptor = OpenAIAdaptor(api_key="sk-test")
        messages = [Message(role="user", content="Hello")]

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            with pytest.raises(ValueError, match="OpenAI API error"):
                await adaptor.call(messages, [])

    @pytest.mark.asyncio
    async def test_call_custom_parameters(self):
        adaptor = OpenAIAdaptor(api_key="sk-test", model="gpt-4")
        messages = [Message(role="user", content="Hello")]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Response",
                        "tool_calls": None,
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await adaptor.call(
                messages, [], temperature=0.5, timeout=30.0
            )

            # Verify the call was made
            assert mock_instance.post.called
            call_args = mock_instance.post.call_args

            # Check that model and temperature are in the payload
            payload = call_args.kwargs.get("json", {})
            assert payload["model"] == "gpt-4"
            assert payload["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_call_with_custom_base_url(self):
        adaptor = OpenAIAdaptor(
            api_key="sk-test", base_url="http://localhost:8000/v1"
        )
        messages = [Message(role="user", content="Hello")]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Response",
                        "tool_calls": None,
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            await adaptor.call(messages, [])

            # Check that the correct base_url was used
            call_args = mock_instance.post.call_args
            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
            assert "http://localhost:8000/v1" in url
