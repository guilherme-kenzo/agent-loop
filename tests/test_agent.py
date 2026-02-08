import pytest
from pydantic import BaseModel

from agent_loop.agent import Agent
from agent_loop.exceptions import ContextSizeExceeded, ToolNotFound
from agent_loop.execution import Message, ToolCall
from agent_loop.model import ModelAdaptor, ModelResponse
from agent_loop.tools import Tool


# --- Test fixtures ---


class FakeModel(ModelAdaptor):
    """Model that returns a canned final response."""

    def __init__(self, response_text: str = "The answer is 42."):
        self.response_text = response_text

    async def call(self, messages, tools, **kwargs):
        return ModelResponse(type="final_response", content=self.response_text)


class ToolThenAnswerModel(ModelAdaptor):
    """Model that calls a tool once, then gives a final response."""

    def __init__(self):
        self.call_count = 0

    async def call(self, messages, tools, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return ModelResponse(
                type="tool_call",
                content="Let me search for that.",
                tool_call=ToolCall(
                    id="call_1",
                    tool_name="echo",
                    arguments={"text": "hello"},
                ),
            )
        return ModelResponse(type="final_response", content="Done: echo: hello")


class NeverFinishModel(ModelAdaptor):
    """Model that always calls a tool and never finishes."""

    async def call(self, messages, tools, **kwargs):
        return ModelResponse(
            type="tool_call",
            content="Calling tool again.",
            tool_call=ToolCall(
                id="call_n",
                tool_name="echo",
                arguments={"text": "loop"},
            ),
        )


class BadToolModel(ModelAdaptor):
    """Model that calls a nonexistent tool."""

    async def call(self, messages, tools, **kwargs):
        return ModelResponse(
            type="tool_call",
            content="Calling missing tool.",
            tool_call=ToolCall(
                id="call_bad",
                tool_name="nonexistent",
                arguments={},
            ),
        )


class EchoInput(BaseModel):
    text: str


class EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    input_model = EchoInput

    async def execute(self, text: str) -> str:
        return f"echo: {text}"


# --- Tests ---


class TestAgentInit:
    def test_defaults(self):
        agent = Agent(model=FakeModel(), tools=[])
        assert agent.max_iterations == 20
        assert agent.max_context_size is None
        assert agent.name == "Agent"

    def test_custom_params(self):
        agent = Agent(
            model=FakeModel(),
            tools=[EchoTool()],
            max_iterations=5,
            max_context_size=1000,
            name="TestAgent",
        )
        assert agent.max_iterations == 5
        assert agent.max_context_size == 1000
        assert agent.name == "TestAgent"
        assert len(agent.tools) == 1


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_simple_response(self):
        agent = Agent(model=FakeModel("Hello!"), tools=[])
        result = await agent.run_async("Hi")
        assert result.state == "completed"
        assert result.response == "Hello!"
        assert result.iterations == 1
        assert len(result.messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        agent = Agent(model=ToolThenAnswerModel(), tools=[EchoTool()])
        result = await agent.run_async("Echo hello")
        assert result.state == "completed"
        assert result.response == "Done: echo: hello"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result == "echo: hello"
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        agent = Agent(model=NeverFinishModel(), tools=[EchoTool()], max_iterations=3)
        result = await agent.run_async("Loop forever")
        assert result.state == "max_iterations"
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        agent = Agent(model=BadToolModel(), tools=[EchoTool()])
        with pytest.raises(ToolNotFound):
            await agent.run_async("Call bad tool")

    @pytest.mark.asyncio
    async def test_context_size_exceeded(self):
        agent = Agent(model=FakeModel("x" * 100), tools=[], max_context_size=10)
        with pytest.raises(ContextSizeExceeded):
            await agent.run_async("A long prompt that exceeds context")


class TestAgentMessages:
    @pytest.mark.asyncio
    async def test_messages_prepended_to_execution(self):
        agent = Agent(model=FakeModel("Got it."), tools=[])
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
        ]
        result = await agent.run_async("And 3+3?", messages=history)
        assert result.state == "completed"
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "What is 2+2?"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "4"
        assert result.messages[2].role == "user"
        assert result.messages[2].content == "And 3+3?"

    @pytest.mark.asyncio
    async def test_messages_passed_to_model(self):
        """Verify the model receives the full message history."""

        class HistoryCapturingModel(ModelAdaptor):
            def __init__(self):
                self.received_messages = []

            async def call(self, messages, tools, **kwargs):
                self.received_messages = list(messages)
                return ModelResponse(type="final_response", content="ok")

        model = HistoryCapturingModel()
        agent = Agent(model=model, tools=[])
        history = [
            Message(role="user", content="prior question"),
            Message(role="assistant", content="prior answer"),
        ]
        await agent.run_async("new question", messages=history)
        assert len(model.received_messages) == 3
        assert model.received_messages[0].content == "prior question"
        assert model.received_messages[1].content == "prior answer"
        assert model.received_messages[2].content == "new question"

    @pytest.mark.asyncio
    async def test_no_messages_defaults_to_original_behavior(self):
        agent = Agent(model=FakeModel("Hello!"), tools=[])
        result = await agent.run_async("Hi")
        assert len(result.messages) == 2  # user + assistant
        assert result.messages[0].content == "Hi"

    def test_sync_run_with_messages(self):
        agent = Agent(model=FakeModel("Done."), tools=[])
        history = [Message(role="user", content="first"), Message(role="assistant", content="reply")]
        result = agent.run("second", messages=history)
        assert result.state == "completed"
        assert len(result.messages) == 4  # 2 history + user + assistant


class TestAgentFindTool:
    def test_find_existing_tool(self):
        agent = Agent(model=FakeModel(), tools=[EchoTool()])
        tool = agent._find_tool("echo")
        assert tool.name == "echo"

    def test_find_missing_tool_raises(self):
        agent = Agent(model=FakeModel(), tools=[])
        with pytest.raises(ToolNotFound):
            agent._find_tool("nonexistent")
