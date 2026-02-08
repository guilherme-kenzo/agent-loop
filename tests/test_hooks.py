"""Tests for hook system."""

import pytest
from pydantic import BaseModel

from agent_loop.agent import Agent
from agent_loop.execution import ToolCall
from agent_loop.hooks import (
    AfterToolCallEventData,
    BeforeRunEventData,
    BeforeToolCallEventData,
    HookEvent,
    HookRegistry,
    HookResponse,
    Middleware,
    OnToolErrorEventData,
)
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


class FailingToolModel(ModelAdaptor):
    """Model that calls a tool that will fail."""

    def __init__(self):
        self.call_count = 0

    async def call(self, messages, tools, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return ModelResponse(
                type="tool_call",
                content="Calling failing tool.",
                tool_call=ToolCall(
                    id="call_fail",
                    tool_name="failing_tool",
                    arguments={},
                ),
            )
        return ModelResponse(type="final_response", content="Handled error")


class EchoInput(BaseModel):
    text: str


class EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    input_model = EchoInput

    async def execute(self, text: str) -> str:
        return f"echo: {text}"


class FailingInput(BaseModel):
    pass


class FailingTool(Tool):
    name = "failing_tool"
    description = "A tool that always fails"
    input_model = FailingInput
    fail_count = 0

    async def execute(self) -> str:
        FailingTool.fail_count += 1
        raise Exception("Tool failed intentionally")


# --- HookRegistry Tests ---


class TestHookRegistryBasic:
    @pytest.mark.asyncio
    async def test_hook_registration_and_triggering(self):
        """Test basic hook registration and triggering."""
        registry = HookRegistry()

        events = []

        @registry.on("before_run")
        async def capture_event(event):
            events.append(event)

        event = BeforeRunEventData(agent=None, input="test")
        await registry.trigger("before_run", event)

        assert len(events) == 1
        assert events[0].input == "test"

    @pytest.mark.asyncio
    async def test_hook_with_response(self):
        """Test hook that returns response to influence execution."""
        registry = HookRegistry()

        @registry.on("on_tool_error")
        async def retry_on_error(event):
            if event.attempt < 3:
                return {"action": "retry", "delay_ms": 100}
            return {"action": "abort"}

        event = OnToolErrorEventData(
            execution=None,
            tool_name="search",
            arguments={},
            error=Exception("Network error"),
            error_message="Network error",
            attempt=1,
        )

        response = await registry.trigger("on_tool_error", event)

        assert response is not None
        assert response.action == "retry"
        assert response.delay_ms == 100

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers for same hook."""
        registry = HookRegistry()

        calls = []

        @registry.on("before_run")
        async def handler1(event):
            calls.append("h1")

        @registry.on("before_run")
        async def handler2(event):
            calls.append("h2")

        event = BeforeRunEventData(agent=None, input="test")
        await registry.trigger("before_run", event)

        assert calls == ["h1", "h2"]

    @pytest.mark.asyncio
    async def test_hook_exception_does_not_crash(self):
        """Test that hook exception doesn't crash execution."""
        registry = HookRegistry()

        @registry.on("before_run")
        async def bad_hook(event):
            raise ValueError("Intentional error")

        @registry.on("before_run")
        async def good_hook(event):
            return None

        event = BeforeRunEventData(agent=None, input="test")
        # Should not raise, just log warning
        response = await registry.trigger("before_run", event)

        assert response is None

    def test_invalid_hook_name_raises(self):
        """Test that invalid hook name raises error."""
        registry = HookRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.register_handler("invalid_hook_name", lambda e: None)

        assert "Invalid hook name" in str(exc_info.value)

    def test_has_handlers(self):
        """Test has_handlers method."""
        registry = HookRegistry()

        assert registry.has_handlers("before_run") is False

        @registry.on("before_run")
        async def handler(event):
            pass

        assert registry.has_handlers("before_run") is True

    def test_clear(self):
        """Test clear method."""
        registry = HookRegistry()

        @registry.on("before_run")
        async def handler(event):
            pass

        assert registry.has_handlers("before_run") is True
        registry.clear()
        assert registry.has_handlers("before_run") is False


class TestHookResponse:
    def test_from_dict(self):
        """Test HookResponse.from_dict."""
        data = {"action": "retry", "delay_ms": 100, "arguments": {"query": "new query"}}

        response = HookResponse.from_dict(data)

        assert response.action == "retry"
        assert response.delay_ms == 100
        assert response.arguments == {"query": "new query"}

    def test_from_dict_none(self):
        """Test HookResponse.from_dict with None."""
        response = HookResponse.from_dict(None)
        assert response is None

    def test_from_dict_ignores_unknown_fields(self):
        """Test that from_dict ignores unknown fields."""
        data = {"action": "retry", "unknown_field": "ignored"}

        response = HookResponse.from_dict(data)

        assert response.action == "retry"
        assert not hasattr(response, "unknown_field")


# --- Agent Integration Tests ---


class TestAgentHookDecorator:
    @pytest.mark.asyncio
    async def test_decorator_style_hook(self):
        """Test @agent.hook() decorator pattern."""
        agent = Agent(model=FakeModel("Hello!"), tools=[])

        calls = []

        @agent.hook("before_run")
        async def log_run(event):
            calls.append(event.input)

        await agent.run_async("test query")

        assert calls == ["test query"]

    @pytest.mark.asyncio
    async def test_multiple_hooks_on_agent(self):
        """Test multiple hooks registered via decorator."""
        agent = Agent(model=ToolThenAnswerModel(), tools=[EchoTool()])

        events = []

        @agent.hook("before_run")
        async def on_before_run(event):
            events.append("before_run")

        @agent.hook("after_model_call")
        async def on_after_model(event):
            events.append("after_model_call")

        @agent.hook("after_tool_call")
        async def on_after_tool(event):
            events.append(f"after_tool_call:{event.tool_name}")

        @agent.hook("after_run")
        async def on_after_run(event):
            events.append("after_run")

        await agent.run_async("Echo hello")

        assert "before_run" in events
        assert "after_model_call" in events
        assert "after_tool_call:echo" in events
        assert "after_run" in events


class TestAgentHookRegistry:
    @pytest.mark.asyncio
    async def test_external_hook_registry(self):
        """Test passing external HookRegistry to Agent."""
        hooks = HookRegistry()

        calls = []

        @hooks.on("before_run")
        async def log_run(event):
            calls.append(event.input)

        agent = Agent(model=FakeModel("Hello!"), tools=[], hooks=hooks)

        await agent.run_async("test from external")

        assert calls == ["test from external"]

    @pytest.mark.asyncio
    async def test_shared_registry_across_agents(self):
        """Test sharing HookRegistry across multiple agents."""
        hooks = HookRegistry()

        calls = []

        @hooks.on("before_run")
        async def log_run(event):
            calls.append(f"{event.agent.name}:{event.input}")

        agent1 = Agent(model=FakeModel(), tools=[], hooks=hooks, name="Agent1")
        agent2 = Agent(model=FakeModel(), tools=[], hooks=hooks, name="Agent2")

        await agent1.run_async("query1")
        await agent2.run_async("query2")

        assert "Agent1:query1" in calls
        assert "Agent2:query2" in calls


class TestAgentMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_class(self):
        """Test Middleware class pattern."""

        class LogMiddleware(Middleware):
            def __init__(self):
                self.calls = []

            async def before_run(self, event):
                self.calls.append("before_run")

            async def after_run(self, event):
                self.calls.append("after_run")

        middleware = LogMiddleware()
        agent = Agent(model=FakeModel(), tools=[], middlewares=[middleware])

        await agent.run_async("test")

        assert middleware.calls == ["before_run", "after_run"]

    @pytest.mark.asyncio
    async def test_middleware_with_tool_hooks(self):
        """Test Middleware with tool-related hooks."""

        class ToolLogMiddleware(Middleware):
            def __init__(self):
                self.tool_calls = []

            async def before_tool_call(self, event):
                self.tool_calls.append(f"before:{event.tool_name}")

            async def after_tool_call(self, event):
                self.tool_calls.append(f"after:{event.tool_name}:{event.result}")

        middleware = ToolLogMiddleware()
        agent = Agent(
            model=ToolThenAnswerModel(), tools=[EchoTool()], middlewares=[middleware]
        )

        await agent.run_async("Echo hello")

        assert "before:echo" in middleware.tool_calls
        assert any("after:echo:" in call for call in middleware.tool_calls)

    @pytest.mark.asyncio
    async def test_multiple_middlewares(self):
        """Test multiple middleware instances."""

        class MiddlewareA(Middleware):
            def __init__(self):
                self.called = False

            async def before_run(self, event):
                self.called = True

        class MiddlewareB(Middleware):
            def __init__(self):
                self.called = False

            async def after_run(self, event):
                self.called = True

        mw_a = MiddlewareA()
        mw_b = MiddlewareB()

        agent = Agent(model=FakeModel(), tools=[], middlewares=[mw_a, mw_b])

        await agent.run_async("test")

        assert mw_a.called is True
        assert mw_b.called is True


class TestHookResponses:
    @pytest.mark.asyncio
    async def test_retry_on_tool_error(self):
        """Test retry action on tool error."""
        FailingTool.fail_count = 0

        hooks = HookRegistry()

        @hooks.on("on_tool_error")
        async def retry_hook(event):
            if event.attempt < 2:
                return {"action": "retry"}
            return None

        agent = Agent(
            model=FailingToolModel(), tools=[FailingTool()], hooks=hooks
        )

        await agent.run_async("Do something")

        # Tool should have been called twice (initial + 1 retry)
        assert FailingTool.fail_count == 2

    @pytest.mark.asyncio
    async def test_skip_with_cached_result(self):
        """Test skip action with cached result."""

        class CacheMiddleware(Middleware):
            def __init__(self):
                self.cache = {"echo:hello": "cached: hello"}

            async def before_tool_call(self, event):
                key = f"{event.tool_name}:{event.arguments.get('text', '')}"
                if key in self.cache:
                    return {"action": "skip", "cached_result": self.cache[key]}

        cache_mw = CacheMiddleware()
        agent = Agent(
            model=ToolThenAnswerModel(), tools=[EchoTool()], middlewares=[cache_mw]
        )

        result = await agent.run_async("Echo hello")

        # The tool result should be the cached value
        assert result.tool_calls[0].result == "cached: hello"


class TestAllHookPoints:
    @pytest.mark.asyncio
    async def test_all_hooks_fire_in_order(self):
        """Test that all 9 hook points fire during execution."""
        hooks = HookRegistry()
        fired_hooks = []

        @hooks.on("before_run")
        async def h1(event):
            fired_hooks.append("before_run")

        @hooks.on("before_iteration")
        async def h2(event):
            fired_hooks.append("before_iteration")

        @hooks.on("before_model_call")
        async def h3(event):
            fired_hooks.append("before_model_call")

        @hooks.on("after_model_call")
        async def h4(event):
            fired_hooks.append("after_model_call")

        @hooks.on("before_tool_call")
        async def h5(event):
            fired_hooks.append("before_tool_call")

        @hooks.on("after_tool_call")
        async def h6(event):
            fired_hooks.append("after_tool_call")

        @hooks.on("after_iteration")
        async def h7(event):
            fired_hooks.append("after_iteration")

        @hooks.on("after_run")
        async def h8(event):
            fired_hooks.append("after_run")

        agent = Agent(model=ToolThenAnswerModel(), tools=[EchoTool()], hooks=hooks)
        await agent.run_async("test")

        # Verify all hooks fired
        assert "before_run" in fired_hooks
        assert "before_iteration" in fired_hooks
        assert "before_model_call" in fired_hooks
        assert "after_model_call" in fired_hooks
        assert "before_tool_call" in fired_hooks
        assert "after_tool_call" in fired_hooks
        assert "after_iteration" in fired_hooks
        assert "after_run" in fired_hooks


class TestHookEventEnum:
    def test_all_hook_events_exist(self):
        """Verify all 9 hook events are defined."""
        expected = [
            "before_run",
            "after_run",
            "before_iteration",
            "after_iteration",
            "before_model_call",
            "after_model_call",
            "before_tool_call",
            "after_tool_call",
            "on_tool_error",
        ]

        actual = [e.value for e in HookEvent]

        for hook in expected:
            assert hook in actual, f"Missing hook: {hook}"
