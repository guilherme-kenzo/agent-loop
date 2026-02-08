"""Hook system for agent-loop.

Enables extensibility without modifying core Agent code.
Follows Flask's before_request/after_request pattern.

Architecture:
- HookRegistry is the CORE implementation
- Decorator (@hooks.on, @agent.hook) and Middleware are convenience wrappers
- Everything goes through HookRegistry
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Available hook points in agent execution."""

    BEFORE_RUN = "before_run"
    AFTER_RUN = "after_run"

    BEFORE_ITERATION = "before_iteration"
    AFTER_ITERATION = "after_iteration"

    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_CALL = "after_model_call"

    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_TOOL_ERROR = "on_tool_error"


# ============================================================================
# Hook Event Data Classes
# ============================================================================


@dataclass
class BeforeRunEventData:
    """Called before agent execution starts."""

    agent: Any  # Agent instance
    input: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AfterRunEventData:
    """Called after agent execution completes."""

    execution: Any  # Execution instance
    total_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BeforeIterationEventData:
    """Called before each agent loop iteration."""

    execution: Any
    iteration_count: int


@dataclass
class AfterIterationEventData:
    """Called after each agent loop iteration."""

    execution: Any
    iteration_count: int
    elapsed_time_ms: float


@dataclass
class BeforeModelCallEventData:
    """Called before calling the model."""

    execution: Any
    messages: List[Any]  # List of Message objects
    tools: List[Any]  # List of Tool objects


@dataclass
class AfterModelCallEventData:
    """Called after model returns response."""

    execution: Any
    model_response: Any  # ModelResponse object
    response_time_ms: float
    token_count: Optional[int] = None


@dataclass
class BeforeToolCallEventData:
    """Called before executing a tool."""

    execution: Any
    tool_name: str
    arguments: Dict[str, Any]
    tool_index: int
    iteration: int


@dataclass
class AfterToolCallEventData:
    """Called after tool execution succeeds."""

    execution: Any
    tool_call: Any  # ToolCall object
    tool_name: str
    result: str
    execution_time_ms: float


@dataclass
class OnToolErrorEventData:
    """Called when tool execution fails."""

    execution: Any
    tool_name: str
    arguments: Dict[str, Any]
    error: Exception
    error_message: str
    attempt: int


# ============================================================================
# Hook Response
# ============================================================================


@dataclass
class HookResponse:
    """What a hook can return to influence execution."""

    action: Optional[str] = None  # 'retry', 'abort', 'skip', etc.
    result: Optional[str] = None  # For tool result override
    cached_result: Optional[str] = None  # For cached results
    arguments: Optional[Dict[str, Any]] = None  # Modified tool arguments
    delay_ms: Optional[int] = None  # Delay before retry

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["HookResponse"]:
        """Convert dict to HookResponse."""
        if data is None:
            return None
        if isinstance(data, HookResponse):
            return data
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Hook Registry
# ============================================================================


class HookRegistry:
    """Central registry for all hooks.

    Supports both decorator-style and direct registration.

    Usage:
        hooks = HookRegistry()

        @hooks.on('after_tool_call')
        async def log_tool(event):
            print(f"Tool: {event.tool_name}")

        # Or direct registration
        async def my_hook(event):
            pass
        hooks.register_handler('after_tool_call', my_hook)
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {
            event.value: [] for event in HookEvent
        }

    def on(self, hook_name: str):
        """Decorator for registering hook handlers.

        Args:
            hook_name: Name of the hook (e.g., 'after_tool_call')

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self.register_handler(hook_name, func)
            return func

        return decorator

    def register_handler(self, hook_name: str, handler: Callable) -> None:
        """Register a hook handler.

        Args:
            hook_name: Name of the hook
            handler: Async function to call

        Raises:
            ValueError: If hook_name is not valid
        """
        if hook_name not in self._handlers:
            valid_hooks = [e.value for e in HookEvent]
            raise ValueError(
                f"Invalid hook name '{hook_name}'. Valid hooks: {valid_hooks}"
            )
        self._handlers[hook_name].append(handler)

    async def trigger(
        self,
        hook_name: str,
        event_data: Any,
    ) -> Optional[HookResponse]:
        """Execute all handlers for a hook.

        Args:
            hook_name: Name of the hook
            event_data: Event data to pass to handlers

        Returns:
            First non-None response from any handler, or None
        """
        handlers = self._handlers.get(hook_name, [])

        for handler in handlers:
            try:
                result = await handler(event_data)
                if result is not None:
                    return HookResponse.from_dict(result)
            except Exception as e:
                # Log but don't fail execution
                logger.warning(f"Hook '{hook_name}' raised exception: {e}")

        return None

    def has_handlers(self, hook_name: str) -> bool:
        """Check if hook has any registered handlers."""
        return len(self._handlers.get(hook_name, [])) > 0

    def clear(self) -> None:
        """Clear all handlers (useful for testing)."""
        for hook_name in self._handlers:
            self._handlers[hook_name] = []


# ============================================================================
# Middleware Base Class (Optional, for stateful handlers)
# ============================================================================


class Middleware:
    """Base class for middleware (stateful hook handlers).

    Override methods for hooks you want to handle.

    Usage:
        class MyMiddleware(Middleware):
            async def after_tool_call(self, event):
                print(f"Tool: {event.tool_name}")

        middleware = MyMiddleware()
        agent = Agent(model=model, tools=tools, middlewares=[middleware])
    """

    async def before_run(self, event: BeforeRunEventData) -> Optional[Dict]:
        pass

    async def after_run(self, event: AfterRunEventData) -> Optional[Dict]:
        pass

    async def before_iteration(
        self, event: BeforeIterationEventData
    ) -> Optional[Dict]:
        pass

    async def after_iteration(self, event: AfterIterationEventData) -> Optional[Dict]:
        pass

    async def before_model_call(
        self, event: BeforeModelCallEventData
    ) -> Optional[Dict]:
        pass

    async def after_model_call(self, event: AfterModelCallEventData) -> Optional[Dict]:
        pass

    async def before_tool_call(self, event: BeforeToolCallEventData) -> Optional[Dict]:
        pass

    async def after_tool_call(self, event: AfterToolCallEventData) -> Optional[Dict]:
        pass

    async def on_tool_error(self, event: OnToolErrorEventData) -> Optional[Dict]:
        pass
