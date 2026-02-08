from agent_loop.adaptors.openai import OpenAIAdaptor

# Conditional imports for optional SDK-based adaptors
try:
    from agent_loop.adaptors.anthropic import AnthropicAdaptor
except ImportError:
    pass

try:
    from agent_loop.adaptors.gemini import GeminiAdaptor
except ImportError:
    pass

try:
    from agent_loop.adaptors.ollama import OllamaAdaptor
except ImportError:
    pass

from agent_loop.agent import Agent
from agent_loop.exceptions import (
    AgentLoopError,
    ContextSizeExceeded,
    MaxIterationsReached,
    ToolExecutionError,
    ToolNotFound,
    ToolValidationError,
)
from agent_loop.execution import Execution, Message, ToolCall
from agent_loop.hooks import (
    AfterIterationEventData,
    AfterModelCallEventData,
    AfterRunEventData,
    AfterToolCallEventData,
    BeforeIterationEventData,
    BeforeModelCallEventData,
    BeforeRunEventData,
    BeforeToolCallEventData,
    HookEvent,
    HookRegistry,
    HookResponse,
    Middleware,
    OnToolErrorEventData,
)
from agent_loop.model import ModelAdaptor, ModelResponse
from agent_loop.tools import Tool, ToolInput

__all__ = [
    # Core
    "Agent",
    "Execution",
    "Message",
    "ModelAdaptor",
    "ModelResponse",
    "OpenAIAdaptor",
    "AnthropicAdaptor",
    "GeminiAdaptor",
    "OllamaAdaptor",
    "Tool",
    "ToolCall",
    "ToolInput",
    # Hooks
    "HookRegistry",
    "HookEvent",
    "HookResponse",
    "Middleware",
    # Hook Event Data
    "BeforeRunEventData",
    "AfterRunEventData",
    "BeforeIterationEventData",
    "AfterIterationEventData",
    "BeforeModelCallEventData",
    "AfterModelCallEventData",
    "BeforeToolCallEventData",
    "AfterToolCallEventData",
    "OnToolErrorEventData",
    # Exceptions
    "AgentLoopError",
    "ContextSizeExceeded",
    "MaxIterationsReached",
    "ToolExecutionError",
    "ToolNotFound",
    "ToolValidationError",
]
