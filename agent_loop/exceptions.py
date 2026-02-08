class AgentLoopError(Exception):
    """Base exception for agent-loop errors."""


class ContextSizeExceeded(AgentLoopError):
    """Raised when message context exceeds max_context_size."""


class MaxIterationsReached(AgentLoopError):
    """Raised when agent hits max_iterations without completing."""


class ToolValidationError(AgentLoopError):
    """Raised when tool input fails Pydantic validation."""


class ToolNotFound(AgentLoopError):
    """Raised when model calls a tool that doesn't exist."""


class ToolExecutionError(AgentLoopError):
    """Raised when tool execution fails critically."""
