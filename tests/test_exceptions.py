import pytest

from agent_loop.exceptions import (
    AgentLoopError,
    ContextSizeExceeded,
    MaxIterationsReached,
    ToolExecutionError,
    ToolNotFound,
    ToolValidationError,
)


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_agent_loop_error(self):
        for exc_class in [
            ContextSizeExceeded,
            MaxIterationsReached,
            ToolValidationError,
            ToolNotFound,
            ToolExecutionError,
        ]:
            assert issubclass(exc_class, AgentLoopError)

    def test_agent_loop_error_inherits_from_exception(self):
        assert issubclass(AgentLoopError, Exception)

    def test_exceptions_carry_message(self):
        err = ToolNotFound("missing_tool")
        assert str(err) == "missing_tool"
