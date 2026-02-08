from dataclasses import dataclass
from typing import Optional, Literal

from agent_loop.execution import Message, ToolCall
from agent_loop.tools import Tool


@dataclass
class ModelResponse:
    type: Literal["final_response", "tool_call"]
    content: str
    tool_call: Optional[ToolCall] = None


class ModelAdaptor:
    async def call(
        self,
        messages: list[Message],
        tools: list[Tool],
        **kwargs,
    ) -> ModelResponse:
        """Call the model with messages and available tools."""
        raise NotImplementedError
