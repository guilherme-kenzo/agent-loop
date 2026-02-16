import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None  # For assistant messages with tool calls


@dataclass
class ToolCall:
    id: str
    tool_name: str
    arguments: dict
    result: str = ""
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    thought_signature: Optional[bytes] = None


@dataclass
class Execution:
    input: str
    response: str = ""
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    state: str = "running"  # "running" | "completed" | "failed" | "max_iterations"
    metadata: dict = field(default_factory=dict)
