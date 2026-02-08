from agent_loop.execution import Execution, Message, ToolCall


class TestMessage:
    def test_basic_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_call_id is None

    def test_tool_message(self):
        msg = Message(role="tool", content="result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"


class TestToolCall:
    def test_tool_call_defaults(self):
        tc = ToolCall(id="1", tool_name="search", arguments={"q": "test"})
        assert tc.result == ""
        assert tc.error is None
        assert tc.timestamp > 0

    def test_tool_call_with_result(self):
        tc = ToolCall(id="1", tool_name="search", arguments={}, result="found it")
        assert tc.result == "found it"


class TestExecution:
    def test_execution_defaults(self):
        ex = Execution(input="test input")
        assert ex.input == "test input"
        assert ex.response == ""
        assert ex.messages == []
        assert ex.tool_calls == []
        assert ex.iterations == 0
        assert ex.state == "running"
        assert ex.metadata == {}

    def test_execution_mutable_defaults_are_independent(self):
        ex1 = Execution(input="a")
        ex2 = Execution(input="b")
        ex1.messages.append(Message(role="user", content="a"))
        assert len(ex2.messages) == 0
