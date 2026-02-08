import pytest
from pydantic import BaseModel

from agent_loop.tools import Tool, ToolInput


class TestToolInput:
    def test_tool_input_is_base_model(self):
        assert issubclass(ToolInput, BaseModel)

    def test_custom_input_model(self):
        class SearchInput(ToolInput):
            query: str
            limit: int = 10

        inp = SearchInput(query="hello")
        assert inp.query == "hello"
        assert inp.limit == 10


class TestTool:
    def test_schema_returns_json_schema(self):
        class MyInput(BaseModel):
            query: str

        class MyTool(Tool):
            name = "my_tool"
            description = "A test tool"
            input_model = MyInput

        tool = MyTool()
        schema = tool.schema()
        assert "properties" in schema
        assert "query" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_raises_not_implemented(self):
        class EmptyInput(BaseModel):
            pass

        class EmptyTool(Tool):
            name = "empty"
            description = "Does nothing"
            input_model = EmptyInput

        tool = EmptyTool()
        with pytest.raises(NotImplementedError):
            await tool.execute()

    @pytest.mark.asyncio
    async def test_custom_tool_execute(self):
        class EchoInput(BaseModel):
            text: str

        class EchoTool(Tool):
            name = "echo"
            description = "Echoes input"
            input_model = EchoInput

            async def execute(self, text: str) -> str:
                return f"echo: {text}"

        tool = EchoTool()
        result = await tool.execute(text="hello")
        assert result == "echo: hello"
