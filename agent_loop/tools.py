from pydantic import BaseModel


class ToolInput(BaseModel):
    """Subclass this for tool-specific input validation."""


class Tool:
    name: str
    description: str
    input_model: type[BaseModel]

    def schema(self) -> dict:
        """Return JSON schema from Pydantic model."""
        return self.input_model.model_json_schema()

    async def execute(self, **kwargs) -> str:
        """Execute tool. Always async; sync tools wrap sync code.

        Pydantic validates inputs before this is called.
        """
        raise NotImplementedError
