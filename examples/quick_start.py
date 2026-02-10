"""Minimal agent-loop example with a hook. Requires OPENAI_API_KEY."""

import os
from pydantic import Field
from agent_loop import Agent, OpenAIAdaptor, Tool, ToolInput


class CityInput(ToolInput):
    city: str = Field(description="City name")


class GetPopulation(Tool):
    name = "get_population"
    description = "Returns the approximate population of a city"
    input_model = CityInput

    async def execute(self, city: str) -> str:
        populations = {"tokyo": "14M", "paris": "2.1M", "new york": "8.3M"}
        return populations.get(city.lower(), "unknown")


agent = Agent(
    model=OpenAIAdaptor(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-mini"),
    tools=[GetPopulation()],
)


@agent.hook("after_tool_call")
async def on_tool_call(event):
    print(f"[hook] {event.tool_name}({event.arguments}) -> {event.result}")


if __name__ == "__main__":
    result = agent.run("What's the population of Tokyo and Paris?")
    print(result.response)
