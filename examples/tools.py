"""Shared tool definitions for minimal agent example."""

import json
from typing import Any

from agent_loop import Tool, ToolInput
from pydantic import BaseModel, Field


class CalculatorInput(ToolInput):
    """Input model for the calculator tool."""

    expression: str = Field(
        ...,
        description="A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
    )


class CalculatorTool(Tool):
    """Simple calculator tool that evaluates math expressions."""

    name = "calculator"
    description = (
        "Evaluates mathematical expressions and returns the result. "
        "Supports basic operations: +, -, *, /, **, //, %"
    )
    input_model = CalculatorInput

    async def execute(self, expression: str) -> str:
        """Execute a math expression safely.

        Args:
            expression: Math expression to evaluate

        Returns:
            Result as string, or error message
        """
        try:
            # Allowed names for safe evaluation
            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
            }
            result = eval(expression, safe_dict)
            return json.dumps({"expression": expression, "result": result})
        except SyntaxError as e:
            return json.dumps({"expression": expression, "error": f"Syntax error: {e}"})
        except ZeroDivisionError:
            return json.dumps(
                {"expression": expression, "error": "Division by zero"}
            )
        except Exception as e:
            return json.dumps({"expression": expression, "error": str(e)})


class WebSearchInput(ToolInput):
    """Input model for the web search tool."""

    query: str = Field(
        ...,
        description="Search query to find information about (e.g., 'weather in São Paulo')",
    )


class WebSearchTool(Tool):
    """Simulated web search tool for demonstration.

    In a real implementation, this would call an actual search API.
    """

    name = "web_search"
    description = (
        "Performs a web search and returns simulated results. "
        "For demonstration purposes, returns mock results."
    )
    input_model = WebSearchInput

    async def execute(self, query: str) -> str:
        """Perform a simulated web search.

        Args:
            query: Search query

        Returns:
            JSON string with search results
        """
        # Mock search results for demonstration
        mock_results = {
            "weather": [
                {
                    "title": "Weather in São Paulo",
                    "url": "https://weather.example.com/sp",
                    "snippet": "São Paulo weather: Partly cloudy, 28°C, humid",
                }
            ],
            "python": [
                {
                    "title": "Python Programming Language",
                    "url": "https://python.org",
                    "snippet": "Python is a high-level programming language known for its simplicity.",
                }
            ],
            "default": [
                {
                    "title": f"Search results for: {query}",
                    "url": "https://search.example.com",
                    "snippet": f"Mock search result for query: {query}",
                }
            ],
        }

        # Return mock results based on query type
        query_lower = query.lower()
        if "weather" in query_lower:
            results = mock_results["weather"]
        elif "python" in query_lower:
            results = mock_results["python"]
        else:
            results = mock_results["default"]

        return json.dumps({"query": query, "results": results, "count": len(results)})
