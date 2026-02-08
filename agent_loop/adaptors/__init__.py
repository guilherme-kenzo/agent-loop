"""Model adaptors for agent-loop.

This module provides implementations of ModelAdaptor for various LLM providers.
"""

from agent_loop.adaptors.openai import OpenAIAdaptor

__all__ = ["OpenAIAdaptor"]

# Conditional imports for optional SDK-based adaptors
try:
    from agent_loop.adaptors.anthropic import AnthropicAdaptor

    __all__.append("AnthropicAdaptor")
except ImportError:
    pass

try:
    from agent_loop.adaptors.gemini import GeminiAdaptor

    __all__.append("GeminiAdaptor")
except ImportError:
    pass

try:
    from agent_loop.adaptors.ollama import OllamaAdaptor

    __all__.append("OllamaAdaptor")
except ImportError:
    pass
