import asyncio
import time
from typing import TYPE_CHECKING, Optional

from pydantic import ValidationError

from agent_loop.exceptions import (
    ContextSizeExceeded,
    ToolExecutionError,
    ToolNotFound,
)
from agent_loop.execution import Execution, Message
from agent_loop.model import ModelAdaptor
from agent_loop.tools import Tool

if TYPE_CHECKING:
    from agent_loop.hooks import HookRegistry, Middleware


class Agent:
    def __init__(
        self,
        model: ModelAdaptor,
        tools: list[Tool],
        max_iterations: int = 20,
        max_context_size: Optional[int] = None,
        name: str = "Agent",
        hooks: Optional["HookRegistry"] = None,
        middlewares: Optional[list["Middleware"]] = None,
    ):
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_context_size = max_context_size
        self.name = name

        # ALWAYS use HookRegistry as the foundation
        # User can pass one, or we create an internal one
        if hooks is None:
            from agent_loop.hooks import HookRegistry

            hooks = HookRegistry()
        self.hooks = hooks

        # Convert middleware to handlers registered in the same HookRegistry
        # This makes middleware just syntactic sugar over HookRegistry
        if middlewares:
            self._register_middlewares(middlewares)

    def _register_middlewares(self, middlewares: list["Middleware"]) -> None:
        """Convert middleware instances to HookRegistry handlers."""
        from agent_loop.hooks import HookEvent

        hook_names = [e.value for e in HookEvent]
        for middleware in middlewares:
            for hook_name in hook_names:
                handler = getattr(middleware, hook_name, None)
                if handler is not None and asyncio.iscoroutinefunction(handler):
                    self.hooks.register_handler(hook_name, handler)

    def hook(self, hook_name: str):
        """Decorator for registering hooks directly on agent.

        Convenience wrapper around internal HookRegistry.on()

        Usage:
            agent = Agent(model=model, tools=tools)

            @agent.hook('after_tool_call')
            async def log_tool(event):
                print(f"Tool: {event.tool_name}")

        Under the hood, this just delegates to self.hooks.on(hook_name)
        """
        return self.hooks.on(hook_name)

    def run(self, input: str, messages: Optional[list[Message]] = None) -> Execution:
        """Run agent synchronously."""
        return asyncio.run(self.run_async(input, messages))

    async def run_async(
        self, input: str, messages: Optional[list[Message]] = None
    ) -> Execution:
        """Run agent asynchronously with hook support."""
        from agent_loop.hooks import (
            AfterIterationEventData,
            AfterModelCallEventData,
            AfterRunEventData,
            AfterToolCallEventData,
            BeforeIterationEventData,
            BeforeModelCallEventData,
            BeforeRunEventData,
            BeforeToolCallEventData,
            OnToolErrorEventData,
        )

        start_time = time.time()

        # BEFORE_RUN
        await self.hooks.trigger(
            "before_run", BeforeRunEventData(agent=self, input=input)
        )

        execution = Execution(input=input)
        if messages:
            execution.messages.extend(messages)
        execution.messages.append(Message(role="user", content=input))

        while execution.iterations < self.max_iterations:
            iteration_start = time.time()

            # BEFORE_ITERATION
            await self.hooks.trigger(
                "before_iteration",
                BeforeIterationEventData(
                    execution=execution, iteration_count=execution.iterations
                ),
            )

            # Context size check
            if self.max_context_size is not None:
                context_size = sum(len(m.content) for m in execution.messages)
                if context_size > self.max_context_size:
                    execution.state = "failed"
                    raise ContextSizeExceeded(
                        f"Context size {context_size} exceeds limit {self.max_context_size}"
                    )

            # BEFORE_MODEL_CALL
            await self.hooks.trigger(
                "before_model_call",
                BeforeModelCallEventData(
                    execution=execution, messages=execution.messages, tools=self.tools
                ),
            )

            # Call model
            model_start = time.time()
            response = await self.model.call(
                messages=execution.messages,
                tools=self.tools,
            )
            model_time = (time.time() - model_start) * 1000

            # AFTER_MODEL_CALL
            await self.hooks.trigger(
                "after_model_call",
                AfterModelCallEventData(
                    execution=execution,
                    model_response=response,
                    response_time_ms=model_time,
                    token_count=None,
                ),
            )

            # Create assistant message with tool_calls if present
            assistant_msg = Message(role="assistant", content=response.content)
            if response.type == "tool_call":
                assistant_msg.tool_calls = [response.tool_call]

            execution.messages.append(assistant_msg)

            if response.type == "final_response":
                execution.response = response.content
                execution.state = "completed"
                execution.iterations += 1
                break

            elif response.type == "tool_call":
                tool_call = response.tool_call
                tool = self._find_tool(tool_call.tool_name)

                # BEFORE_TOOL_CALL
                before_tool_response = await self.hooks.trigger(
                    "before_tool_call",
                    BeforeToolCallEventData(
                        execution=execution,
                        tool_name=tool_call.tool_name,
                        arguments=tool_call.arguments,
                        tool_index=len(execution.tool_calls),
                        iteration=execution.iterations,
                    ),
                )

                # Check for skip action (e.g., cached result)
                if (
                    before_tool_response
                    and before_tool_response.action == "skip"
                    and before_tool_response.cached_result is not None
                ):
                    result = before_tool_response.cached_result
                    tool_call.result = result
                else:
                    # Execute tool (with retry loop)
                    tool_attempt = 0
                    result = None
                    tool_error = None
                    max_attempts = 3

                    while tool_attempt < max_attempts:
                        tool_start = time.time()
                        tool_attempt += 1

                        try:
                            validated = tool.input_model(**tool_call.arguments)
                            result = await tool.execute(**validated.model_dump())
                            tool_call.result = result
                            tool_error = None
                            break

                        except ValidationError as e:
                            tool_error = str(e)
                            result = f"Validation error: {e}"
                            tool_call.error = tool_error

                        except ToolExecutionError as e:
                            tool_error = str(e)
                            result = f"Tool error: {e}"
                            tool_call.error = tool_error

                        except Exception as e:
                            tool_error = str(e)
                            result = f"Error: {e}"
                            tool_call.error = tool_error

                        # ON_TOOL_ERROR - ask hooks if we should retry
                        if tool_error:
                            hook_response = await self.hooks.trigger(
                                "on_tool_error",
                                OnToolErrorEventData(
                                    execution=execution,
                                    tool_name=tool_call.tool_name,
                                    arguments=tool_call.arguments,
                                    error=Exception(tool_error),
                                    error_message=tool_error,
                                    attempt=tool_attempt,
                                ),
                            )

                            if hook_response and hook_response.action == "retry":
                                # Delay if specified
                                if hook_response.delay_ms:
                                    await asyncio.sleep(hook_response.delay_ms / 1000)
                                # Update arguments if provided
                                if hook_response.arguments:
                                    tool_call.arguments = hook_response.arguments
                                continue  # Retry
                            else:
                                # Don't retry, use error result
                                break

                tool_time = (time.time() - iteration_start) * 1000

                # AFTER_TOOL_CALL
                await self.hooks.trigger(
                    "after_tool_call",
                    AfterToolCallEventData(
                        execution=execution,
                        tool_call=tool_call,
                        tool_name=tool_call.tool_name,
                        result=result or "",
                        execution_time_ms=tool_time,
                    ),
                )

                execution.tool_calls.append(tool_call)
                execution.messages.append(
                    Message(
                        role="tool",
                        content=result or "",
                        tool_call_id=tool_call.id,
                    )
                )

            execution.iterations += 1

            # AFTER_ITERATION
            iteration_time = (time.time() - iteration_start) * 1000
            await self.hooks.trigger(
                "after_iteration",
                AfterIterationEventData(
                    execution=execution,
                    iteration_count=execution.iterations,
                    elapsed_time_ms=iteration_time,
                ),
            )

        if execution.iterations >= self.max_iterations:
            execution.state = "max_iterations"

        # AFTER_RUN
        total_time = (time.time() - start_time) * 1000
        await self.hooks.trigger(
            "after_run",
            AfterRunEventData(execution=execution, total_time_ms=total_time),
        )

        return execution

    def _find_tool(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ToolNotFound(f"Tool '{name}' not found")
