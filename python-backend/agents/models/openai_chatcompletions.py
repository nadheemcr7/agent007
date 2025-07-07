from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from .. import _debug
from ..agent_output import AgentOutputSchemaBase
from ..exceptions import AgentsException, UserError
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..logger import logger
from ..tool import Tool
from ..tracing import generation_span
from ..tracing.span_data import GenerationSpanData
from ..tracing.spans import Span
from ..usage import Usage
from .chatcmpl_converter import Converter
from .chatcmpl_helpers import HEADERS, ChatCmplHelpers
from .chatcmpl_stream_handler import ChatCmplStreamHandler
from .fake_id import FAKE_RESPONSES_ID
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


class OpenAIChatCompletionsModel(Model):
    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client

    def _convert_not_given_to_none(self, value: Any) -> Any:
        """Converts NOT_GIVEN sentinel to None, otherwise returns the value."""
        return None if value is NOT_GIVEN else value

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict() | {"base_url": str(self._client.base_url)},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
                prompt=prompt,
            )

            first_choice = response.choices[0]
            message = first_choice.message

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                if message is not None:
                    logger.debug(
                        "LLM resp:\n%s\n",
                        json.dumps(message.model_dump(), indent=2),
                    )
                else:
                    logger.debug(
                        "LLM resp had no message. finish_reason: %s",
                        first_choice.finish_reason,
                    )

            # Safe access to usage details for Groq compatibility
            input_tokens_details_cached = 0
            if hasattr(response.usage, 'prompt_tokens_details') and \
               getattr(response.usage, 'prompt_tokens_details') is not None:
                input_tokens_details_cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)

            output_tokens_details_reasoning = 0
            if hasattr(response.usage, 'completion_tokens_details') and \
               getattr(response.usage, 'completion_tokens_details') is not None:
                output_tokens_details_reasoning = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0)

            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=input_tokens_details_cached
                    ),
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=output_tokens_details_reasoning
                    ),
                )
                if response.usage
                else Usage()
            )

            if tracing.include_data():
                span_generation.span_data.output = (
                    [message.model_dump()] if message is not None else []
                )
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            items = Converter.message_to_output_items(message) if message is not None else []

            return ModelResponse(
                output=items,
                usage=usage,
                response_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """
        Yields a partial message as it is generated, as well as the usage information.
        """
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict() | {"base_url": str(self._client.base_url)},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response_tuple = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=True,
                prompt=prompt,
            )

            if not isinstance(response_tuple, tuple) or len(response_tuple) != 2:
                logger.error(f"Expected tuple[Response, AsyncStream] for streaming, but got type: {type(response_tuple)}")
                raise TypeError(f"Expected tuple[Response, AsyncStream] for streaming call, but received {type(response_tuple)}")
            
            final_sdk_response, stream = response_tuple

            final_response: Response | None = None
            async for chunk in ChatCmplStreamHandler.handle_stream(final_sdk_response, stream):
                yield chunk

                if chunk.type == "response.completed":
                    final_response = chunk.response

            if tracing.include_data() and final_response:
                span_generation.span_data.output = [final_response.model_dump()]

            if final_response and final_response.usage:
                span_generation.span_data.usage = {
                    "input_tokens": final_response.usage.input_tokens,
                    "output_tokens": final_response.usage.output_tokens,
                }

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
        prompt: ResponsePromptParam | None = None,
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False],
        prompt: ResponsePromptParam | None = None,
    ) -> ChatCompletion: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: ResponsePromptParam | None = None,
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        converted_messages = Converter.items_to_messages(input)

        # Add JSON instruction for structured output
        original_system_instructions = system_instructions
        if output_schema:
            json_instruction = "Your response should be in JSON format."
            if original_system_instructions:
                if "json" not in original_system_instructions.lower():
                    system_instructions = f"{original_system_instructions}\n\n{json_instruction}"
            else:
                system_instructions = json_instruction

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        if tracing.include_data():
            span.span_data.input = converted_messages

        _g = self._convert_not_given_to_none

        # Handle parallel tool calls for Groq compatibility
        original_parallel_tool_calls_value = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else NOT_GIVEN
        )
        parallel_tool_calls_param = _g(original_parallel_tool_calls_value)

        # Handle tool_choice for Groq compatibility
        original_tool_choice_param = Converter.convert_tool_choice(model_settings.tool_choice)
        groq_tool_choice = "auto"

        if isinstance(original_tool_choice_param, str):
            if original_tool_choice_param in ["none", "auto", "required"]:
                groq_tool_choice = original_tool_choice_param
        elif isinstance(original_tool_choice_param, dict):
            groq_tool_choice = "auto"

        # Handle response format for Groq compatibility
        response_format_param = NOT_GIVEN
        if output_schema:
            response_format_param = {"type": "json_object"}
        else:
            response_format_param = _g(Converter.convert_response_format(output_schema))

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                f"{json.dumps(converted_messages, indent=2)}\n"
                f"Tools:\n{json.dumps(converted_tools, indent=2)}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {groq_tool_choice}\n"
                f"Response format: {response_format_param}\n"
            )

        # Apply parameter conversion
        reasoning_effort_param = _g(model_settings.reasoning.effort if model_settings.reasoning else None)
        store_param = _g(ChatCmplHelpers.get_store_param(self._get_client(), model_settings))

        # Build parameters dictionary
        params = {
            "model": self.model,
            "messages": converted_messages,
            "tools": _g(converted_tools or NOT_GIVEN),
            "temperature": _g(model_settings.temperature),
            "top_p": _g(model_settings.top_p),
            "frequency_penalty": _g(model_settings.frequency_penalty),
            "presence_penalty": _g(model_settings.presence_penalty),
            "max_tokens": _g(model_settings.max_tokens),
            "tool_choice": _g(groq_tool_choice),
            "response_format": _g(response_format_param),
            "parallel_tool_calls": parallel_tool_calls_param,
            "stream": stream,
            "store": store_param,
            "reasoning_effort": reasoning_effort_param,
            "extra_headers": {**HEADERS, **(_g(model_settings.extra_headers) or {})},
            "extra_query": _g(model_settings.extra_query),
            "extra_body": _g(model_settings.extra_body),
            "metadata": _g(model_settings.metadata),
            **(model_settings.extra_args or {}),
        }

        # Filter out None parameters
        final_params = {k: v for k, v in params.items() if v is not None or k in ["messages", "model", "stream"]}
        if final_params.get("tools") == []:
            del final_params["tools"]
        
        try:
            ret = await self._get_client().chat.completions.create(**final_params)
        except Exception as e:
            logger.error(f"Failed to call chat.completions.create: {e}")
            raise

        if stream:
            response = Response(
                id=FAKE_RESPONSES_ID,
                created_at=time.time(),
                model=self.model,
                object="response",
                tool_choice=cast(Literal["auto", "required", "none"], original_tool_choice_param)
                if original_tool_choice_param is not NOT_GIVEN
                else "auto",
                output=[],
                top_p=model_settings.top_p,
                temperature=model_settings.temperature,
                tools=[],
                parallel_tool_calls=original_parallel_tool_calls_value or False,
                reasoning=model_settings.reasoning,
            )
            return response, ret
        else:
            return ret

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ValueError("OpenAIChatCompletionsModel was not initialized with an OpenAI/Groq client.")
        return self._client