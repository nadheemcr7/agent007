# from __future__ import annotations

# import json
# import time
# from collections.abc import AsyncIterator
# from typing import TYPE_CHECKING, Any, Literal, cast, overload

# from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
# from openai.types import ChatModel
# from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
# from openai.types.chat.chat_completion import Choice
# from openai.types.responses import Response
# from openai.types.responses.response_prompt_param import ResponsePromptParam
# from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

# from .. import _debug
# from ..agent_output import AgentOutputSchemaBase
# from ..handoffs import Handoff
# from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
# from ..logger import logger
# from ..tool import Tool
# from ..tracing import generation_span
# from ..tracing.span_data import GenerationSpanData
# from ..tracing.spans import Span
# from ..usage import Usage
# from .chatcmpl_converter import Converter
# from .chatcmpl_helpers import HEADERS, ChatCmplHelpers
# from .chatcmpl_stream_handler import ChatCmplStreamHandler
# from .fake_id import FAKE_RESPONSES_ID
# from .interface import Model, ModelTracing

# if TYPE_CHECKING:
#     from ..model_settings import ModelSettings


# class OpenAIChatCompletionsModel(Model):
#     def __init__(
#         self,
#         model: str | ChatModel,
#         openai_client: AsyncOpenAI,
#     ) -> None:
#         self.model = model
#         self._client = openai_client

#     def _non_null_or_not_given(self, value: Any) -> Any:
#         return value if value is not None else NOT_GIVEN

#     async def get_response(
#         self,
#         system_instructions: str | None,
#         input: str | list[TResponseInputItem],
#         model_settings: ModelSettings,
#         tools: list[Tool],
#         output_schema: AgentOutputSchemaBase | None,
#         handoffs: list[Handoff],
#         tracing: ModelTracing,
#         previous_response_id: str | None,
#         prompt: ResponsePromptParam | None = None,
#     ) -> ModelResponse:
#         with generation_span(
#             model=str(self.model),
#             model_config=model_settings.to_json_dict() | {"base_url": str(self._client.base_url)},
#             disabled=tracing.is_disabled(),
#         ) as span_generation:
#             response = await self._fetch_response(
#                 system_instructions,
#                 input,
#                 model_settings,
#                 tools,
#                 output_schema,
#                 handoffs,
#                 span_generation,
#                 tracing,
#                 stream=False,
#                 prompt=prompt,
#             )

#             message: ChatCompletionMessage | None = None
#             first_choice: Choice | None = None
#             if response.choices and len(response.choices) > 0:
#                 first_choice = response.choices[0]
#                 message = first_choice.message

#             if _debug.DONT_LOG_MODEL_DATA:
#                 logger.debug("Received model response")
#             else:
#                 if message is not None:
#                     logger.debug(
#                         "LLM resp:\n%s\n",
#                         json.dumps(message.model_dump(), indent=2, ensure_ascii=False),
#                     )
#                 else:
#                     finish_reason = first_choice.finish_reason if first_choice else "-"
#                     logger.debug(f"LLM resp had no message. finish_reason: {finish_reason}")

#             usage = (
#                 Usage(
#                     requests=1,
#                     input_tokens=response.usage.prompt_tokens,
#                     output_tokens=response.usage.completion_tokens,
#                     total_tokens=response.usage.total_tokens,
#                     input_tokens_details=InputTokensDetails(
#                         cached_tokens=getattr(
#                             response.usage.prompt_tokens_details, "cached_tokens", 0
#                         )
#                         or 0,
#                     ),
#                     output_tokens_details=OutputTokensDetails(
#                         reasoning_tokens=getattr(
#                             response.usage.completion_tokens_details, "reasoning_tokens", 0
#                         )
#                         or 0,
#                     ),
#                 )
#                 if response.usage
#                 else Usage()
#             )
#             if tracing.include_data():
#                 span_generation.span_data.output = (
#                     [message.model_dump()] if message is not None else []
#                 )
#             span_generation.span_data.usage = {
#                 "input_tokens": usage.input_tokens,
#                 "output_tokens": usage.output_tokens,
#             }

#             items = Converter.message_to_output_items(message) if message is not None else []

#             return ModelResponse(
#                 output=items,
#                 usage=usage,
#                 response_id=None,
#             )

#     async def stream_response(
#         self,
#         system_instructions: str | None,
#         input: str | list[TResponseInputItem],
#         model_settings: ModelSettings,
#         tools: list[Tool],
#         output_schema: AgentOutputSchemaBase | None,
#         handoffs: list[Handoff],
#         tracing: ModelTracing,
#         previous_response_id: str | None,
#         prompt: ResponsePromptParam | None = None,
#     ) -> AsyncIterator[TResponseStreamEvent]:
#         """
#         Yields a partial message as it is generated, as well as the usage information.
#         """
#         with generation_span(
#             model=str(self.model),
#             model_config=model_settings.to_json_dict() | {"base_url": str(self._client.base_url)},
#             disabled=tracing.is_disabled(),
#         ) as span_generation:
#             response, stream = await self._fetch_response(
#                 system_instructions,
#                 input,
#                 model_settings,
#                 tools,
#                 output_schema,
#                 handoffs,
#                 span_generation,
#                 tracing,
#                 stream=True,
#                 prompt=prompt,
#             )

#             final_response: Response | None = None
#             async for chunk in ChatCmplStreamHandler.handle_stream(response, stream):
#                 yield chunk

#                 if chunk.type == "response.completed":
#                     final_response = chunk.response

#             if tracing.include_data() and final_response:
#                 span_generation.span_data.output = [final_response.model_dump()]

#             if final_response and final_response.usage:
#                 span_generation.span_data.usage = {
#                     "input_tokens": final_response.usage.input_tokens,
#                     "output_tokens": final_response.usage.output_tokens,
#                 }

#     @overload
#     async def _fetch_response(
#         self,
#         system_instructions: str | None,
#         input: str | list[TResponseInputItem],
#         model_settings: ModelSettings,
#         tools: list[Tool],
#         output_schema: AgentOutputSchemaBase | None,
#         handoffs: list[Handoff],
#         span: Span[GenerationSpanData],
#         tracing: ModelTracing,
#         stream: Literal[True],
#         prompt: ResponsePromptParam | None = None,
#     ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

#     @overload
#     async def _fetch_response(
#         self,
#         system_instructions: str | None,
#         input: str | list[TResponseInputItem],
#         model_settings: ModelSettings,
#         tools: list[Tool],
#         output_schema: AgentOutputSchemaBase | None,
#         handoffs: list[Handoff],
#         span: Span[GenerationSpanData],
#         tracing: ModelTracing,
#         stream: Literal[False],
#         prompt: ResponsePromptParam | None = None,
#     ) -> ChatCompletion: ...

#     async def _fetch_response(
#         self,
#         system_instructions: str | None,
#         input: str | list[TResponseInputItem],
#         model_settings: ModelSettings,
#         tools: list[Tool],
#         output_schema: AgentOutputSchemaBase | None,
#         handoffs: list[Handoff],
#         span: Span[GenerationSpanData],
#         tracing: ModelTracing,
#         stream: bool = False,
#         prompt: ResponsePromptParam | None = None,
#     ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
#         converted_messages = Converter.items_to_messages(input)

#         if system_instructions:
#             converted_messages.insert(
#                 0,
#                 {
#                     "content": system_instructions,
#                     "role": "system",
#                 },
#             )
#         if tracing.include_data():
#             span.span_data.input = converted_messages

#         parallel_tool_calls = (
#             True
#             if model_settings.parallel_tool_calls and tools and len(tools) > 0
#             else False
#             if model_settings.parallel_tool_calls is False
#             else NOT_GIVEN
#         )
#         tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
#         response_format = Converter.convert_response_format(output_schema)

#         converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

#         for handoff in handoffs:
#             converted_tools.append(Converter.convert_handoff_tool(handoff))

#         if _debug.DONT_LOG_MODEL_DATA:
#             logger.debug("Calling LLM")
#         else:
#             logger.debug(
#                 f"{json.dumps(converted_messages, indent=2, ensure_ascii=False)}\n"
#                 f"Tools:\n{json.dumps(converted_tools, indent=2, ensure_ascii=False)}\n"
#                 f"Stream: {stream}\n"
#                 f"Tool choice: {tool_choice}\n"
#                 f"Response format: {response_format}\n"
#             )

#         reasoning_effort = model_settings.reasoning.effort if model_settings.reasoning else None
#         store = ChatCmplHelpers.get_store_param(self._get_client(), model_settings)

#         stream_options = ChatCmplHelpers.get_stream_options_param(
#             self._get_client(), model_settings, stream=stream
#         )

#         ret = await self._get_client().chat.completions.create(
#             model=self.model,
#             messages=converted_messages,
#             tools=converted_tools or NOT_GIVEN,
#             temperature=self._non_null_or_not_given(model_settings.temperature),
#             top_p=self._non_null_or_not_given(model_settings.top_p),
#             frequency_penalty=self._non_null_or_not_given(model_settings.frequency_penalty),
#             presence_penalty=self._non_null_or_not_given(model_settings.presence_penalty),
#             max_tokens=self._non_null_or_not_given(model_settings.max_tokens),
#             tool_choice=tool_choice,
#             response_format=response_format,
#             parallel_tool_calls=parallel_tool_calls,
#             stream=stream,
#             stream_options=self._non_null_or_not_given(stream_options),
#             store=self._non_null_or_not_given(store),
#             reasoning_effort=self._non_null_or_not_given(reasoning_effort),
#             extra_headers={**HEADERS, **(model_settings.extra_headers or {})},
#             extra_query=model_settings.extra_query,
#             extra_body=model_settings.extra_body,
#             metadata=self._non_null_or_not_given(model_settings.metadata),
#             **(model_settings.extra_args or {}),
#         )

#         if isinstance(ret, ChatCompletion):
#             return ret

#         response = Response(
#             id=FAKE_RESPONSES_ID,
#             created_at=time.time(),
#             model=self.model,
#             object="response",
#             output=[],
#             tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
#             if tool_choice != NOT_GIVEN
#             else "auto",
#             top_p=model_settings.top_p,
#             temperature=model_settings.temperature,
#             tools=[],
#             parallel_tool_calls=parallel_tool_calls or False,
#             reasoning=model_settings.reasoning,
#         )
#         return response, ret

#     def _get_client(self) -> AsyncOpenAI:
#         if self._client is None:
#             self._client = AsyncOpenAI()
#         return self._client


















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
        openai_client: AsyncOpenAI, # This is now either an OpenAI or Groq client
    ) -> None:
        self.model = model
        self._client = openai_client

    # IMPORTANT: This helper converts NOT_GIVEN to None.
    # This is crucial because Groq's underlying httpx client cannot serialize NOT_GIVEN.
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
            # Call _fetch_response with stream=False, expecting ChatCompletion
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

            # Groq's CompletionUsage object does not have prompt_tokens_details or completion_tokens_details
            # Safely get these or default to 0.
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
            # Call _fetch_response with stream=True, expecting tuple
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

            # Ensure response_tuple is indeed a tuple of expected types
            if not isinstance(response_tuple, tuple) or len(response_tuple) != 2 or \
               not isinstance(response_tuple[0], Response) or \
               not isinstance(response_tuple[1], AsyncStream):
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

            # Apply the same safe access for usage details in streaming
            if final_response and final_response.usage:
                input_tokens_details_cached = 0
                if hasattr(final_response.usage, 'prompt_tokens_details') and \
                   getattr(final_response.usage, 'prompt_tokens_details') is not None:
                    input_tokens_details_cached = getattr(final_response.usage.prompt_tokens_details, "cached_tokens", 0)

                output_tokens_details_reasoning = 0
                if hasattr(final_response.usage, 'completion_tokens_details') and \
                   getattr(final_response.usage, 'completion_tokens_details') is not None:
                    output_tokens_details_reasoning = getattr(final_response.usage.completion_tokens_details, "reasoning_tokens", 0)

                span_generation.span_data.usage = {
                    "input_tokens": final_response.usage.input_tokens,
                    "output_tokens": final_response.usage.output_tokens,
                    "input_tokens_details": {"cached_tokens": input_tokens_details_cached},
                    "output_tokens_details": {"reasoning_tokens": output_tokens_details_reasoning},
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

        # --- FIX: Inject "json" keyword into system_instructions if output_schema is present ---
        original_system_instructions = system_instructions # Keep original for later reference if needed
        if output_schema:
            json_instruction = "Your response should be in JSON format."
            if original_system_instructions:
                if "json" not in original_system_instructions.lower():
                    system_instructions = f"{original_system_instructions}\n\n{json_instruction}"
            else:
                system_instructions = json_instruction
            logger.debug(f"Adjusted system instructions for JSON output: {system_instructions}")
        # --- END FIX ---

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

        # Use the new helper to ensure parameters are None instead of NOT_GIVEN
        _g = self._convert_not_given_to_none

        # For Groq, parallel_tool_calls is not directly configurable via this param
        # The tool_choice param implicitly controls parallel execution in some APIs
        # Setting to None will exclude it from the request if it's NOT_GIVEN or False
        # which is safer for Groq.
        original_parallel_tool_calls_value = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else NOT_GIVEN
        )
        parallel_tool_calls_param = _g(original_parallel_tool_calls_value)


        # Handle tool_choice specifically for Groq's API
        # Groq only accepts 'none', 'auto', or 'required' as string values.
        # Store the original `tool_choice` for the Response object construction later
        original_tool_choice_param = Converter.convert_tool_choice(model_settings.tool_choice)
        groq_tool_choice = "auto" # Default to 'auto' if not explicitly handled

        if isinstance(original_tool_choice_param, str):
            if original_tool_choice_param in ["none", "auto", "required"]:
                groq_tool_choice = original_tool_choice_param
            elif original_tool_choice_param == "any":
                groq_tool_choice = "auto"
        elif isinstance(original_tool_choice_param, dict):
            # If it's a dict (e.g., to force a specific tool),
            # Groq's error suggests it doesn't accept this.
            # We'll default to "auto" to prevent immediate errors,
            # acknowledging that specific tool forcing might behave differently.
            groq_tool_choice = "auto"
        # If original_tool_choice_param was NOT_GIVEN or None, groq_tool_choice remains "auto"
        # and _g will convert it to None before passing if None is the final value.

        # --- FIX FOR JSON SCHEMA ERROR (revisited) ---
        # Groq models currently do not support the 'json_schema' response format directly.
        # However, they often support 'json_object' to strongly instruct the model to produce JSON.
        # If an output_schema is provided, we now explicitly set response_format to 'json_object'.
        response_format_param = NOT_GIVEN # Default to NOT_GIVEN
        if output_schema:
            # When an output_schema is provided, the SDK expects a structured JSON response.
            # Groq models generally do not support the specific 'json_schema' format,
            # but they do support 'json_object' which strongly encourages JSON output.
            # We explicitly set this to guide the model.
            response_format_param = {"type": "json_object"}
            # The 'json' keyword has been added to system_instructions above, so no need to log here.
        else:
            # If no output_schema, use the default conversion, but ensure it's None if it was NOT_GIVEN
            response_format_param = _g(Converter.convert_response_format(output_schema))
        # --- END FIX ---


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
                f"Tool choice: {groq_tool_choice}\n" # Log the Groq-compatible value
                f"Response format: {response_format_param}\n" # Log the new parameter
            )

        # Apply _g (convert_not_given_to_none) to all potentially NOT_GIVEN parameters
        reasoning_effort_param = _g(model_settings.reasoning.effort if model_settings.reasoning else None)
        store_param = _g(ChatCmplHelpers.get_store_param(self._get_client(), model_settings))
        
        # Removed stream_options as it's causing TypeError with Groq client
        # stream_options = ChatCmplHelpers.get_stream_options_param(
        #     self._get_client(), model_settings, stream=stream
        # )

        # Build parameters dictionary
        params = {
            "model": self.model,
            "messages": converted_messages,
            # Apply _g to each parameter that might be NOT_GIVEN
            "tools": _g(converted_tools or NOT_GIVEN), # tools can be NOT_GIVEN if empty
            "temperature": _g(model_settings.temperature),
            "top_p": _g(model_settings.top_p),
            "frequency_penalty": _g(model_settings.frequency_penalty),
            "presence_penalty": _g(model_settings.presence_penalty),
            "max_tokens": _g(model_settings.max_tokens),
            "tool_choice": _g(groq_tool_choice), # Use the Groq-compatible tool_choice
            "response_format": _g(response_format_param), # Use the new parameter
            "parallel_tool_calls": parallel_tool_calls_param, # Use the handled parallel_tool_calls_param
            "stream": stream,
            "store": store_param,
            "reasoning_effort": reasoning_effort_param,
            "extra_headers": {**HEADERS, **(_g(model_settings.extra_headers) or {})},
            "extra_query": _g(model_settings.extra_query),
            "extra_body": _g(model_settings.extra_body),
            "metadata": _g(model_settings.metadata),
            **(model_settings.extra_args or {}),
        }

        # Filter out parameters that are None IF the API consistently rejects them.
        # For Groq, it's safer to explicitly remove keys that would send None,
        # especially for parameters that are expected to be present with a valid type.
        final_params = {k: v for k, v in params.items() if v is not None or k in ["messages", "model", "stream"]}
        # Ensure 'tools' is not present if it's an empty list (equivalent to NOT_GIVEN)
        if final_params.get("tools") == []:
            del final_params["tools"]
        
        try: # ADDED TRY-EXCEPT BLOCK HERE
            print(f"DEBUG: openai_chatcompletions - Calling chat.completions.create with params: {json.dumps(final_params, indent=2)}")
            ret = await self._get_client().chat.completions.create(**final_params)
            print("DEBUG: openai_chatcompletions - chat.completions.create call successful.")
        except Exception as e:
            print(f"ERROR: openai_chatcompletions - Failed to call chat.completions.create: {e}")
            raise # Re-raise the exception after logging

        if stream:
            # If streaming, 'ret' is AsyncStream[ChatCompletionChunk].
            # Construct the internal SDK Response object and return as a tuple.
            response = Response(
                id=FAKE_RESPONSES_ID,
                created_at=time.time(),
                model=self.model,
                object="response",
                tool_choice=cast(Literal["auto", "required", "none"], original_tool_choice_param)
                if original_tool_choice_param is not NOT_GIVEN
                else "auto",
                output=[], # output is usually set later by the stream handler
                top_p=model_settings.top_p,
                temperature=model_settings.temperature,
                tools=[], # tools are usually set later
                parallel_tool_calls=original_parallel_tool_calls_value or False, # FIX: Used correct variable here
                reasoning=model_settings.reasoning,
            )
            return response, ret
        else:
            # If not streaming, 'ret' should be ChatCompletion (from OpenAI or Groq).
            # The type hints are just for static analysis; at runtime, we expect
            # an object with .choices attribute.
            return ret


    def _get_client(self) -> AsyncOpenAI:
        # This method should ideally not be self._client = None
        # It should return the already initialized self._client
        # The initialization happens in GroqProvider's _get_client
        # This method is just a getter for the client instance.
        if self._client is None:
            # This case should ideally not be hit if GroqChatCompletionsModel is properly initialized
            # with a client. If it is, it means there's a deeper issue in client passing.
            raise ValueError("OpenAIChatCompletionsModel was not initialized with an OpenAI/Groq client.")
        return self._client