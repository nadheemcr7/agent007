"""Stream handler for chat completion responses."""

import json
import logging
from typing import AsyncIterator, Any, Dict
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import Response

logger = logging.getLogger(__name__)

class StreamEvent:
    """Base class for stream events."""
    def __init__(self, event_type: str, data: Any = None):
        self.type = event_type
        self.data = data

class ResponseCompletedEvent(StreamEvent):
    """Event indicating response completion."""
    def __init__(self, response: Response):
        super().__init__("response.completed")
        self.response = response

class ChatCmplStreamHandler:
    """Handler for streaming chat completion responses."""
    
    @staticmethod
    async def handle_stream(
        response: Response, 
        stream: AsyncStream[ChatCompletionChunk]
    ) -> AsyncIterator[StreamEvent]:
        """Handle streaming response and yield events."""
        
        accumulated_content = ""
        accumulated_tool_calls = []
        
        try:
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle content
                    if delta.content:
                        accumulated_content += delta.content
                        yield StreamEvent("content.delta", {"content": delta.content})
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.function:
                                accumulated_tool_calls.append({
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                })
                                yield StreamEvent("tool_call.delta", {"tool_call": tool_call})
                    
                    # Check for finish reason
                    if choice.finish_reason:
                        yield StreamEvent("choice.finished", {"finish_reason": choice.finish_reason})
            
            # Create final response
            final_response = Response(
                id=response.id,
                created_at=response.created_at,
                model=response.model,
                object="response",
                output=[],
                tool_choice=response.tool_choice,
                top_p=response.top_p,
                temperature=response.temperature,
                tools=response.tools,
                parallel_tool_calls=response.parallel_tool_calls,
                reasoning=response.reasoning,
            )
            
            # Add usage information if available
            if hasattr(stream, 'usage') and stream.usage:
                final_response.usage = stream.usage
            
            yield ResponseCompletedEvent(final_response)
            
        except Exception as e:
            logger.error(f"Error in stream handler: {e}")
            yield StreamEvent("error", {"error": str(e)})