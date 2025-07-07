"""Helper utilities for chat completion models."""

from typing import Dict, Any, Optional
from openai import AsyncOpenAI

# Default headers for API requests
HEADERS = {
    "User-Agent": "airline-agent/1.0"
}

class ChatCmplHelpers:
    @staticmethod
    def get_store_param(client: AsyncOpenAI, model_settings: Any) -> Optional[str]:
        """Get store parameter for the request."""
        # Return None for now as this is provider-specific
        return None
    
    @staticmethod
    def get_stream_options_param(client: AsyncOpenAI, model_settings: Any, stream: bool = False) -> Optional[Dict[str, Any]]:
        """Get stream options parameter for the request."""
        if stream:
            return {"include_usage": True}
        return None