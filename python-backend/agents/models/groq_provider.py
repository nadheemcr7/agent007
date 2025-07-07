from __future__ import annotations

import httpx
import os
from groq import Groq, AsyncGroq
from .interface import Model, ModelProvider
from .openai_chatcompletions import OpenAIChatCompletionsModel
# Keeping for structural consistency, though not directly used by Groq in this setup
from .openai_responses import OpenAIResponsesModel 

DEFAULT_MODEL: str = "llama3-8b-8192" # Default Groq model

_http_client: httpx.AsyncClient | None = None

def shared_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient() 
    return _http_client

class GroqChatCompletionsModel(OpenAIChatCompletionsModel):
    """
    A model class that adapts the generic chat completions model to Groq's API.
    It inherits from OpenAIChatCompletionsModel and simply passes the Groq client.
    The necessary parameter adjustments for Groq compatibility are handled
    within the OpenAIChatCompletionsModel's _fetch_response method.
    """
    def __init__(self, model: str, groq_client: AsyncGroq):
        # Pass the Groq client as the openai_client.
        super().__init__(model=model, openai_client=groq_client)

class GroqProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        groq_client: AsyncGroq | None = None,
    ) -> None:
        """Create a new Groq provider.
                Args:
            api_key: The API key to use for the Groq client. If not provided, we will use the
                default API key from the environment variable (GROQ_API_KEY).
            base_url: The base URL to use for the Groq client.
            groq_client: An optional Groq client to use.
        """
        if groq_client is not None:
            assert api_key is None and base_url is None, (
                "Don't provide api_key or base_url if you provide groq_client"
            )
            self._client: AsyncGroq | None = groq_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            # Get API key from environment if not provided explicitly
            api_key_to_use = self._stored_api_key or os.getenv("GROQ_API_KEY")
            if not api_key_to_use:
                raise ValueError(
                    "GROQ_API_KEY must be set either by passing api_key to the GroqProvider "
                    "or by setting the GROQ_API_KEY environment variable."
                )
            self._client = AsyncGroq(
                api_key=api_key_to_use,
                base_url=self._stored_base_url,
                http_client=shared_http_client(),
            )
        return self._client

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        client = self._get_client()
        return GroqChatCompletionsModel(model=model_name, groq_client=client)