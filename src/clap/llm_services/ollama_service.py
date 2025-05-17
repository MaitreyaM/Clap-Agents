import os
import json
import uuid
from typing import Any, Dict, List, Optional

_OPENAI_LIB_AVAILABLE = False
_AsyncOpenAI_Placeholder_Type = Any
_OpenAIError_Placeholder_Type = type(Exception)

try:
    from openai import AsyncOpenAI as ImportedAsyncOpenAI, OpenAIError as ImportedOpenAIError
    _AsyncOpenAI_Placeholder_Type = ImportedAsyncOpenAI
    _OpenAIError_Placeholder_Type = ImportedOpenAIError
    _OPENAI_LIB_AVAILABLE = True
except ImportError:
    pass

from colorama import Fore
from .base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall

OLLAMA_OPENAI_COMPAT_BASE_URL = "http://localhost:11434/v1"

class OllamaOpenAICompatService(LLMServiceInterface): 
    """
    LLM Service implementation using the OpenAI SDK configured for a
    local Ollama instance's OpenAI-compatible API.
    """
    _client: _AsyncOpenAI_Placeholder_Type

    def __init__(
        self,
        base_url: str = OLLAMA_OPENAI_COMPAT_BASE_URL,
        api_key: str = "ollama", # Dummy
        default_model: Optional[str] = None
    ):

        """
        Initializes the service using the OpenAI client pointed at Ollama.

        Args:
            base_url: The base URL for the Ollama OpenAI compatibility endpoint.
            api_key: Dummy API key for the OpenAI client (Ollama ignores it).
            default_model: Optional default Ollama model name to use if not specified in calls.
        """
        if not _OPENAI_LIB_AVAILABLE:
            raise ImportError(
                "The 'openai' Python library is required to use OllamaOpenAICompatService. "
                "Install with 'pip install openai' or 'pip install \"clap-agents[ollama]\"' (if ollama extra includes openai)."
            )
        self.default_model = default_model
        try:
            self._client = _AsyncOpenAI_Placeholder_Type(base_url=base_url, api_key=api_key)
            # print(f"OllamaService: Initialized OpenAI client for Ollama at {base_url}")
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize OpenAI client for Ollama: {e}{Fore.RESET}"); raise

    async def get_llm_response(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: str = "auto", temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> StandardizedLLMResponse:
        """
        Sends messages via the OpenAI SDK (to Ollama's endpoint)
        and returns a standardized response.
        """
        if not _OPENAI_LIB_AVAILABLE: raise RuntimeError("OpenAI library not available for Ollama service.")
        request_model = model or self.default_model
        if not request_model: raise ValueError("Ollama model name not specified.")
        try:
            api_kwargs: Dict[str, Any] = {"messages": messages, "model": request_model}
            if tools and tool_choice != "none":
                api_kwargs["tools"] = tools
                if isinstance(tool_choice, dict) or tool_choice in ["auto", "required", "none"]: api_kwargs["tool_choice"] = tool_choice
            else: api_kwargs["tools"] = None; api_kwargs["tool_choice"] = None
            if temperature is not None: api_kwargs["temperature"] = temperature
            if max_tokens is not None: api_kwargs["max_tokens"] = max_tokens
            api_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}
            # print(f"OllamaService: Sending request to model '{request_model}'")
            response = await self._client.chat.completions.create(**api_kwargs)
            message = response.choices[0].message
            text_content = message.content
            tool_calls_std: List[LLMToolCall] = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc.id and tc.function and tc.function.name and tc.function.arguments is not None:
                        tool_calls_std.append(LLMToolCall(id=tc.id, function_name=tc.function.name, function_arguments_json_str=tc.function.arguments))
                    else: print(f"{Fore.YELLOW}Warning: Incomplete tool_call from Ollama: {tc}{Fore.RESET}")
            return StandardizedLLMResponse(text_content=text_content, tool_calls=tool_calls_std)
        except _OpenAIError_Placeholder_Type as e: # Use placeholder
            err_msg = f"Ollama (OpenAI Compat) API Error: {e}"
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'text'): err_msg += f" - Details: {e.response.text}"
            print(f"{Fore.RED}{err_msg}{Fore.RESET}")
            return StandardizedLLMResponse(text_content=err_msg)
        except Exception as e:
            print(f"{Fore.RED}Unexpected error with Ollama (OpenAI Compat): {e}{Fore.RESET}")
            return StandardizedLLMResponse(text_content=f"Ollama Unexpected Error: {e}")


    async def close(self):
        if _OPENAI_LIB_AVAILABLE and hasattr(self, '_client') and self._client:
            if hasattr(self._client, "close"): await self._client.close() # For openai >1.0
            elif hasattr(self._client, "_client") and hasattr(self._client._client, "is_closed"): # For httpx client in openai <1.0
                 if not self._client._client.is_closed: await self._client._client.aclose() # type: ignore
        # print("OllamaService: Client closed.")
# --- END OF FILE ---