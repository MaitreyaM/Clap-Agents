import os
import json
import uuid 
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError:
    raise ImportError("OpenAI SDK not found. Install with: pip install openai")

from colorama import Fore

from .base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall

OLLAMA_OPENAI_COMPAT_BASE_URL = "http://localhost:11434/v1" 

class OllamaOpenAICompatService(LLMServiceInterface):
    """
    LLM Service implementation using the OpenAI SDK configured for a
    local Ollama instance's OpenAI-compatible API.
    """
    _client: AsyncOpenAI

    def __init__(
        self,
        base_url: str = OLLAMA_OPENAI_COMPAT_BASE_URL,
        api_key: str = "ollama", 
        default_model: Optional[str] = None 
    ):
        """
        Initializes the service using the OpenAI client pointed at Ollama.

        Args:
            base_url: The base URL for the Ollama OpenAI compatibility endpoint.
            api_key: Dummy API key for the OpenAI client (Ollama ignores it).
            default_model: Optional default Ollama model name to use if not specified in calls.
        """
        self.default_model = default_model
        try:
            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            print(f"OllamaOpenAICompatService: Initialized OpenAI client for Ollama at {base_url}")
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize OpenAI client for Ollama: {e}{Fore.RESET}")
            raise

    async def get_llm_response(
        self,
        model: str, 
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto", 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> StandardizedLLMResponse:
        """
        Sends messages via the OpenAI SDK (to Ollama's endpoint)
        and returns a standardized response.
        """
        request_model = model or self.default_model
        if not request_model:
            raise ValueError("Ollama model name not specified in call or as default.")

        try:
            api_kwargs: Dict[str, Any] = {
                "messages": messages,
                "model": request_model,
            }
            if tools and tool_choice != "none":
                api_kwargs["tools"] = tools
                if isinstance(tool_choice, dict) or tool_choice in ["auto", "required", "none"]: 
                    api_kwargs["tool_choice"] = tool_choice
            else: 
                 api_kwargs["tools"] = None
                 api_kwargs["tool_choice"] = None


            if temperature is not None: api_kwargs["temperature"] = temperature
            if max_tokens is not None: api_kwargs["max_tokens"] = max_tokens
           
            api_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}


            print(f"OllamaOpenAICompatService: Sending request to model '{request_model}' with tool_choice: {api_kwargs.get('tool_choice')}")
            # print(f"OllamaOpenAICompatService: Request messages: {json.dumps(messages, indent=2)}")
            # if api_kwargs.get("tools"):
            # print(f"OllamaOpenAICompatService: Request tools: {json.dumps(api_kwargs['tools'], indent=2)}")


            response = await self._client.chat.completions.create(**api_kwargs)
            # print(f"OllamaOpenAICompatService: Raw response from Ollama: {response}")


            message = response.choices[0].message
            text_content = message.content
            tool_calls_standardized: List[LLMToolCall] = []

            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc.id and tc.function and tc.function.name and tc.function.arguments is not None:
                        tool_calls_standardized.append(
                            LLMToolCall(
                                id=tc.id,
                                function_name=tc.function.name,
                                function_arguments_json_str=tc.function.arguments 
                            )
                        )
                    else:
                        print(f"{Fore.YELLOW}Warning: Received incomplete tool_call from Ollama (via OpenAI compat): {tc}{Fore.RESET}")


            return StandardizedLLMResponse(
                text_content=text_content,
                tool_calls=tool_calls_standardized
            )

        except OpenAIError as e: 
            print(f"{Fore.RED}Ollama (via OpenAI Compat) API Error: {e}{Fore.RESET}")
            error_message = f"Ollama (OpenAI Compat) API Error: {e}"
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'text'):
                error_message += f" - Details: {e.response.text}"
            return StandardizedLLMResponse(text_content=error_message)
        except Exception as e:
            print(f"{Fore.RED}Unexpected error calling Ollama (via OpenAI Compat) LLM API: {e}{Fore.RESET}")
            return StandardizedLLMResponse(text_content=f"Ollama (OpenAI Compat) Unexpected Error: {e}")

    async def close(self):
        """Closes the underlying AsyncOpenAI client."""
        print("OllamaOpenAICompatService: Closing OpenAI client...")
        if hasattr(self, '_client') and self._client:
            try:
                await self._client.close()
                print("OllamaOpenAICompatService: Client closed.")
            except Exception as e:
                print(f"OllamaOpenAICompatService: Error closing client: {e}")

