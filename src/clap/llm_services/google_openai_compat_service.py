# --- START OF agentic_patterns/llm_services/google_openai_compat_service.py ---

import os
import json
import uuid 
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError:
    raise ImportError("OpenAI SDK not found. Please install it using: pip install openai")

from colorama import Fore

from .base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall

GOOGLE_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

class GoogleOpenAICompatService(LLMServiceInterface):
    """
    LLM Service implementation using the OpenAI SDK configured for Google's
    Generative Language API (Gemini models via compatibility layer).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = GOOGLE_COMPAT_BASE_URL):
        """
        Initializes the service using the OpenAI client pointed at Google's endpoint.

        Args:
            api_key: Optional Google API key. If None, uses GOOGLE_API_KEY env var.
            base_url: The base URL for the Google compatibility endpoint.
        """
        effective_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not effective_key:
            raise ValueError("Google API Key not provided or found in environment variables (GOOGLE_API_KEY).")

        try:
            self.client = AsyncOpenAI(
                api_key=effective_key,
                base_url=base_url,
            )
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize OpenAI client for Google: {e}{Fore.RESET}")
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
        Sends messages via the OpenAI SDK (to Google's endpoint) and returns a standardized response.

        Args:
            model: The Google model identifier (e.g., "gemini-1.5-flash").
            messages: Chat history in the OpenAI dictionary format.
            tools: Tool schemas in the OpenAI function format.
            tool_choice: Tool choice setting ("auto", "none", etc.).
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            A StandardizedLLMResponse object.

        Raises:
            OpenAIError: If the API call fails.
            Exception: For other unexpected errors.
        """
        try:
            api_kwargs = {
                "messages": messages,
                "model": model,
                "tool_choice": tool_choice if tools else None,
                "tools": tools if tools else None,
            }
            if temperature is not None: api_kwargs["temperature"] = temperature
            if max_tokens is not None: api_kwargs["max_tokens"] = max_tokens
            api_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}


            response = await self.client.chat.completions.create(**api_kwargs)

            message = response.choices[0].message
            text_content = message.content
            tool_calls: List[LLMToolCall] = []

            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_call_id = getattr(tc, 'id', None)
                    if not tool_call_id:
                        tool_call_id = f"compat_call_{uuid.uuid4().hex[:6]}" 
                        print(f"{Fore.YELLOW}Warning: Tool call from Google compat layer missing ID. Generated fallback: {tool_call_id}{Fore.RESET}")

                    if tc.function:
                        tool_calls.append(
                            LLMToolCall(
                                id=tool_call_id,
                                function_name=tc.function.name,
                                function_arguments_json_str=tc.function.arguments
                            )
                        )

            return StandardizedLLMResponse(
                text_content=text_content,
                tool_calls=tool_calls
            )

        except OpenAIError as e:
            print(f"{Fore.RED}Google (via OpenAI Compat Layer) API Error: {e}{Fore.RESET}")
            raise
        except Exception as e:
            print(f"{Fore.RED}Error calling Google (via OpenAI Compat Layer) LLM API: {e}{Fore.RESET}")
            raise

