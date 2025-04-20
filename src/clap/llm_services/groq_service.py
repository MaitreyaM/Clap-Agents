# --- START OF agentic_patterns/llm_services/groq_service.py ---

from typing import Any, Dict, List, Optional

from groq import AsyncGroq, GroqError # Import AsyncGroq and potential errors
from colorama import Fore # For error printing

# Import the base interface and response structures
from .base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall

class GroqService(LLMServiceInterface):
    """LLM Service implementation for the Groq API."""

    def __init__(self, client: Optional[AsyncGroq] = None):
        """
        Initializes the Groq service.

        Args:
            client: An optional pre-configured AsyncGroq client.
                    If None, a new client will be created using environment variables.
        """
        self.client = client or AsyncGroq()
        # Add any other Groq-specific initialization here if needed

    async def get_llm_response(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        # Add other relevant Groq parameters if desired, e.g., temperature, max_tokens
        # temperature: Optional[float] = None,
        # max_tokens: Optional[int] = None,
    ) -> StandardizedLLMResponse:
        """
        Sends messages to the Groq API and returns a standardized response.

        Args:
            model: The Groq model identifier (e.g., "llama-3.3-70b-versatile").
            messages: Chat history in the OpenAI/Groq dictionary format.
            tools: Tool schemas in the OpenAI/Groq function format.
            tool_choice: Tool choice setting ("auto", "none", etc.).

        Returns:
            A StandardizedLLMResponse object.

        Raises:
            GroqError: If the API call fails.
            Exception: For other unexpected errors.
        """
        try:
            api_kwargs = {
                "messages": messages,
                "model": model,
                # Pass other parameters if added to method signature
                # "temperature": temperature,
                # "max_tokens": max_tokens,
            }
            if tools:
                api_kwargs["tools"] = tools
                api_kwargs["tool_choice"] = tool_choice

            # Call the Groq API asynchronously using the correct method name
            response = await self.client.chat.completions.create(**api_kwargs)

            # Process the response
            message = response.choices[0].message
            text_content = message.content
            tool_calls: List[LLMToolCall] = []

            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc.function: # Check if function attribute exists
                        tool_calls.append(
                            LLMToolCall(
                                id=tc.id,
                                function_name=tc.function.name,
                                function_arguments_json_str=tc.function.arguments
                            )
                        )

            # Return the standardized response
            return StandardizedLLMResponse(
                text_content=text_content,
                tool_calls=tool_calls
            )

        except GroqError as e:
            # Catch specific Groq errors for potentially better handling
            print(f"{Fore.RED}Groq API Error: {e}{Fore.RESET}")
            # Re-raise or handle as needed, maybe return an error response?
            # For now, re-raise to signal failure clearly
            raise
        except Exception as e:
            print(f"{Fore.RED}Error calling Groq LLM API: {e}{Fore.RESET}")
            # Depending on desired behavior, could return a StandardizedLLMResponse
            # with error info in text_content, or re-raise. Re-raising is cleaner.
            raise

# --- END OF agentic_patterns/llm_services/groq_service.py ---