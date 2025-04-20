

import asyncio 
from typing import Optional, List, Dict, Any 
# Assuming Groq client and specific API response types might be needed
# from groq import Groq # Already imported elsewhere, ensure available
# from groq.types.chat.chat_completion import ChatCompletion # Example type hint
# from groq.types.chat.chat_completion_message import ChatCompletionMessage # Example type hint
from groq import AsyncGroq


GroqClient = Any
ChatCompletionMessage = Any


async def completions_create(
    client: AsyncGroq,
    messages: List[Dict[str, Any]], # Use more specific types if available
    model: str,
    tools: Optional[List[Dict[str, Any]]] = None, # Added tools parameter
    tool_choice: str = "auto" # Added tool_choice parameter ("auto", "none", or {"type": "function", "function": {"name": "my_function"}})
) -> ChatCompletionMessage: # Changed return type
    """
    Sends an asynchronous request to the client's completions endpoint, supporting tool use.

    Args:
        client: The API client object (e.g., Groq) supporting async operations.
        messages: A list of message objects for the chat history.
        model: The model to use.
        tools: A list of tool schemas the model can use.
        tool_choice: Controls how the model uses tools.

    Returns:
        The message object from the API response, which might contain content or tool calls.
    """
    try:
        # Prepare arguments, only include tools/tool_choice if tools are provided
        api_kwargs = {
            "messages": messages,
            "model": model,
        }
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = tool_choice

        # Changed .acreate to .create based on Groq async documentation
        response = await client.chat.completions.create(**api_kwargs)
        # Return the entire message object from the first choice
        return response.choices[0].message
    except Exception as e:
        # Handle potential API errors
        print(f"Error calling LLM API asynchronously: {e}")
        # Return a custom message or re-raise depending on desired error handling
        # Returning a placeholder error message object might be useful
        class ErrorMessage:
             content = f"Error communicating with LLM: {e}"
             tool_calls = None
             role = "assistant"
        return ErrorMessage()


def build_prompt_structure(
    role: str,
    content: Optional[str] = None, # Content is optional now
    tag: str = "",
    tool_calls: Optional[List[Dict[str, Any]]] = None, # Added for assistant messages
    tool_call_id: Optional[str] = None # Added for tool messages
) -> dict:
    """
    Builds a structured message dictionary for the chat API.

    Args:
        role: The role ('system', 'user', 'assistant', 'tool').
        content: The text content of the message (required for user, system, tool roles).
        tag: An optional tag to wrap the content (legacy, consider removing).
        tool_calls: A list of tool calls requested by the assistant.
        tool_call_id: The ID of the tool call this message is a response to (for role 'tool').

    Returns:
        A dictionary representing the structured message.
    """
    message: Dict[str, Any] = {"role": role}
    if content is not None:
        if tag: # Apply legacy tag if provided
             content = f"<{tag}>{content}</{tag}>"
        message["content"] = content

    # Add tool_calls if provided (only for assistant role)
    if role == "assistant" and tool_calls:
        message["tool_calls"] = tool_calls

    # Add tool_call_id if provided (only for tool role)
    if role == "tool" and tool_call_id:
        message["tool_call_id"] = tool_call_id
        if content is None: # Tool role requires content
             raise ValueError("Content is required for role 'tool'.")

    # Basic validation
    if role == "tool" and not tool_call_id:
         raise ValueError("tool_call_id is required for role 'tool'.")
    if role != "assistant" and tool_calls:
         raise ValueError("tool_calls can only be added to 'assistant' role messages.")

    return message


def update_chat_history(
    history: list,
    message: ChatCompletionMessage | Dict[str, Any] # Accept API message object or manually created dict
    ):
    """
    Updates the chat history by appending a message object or a manually created message dict.

    Args:
        history (list): The list representing the current chat history.
        message: The message object from the API response or a dict created by build_prompt_structure.
    """
    # If it's an API message object, convert it to the expected dict format
    if hasattr(message, "role"): # Basic check if it looks like an API message object
        msg_dict = {"role": message.role}
        if hasattr(message, "content") and message.content is not None:
            msg_dict["content"] = message.content
        if hasattr(message, "tool_calls") and message.tool_calls:
             # Assuming message.tool_calls is already in the correct list[dict] format
            msg_dict["tool_calls"] = message.tool_calls
        # Add other relevant fields if needed
        history.append(msg_dict)
    elif isinstance(message, dict) and "role" in message:
        # If it's already a dictionary (e.g., from build_prompt_structure)
        history.append(message)
    else:
        raise TypeError("Invalid message type provided to update_chat_history.")


class ChatHistory(list):
    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None, total_length: int = -1): # Type hint messages
        if messages is None:
            messages = []
        super().__init__(messages)
        self.total_length = total_length # Note: total_length logic might need adjustment for tool calls/responses

    def append(self, msg: Dict[str, Any]): # Expecting message dictionaries now
        if not isinstance(msg, dict) or "role" not in msg:
            raise TypeError("ChatHistory can only append message dictionaries with a 'role'.")

        # Simple length check, might need refinement based on token count or message types
        if self.total_length > 0 and len(self) == self.total_length:
            self.pop(0) # Remove the oldest message (index 0)
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None, total_length: int = -1):
        super().__init__(messages, total_length)

    def append(self, msg: Dict[str, Any]):
        if not isinstance(msg, dict) or "role" not in msg:
            raise TypeError("ChatHistory can only append message dictionaries with a 'role'.")

        # Keep the first message (system prompt) fixed
        if self.total_length > 0 and len(self) == self.total_length:
            if len(self) > 1: # Ensure there's more than just the system prompt to remove
                 self.pop(1) # Remove the second oldest message (index 1)
            else:
                 # Cannot append if length is 1 and fixed
                 print("Warning: Cannot append to FixedFirstChatHistory of size 1.")
                 return
        # Only call super().append if there's space or an item was removed
        if self.total_length <= 0 or len(self) < self.total_length:
             super().append(msg)


# --- END OF ASYNC MODIFIED completions.py ---