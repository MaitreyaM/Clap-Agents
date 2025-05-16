

import asyncio 
from typing import Optional, List, Dict, Any 
# from groq import Groq 
# from groq.types.chat.chat_completion import ChatCompletion 
# from groq.types.chat.chat_completion_message import ChatCompletionMessage 
from groq import AsyncGroq


GroqClient = Any
ChatCompletionMessage = Any


async def completions_create(
    client: AsyncGroq,
    messages: List[Dict[str, Any]], 
    model: str,
    tools: Optional[List[Dict[str, Any]]] = None, 
    tool_choice: str = "auto" 
) -> ChatCompletionMessage: 
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
        
        api_kwargs = {
            "messages": messages,
            "model": model,
        }
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = tool_choice

        
        response = await client.chat.completions.create(**api_kwargs)
        
        return response.choices[0].message
    except Exception as e:
        
        print(f"Error calling LLM API asynchronously: {e}")
        
        
        class ErrorMessage:
             content = f"Error communicating with LLM: {e}"
             tool_calls = None
             role = "assistant"
        return ErrorMessage()


def build_prompt_structure(
    role: str,
    content: Optional[str] = None, 
    tag: str = "",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    tool_call_id: Optional[str] = None 
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

    
    if role == "assistant" and tool_calls:
        message["tool_calls"] = tool_calls

   
    if role == "tool" and tool_call_id:
        message["tool_call_id"] = tool_call_id
        if content is None: 
             raise ValueError("Content is required for role 'tool'.")

    
    if role == "tool" and not tool_call_id:
         raise ValueError("tool_call_id is required for role 'tool'.")
    if role != "assistant" and tool_calls:
         raise ValueError("tool_calls can only be added to 'assistant' role messages.")

    return message


def update_chat_history(
    history: list,
    message: ChatCompletionMessage | Dict[str, Any] 
    ):
    """
    Updates the chat history by appending a message object or a manually created message dict.

    Args:
        history (list): The list representing the current chat history.
        message: The message object from the API response or a dict created by build_prompt_structure.
    """
    
    if hasattr(message, "role"): # Basic check if it looks like an API message object
        msg_dict = {"role": message.role}
        if hasattr(message, "content") and message.content is not None:
            msg_dict["content"] = message.content
        if hasattr(message, "tool_calls") and message.tool_calls:
             
            msg_dict["tool_calls"] = message.tool_calls
        
        history.append(msg_dict)
    elif isinstance(message, dict) and "role" in message:
        
        history.append(message)
    else:
        raise TypeError("Invalid message type provided to update_chat_history.")


class ChatHistory(list):
    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None, total_length: int = -1): 
        if messages is None:
            messages = []
        super().__init__(messages)
        self.total_length = total_length 

    def append(self, msg: Dict[str, Any]): 
        if not isinstance(msg, dict) or "role" not in msg:
            raise TypeError("ChatHistory can only append message dictionaries with a 'role'.")

        
        if self.total_length > 0 and len(self) == self.total_length:
            self.pop(0) 
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None, total_length: int = -1):
        super().__init__(messages, total_length)

    def append(self, msg: Dict[str, Any]):
        if not isinstance(msg, dict) or "role" not in msg:
            raise TypeError("ChatHistory can only append message dictionaries with a 'role'.")

        
        if self.total_length > 0 and len(self) == self.total_length:
            if len(self) > 1:
                 self.pop(1) 
            else:
                 
                 print("Warning: Cannot append to FixedFirstChatHistory of size 1.")
                 return
        
        if self.total_length <= 0 or len(self) < self.total_length:
             super().append(msg)


