# --- START OF ASYNC MODIFIED tool_agent.py (Init Fix) ---

import json
import asyncio
from typing import List, Dict, Any, Optional

from colorama import Fore
from dotenv import load_dotenv
from groq import AsyncGroq

from clap.tool_pattern.tool import Tool
from clap.mcp_client.client import MCPClientManager
from clap.utils.completions import build_prompt_structure
from clap.utils.completions import ChatHistory
from clap.utils.completions import completions_create
from clap.utils.completions import update_chat_history
from mcp import types as mcp_types

load_dotenv()

NATIVE_TOOL_SYSTEM_PROMPT = """
You are a helpful assistant. Use the available tools (local or remote) if necessary to answer the user's request.
If you use a tool, you will be given the results, and then you should provide the final response to the user based on those results.
If no tool is needed, answer directly.
"""

class ToolAgent:
    """
    A simple agent that uses native tool calling asynchronously to answer user queries.
    Supports both local Python tools and remote MCP tools via an MCPClientManager.
    It makes one attempt to call tools if needed, processes the results,
    and then generates a final response.
    """

    def __init__(
        self,
        tools: Optional[Tool | List[Tool]] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        mcp_server_names: Optional[List[str]] = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = NATIVE_TOOL_SYSTEM_PROMPT,
    ) -> None:
        self.client = AsyncGroq()
        self.model = model
        self.system_prompt = system_prompt

        if tools is None:
            self.local_tools = []
        elif isinstance(tools, list):
            self.local_tools = tools
        else:
            self.local_tools = [tools]

        self.local_tools_dict = {tool.name: tool for tool in self.local_tools}
        self.local_tool_schemas = [tool.fn_schema for tool in self.local_tools]

        self.mcp_manager = mcp_manager
        self.mcp_server_names = mcp_server_names or []
        self.remote_tools_dict: Dict[str, mcp_types.Tool] = {}
        self.remote_tool_server_map: Dict[str, str] = {}


    async def _get_combined_tool_schemas(self) -> List[Dict[str, Any]]:
        """Fetches remote tools and combines their schemas with local ones."""
        all_schemas = list(self.local_tool_schemas) # Start with local schemas
        self.remote_tools_dict = {} # Reset remote tools for this run
        self.remote_tool_server_map = {}

        if self.mcp_manager and self.mcp_server_names:
            fetch_tasks = [
                self.mcp_manager.list_remote_tools(name)
                for name in self.mcp_server_names
            ]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for server_name, result in zip(self.mcp_server_names, results):
                if isinstance(result, Exception):
                    print(f"{Fore.RED}Error listing tools from MCP server '{server_name}': {result}{Fore.RESET}")
                    continue

                if isinstance(result, list):
                    for tool in result:
                        if isinstance(tool, mcp_types.Tool):
                           if tool.name in self.local_tools_dict:
                               print(f"{Fore.YELLOW}Warning: Remote tool '{tool.name}' from server '{server_name}' conflicts with local tool. Local tool will be used.{Fore.RESET}")
                               continue
                           if tool.name in self.remote_tools_dict:
                               print(f"{Fore.YELLOW}Warning: Remote tool '{tool.name}' from server '{server_name}' conflicts with another remote tool from server '{self.remote_tool_server_map[tool.name]}'. Skipping duplicate.{Fore.RESET}")
                               continue

                           self.remote_tools_dict[tool.name] = tool
                           self.remote_tool_server_map[tool.name] = server_name

                           translated_schema = {
                               "type": "function",
                               "function": {
                                   "name": tool.name,
                                   "description": tool.description or "",
                                   "parameters": tool.inputSchema
                               }
                           }
                           all_schemas.append(translated_schema)
                        else:
                             print(f"{Fore.YELLOW}Warning: Received non-Tool object from {server_name}: {type(tool)}{Fore.RESET}")

        print(f"{Fore.BLUE}Total tools available to LLM: {len(all_schemas)}{Fore.RESET}")
        return all_schemas

    async def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Processes tool calls requested by the LLM asynchronously, dispatches execution
        to local or remote tools, and collects results formatted as 'tool' role messages.
        """
        observation_messages = []
        if not isinstance(tool_calls, list):
             print(f"{Fore.RED}Error: Expected a list of tool_calls, got {type(tool_calls)}{Fore.RESET}")
             return observation_messages

        tasks = [self._execute_single_tool_call(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
             if isinstance(result, dict):
                  if len(result) == 1:
                      tool_call_id, result_str = list(result.items())[0]
                      observation_messages.append(
                           build_prompt_structure(role="tool", content=result_str, tool_call_id=tool_call_id)
                      )
                  else:
                       print(f"{Fore.RED}Error: Unexpected result format from tool execution: {result}{Fore.RESET}")
             elif isinstance(result, Exception):
                  print(f"{Fore.RED}Error during concurrent tool execution: {result}{Fore.RESET}")
             else:
                  print(f"{Fore.RED}Error: Unexpected item in tool execution results: {result}{Fore.RESET}")

        return observation_messages

    async def _execute_single_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Helper to execute a single tool call (local or remote)."""
        tool_call_id = getattr(tool_call, 'id', 'error_no_id')
        function_call = getattr(tool_call, 'function', None)
        tool_name = getattr(function_call, 'name', 'error_unknown_name')
        result_str = f"Error: Processing failed for tool call '{tool_name}' (id: {tool_call_id})."

        try:
            if not function_call:
                 raise ValueError("Invalid tool_call object structure: missing 'function'.")

            arguments_str = getattr(function_call, 'arguments', '{}')
            arguments = json.loads(arguments_str)

            if tool_name in self.local_tools_dict:
                tool = self.local_tools_dict[tool_name]
                print(f"{Fore.GREEN}\nExecuting Local Tool: {tool_name}{Fore.RESET}")
                print(f"Tool call ID: {tool_call_id}")
                print(f"Arguments: {arguments}")
                result = await tool.run(**arguments)
            elif tool_name in self.remote_tool_server_map and self.mcp_manager:
                server_name = self.remote_tool_server_map[tool_name]
                print(f"{Fore.CYAN}\nExecuting Remote MCP Tool: {tool_name} on {server_name}{Fore.RESET}")
                print(f"Tool call ID: {tool_call_id}")
                print(f"Arguments: {arguments}")
                result = await self.mcp_manager.call_remote_tool(server_name, tool_name, arguments)
            else:
                print(f"{Fore.RED}Error: Tool '{tool_name}' not found locally or in known remote servers.{Fore.RESET}")
                result_str = f"Error: Tool '{tool_name}' is not available."
                return {tool_call_id: result_str}

            if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                result_str = str(result)
            else:
                try: result_str = json.dumps(result)
                except TypeError: result_str = str(result)
            print(f"{Fore.GREEN}Tool '{tool_name}' result: {result_str[:100]}...{Fore.RESET}")

        except json.JSONDecodeError:
            print(f"{Fore.RED}Error: Could not decode arguments for tool {tool_name}: {arguments_str}{Fore.RESET}")
            result_str = f"Error: Invalid arguments JSON provided for {tool_name}"
        except Exception as e:
             print(f"{Fore.RED}Error processing or running tool {tool_name} (id: {tool_call_id}): {e}{Fore.RESET}")
             result_str = f"Error executing tool {tool_name}: {e}"

        return {tool_call_id: result_str}

    async def run(
        self,
        user_msg: str,
    ) -> str:
        """
        Handles the asynchronous interaction: user message -> LLM (tool decision) ->
        execute tools (local or remote) -> LLM (final response).
        """
        combined_tool_schemas = await self._get_combined_tool_schemas()

        initial_user_message = build_prompt_structure(role="user", content=user_msg)
        chat_history = ChatHistory(
            [
                build_prompt_structure(role="system", content=self.system_prompt),
                initial_user_message,
            ]
        )

        print(f"{Fore.CYAN}\n--- Calling LLM for Tool Decision ---{Fore.RESET}")
        assistant_message_1 = await completions_create(
            self.client,
            messages=list(chat_history),
            model=self.model,
            tools=combined_tool_schemas,
            tool_choice="auto"
        )

        update_chat_history(chat_history, assistant_message_1)

        final_response = "Agent encountered an issue."

        if hasattr(assistant_message_1, 'tool_calls') and assistant_message_1.tool_calls:
            print(f"{Fore.YELLOW}\nAssistant requests tool calls:{Fore.RESET}")
            observation_messages = await self.process_tool_calls(assistant_message_1.tool_calls)
            print(f"{Fore.BLUE}\nObservations prepared for LLM: {observation_messages}{Fore.RESET}")

            for obs_msg in observation_messages:
                update_chat_history(chat_history, obs_msg)

            print(f"{Fore.CYAN}\n--- Calling LLM for Final Response ---{Fore.RESET}")
            assistant_message_2 = await completions_create(
                self.client,
                messages=list(chat_history),
                model=self.model,
            )
            final_response = str(assistant_message_2.content) if assistant_message_2.content else "Agent did not provide a final response after using tools."

        elif assistant_message_1.content is not None:
            print(f"{Fore.CYAN}\nAssistant provided direct response (no tools used):{Fore.RESET}")
            final_response = assistant_message_1.content
        else:
            print(f"{Fore.RED}Error: Assistant message has neither content nor tool calls.{Fore.RESET}")
            final_response = "Error: Received an unexpected empty response from the assistant."

        print(f"{Fore.GREEN}\nFinal Response:\n{final_response}{Fore.RESET}")
        return final_response

