
import json
import re
from typing import List, Dict, Any, Optional
import asyncio

from colorama import Fore
from dotenv import load_dotenv


from clap.llm_services.base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall
from clap.tool_pattern.tool import Tool
from clap.mcp_client.client import MCPClientManager, SseServerConfig
from clap.utils.completions import build_prompt_structure, ChatHistory, update_chat_history
from mcp import types as mcp_types


load_dotenv()

CORE_SYSTEM_PROMPT = """
You are an AI assistant that uses the ReAct (**Reason**->**Act**) process to answer questions and perform tasks using available tools (both local and remote MCP tools).

**Your Interaction Loop:**
1.  **Thought:** You MUST first analyze the query/situation and formulate a plan. Start your response **only** with your thought process, prefixed with "**Thought:**" on a new line.
2.  **Action Decision:** Based on your thought, decide if a tool is needed.
3.  **Observation:** If a tool is called, the system will provide the result. Analyze this in your next Thought.
4.  **Final Response:** When you have enough information, provide the final answer. Start this **only** with "**Final Response:**" on a new line, following your final thought.

**Output Syntax:**

*   **For Tool Use:**
    Thought: [Your reasoning and plan to use a tool]
    *(System executes tool based on your thought's intent)*

*   **After Observation:**
    Thought: [Your analysis of the observation and next step]
    *(Either signal another tool use implicitly or provide final response)*

*   **For Final Answer:**
    Thought: [Your final reasoning]
    Final Response: [Your final answer to the user]

---

**Constraint:** Always begin your response content with "Thought:". If providing the final answer, include "Final Response:" after the final thought. Do not add any other text before "Thought:" or "Final Response:" on their respective lines.
"""

class ReactAgent:
    """
    Async ReAct agent supporting local and remote MCP tools, using a configurable LLM service.
    """

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        model: str, 
        tools: Optional[List[Tool]] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        mcp_server_names: Optional[List[str]] = None,
        system_prompt: str = "",
    ) -> None:
        self.llm_service = llm_service
        self.model = model
        self.system_prompt = (system_prompt + "\n\n" + CORE_SYSTEM_PROMPT).strip()

        self.local_tools = tools if tools else []
        self.local_tools_dict = {tool.name: tool for tool in self.local_tools}
        self.local_tool_schemas = [tool.fn_schema for tool in self.local_tools]

        self.mcp_manager = mcp_manager
        self.mcp_server_names = mcp_server_names or []
        self.remote_tools_dict: Dict[str, mcp_types.Tool] = {}
        self.remote_tool_server_map: Dict[str, str] = {}

    async def _get_combined_tool_schemas(self) -> List[Dict[str, Any]]:
        all_schemas = list(self.local_tool_schemas)
        self.remote_tools_dict = {}
        self.remote_tool_server_map = {}
        if self.mcp_manager and self.mcp_server_names:
            fetch_tasks = [self.mcp_manager.list_remote_tools(name) for name in self.mcp_server_names]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for server_name, result in zip(self.mcp_server_names, results):
                if isinstance(result, Exception):
                    print(f"{Fore.RED}Error listing tools from MCP server '{server_name}': {result}{Fore.RESET}")
                    continue
                if isinstance(result, list):
                    for tool in result:
                        if isinstance(tool, mcp_types.Tool):
                           if tool.name in self.local_tools_dict: continue 
                           if tool.name in self.remote_tools_dict: continue # Skip conflicts
                           self.remote_tools_dict[tool.name] = tool
                           self.remote_tool_server_map[tool.name] = server_name
                           translated_schema = {"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema}}
                           all_schemas.append(translated_schema)
                        else: print(f"{Fore.YELLOW}Warning: Received non-Tool object from {server_name}: {type(tool)}{Fore.RESET}")
        print(f"{Fore.BLUE}Total tools available to LLM: {len(all_schemas)}{Fore.RESET}")
        return all_schemas

    async def process_tool_calls(self, tool_calls: List[LLMToolCall]) -> Dict[str, Any]:
        observations = {}
        if not isinstance(tool_calls, list):
             print(f"{Fore.RED}Error: Expected a list of LLMToolCall, got {type(tool_calls)}{Fore.RESET}")
             return observations
        tasks = [self._execute_single_tool_call(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, dict): observations.update(result)
            elif isinstance(result, Exception): print(f"{Fore.RED}Error during concurrent tool execution: {result}{Fore.RESET}")
            else: print(f"{Fore.RED}Error: Unexpected item in tool execution results: {result}{Fore.RESET}")
        return observations

    async def _execute_single_tool_call(self, tool_call: LLMToolCall) -> Dict[str, Any]: # Type hint changed
        tool_call_id = tool_call.id
        tool_name = tool_call.function_name
        result_str = f"Error: Processing failed for tool call '{tool_name}' (id: {tool_call_id})."
        try:
            arguments = json.loads(tool_call.function_arguments_json_str)
            if tool_name in self.local_tools_dict:
                tool = self.local_tools_dict[tool_name]
                print(f"{Fore.GREEN}\nExecuting Local Tool: {tool_name}{Fore.RESET}...")
                result = await tool.run(**arguments)
            elif tool_name in self.remote_tool_server_map and self.mcp_manager:
                server_name = self.remote_tool_server_map[tool_name]
                print(f"{Fore.CYAN}\nExecuting Remote MCP Tool: {tool_name} on {server_name}{Fore.RESET}...")
                result = await self.mcp_manager.call_remote_tool(server_name, tool_name, arguments)
            else:
                print(f"{Fore.RED}Error: Tool '{tool_name}' not found.{Fore.RESET}")
                result_str = f"Error: Tool '{tool_name}' is not available."
                return {tool_call_id: result_str}

            if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                result_str = str(result)
            else:
                try: result_str = json.dumps(result)
                except TypeError: result_str = str(result)
            print(f"{Fore.GREEN}Tool '{tool_name}' result: {result_str[:100]}...{Fore.RESET}")
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error decoding arguments for {tool_name}: {tool_call.function_arguments_json_str}{Fore.RESET}")
            result_str = f"Error: Invalid arguments JSON provided for {tool_name}"
        except Exception as e:
             print(f"{Fore.RED}Error executing tool {tool_name} (id: {tool_call_id}): {e}{Fore.RESET}")
             result_str = f"Error executing tool {tool_name}: {e}"
        return {tool_call_id: result_str}


    async def run(
        self,
        user_msg: str,
        max_rounds: int = 5,
    ) -> str:
        combined_tool_schemas = await self._get_combined_tool_schemas()

        initial_user_message = build_prompt_structure(role="user", content=user_msg)
        chat_history = ChatHistory(
            [
                build_prompt_structure(role="system", content=self.system_prompt),
                initial_user_message,
            ]
        )

        final_response = "Agent failed to produce a response."

        for round_num in range(max_rounds):
            print(Fore.CYAN + f"\n--- Round {round_num + 1} ---")
            current_tools = combined_tool_schemas if combined_tool_schemas else None
            current_tool_choice = "auto" if current_tools else "none"

            llm_response: StandardizedLLMResponse = await self.llm_service.get_llm_response(
                model=self.model,
                messages=list(chat_history),
                tools=current_tools,
                tool_choice=current_tool_choice
            )
            # --- End Change ---

            assistant_content = llm_response.text_content # Use standardized response field
            extracted_thought = None
            potential_final_response = None

            if assistant_content is not None:
                 lines = assistant_content.strip().split('\n')
                 thought_lines = []
                 response_lines = []
                 in_thought = False
                 in_response = False
                 for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("Thought:"):
                        in_thought = True; in_response = False
                        thought_content = stripped_line[len("Thought:"):].strip()
                        if thought_content: thought_lines.append(thought_content)
                    elif stripped_line.startswith("Final Response:"):
                         in_response = True; in_thought = False
                         response_content = stripped_line[len("Final Response:"):].strip()
                         if response_content: response_lines.append(response_content)
                    elif in_thought: thought_lines.append(line)
                    elif in_response: response_lines.append(line)
                 if thought_lines:
                     extracted_thought = "\n".join(thought_lines).strip()
                     print(f"{Fore.MAGENTA}\nThought: {extracted_thought}{Fore.RESET}")
                 if response_lines:
                      potential_final_response = "\n".join(response_lines).strip()

            assistant_msg_dict: Dict[str, Any] = {"role": "assistant"}
            if assistant_content:
                assistant_msg_dict["content"] = assistant_content 
            if llm_response.tool_calls:
                 assistant_msg_dict["tool_calls"] = [
                     {
                         "id": tc.id,
                         "type": "function", # Assuming 'function' type
                         "function": {
                             "name": tc.function_name,
                             "arguments": tc.function_arguments_json_str,
                         }
                     } for tc in llm_response.tool_calls
                 ]
            update_chat_history(chat_history, assistant_msg_dict)


            has_tool_calls = bool(llm_response.tool_calls)

            if has_tool_calls:
                print(f"{Fore.YELLOW}\nAssistant requests tool calls:{Fore.RESET}")
                observations = await self.process_tool_calls(llm_response.tool_calls)
                print(f"{Fore.BLUE}\nObservations: {observations}{Fore.RESET}")

                for tool_call in llm_response.tool_calls:
                     tool_call_id = tool_call.id
                     result = observations.get(tool_call_id, "Error: Observation not found.")
                     tool_message = build_prompt_structure(role="tool", content=str(result), tool_call_id=tool_call_id)
                     update_chat_history(chat_history, tool_message)

            elif potential_final_response is not None:
                print(f"{Fore.CYAN}\nAssistant provides final response:{Fore.RESET}")
                final_response = potential_final_response
                print(f"{Fore.GREEN}{final_response}{Fore.RESET}")
                return final_response

            elif assistant_content is not None and not has_tool_calls:
                 print(f"{Fore.YELLOW}\nAssistant provided content without 'Final Response:' prefix and no tool calls.{Fore.RESET}")
                 final_response = assistant_content.strip()
                 print(f"{Fore.GREEN}{final_response}{Fore.RESET}")
                 return final_response


            elif not has_tool_calls and assistant_content is None:
                 print(f"{Fore.RED}Error: Assistant message has neither content nor tool calls.{Fore.RESET}")
                 final_response = "Error: Received an unexpected empty or invalid response from the assistant."
                 return final_response

        print(f"{Fore.YELLOW}\nMaximum rounds ({max_rounds}) reached.{Fore.RESET}")
        if potential_final_response and not has_tool_calls:
             final_response = potential_final_response
             print(f"{Fore.GREEN}(Last response from agent): {final_response}{Fore.RESET}")
        elif assistant_content and not has_tool_calls:
             final_response = assistant_content.strip() # Use stripped content
             print(f"{Fore.GREEN}(Last raw content from agent): {final_response}{Fore.RESET}")
        else:
            final_response = "Agent stopped after maximum rounds without reaching a final answer."
            print(f"{Fore.YELLOW}{final_response}{Fore.RESET}")

        return final_response

