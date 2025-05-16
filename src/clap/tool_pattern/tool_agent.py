
import json
import asyncio
from typing import List, Dict, Any, Optional, Union # Added Union

from colorama import Fore
from dotenv import load_dotenv

from clap.tool_pattern.tool import Tool
from clap.mcp_client.client import MCPClientManager
from clap.utils.completions import build_prompt_structure, ChatHistory, update_chat_history
from clap.llm_services.base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall
from clap.vector_stores.base import VectorStoreInterface, QueryResult

from clap.multiagent_pattern.agent import VECTOR_QUERY_TOOL_SCHEMA



try:
    from mcp import types as mcp_types
except ImportError:
    mcp_types = None

load_dotenv()

NATIVE_TOOL_SYSTEM_PROMPT = """
You are a helpful assistant. Use the available tools (local functions, remote MCP tools, or vector_query for document retrieval) if necessary to answer the user's request.
If you use a tool, you will be given the results, and then you should provide the final response to the user based on those results.
If no tool is needed, answer directly.
When using vector_query, the 'query' argument should be the user's main question.
"""

class ToolAgent:
    """
    A simple agent that uses LLM native tool calling asynchronously.
    Supports local, remote MCP tools, and RAG via vector_query tool, using an LLMServiceInterface.
    Makes one attempt to call tools if needed, processes results,
    and then generates a final response.
    """

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        model: str,
        tools: Optional[Union[Tool, List[Tool]]] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        mcp_server_names: Optional[List[str]] = None,
        vector_store: Optional[VectorStoreInterface] = None, 
        system_prompt: str = NATIVE_TOOL_SYSTEM_PROMPT,
    ) -> None:
        if not isinstance(llm_service, LLMServiceInterface):
            raise TypeError("llm_service must be an instance of LLMServiceInterface.")
        if not model or not isinstance(model, str):
            raise ValueError("A valid model name (string) is required.")

        self.llm_service = llm_service
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
        self.remote_tools_dict: Dict[str, Any] = {}
        self.remote_tool_server_map: Dict[str, str] = {}

        self.vector_store = vector_store 

    async def _get_combined_tool_schemas(self) -> List[Dict[str, Any]]:
        all_schemas = list(self.local_tool_schemas)
        
        self.remote_tools_dict = {}
        self.remote_tool_server_map = {}
        if self.mcp_manager and self.mcp_server_names and mcp_types:
            fetch_tasks = [self.mcp_manager.list_remote_tools(name) for name in self.mcp_server_names]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for server_name, result in zip(self.mcp_server_names, results):
                if isinstance(result, Exception): print(f"{Fore.RED}ToolAgent: Error listing MCP tools '{server_name}': {result}{Fore.RESET}"); continue
                if isinstance(result, list):
                    for tool_obj in result: 
                        if isinstance(tool_obj, mcp_types.Tool):
                           if tool_obj.name in self.local_tools_dict: print(f"{Fore.YELLOW}ToolAgent Warning: MCP tool '{tool_obj.name}' conflicts with local. Skipping.{Fore.RESET}"); continue
                           
                           if self.vector_store and tool_obj.name == VECTOR_QUERY_TOOL_SCHEMA["function"]["name"]:
                               print(f"{Fore.YELLOW}ToolAgent Warning: MCP tool '{tool_obj.name}' conflicts with built-in vector_query tool. Skipping MCP version.{Fore.RESET}"); continue
                           if tool_obj.name in self.remote_tools_dict: print(f"{Fore.YELLOW}ToolAgent Warning: MCP tool '{tool_obj.name}' conflicts with another remote. Skipping.{Fore.RESET}"); continue
                           self.remote_tools_dict[tool_obj.name] = tool_obj
                           self.remote_tool_server_map[tool_obj.name] = server_name
                           translated_schema = {"type": "function", "function": {"name": tool_obj.name, "description": tool_obj.description or "", "parameters": tool_obj.inputSchema or {"type": "object", "properties": {}}}}
                           all_schemas.append(translated_schema)
                        else: print(f"{Fore.YELLOW}ToolAgent Warning: Non-Tool object from {server_name}: {type(tool_obj)}{Fore.RESET}")

        
        if self.vector_store:
            
            if not any(schema["function"]["name"] == VECTOR_QUERY_TOOL_SCHEMA["function"]["name"] for schema in all_schemas):
                all_schemas.append(VECTOR_QUERY_TOOL_SCHEMA)
                print(f"{Fore.BLUE}ToolAgent: Vector query tool is available.{Fore.RESET}")
        
        print(f"{Fore.BLUE}ToolAgent: Total tools available to LLM: {len(all_schemas)}{Fore.RESET}")
        return all_schemas

    async def _execute_single_tool_call(self, tool_call: LLMToolCall) -> Dict[str, Any]:
        tool_call_id = tool_call.id
        tool_name = tool_call.function_name
        result_str = f"Error: Processing tool call '{tool_name}' (id: {tool_call_id})."
        try:
            arguments = json.loads(tool_call.function_arguments_json_str)

            
            if self.vector_store and tool_name == VECTOR_QUERY_TOOL_SCHEMA["function"]["name"]:
                print(f"{Fore.CYAN}\nToolAgent: Executing Vector Store Query Tool: {tool_name} (ID: {tool_call_id}) Args: {arguments}{Fore.RESET}")
                query_text = arguments.get("query")
                
                top_k_value_from_llm = arguments.get("top_k")
                default_top_k_from_schema = VECTOR_QUERY_TOOL_SCHEMA["function"]["parameters"]["properties"]["top_k"].get("default", 3)
                top_k = default_top_k_from_schema
                if top_k_value_from_llm is not None:
                    try: top_k = int(top_k_value_from_llm)
                    except (ValueError, TypeError):
                        print(f"{Fore.YELLOW}ToolAgent Warning: LLM provided top_k '{top_k_value_from_llm}' is invalid. Using default: {default_top_k_from_schema}.{Fore.RESET}")

                if not query_text:
                    result_str = "Error: 'query' argument required for vector_query tool."
                else:
                    query_results: QueryResult = await self.vector_store.aquery(
                        query_texts=[query_text], n_results=top_k,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    formatted_chunks_for_llm = []
                    current_length = 0
                    max_obs_len = 4000 
                    retrieved_docs = query_results.get("documents")
                    retrieved_ids = query_results.get("ids")
                    if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 and \
                       retrieved_ids and isinstance(retrieved_ids, list) and len(retrieved_ids) > 0:
                        docs_for_query, ids_for_query = retrieved_docs[0], retrieved_ids[0]
                        metas_list, distances_list = query_results.get("metadatas"), query_results.get("distances")
                        metas_for_query = metas_list[0] if metas_list and len(metas_list) > 0 else [None] * len(docs_for_query)
                        distances_for_query = distances_list[0] if distances_list and len(distances_list) > 0 else [None] * len(docs_for_query)
                        for i, doc_content_item in enumerate(docs_for_query):
                            current_meta = metas_for_query[i] if i < len(metas_for_query) else None
                            current_id = str(ids_for_query[i]) if i < len(ids_for_query) else "N/A"
                            current_distance = distances_for_query[i] if i < len(distances_for_query) and distances_for_query[i] is not None else float('nan')
                            meta_str = json.dumps(current_meta, ensure_ascii=False) if current_meta else "{}"
                            current_chunk_formatted = (
                                f"--- Retrieved Chunk {str(i+1)} (ID: {current_id}, Distance: {current_distance:.4f}) ---\n"
                                f"Metadata: {meta_str}\nContent: {str(doc_content_item)}\n\n")
                            chunk_len = len(current_chunk_formatted)
                            if current_length + chunk_len <= max_obs_len:
                                formatted_chunks_for_llm.append(current_chunk_formatted)
                                current_length += chunk_len
                            else:
                                print(f"{Fore.YELLOW}ToolAgent: Obs limit ({max_obs_len}) reached. Included {len(formatted_chunks_for_llm)} chunks.{Fore.RESET}"); break
                        if formatted_chunks_for_llm:
                            header = f"Retrieved {len(formatted_chunks_for_llm)} relevant document chunks (out of {len(docs_for_query)} found):\n\n"
                            result_str = header + "".join(formatted_chunks_for_llm).strip()
                        else: result_str = "No relevant documents found (or chunks too long for limit)."
                    else: result_str = "No relevant documents found in vector store for query."
           

            elif tool_name in self.local_tools_dict:
                tool_instance = self.local_tools_dict[tool_name]
                result = await tool_instance.run(**arguments)
                if not isinstance(result, str): result_str = json.dumps(result, ensure_ascii=False)
                else: result_str = result
            elif tool_name in self.remote_tool_server_map and self.mcp_manager:
                server_name = self.remote_tool_server_map[tool_name]
                result_str = await self.mcp_manager.call_remote_tool(server_name, tool_name, arguments)
            else:
                result_str = f"Error: Tool '{tool_name}' not available."
                print(f"{Fore.RED}ToolAgent: {result_str}{Fore.RESET}")
                return {tool_call_id: result_str}
            
            print(f"{Fore.GREEN}ToolAgent: Tool '{tool_name}' observation: {result_str[:150]}...{Fore.RESET}")
        except json.JSONDecodeError:
            result_str = f"Error: Invalid arguments JSON for {tool_name}."
            print(f"{Fore.RED}ToolAgent: {result_str} Data: {tool_call.function_arguments_json_str}{Fore.RESET}")
        except Exception as e:
            result_str = f"Error executing tool {tool_name}: {e}"
            print(f"{Fore.RED}ToolAgent: Error for tool {tool_name} (ID: {tool_call_id}): {e}{Fore.RESET}")
        return {tool_call_id: result_str}

    
    async def process_tool_calls(self, tool_calls: List[LLMToolCall]) -> List[Dict[str, Any]]:
        observation_messages = []
        if not isinstance(tool_calls, list): return observation_messages
        tasks = [self._execute_single_tool_call(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, dict) and len(result) == 1:
                  tool_call_id, result_str = list(result.items())[0]
                  observation_messages.append(build_prompt_structure(role="tool", content=result_str, tool_call_id=tool_call_id))
             elif isinstance(result, Exception): print(f"{Fore.RED}ToolAgent: Error in concurrent tool execution: {result}{Fore.RESET}")
             else: print(f"{Fore.RED}ToolAgent: Unexpected item in tool results: {result}{Fore.RESET}")
        return observation_messages

    async def run(self, user_msg: str) -> str:
        combined_tool_schemas = await self._get_combined_tool_schemas()
        initial_user_message = build_prompt_structure(role="user", content=user_msg)
        chat_history = ChatHistory(
            [build_prompt_structure(role="system", content=self.system_prompt), initial_user_message]
        )
        llm_response_1: StandardizedLLMResponse = await self.llm_service.get_llm_response(
            model=self.model, messages=list(chat_history),
            tools=combined_tool_schemas if combined_tool_schemas else None,
            tool_choice="auto" if combined_tool_schemas else "none"
        )
        assistant_msg_1_dict: Dict[str, Any] = {"role": "assistant"}
        if llm_response_1.text_content: assistant_msg_1_dict["content"] = llm_response_1.text_content
        if llm_response_1.tool_calls:
            assistant_msg_1_dict["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function_name, "arguments": tc.function_arguments_json_str}}
                for tc in llm_response_1.tool_calls
            ]
        if "content" in assistant_msg_1_dict or "tool_calls" in assistant_msg_1_dict:
            update_chat_history(chat_history, assistant_msg_1_dict)

        final_response = "ToolAgent encountered an issue."
        if llm_response_1.tool_calls:
            observation_messages = await self.process_tool_calls(llm_response_1.tool_calls)
            for obs_msg in observation_messages: update_chat_history(chat_history, obs_msg)
            llm_response_2: StandardizedLLMResponse = await self.llm_service.get_llm_response(
                model=self.model, messages=list(chat_history)
            )
            final_response = llm_response_2.text_content if llm_response_2.text_content else "Agent provided no final response after using tools."
        elif llm_response_1.text_content is not None:
            final_response = llm_response_1.text_content
        else:
            print(f"{Fore.RED}ToolAgent Error: LLM message has neither content nor tool_calls.{Fore.RESET}")
            final_response = "Error: ToolAgent received an unexpected empty response from the LLM."
        return final_response
