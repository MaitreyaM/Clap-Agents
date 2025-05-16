
import json
import re
from typing import List, Dict, Any, Optional
import asyncio

from colorama import Fore
from dotenv import load_dotenv


from clap.llm_services.base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall
from clap.tool_pattern.tool import Tool
from clap.mcp_client.client import MCPClientManager
# from clap.mcp_client.client import MCPClientManager, SseServerConfig 
from clap.utils.completions import build_prompt_structure, ChatHistory, update_chat_history
from clap.vector_stores.base import VectorStoreInterface, QueryResult
from clap.multiagent_pattern.agent import VECTOR_QUERY_TOOL_SCHEMA

try:
    from mcp import types as mcp_types
except ImportError:
    mcp_types = None 


load_dotenv()

CORE_SYSTEM_PROMPT = """
You are an AI assistant using the ReAct (Reason->Act) process with tools (local, remote MCP, `vector_query`).

**ReAct Loop:**
1.  **Thought:** REQUIRED start. Analyze the query/situation and plan next action. Start response ONLY with "Thought:".
2.  **Action Decision:** Decide if a tool is needed. **If using `vector_query`, the `query` argument MUST be the 'User Query:' from the main prompt.** Determine arguments for other tools based on your reasoning.
3.  **Tool Call / Next Step:** Use standard tool call format if applicable. If no tool call, proceed to step 5 (or 6 if done).
4.  **Observation:** (System provides tool results if a tool was called).
5.  **Thought:** Analyze observation (if any) and decide next step (another tool or final response).
6.  **Final Response:** REQUIRED end for the final answer. Must immediately follow the last Thought.

**Output Format:** Always start responses with "Thought:". Use "Final Response:" ONLY for the final answer, directly after the concluding Thought. No extra text before these prefixes. Be precise with tool arguments.
"""

class ReactAgent:
    """
    Async ReAct agent supporting local tools, remote MCP tools, and vector store queries,
    using a configurable LLM service.
    """

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        model: str,
        agent_name: str = "ReactAgent", 
        tools: Optional[List[Tool]] = None,
        mcp_manager: Optional[MCPClientManager] = None, 
        mcp_server_names: Optional[List[str]] = None,  
        vector_store: Optional[VectorStoreInterface] = None, 
        system_prompt: str = "",
    ) -> None:
        self.llm_service = llm_service
        self.model = model
        self.agent_name = agent_name
        self.system_prompt = (system_prompt + "\n\n" + CORE_SYSTEM_PROMPT).strip()

        
        self.local_tools = tools if tools else []
        self.local_tools_dict = {tool.name: tool for tool in self.local_tools}
        self.local_tool_schemas = [tool.fn_schema for tool in self.local_tools]

        
        self.mcp_manager = mcp_manager
        self.mcp_server_names = mcp_server_names or []
        self.remote_tools_dict: Dict[str, Any] = {} 
        self.remote_tool_server_map: Dict[str, str] = {}

        
        self.vector_store = vector_store


    async def _get_combined_tool_schemas(self) -> List[Dict[str, Any]]:
        """Combines schemas for local tools, remote MCP tools, and vector store query tool."""
        all_schemas = list(self.local_tool_schemas) 

        
        if self.vector_store:
            all_schemas.append(VECTOR_QUERY_TOOL_SCHEMA)
            print(f"{Fore.BLUE}[{self.agent_name}] Vector query tool is available.{Fore.RESET}")

        
        self.remote_tools_dict = {}
        self.remote_tool_server_map = {}
        if self.mcp_manager and self.mcp_server_names and mcp_types:
            fetch_tasks = [self.mcp_manager.list_remote_tools(name) for name in self.mcp_server_names]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for server_name, result in zip(self.mcp_server_names, results):
                if isinstance(result, Exception):
                    print(f"{Fore.RED}[{self.agent_name}] Error listing tools from MCP server '{server_name}': {result}{Fore.RESET}")
                    continue
                if isinstance(result, list):
                    for tool in result:
                         
                        if isinstance(tool, mcp_types.Tool):
                            if tool.name in self.local_tools_dict or tool.name == VECTOR_QUERY_TOOL_SCHEMA["function"]["name"]:
                                print(f"{Fore.YELLOW}Warning: Remote MCP tool '{tool.name}' conflicts with a local/vector tool. Skipping.{Fore.RESET}")
                                continue
                            if tool.name in self.remote_tools_dict:
                                print(f"{Fore.YELLOW}Warning: Remote MCP tool '{tool.name}' conflicts with another remote tool. Skipping duplicate.{Fore.RESET}")
                                continue

                            self.remote_tools_dict[tool.name] = tool
                            self.remote_tool_server_map[tool.name] = server_name
                            
                            translated_schema = {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "parameters": tool.inputSchema or {"type": "object", "properties": {}} # Handle potentially missing schema
                                }
                            }
                            all_schemas.append(translated_schema)
                        else:
                            print(f"{Fore.YELLOW}Warning: Received non-Tool object from {server_name}: {type(tool)}{Fore.RESET}")

        print(f"{Fore.BLUE}[{self.agent_name}] Total tools available to LLM: {len(all_schemas)}{Fore.RESET}")
        # print(f"Schemas: {json.dumps(all_schemas, indent=2)}") 
        return all_schemas



    async def _execute_single_tool_call(self, tool_call: LLMToolCall) -> Dict[str, Any]:
        """
        Executes a single tool call (local, remote MCP, or vector query),
        handling observation length limits for vector queries.
        """
        tool_call_id = tool_call.id
        tool_name = tool_call.function_name
        result_str = f"Error: Processing failed for tool call '{tool_name}' (id: {tool_call_id})." # Default error message

        try:
            arguments = json.loads(tool_call.function_arguments_json_str)

            
            if tool_name == VECTOR_QUERY_TOOL_SCHEMA["function"]["name"]:
                if not self.vector_store:
                    print(f"{Fore.RED}Error: Agent {self.agent_name} received call for '{tool_name}' but has no vector store configured.{Fore.RESET}")
                    result_str = f"Error: Vector store not available for agent {self.agent_name}."
                else:
                    print(f"{Fore.CYAN}\n[{self.agent_name}] Executing Vector Store Query Tool: {tool_name}{Fore.RESET}")
                    print(f"Tool call ID: {tool_call_id}") 
                    print(f"Arguments: {arguments}")     

                    query_text = arguments.get("query")
                    
                    top_k_value_from_llm = arguments.get("top_k")
                    default_top_k_from_schema = VECTOR_QUERY_TOOL_SCHEMA["function"]["parameters"]["properties"]["top_k"].get("default", 3)
                    top_k = default_top_k_from_schema 

                    if top_k_value_from_llm is not None:
                        try:
                            top_k = int(top_k_value_from_llm)
                        except (ValueError, TypeError):
                            print(f"{Fore.YELLOW}Warning: LLM provided top_k '{top_k_value_from_llm}' is not a valid integer. Using schema default: {default_top_k_from_schema}.{Fore.RESET}")
                            


                    if not query_text:
                         result_str = "Error: 'query' argument is required for vector_query tool."
                    else:
                        query_results: QueryResult = await self.vector_store.aquery(
                            query_texts=[query_text],
                            n_results=top_k, 
                            include=["documents", "metadatas", "distances"]
                        )

                        formatted_chunks_for_llm = []
                        current_length = 0
                        max_obs_len = 4000 

                        
                        retrieved_docs = query_results.get("documents")
                        retrieved_ids = query_results.get("ids")

                        if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 and \
                           retrieved_ids and isinstance(retrieved_ids, list) and len(retrieved_ids) > 0:

                            docs_for_query = retrieved_docs[0]
                            ids_for_query = retrieved_ids[0]
                            
                            metas_for_query = []
                            if query_results.get("metadatas") and isinstance(query_results["metadatas"], list) and len(query_results["metadatas"]) > 0:
                                metas_for_query = query_results["metadatas"][0] 
                            else:
                                metas_for_query = [None] * len(docs_for_query)

                            distances_for_query = []
                            if query_results.get("distances") and isinstance(query_results["distances"], list) and len(query_results["distances"]) > 0:
                                distances_for_query = query_results["distances"][0] 
                            else:
                                distances_for_query = [None] * len(docs_for_query)


                            for i, doc_content_item in enumerate(docs_for_query): 
                                current_meta = metas_for_query[i] if i < len(metas_for_query) else None
                                current_id = str(ids_for_query[i]) if i < len(ids_for_query) else "N/A"
                                current_distance = distances_for_query[i] if i < len(distances_for_query) and distances_for_query[i] is not None else float('nan')

                                meta_str = json.dumps(current_meta, ensure_ascii=False) if current_meta else "{}"
                                
                                
                                current_chunk_formatted = (
                                    f"--- Retrieved Chunk {str(i+1)} (ID: {current_id}, Distance: {current_distance:.4f}) ---\n"
                                    f"Metadata: {meta_str}\n" 
                                    f"Content: {str(doc_content_item)}\n\n" 
                                )
                                
                                chunk_len = len(current_chunk_formatted)

                                if current_length + chunk_len <= max_obs_len:
                                    formatted_chunks_for_llm.append(current_chunk_formatted)
                                    current_length += chunk_len
                                else:
                                    print(f"{Fore.YELLOW}[{self.agent_name}] Observation limit ({max_obs_len} chars) reached. Included {len(formatted_chunks_for_llm)} full chunks out of {len(docs_for_query)} retrieved.{Fore.RESET}")
                                    break
                            
                            if formatted_chunks_for_llm:
                                header = f"Retrieved {len(formatted_chunks_for_llm)} relevant document chunks (out of {len(docs_for_query)} found for the query):\n\n"
                                result_str = header + "".join(formatted_chunks_for_llm).strip()
                            else:
                                result_str = "No relevant documents found (or all retrieved documents were too long to fit context limit)."
                        else: 
                             result_str = "No relevant documents found in vector store for the query."

        
            elif tool_name in self.local_tools_dict:
                tool = self.local_tools_dict[tool_name]
                print(f"{Fore.GREEN}\n[{self.agent_name}] Executing Local Tool: {tool_name}{Fore.RESET}")
                print(f"Tool call ID: {tool_call_id}")
                print(f"Arguments: {arguments}")
                result = await tool.run(**arguments)
               
                if not isinstance(result, str):
                    try:
                        result_str = json.dumps(result, ensure_ascii=False)
                    except TypeError: 
                        result_str = str(result)
                else:
                    result_str = result

           
            elif tool_name in self.remote_tool_server_map and self.mcp_manager:
                server_name = self.remote_tool_server_map[tool_name]
                print(f"{Fore.CYAN}\n[{self.agent_name}] Executing Remote MCP Tool: {tool_name} on {server_name}{Fore.RESET}")
                print(f"Tool call ID: {tool_call_id}")
                print(f"Arguments: {arguments}")
                
                result_str = await self.mcp_manager.call_remote_tool(server_name, tool_name, arguments)

            
            else:
                print(f"{Fore.RED}Error: Tool '{tool_name}' not found locally, remotely, or as vector query.{Fore.RESET}")
                result_str = f"Error: Tool '{tool_name}' is not available."

            
            print(f"{Fore.GREEN}Tool '{tool_name}' observation prepared: {result_str[:150]}...{Fore.RESET}")


        except json.JSONDecodeError:
            print(f"{Fore.RED}Error decoding arguments for {tool_name}: {tool_call.function_arguments_json_str}{Fore.RESET}")
            result_str = f"Error: Invalid arguments JSON provided for {tool_name}."
        except Exception as e:
            print(f"{Fore.RED}Error executing/processing tool {tool_name} (id: {tool_call_id}): {e}{Fore.RESET}")
          
            result_str = f"Error during execution of tool {tool_name}: {e}"

        
        return {tool_call_id: result_str}




    
    async def process_tool_calls(self, tool_calls: List[LLMToolCall]) -> Dict[str, Any]:
        """Processes multiple tool calls concurrently."""
        observations = {}
        if not isinstance(tool_calls, list):
            print(f"{Fore.RED}Error: Expected a list of LLMToolCall, got {type(tool_calls)}{Fore.RESET}")
            return observations 

        tasks = [self._execute_single_tool_call(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict) and len(result) == 1:
                observations.update(result)
            elif isinstance(result, Exception):
                 
                 print(f"{Fore.RED}Error during concurrent tool execution gather: {result}{Fore.RESET}")
                 
                 # observations[f"error_{len(observations)}"] = f"Tool execution failed: {result}"
            else:
                 print(f"{Fore.RED}Error: Unexpected item in tool execution results: {result}{Fore.RESET}")

        return observations


    async def run(
        self,
        user_msg: str,
        max_rounds: int = 5,
    ) -> str:
        """Runs the ReAct loop for the agent."""
        print(f"--- [{self.agent_name}] Starting ReAct Loop ---")
        combined_tool_schemas = await self._get_combined_tool_schemas()

        initial_user_message = build_prompt_structure(role="user", content=user_msg)
        chat_history = ChatHistory(
            [
                build_prompt_structure(role="system", content=self.system_prompt),
                initial_user_message,
            ]
        )

        final_response = f"Agent {self.agent_name} failed to produce a final response."

        for round_num in range(max_rounds):
            print(Fore.CYAN + f"\n--- [{self.agent_name}] Round {round_num + 1}/{max_rounds} ---")

            
            current_tools = combined_tool_schemas if combined_tool_schemas else None
            current_tool_choice = "auto" if current_tools else "none"

            print(f"[{self.agent_name}] Calling LLM...")
            llm_response: StandardizedLLMResponse = await self.llm_service.get_llm_response(
                model=self.model,
                messages=list(chat_history),
                tools=current_tools,
                tool_choice=current_tool_choice
            )

            assistant_content = llm_response.text_content
            llm_tool_calls = llm_response.tool_calls 

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
                    elif in_thought:
                        
                        thought_lines.append(line) 
                    elif in_response:
                        response_lines.append(line) 

                if thought_lines:
                    extracted_thought = "\n".join(thought_lines).strip()
                    print(f"{Fore.MAGENTA}\n[{self.agent_name}] Thought:\n{extracted_thought}{Fore.RESET}")
                else:
                     print(f"{Fore.YELLOW}Warning: No 'Thought:' prefix found in LLM response content.{Fore.RESET}")
                     

                if response_lines:
                    potential_final_response = "\n".join(response_lines).strip()
                    

            
            assistant_msg_dict: Dict[str, Any] = {"role": "assistant"}
            if assistant_content: 
                assistant_msg_dict["content"] = assistant_content
            if llm_tool_calls:
                 
                 assistant_msg_dict["tool_calls"] = [
                     {
                         "id": tc.id,
                         "type": "function", 
                         "function": {
                             "name": tc.function_name,
                             "arguments": tc.function_arguments_json_str,
                         }
                     } for tc in llm_tool_calls
                 ]
            
            update_chat_history(chat_history, assistant_msg_dict)

            
            if llm_tool_calls:
                print(f"{Fore.YELLOW}\n[{self.agent_name}] Assistant requests tool calls:{Fore.RESET}")
                
                observations = await self.process_tool_calls(llm_tool_calls)
                print(f"{Fore.BLUE}\n[{self.agent_name}] Observations generated: {len(observations)} items.{Fore.RESET}")

                if not observations:
                     print(f"{Fore.RED}Error: Tool processing failed to return any observations.{Fore.RESET}")
                     
                     error_message = build_prompt_structure(role="user", content="System Error: Tool execution failed to produce results. Please try again or proceed without tool results.")
                     update_chat_history(chat_history, error_message)
                     continue 


                
                tool_messages_added = 0
                for tool_call in llm_tool_calls:
                    tool_call_id = tool_call.id
                    observation_content = observations.get(tool_call_id)
                    if observation_content is None:
                         print(f"{Fore.RED}Error: Observation missing for tool call ID {tool_call_id}.{Fore.RESET}")
                         observation_content = f"Error: Result for tool call {tool_call_id} was not found."

                    tool_message = build_prompt_structure(
                         role="tool",
                         content=str(observation_content), 
                         tool_call_id=tool_call_id
                    )
                    update_chat_history(chat_history, tool_message)
                    tool_messages_added += 1

                if tool_messages_added == 0:
                     print(f"{Fore.RED}Critical Error: No tool messages were added to history despite tool calls being present.{Fore.RESET}")
                     
                     return f"Error: Agent {self.agent_name} failed during tool observation processing."


            elif potential_final_response is not None:
                
                print(f"{Fore.GREEN}\n[{self.agent_name}] Assistant provides final response:{Fore.RESET}")
                final_response = potential_final_response
                print(f"{Fore.GREEN}{final_response}{Fore.RESET}")
                return final_response

            elif assistant_content is not None and not llm_tool_calls:
                
                print(f"{Fore.YELLOW}\n[{self.agent_name}] Assistant provided content without 'Final Response:' prefix and no tool calls. Treating as final answer.{Fore.RESET}")
                final_response = assistant_content.strip() 
                 
                if final_response.startswith("Thought:"):
                     final_response = final_response[len("Thought:"):].strip()
                print(f"{Fore.GREEN}{final_response}{Fore.RESET}")
                return final_response

            elif not llm_tool_calls and assistant_content is None:
                
                print(f"{Fore.RED}Error: Assistant message has neither content nor tool calls.{Fore.RESET}")
                final_response = f"Error: Agent {self.agent_name} received an empty response from the LLM."
                return final_response
            


        
        print(f"{Fore.YELLOW}\n[{self.agent_name}] Maximum rounds ({max_rounds}) reached.{Fore.RESET}")

        if potential_final_response and not llm_tool_calls:
             final_response = potential_final_response
             print(f"{Fore.GREEN}(Last response from agent {self.agent_name}): {final_response}{Fore.RESET}")
        elif assistant_content and not llm_tool_calls:
             
             final_response = assistant_content.strip()
             if final_response.startswith("Thought:"):
                  final_response = final_response[len("Thought:"):].strip()
             print(f"{Fore.GREEN}(Last raw content from agent {self.agent_name}): {final_response}{Fore.RESET}")
        else:
            final_response = f"Agent {self.agent_name} stopped after maximum rounds without reaching a final answer."
            print(f"{Fore.YELLOW}{final_response}{Fore.RESET}")

        return final_response

