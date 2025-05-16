
import asyncio
import json
from textwrap import dedent
from typing import Any, List, Optional, Dict 

from clap.llm_services.base import LLMServiceInterface
from clap.llm_services.groq_service import GroqService 

from clap.mcp_client.client import MCPClientManager

from clap.tool_pattern.tool import Tool

from clap.vector_stores.base import VectorStoreInterface




VECTOR_QUERY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "vector_query",
        "description": "Queries the configured vector store for relevant information based on the input query text. Use this to find context from stored documents before answering complex questions or summarizing information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query text to search for relevant documents."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The maximum number of relevant document chunks to retrieve. Defaults to 3.",
                    "default": 3
                },
                
            },
            "required": ["query"]
        }
    }
}


class Agent:
    """
    Represents an AI agent using a configurable LLM Service.
    Can work in a team, use local/remote MCP tools, and optionally a vector store.

    Args:
        name (str): Agent name.
        backstory (str): Agent background/persona.
        task_description (str): Description of the agent's specific task.
        task_expected_output (str): Expected output format.
        tools (Optional[List[Tool]]): Local tools for the agent.
        model (str): Model identifier string (passed to llm_service).
        llm_service (Optional[LLMServiceInterface]): Service for LLM calls (defaults to GroqService).
        mcp_manager (Optional[MCPClientManager]): Shared MCP client manager.
        mcp_server_names (Optional[List[str]]): MCP servers this agent uses.
        vector_store (Optional[VectorStoreInterface]): Vector store instance for RAG. 
        # embedding_function(Optional[EmbeddingFunction]): EF if needed by agent. 

    """
    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str = "No specific task assigned; await runtime user message.",
        task_expected_output: str = "",
        tools: Optional[List[Tool]] = None,
        model: str = "llama-3.3-70b-versatile", 
        llm_service: Optional[LLMServiceInterface] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        mcp_server_names: Optional[List[str]] = None,
        vector_store: Optional[VectorStoreInterface] = None, 
        # embedding_function: Optional[EmbeddingFunction] = None,

    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.mcp_manager = mcp_manager
        self.mcp_server_names = mcp_server_names or []
        self.local_tools = tools or [] 

        self.vector_store = vector_store
        # self.embedding_function = embedding_function 

        llm_service_instance = llm_service or GroqService()

       
        from clap.react_pattern.react_agent import ReactAgent 
        self.react_agent = ReactAgent(
            agent_name=self.name, 
            llm_service=llm_service_instance,
            model=model,
            system_prompt=self.backstory,
            tools=self.local_tools, 
            mcp_manager=self.mcp_manager,
            mcp_server_names=self.mcp_server_names,
            vector_store=self.vector_store 
        )

        self.dependencies: List['Agent'] = []
        self.dependents: List['Agent'] = []
        self.received_context: dict[str, Any] = {}

        from clap.multiagent_pattern.team import Team 
        Team.register_agent(self)


    def __repr__(self): return f"{self.name}"

    def __rshift__(self, other: 'Agent') -> 'Agent': self.add_dependent(other); return other
    def __lshift__(self, other: 'Agent') -> 'Agent': self.add_dependency(other); return other
    def __rrshift__(self, other: List['Agent'] | 'Agent'): self.add_dependency(other); return self
    def __rlshift__(self, other: List['Agent'] | 'Agent'): self.add_dependent(other); return self

    def add_dependency(self, other: 'Agent' | List['Agent']):
        AgentClass = type(self)
        if isinstance(other, AgentClass):
            if other not in self.dependencies: self.dependencies.append(other)
            if self not in other.dependents: other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, AgentClass) for item in other):
            for item in other:
                 if item not in self.dependencies: self.dependencies.append(item)
                 if self not in item.dependents: item.dependents.append(self)
        else: raise TypeError("The dependency must be an instance or list of Agent.")
    def add_dependent(self, other: 'Agent' | List['Agent']):
        AgentClass = type(self)
        if isinstance(other, AgentClass):
            if self not in other.dependencies: other.dependencies.append(self)
            if other not in self.dependents: self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, AgentClass) for item in other):
            for item in other:
                if self not in item.dependencies: item.dependencies.append(self)
                if item not in self.dependents: self.dependents.append(item)
        else: raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, sender_name: str, input_data: Any):
        self.received_context[sender_name] = input_data

    def create_prompt(self) -> str:
        """Creates the initial prompt for the agent's task execution."""
        context_str = "\n---\n".join(
            f"Context from {name}:\n{json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, dict) else str(data)}"
            for name, data in self.received_context.items()
        )
        if not context_str:
            context_str = "No context received from other agents."

        vector_store_info = ""
        user_query = self.task_description 

        if self.vector_store:
            vector_store_info = "\nVector Store Available: Use the 'vector_query' tool with the User Query below to find relevant context before answering factual questions."

        task_info = f"""
        User Query: {user_query}
        Task: Answer the User Query. {vector_store_info or ''} Use context from other agents if provided.
        Expected Output: {self.task_expected_output or 'Produce a meaningful response to complete the task.'}
        """.strip()

        prompt = dedent(f"""
        Agent: {self.name}
        Persona: {self.backstory}
        Team Context: {context_str}

        {task_info}

        Execute now following the ReAct pattern. If using 'vector_query', use the User Query text as the 'query' argument.
        """).strip()
        return prompt

    async def run(self, user_msg: Optional[str] = None) -> dict[str, Any]: 
        """Runs the agent's task using its configured ReactAgent.
        """
        print(f"Agent {self.name}: Preparing to run...")

        current_task = user_msg if user_msg is not None else self.task_description
        if not user_msg and self.task_description == "No specific task assigned; await runtime user message.":
             print(f"Agent {self.name}: Warning - Running without a specific user_msg or a meaningful pre-set task_description.")

        
        original_task_description = self.task_description
        self.task_description = current_task 

        msg = self.create_prompt()
        
        self.task_description = original_task_description 

        print(f"Agent {self.name}: Running ReactAgent...")
        raw_output = await self.react_agent.run(user_msg=msg) 
        output_data = {"output": raw_output}

        print(f"Agent {self.name}: Passing context to {len(self.dependents)} dependents...")
        for dependent in self.dependents:
            dependent.receive_context(self.name, output_data)

        return output_data

