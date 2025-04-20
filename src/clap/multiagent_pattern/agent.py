

import asyncio
import json
from textwrap import dedent
from typing import Any, List, Optional

from clap.llm_services.base import LLMServiceInterface
from clap.llm_services.groq_service import GroqService
from clap.mcp_client.client import MCPClientManager
from clap.react_pattern.react_agent import ReactAgent
from clap.tool_pattern.tool import Tool

class Agent:
    """
    Represents an AI agent using a configurable LLM Service.
    Can work in a team and use local or remote MCP tools.

    Args:
        name (str): Agent name.
        backstory (str): Agent background/persona.
        task_description (str): Description of the agent's specific task.
        task_expected_output (str): Expected output format.
        tools (Optional[List[Tool]]): Local tools for the agent.
        llm (str): Model identifier string (passed to llm_service).
        llm_service (Optional[LLMServiceInterface]): Service for LLM calls (defaults to GroqService).
        mcp_manager (Optional[MCPClientManager]): Shared MCP client manager.
        mcp_server_names (Optional[List[str]]): MCP servers this agent uses.
    """
    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: Optional[List[Tool]] = None,
        model: str = "llama-3.3-70b-versatile", 
        llm_service: Optional[LLMServiceInterface] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        mcp_server_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.mcp_manager = mcp_manager
        self.mcp_server_names = mcp_server_names or []

        llm_service_instance = llm_service or GroqService()

        self.react_agent = ReactAgent(
            llm_service=llm_service_instance,
            model=model,
            system_prompt=self.backstory,
            tools=tools or [],
            mcp_manager=self.mcp_manager,
            mcp_server_names=self.mcp_server_names
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
    def receive_context(self, sender_name: str, input_data: Any): self.received_context[sender_name] = input_data
    def create_prompt(self) -> str:
        context_str = "\n---\n".join(f"Context from {name}:\n{json.dumps(data, indent=2) if isinstance(data, dict) else str(data)}" for name, data in self.received_context.items())
        if not context_str: context_str = "No context received from other agents."
        prompt = dedent(f"""
        You are an AI agent named {self.name}. Your backstory: {self.backstory}
        You are part of a team of agents working together to complete a task.
        Your immediate task is described below. Use the provided context from other agents if relevant.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output or 'Produce a meaningful response to complete the task.'}
        </task_expected_output>

        <context>
        {context_str}
        </context>

        Now, execute your task based on the description, context, and expected output. Your response:
        """).strip(); return prompt
    async def run(self) -> dict[str, Any]:
        msg = self.create_prompt()
        raw_output = await self.react_agent.run(user_msg=msg)
        output_data = {"output": raw_output}
        for dependent in self.dependents:
            dependent.receive_context(self.name, output_data)
        return output_data

