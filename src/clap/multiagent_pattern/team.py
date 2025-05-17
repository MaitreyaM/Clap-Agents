
import asyncio
from collections import deque
from typing import Any, Dict, List , Optional
from colorama import Fore
from clap.utils.logging import fancy_print
from clap.multiagent_pattern.agent import Agent

_GRAPHVIZ_AVAILABLE = False
_Digraph_Placeholder_Type = Any 

try:
    from graphviz import Digraph as ImportedDigraph
    from graphviz.backend import ExecutableNotFound as GraphvizExecutableNotFound 
    _Digraph_Placeholder_Type = ImportedDigraph
    _GRAPHVIZ_AVAILABLE = True
except ImportError:
    pass

class Team:
    """
    A class representing a team of agents working together asynchronously.
    Supports parallel execution of agents where dependencies allow.

    Attributes:
        current_team (Team | None): Class-level variable to track the active Team context. None if no team context is active.
        agents (list[Agent]): A list of agents in the team.
        results (dict[str, Any]): Stores the final results of each agent run.
    """
    current_team = None

    def __init__(self):
        self.agents: List[Agent] = []
        self.results: dict[str, Any] = {}

    def __enter__(self): Team.current_team = self; return self
    def __exit__(self, exc_type, exc_val, exc_tb): Team.current_team = None
    def add_agent(self, agent: Agent):
        if agent not in self.agents: self.agents.append(agent)
    @staticmethod
    def register_agent(agent: Agent):
        if Team.current_team is not None: Team.current_team.add_agent(agent)

    
    def topological_sort(self) -> List[Agent]:
        in_degree: Dict[Agent, int] = {agent: 0 for agent in self.agents}
        adj: Dict[Agent, List[Agent]] = {agent: [] for agent in self.agents}
        agent_map: Dict[str, Agent] = {agent.name: agent for agent in self.agents}
        for agent in self.agents:
            valid_dependencies = [dep for dep in agent.dependencies if dep in self.agents]
            agent.dependencies = valid_dependencies
            for dependency in agent.dependencies:
                if dependency in agent_map.values():
                    adj[dependency].append(agent)
                    in_degree[agent] += 1
        queue: deque[Agent] = deque([agent for agent in self.agents if in_degree[agent] == 0])
        sorted_agents: List[Agent] = []
        processed_edges = 0
        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)
            for potential_dependent in self.agents:
                 if current_agent in potential_dependent.dependencies:
                      in_degree[potential_dependent] -= 1
                      processed_edges +=1
                      if in_degree[potential_dependent] == 0:
                           queue.append(potential_dependent)
        if len(sorted_agents) != len(self.agents):
             detected_agents = {a.name for a in sorted_agents}
             missing_agents = {a.name for a in self.agents} - detected_agents
             remaining_degrees = {a.name: in_degree[a] for a in self.agents if a not in sorted_agents}
             raise ValueError(
                 "Circular dependencies detected. Cannot perform topological sort. "
                 f"Agents processed: {list(detected_agents)}. "
                 f"Agents potentially in cycle: {list(missing_agents)}. "
                 f"Remaining degrees: {remaining_degrees}"
             )
        return sorted_agents

    def plot(self) -> Optional[_Digraph_Placeholder_Type]:
        """
        Generates a visual representation of the agent team's dependency graph.
        Requires the 'graphviz' Python library and the Graphviz system software to be installed.
        If Graphviz is not available, prints a warning and returns None.

        Returns:
            Optional[graphviz.Digraph]: A Digraph object if successful, else None.
        """
        if not _GRAPHVIZ_AVAILABLE:
            print(
                f"{Fore.YELLOW}CLAP Warning: The 'graphviz' Python library is not installed. "
                f"To use the plot() feature, please install it (e.g., pip install \"clap-agents[viz]\").{Fore.RESET}"
            )
            return None
        
        dot: Optional[_Digraph_Placeholder_Type] = None
        try:
            dot = _Digraph_Placeholder_Type(format="png") 
            for agent in self.agents:
                dot.node(agent.name) 
            for agent in self.agents:
                 for dependent in agent.dependents: 
                      if dependent in self.agents: 
                          dot.edge(agent.name, dependent.name) 
            print(f"{Fore.GREEN}Team dependency graph created. To render, call .render() or .view() on the returned object.{Fore.RESET}")
            return dot
        except NameError: 
            print(f"{Fore.YELLOW}CLAP Warning: Graphviz Digraph class not found (likely import issue). Plotting unavailable.{Fore.RESET}")
            return None
        except GraphvizExecutableNotFound:
            print(
                f"{Fore.RED}CLAP Error: Graphviz system software (dot executable) not found in PATH. "
                f"Team.plot() requires it to generate images. Please install Graphviz for your OS and ensure 'dot' is in your PATH.{Fore.RESET}"
            )
            return None
        except Exception as e:
            print(f"{Fore.RED}CLAP Error: An unexpected error occurred during graph plotting: {e}{Fore.RESET}")
            return None

    async def run(self):
        """
        Runs all agents in the team asynchronously, executing them in parallel
        when their dependencies are met. Compatible with Python 3.10+.
        """
        try:
            sorted_agents = self.topological_sort()
        except ValueError as e:
            print(f"{Fore.RED}Error during team setup: {e}{Fore.RESET}")
            return

        self.results = {}
        agent_tasks: Dict[Agent, asyncio.Task] = {}
        try:
            tasks_to_gather = []
            for agent in sorted_agents:
                task = asyncio.create_task(self._run_agent_task(agent, agent_tasks))
                agent_tasks[agent] = task
                tasks_to_gather.append(task)

            await asyncio.gather(*tasks_to_gather)
            print(f"{Fore.BLUE}--- All agent tasks finished ---{Fore.RESET}")

        except Exception as e:
            print(f"{Fore.RED}One or more agents failed during execution:{Fore.RESET}")
            print(f"{Fore.RED}- Error: {e}{Fore.RESET}")
            


    async def _run_agent_task(self, agent: Agent, all_tasks: Dict[Agent, asyncio.Task]):
        """
        An internal async function that wraps the execution of a single agent.
        It waits for dependencies to complete before running the agent.
        """
        dependency_tasks = [
            all_tasks[dep] for dep in agent.dependencies if dep in all_tasks
        ]
        if dependency_tasks:
            print(f"{Fore.YELLOW}Agent {agent.name} waiting for dependencies: {[dep.name for dep in agent.dependencies if dep in all_tasks]}...{Fore.RESET}")
            
            await asyncio.gather(*dependency_tasks)
            print(f"{Fore.GREEN}Agent {agent.name} dependencies met.{Fore.RESET}")

        fancy_print(f"STARTING AGENT: {agent.name}")
        try:
            agent_result = await agent.run()
            self.results[agent.name] = agent_result

            if isinstance(agent_result, dict) and 'output' in agent_result:
                 print(f"{Fore.GREEN}Agent {agent.name} Result:\n{agent_result['output']}{Fore.RESET}")
            else:
                 print(f"{Fore.YELLOW}Agent {agent.name} Result (raw):\n{str(agent_result)}{Fore.RESET}")
            fancy_print(f"FINISHED AGENT: {agent.name}")

        except Exception as e:
            fancy_print(f"ERROR IN AGENT: {agent.name}")
            print(f"{Fore.RED}Agent {agent.name} failed: {e}{Fore.RESET}")
            self.results[agent.name] = {"error": str(e)}
            raise

