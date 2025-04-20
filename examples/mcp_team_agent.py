
import os
import asyncio
from dotenv import load_dotenv
from pydantic import HttpUrl
from clap import Team
from clap import Agent
from clap import duckduckgo_search 
from clap import tool 
from clap import MCPClientManager, SseServerConfig
load_dotenv()

@tool
def count_words(text: str) -> int:
    """Counts the number of words in a given text."""
    return len(text.split())

async def run_agile_team_with_mcp():
    """Defines and runs the Agile benefits team, potentially using MCP tools."""
    mcp_server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000"))
       
    }
    mcp_manager = MCPClientManager(mcp_server_configs)

    topic = "the benefits of Agile methodology in software development"
    
    with Team() as team:
        researcher = Agent(
            name="Web_Researcher",
            backstory="You are an expert web researcher. You can use local search tools or potentially remote MCP tools.",
            task_description=f"Search the web for information on '{topic}'. Also calculate 5 + 3 using the 'add' tool.", # Modified task to use 'add'
            task_expected_output="Raw search results AND the result of 5 + 3.",
            tools=[duckduckgo_search],
            mcp_manager=mcp_manager, # Pass the shared manager
            mcp_server_names=["adder_server"] # Tell it which server(s) to use
        )

        summarizer = Agent(
            name="Content_Summarizer",
            backstory="You are an expert analyst. You can count words locally.",
            task_description="Analyze the provided context (search results and addition result) and extract the main benefits. Also count the words in the addition result.",
            task_expected_output="A concise bullet-point list summarizing key benefits, and the word count.",
            tools=[count_words],
        )
        reporter = Agent(
            name="Report_Writer",
            backstory="You are a skilled writer.",
            task_description="Take the summarized key points and word count, and write a short paragraph.",
            task_expected_output="A single paragraph summarizing the benefits and mentioning the word count."
        )

        researcher >> summarizer >> reporter
        await team.run()
        await mcp_manager.disconnect_all()

asyncio.run(run_agile_team_with_mcp())