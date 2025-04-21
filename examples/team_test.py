
import os
import asyncio # Import asyncio
from dotenv import load_dotenv
from clap import Team
from clap import Agent
from clap import duckduckgo_search

load_dotenv()

topic = "the benefits of Agile methodology in software development"

async def run_team():
    with Team() as team:
        researcher = Agent(
            name="Web_Researcher",
            backstory="You are an expert web researcher.",
            task_description=f"Search the web for information on '{topic}'.",
            task_expected_output="Raw search results.",
            tools=[duckduckgo_search]
        )

        summarizer = Agent(
            name="Content_Summarizer",
            backstory="You are an expert analyst.",
            task_description="Analyze the provided context (search results) and extract the main benefits.",
            task_expected_output="A concise bullet-point list summarizing key benefits."
        )

        reporter = Agent(
            name="Report_Writer",
            backstory="You are a skilled writer.",
            task_description="Take the summarized key points and write a short paragraph.",
            task_expected_output="A single paragraph summarizing the benefits."
        )

        researcher >> summarizer >> reporter

        await team.run()



asyncio.run(run_team())

