
import asyncio
import os
from dotenv import load_dotenv

from clap import Team
from clap import Agent
from clap import GoogleOpenAICompatService

load_dotenv()

async def run_google_team():
    google_llm_service = GoogleOpenAICompatService()
    gemini_model = "gemini-2.5-pro-exp-03-25" 
    topic = "the benefits of using asynchronous programming in Python"
    with Team() as team:
        planner = Agent(
            name="Topic_Planner",
            backstory="Expert in outlining content.",
            task_description=f"Create a short, 3-bullet point outline for explaining '{topic}'.",
            task_expected_output="A 3-item bullet list.",
            llm_service=google_llm_service,
            model=gemini_model,
            
        )

        writer = Agent(
            name="Content_Writer",
            backstory="Skilled technical writer.",
            task_description="Take the outline provided in the context and write a concise paragraph explaining the topic.",
            task_expected_output="One paragraph based on the outline.",
            llm_service=google_llm_service,
            model=gemini_model,
           
        )

        # Define dependency
        planner >> writer

        
        await team.run()

asyncio.run(run_google_team())
