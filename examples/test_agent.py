
import os
import asyncio 
from dotenv import load_dotenv
from clap import ToolAgent
from clap import duckduckgo_search

load_dotenv()

async def main():
    agent = ToolAgent(tools=duckduckgo_search, model="llama-3.3-70b-versatile")
    user_query = "Search the web for recent news about AI advancements."
    response = await agent.run(user_msg=user_query)
    print(f"Response:\n{response}")

asyncio.run(main())