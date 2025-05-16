import os
import asyncio 
from dotenv import load_dotenv
from clap import ToolAgent
from clap import duckduckgo_search
from clap import GroqService
from clap.utils.completions import GroqClient

load_dotenv()

async def main():
    agent = ToolAgent(llm_service=GroqService(),tools=duckduckgo_search, model="meta-llama/llama-4-maverick-17b-128e-instruct")
    user_query = "Search the web for recent news about AI advancements."
    response = await agent.run(user_msg=user_query)
    print(f"Response:\n{response}")

asyncio.run(main())