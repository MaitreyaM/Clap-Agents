
import asyncio
import os
from dotenv import load_dotenv
import crawl4ai
from clap import ToolAgent
from clap.tools import scrape_url, extract_text_by_query
from clap.llm_services import GroqService

load_dotenv()

async def main():
    agent = ToolAgent(
        llm_service=GroqService(),
        tools=[scrape_url, extract_text_by_query], 
        model="llama-3.3-70b-versatile" 
    )
    query1 = "Can you scrape the content of https://docs.agno.com/introduction for me?"
    response1 = await agent.run(user_msg=query1)
    
    print(response1)
    

    await asyncio.sleep(1)

    query2 = "Can you look at https://docs.agno.com/introduction and tell me what it says about 'Agent Teams'?"
    response2 = await agent.run(user_msg=query2)
    print(response2)
    

    

asyncio.run(main())
