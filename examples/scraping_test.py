
import asyncio
import os
from dotenv import load_dotenv
import crawl4ai
from clap import ToolAgent
from clap import scrape_url, extract_text_by_query

load_dotenv()

async def main():
    agent = ToolAgent(
        tools=[scrape_url, extract_text_by_query], 
        model="llama-3.3-70b-versatile" 
    )
    query1 = "Can you scrape the content of https://docs.agno.com/introduction for me?"
    response1 = await agent.run(user_msg=query1)
    
    print(response1[:500] + "...")
    

    await asyncio.sleep(1)

    query2 = "Look for the term 'library' on the page https://docs.agno.com/introduction and show me the surrounding text."
    response2 = await agent.run(user_msg=query2)
    print(response2)
    

    

asyncio.run(main())

