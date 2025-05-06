import asyncio
import os
from dotenv import load_dotenv
from clap import ToolAgent
from clap import send_email, fetch_recent_emails

load_dotenv()

async def main():

    agent = ToolAgent(
        tools=[send_email, fetch_recent_emails], 
        model="llama-3.3-70b-versatile"
    )

   
    query1 = "Check my INBOX and tell me the last 2 emails."
    response1 = await agent.run(user_msg=query1)
    print(response1)

    # await asyncio.sleep(1) # Small delay

    # test_recipient = "youremail@gmail.com" 
    
    # query2 = f"Draft and send an email to {test_recipient}. Subject should be 'Agent Test' and body should be 'Hello from the CLAP Framework! and write a 30 words message thanking them for using our frame work'"
    # response2 = await agent.run(user_msg=query2)


asyncio.run(main())
