import asyncio
import os
from dotenv import load_dotenv
# --- START DEBUG BLOCK ---
print(f"Attempting to load .env file.")
print(f"Current Working Directory: {os.getcwd()}") # Check if this is Cognitive-Layer/
env_path = os.path.join(os.getcwd(), '.env')
print(f"Expected .env path: {env_path}")
print(f"Does .env exist at expected path? {os.path.exists(env_path)}")
# Load the .env file
loaded_dotenv = load_dotenv()
print(f"load_dotenv() returned: {loaded_dotenv}") # Should be True if file was found
# Explicitly check the environment variables *after* loading
loaded_user = os.getenv("SMTP_USERNAME")
loaded_pass_exists = bool(os.getenv("SMTP_PASSWORD")) # Don't print the password itself
print(f"os.getenv('SMTP_USERNAME') after load: {loaded_user}")
print(f"os.getenv('SMTP_PASSWORD') is set after load: {loaded_pass_exists}")
print("--- END DEBUG BLOCK ---")
# Check if variables are still None/empty here
if not loaded_user or not loaded_pass_exists:
    print("\n*** ERROR: Environment variables NOT correctly loaded! Check .env file location and content. ***\n")
# --- Original imports ---
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

    await asyncio.sleep(1) # Small delay

    test_recipient = "maitreyamishra04@gmail.com" 
    
    query2 = f"Draft and send an email to {test_recipient}. Subject should be 'Agent Test' and body should be 'Hello from the CLAP Framework! and write a 30 words message thanking them for using our frame work'"
    response2 = await agent.run(user_msg=query2)


asyncio.run(main())