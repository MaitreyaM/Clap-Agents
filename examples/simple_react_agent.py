import asyncio
import os
from dotenv import load_dotenv
from clap import ReactAgent, tool, GroqService

load_dotenv() 
@tool
def get_word_length(word: str) -> int:
    """Calculates the length of a word."""
    print(f"[Local Tool] Calculating length of: {word}")
    return len(word)

async def main():
    groq_service = GroqService() 
    agent = ReactAgent(
        llm_service=groq_service,
        model="llama-3.3-70b-versatile", 
        tools=[get_word_length], 
        system_prompt="You are a helpful assistant." # Optional 
    )

    user_query = "How many letters are in the word 'framework'?"
    response = await agent.run(user_msg=user_query)
    
    print(response)
    
asyncio.run(main())