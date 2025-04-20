import os
from dotenv import load_dotenv
from clap import ReactAgent
from clap import GoogleOpenAICompatService
from clap import tool
import asyncio

load_dotenv()

@tool
def get_circle_area(radius: float) -> float:
    """Calculates the area of a circle given its radius."""
    print(f"[Local Tool] Calculating area for radius: {radius}")
    return 3.14159 * (radius ** 2)

async def main():
    google_llm_service = GoogleOpenAICompatService()

    agent = ReactAgent(
        llm_service=google_llm_service, # Pass the correct service instance
        model="gemini-2.5-pro-exp-03-25", # Specify a Gemini model name
        tools=[get_circle_area],
    )

    user_query = "What is the area of a circle with a radius of 5?"
    response = await agent.run(user_msg=user_query)
    print(response)
    
asyncio.run(main())