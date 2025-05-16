
import asyncio
import os
from dotenv import load_dotenv
from pydantic import HttpUrl

from clap import ReactAgent
from clap import GoogleOpenAICompatService
from clap import MCPClientManager, SseServerConfig
from clap import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies integer a by integer b."""
    print(f"[Local Tool] Multiplying: {a} * {b}")
    return a * b

async def main():
    
    server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
        "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001"))
    }
    manager = MCPClientManager(server_configs)

    # 3. Instantiate the Google LLM Service
    google_llm_service = GoogleOpenAICompatService()
    gemini_model = "gemini-2.5-pro-exp-03-25" # Or your preferred compatible model

    agent = ReactAgent(
        llm_service=google_llm_service,
        model=gemini_model,
        tools=[multiply],
        mcp_manager=manager, 
        mcp_server_names=["adder_server","subtract_server"] 
    )

    user_query = "Calculate (10 + 5) * 3"

    response = await agent.run(user_msg=user_query)

    await manager.disconnect_all()


asyncio.run(main())

