
import asyncio
import os
from dotenv import load_dotenv
from pydantic import HttpUrl 
from clap import ReactAgent
from clap import MCPClientManager, SseServerConfig
from clap import tool
from clap import GroqService

@tool
def multiply(a: int, b: int) -> int:
    """Subtracts integer b from integer a."""
    print(f"[Local Tool] Multiplying: {a} * {b}")
    return a * b

async def main():
    load_dotenv() 
    groq_llm_service = GroqService()

    server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
        "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001"))
    }
    manager = MCPClientManager(server_configs)

    agent = ReactAgent(
        llm_service=groq_llm_service,
        model="llama3-70b-8192",
        tools=[multiply], 
        mcp_manager=manager,
        mcp_server_names=["adder_server","subtract_server"] 
    )

    user_query = "Calculate ((15 + 7) - 5) * 2"
    response = await agent.run(user_msg=user_query)
    print(response)
    await manager.disconnect_all()

asyncio.run(main())