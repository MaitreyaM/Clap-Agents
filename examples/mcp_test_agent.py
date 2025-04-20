
import os
import asyncio
from dotenv import load_dotenv
from pydantic import HttpUrl
from clap import ToolAgent
from clap import MCPClientManager, SseServerConfig

load_dotenv()

async def main():
    server_name = "adder_server"
    server_configs = {
        server_name: SseServerConfig(url=HttpUrl("http://localhost:8000"))
    }
    manager = MCPClientManager(server_configs)

    agent = ToolAgent(
        tools=[],
        mcp_manager=manager,
        mcp_server_names=[server_name],
        model="llama-3.3-70b-versatile" 
    )

    user_query = "What is 123 plus 456?"

    response = await agent.run(user_msg=user_query)
    await manager.disconnect_all()


asyncio.run(main())
