
import os
import asyncio
from dotenv import load_dotenv
from pydantic import HttpUrl
from clap import ToolAgent
from clap import MCPClientManager, SseServerConfig

load_dotenv()

async def main():
    server_configs = {
        server_name: SseServerConfig(url=HttpUrl("http://localhost:8000"))
    }
    manager = MCPClientManager(server_configs)
    agent = ToolAgent(
        mcp_manager=manager,
        mcp_server_names=["adder_server"],
        model="meta-llama/llama-4-scout-17b-16e-instruct" 
    )

    user_query = "What is 123 plus 456?"

    response = await agent.run(user_msg=user_query)
    await manager.disconnect_all()


asyncio.run(main())
