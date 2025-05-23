import asyncio
import os
from dotenv import load_dotenv
from typing import Optional

from pydantic import HttpUrl


from clap import MCPClientManager, SseServerConfig

load_dotenv()

def print_header(title: str):
    print(f"\n{'='*10} {title.upper()} {'='*10}")

async def run_mcp_test_cycle(
    manager: MCPClientManager,
    server_name: str,
    tool_to_call: Optional[str] = None,
    tool_args: Optional[dict] = None
):
    print_header(f"TEST CYCLE FOR {server_name.upper()}")
    try:
        print(f"Attempting to list tools from '{server_name}'...")
        tools = await manager.list_remote_tools(server_name)
        print(f"Found tools on '{server_name}': {[t.name for t in tools] if tools else 'None'}")

        if tool_to_call and tool_args:
            if any(t.name == tool_to_call for t in tools):
                print(f"Attempting to call tool '{tool_to_call}' on '{server_name}' with args: {tool_args}...")
                result = await manager.call_remote_tool(server_name, tool_to_call, tool_args)
                print(f"Result from '{tool_to_call}' on '{server_name}': {result}")
            else:
                print(f"Tool '{tool_to_call}' not found on server '{server_name}'.")

    except RuntimeError as e:
        print(f"RuntimeError during {server_name} cycle: {e}")
    except Exception as e:
        print(f"Unexpected error during {server_name} cycle: {type(e).__name__} - {e}")


async def main():
    print("MCP Robustness Test: Starting...")
    print("Ensure MCP servers ('adder_server' on 8000, 'subtract_server' on 8001) are running.")

    mcp_server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
        "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001")),
        "non_existent_server": SseServerConfig(url=HttpUrl("http://localhost:8009")) # To test connection error
    }
    manager: Optional[MCPClientManager] = None

    try:
        manager = MCPClientManager(server_configs=mcp_server_configs)
        print("MCPClientManager initialized.")

        await run_mcp_test_cycle(manager, "adder_server", tool_to_call="add", tool_args={"a": 10, "b": 5})

        print_header("SIMULATING AN EXTERNAL ERROR")
        try:
            
            print("Raising a simulated application error...")
            raise ValueError("Simulated application error after first MCP interaction.")
        except ValueError as app_error:
            print(f"Caught simulated application error: {app_error}")
            print("Proceeding to test other MCP operations and cleanup...")

        await run_mcp_test_cycle(manager, "subtract_server", tool_to_call="sub", tool_args={"a": 20, "b": 3})

        await run_mcp_test_cycle(manager, "non_existent_server")


    except Exception as e:
        print(f"Critical error in main test execution: {type(e).__name__} - {e}")
    finally:
        if manager:
            print_header("FINAL MCP CLEANUP")
            print("Calling manager.disconnect_all()...")
            await manager.disconnect_all()
            print("manager.disconnect_all() completed.")
        else:
            print("MCPClientManager was not initialized.")

    print("\nMCP Robustness Test: Finished.")

if __name__ == "__main__":
    asyncio.run(main())
