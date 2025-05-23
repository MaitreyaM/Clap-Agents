
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl 

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from colorama import Fore

class SseServerConfig(BaseModel):
    """Configuration for connecting to an MCP server via SSE."""
    url: HttpUrl = Field(description="The base URL of the MCP SSE server.")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Optional headers for the connection.")

class MCPClientManager:
    """
    Manages connections and interactions with multiple MCP servers via SSE.

    Handles connecting, disconnecting, listing tools, and calling tools on
    configured MCP servers accessible over HTTP/S.
    """

    def __init__(self, server_configs: Dict[str, SseServerConfig]):
        """
        Initializes the manager with SSE server configurations.

        Args:
            server_configs: A dictionary where keys are logical server names
                            and values are SseServerConfig objects.
        """
        if not isinstance(server_configs, dict):
             raise TypeError("server_configs must be a dictionary.")
        self.server_configs = server_configs
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self._connect_locks: Dict[str, asyncio.Lock] = {
             name: asyncio.Lock() for name in server_configs
        }
        self._manager_lock = asyncio.Lock() # General lock for manager state

    async def _ensure_connected(self, server_name: str):
        """
        Ensures a connection via SSE to the specified server is active.

        Args:
            server_name: The logical name of the server to connect to.

        Raises:
            ValueError: If the server configuration is not found or URL is invalid.
            RuntimeError: If connection or initialization fails.
        """
        if server_name in self.sessions:
            return

        connect_lock = self._connect_locks.get(server_name)
        if not connect_lock:
             raise ValueError(f"Configuration or lock for server '{server_name}' not found.")

        async with connect_lock:
            if server_name in self.sessions:
                return

            config = self.server_configs.get(server_name)
            if not config:
                raise ValueError(f"Configuration for server '{server_name}' not found.")

            print(f"{Fore.YELLOW}Attempting to connect to MCP server via SSE: {server_name} at {config.url}{Fore.RESET}")

            # Construct the specific SSE endpoint URL (often /sse)
            sse_url = str(config.url).rstrip('/') + "/sse" 

            exit_stack = AsyncExitStack()
            try:
                
                sse_transport = await exit_stack.enter_async_context(
                    sse_client(url=sse_url, headers=config.headers)
                )
                read_stream, write_stream = sse_transport

                
                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                
                await session.initialize()

                async with self._manager_lock:
                    self.sessions[server_name] = session
                    self.exit_stacks[server_name] = exit_stack
                print(f"{Fore.GREEN}Successfully connected to MCP server via SSE: {server_name}{Fore.RESET}")

            except Exception as e:
                await exit_stack.aclose()
                print(f"{Fore.RED}Failed to connect to MCP server '{server_name}' via SSE: {e}{Fore.RESET}")
                raise RuntimeError(f"SSE connection to '{server_name}' failed.") from e

    async def disconnect(self, server_name: str):
        """
        Disconnects from a specific server and cleans up resources.

        Args:
            server_name: The logical name of the server to disconnect from.
        """
        async with self._manager_lock:
             if server_name in self.sessions:
                 print(f"{Fore.YELLOW}Disconnecting from MCP server: {server_name}...{Fore.RESET}")
                 exit_stack = self.exit_stacks.pop(server_name)
                 del self.sessions[server_name]
                 await exit_stack.aclose()
                 print(f"{Fore.GREEN}Disconnected from MCP server: {server_name}{Fore.RESET}")

        
    async def disconnect_all(self):
        server_names = list(self.sessions.keys())
        print(f"{Fore.YELLOW}MCPClientManager: Disconnecting from all servers ({len(server_names)})...{Fore.RESET}")
        for name in server_names:
            try:
                await self.disconnect(name)
            except Exception as e:
                print(f"{Fore.RED}MCPClientManager: Error during disconnect of '{name}': {e}{Fore.RESET}")
        print(f"{Fore.GREEN}MCPClientManager: Finished disconnecting all servers.{Fore.RESET}")

    async def list_remote_tools(self, server_name: str) -> List[types.Tool]:
        """
        Lists tools available on a specific connected SSE server.

        Args:
            server_name: The logical name of the server.

        Returns:
            A list of mcp.types.Tool objects provided by the server.
        """
        await self._ensure_connected(server_name)
        session = self.sessions.get(server_name)
        if not session:
             raise RuntimeError(f"Failed to get session for '{server_name}' after ensuring connection.")

        try:
            print(f"{Fore.CYAN}Listing tools for server: {server_name}...{Fore.RESET}")
            tool_list_result = await session.list_tools()
            print(f"{Fore.CYAN}Found {len(tool_list_result.tools)} tools on {server_name}.{Fore.RESET}")
            return tool_list_result.tools
        except Exception as e:
            print(f"{Fore.RED}Error listing tools for server '{server_name}': {e}{Fore.RESET}")
            raise RuntimeError(f"Failed to list tools for '{server_name}'.") from e

    async def call_remote_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """
        Calls a tool on a specific connected SSE server.

        Args:
            server_name: The logical name of the server.
            tool_name: The name of the tool to call.
            arguments: A dictionary of arguments for the tool.

        Returns:
            A string representation of the tool's result content.
        """
        await self._ensure_connected(server_name)
        session = self.sessions.get(server_name)
        if not session:
             raise RuntimeError(f"Failed to get session for '{server_name}' after ensuring connection.")

        print(f"{Fore.CYAN}Calling remote tool '{tool_name}' on server '{server_name}' with args: {arguments}{Fore.RESET}")
        try:
             result: types.CallToolResult = await session.call_tool(tool_name, arguments)

             if result.isError:
                 error_content = result.content[0] if result.content else None
                 error_text = getattr(error_content, 'text', 'Unknown tool error')
                 print(f"{Fore.RED}MCP Tool '{tool_name}' on server '{server_name}' returned an error: {error_text}{Fore.RESET}")
                 raise RuntimeError(f"Tool call error on {server_name}.{tool_name}: {error_text}")
             else:
                  response_parts = []
                  for content_item in result.content:
                      if isinstance(content_item, types.TextContent):
                          response_parts.append(content_item.text)
                      elif isinstance(content_item, types.ImageContent):
                           response_parts.append(f"[Image Content Received: {content_item.mimeType}]")
                      elif isinstance(content_item, types.EmbeddedResource):
                           response_parts.append(f"[Embedded Resource Received: {content_item.resource.uri}]")
                      else:
                           response_parts.append(f"[Unsupported content type: {getattr(content_item, 'type', 'unknown')}]")
                  combined_response = "\n".join(response_parts)
                  print(f"{Fore.GREEN}Tool '{tool_name}' result from '{server_name}': {combined_response[:100]}...{Fore.RESET}")
                  return combined_response

        except Exception as e:
             print(f"{Fore.RED}Error calling tool '{tool_name}' on server '{server_name}': {e}{Fore.RESET}")
             raise RuntimeError(f"Failed to call tool '{tool_name}' on '{server_name}'.") from e
