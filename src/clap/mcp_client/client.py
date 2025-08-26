import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from enum import Enum 

from pydantic import BaseModel, Field, HttpUrl , model_validator
from colorama import Fore

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client, StdioServerParameters


# Define an Enum for transport types for clarity and type safety
class TransportType(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"
    STDIO = 'stdio'

# Rename SseServerConfig to ServerConfig and add the transport field
class ServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""
    transport: TransportType = Field(
        default=TransportType.STREAMABLE_HTTP,
        description="The MCP transport to use."
    )
    # Fields for HTTP-based transports
    url: Optional[HttpUrl] = Field(default=None, description="The base URL for HTTP/SSE based MCP servers.")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Optional headers for the connection.")
    
    # Fields for STDIO transport
    command: Optional[str] = Field(default=None, description="The command to execute for stdio-based servers.")
    args: Optional[List[str]] = Field(default=None, description="A list of arguments for the command.")

    @model_validator(mode='after')
    def check_transport_params(self) -> 'ServerConfig':
        """Ensures the correct parameters are provided for the chosen transport."""
        if self.transport in [TransportType.STREAMABLE_HTTP, TransportType.SSE]:
            if not self.url:
                raise ValueError(f"'url' is required for '{self.transport.value}' transport")
        elif self.transport == TransportType.STDIO:
            if not self.command:
                raise ValueError(f"'command' is required for '{self.transport.value}' transport")
        
        return self

# Update the class docstring to be more generic
class MCPClientManager:
    """
    Manages connections and interactions with multiple MCP servers via supported
    MCP transports (Streamable HTTP, SSE).

    Handles connecting, disconnecting, listing tools, and calling tools on
    configured MCP servers accessible over HTTP/S.
    """

    # Update the __init__ method to use the new ServerConfig
    def __init__(self, server_configs: Dict[str, ServerConfig]):
        """
        Initializes the manager with server configurations.

        Args:
            server_configs: A dictionary where keys are logical server names
                            and values are ServerConfig objects.
        """
        if not isinstance(server_configs, dict):
             raise TypeError("server_configs must be a dictionary.")
        self.server_configs = server_configs
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self._connect_locks: Dict[str, asyncio.Lock] = {
             name: asyncio.Lock() for name in server_configs
        }
        self._manager_lock = asyncio.Lock()

    async def _ensure_connected(self, server_name: str):
        """
        Ensures a connection to the specified server is active using the configured transport.

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

            exit_stack = AsyncExitStack()
            try:
                if config.transport == TransportType.STDIO:
                    if not config.command:
                        raise ValueError("Cannot connect to STDIO server without a 'command'.")
                    
                    print(f"{Fore.YELLOW}Attempting to connect to MCP server '{server_name}' using STDIO transport...{Fore.RESET}")
                    print(f"  Command: {config.command} {' '.join(config.args or [])}")

                    server_params = StdioServerParameters(command=config.command, args=config.args or [])
                    stdio_transport_streams = await exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read_stream, write_stream = stdio_transport_streams
                    transport_name = "STDIO"
                # Core logic change - branch based on the configured transport
                elif config.transport == TransportType.STREAMABLE_HTTP:
                    mcp_endpoint_url = str(config.url).rstrip('/')
                    print(f"{Fore.YELLOW}Attempting to connect to MCP server '{server_name}' at {mcp_endpoint_url} using Streamable HTTP transport...{Fore.RESET}")

                    # Use the new streamablehttp_client
                    http_transport = await exit_stack.enter_async_context(
                        streamablehttp_client(url=mcp_endpoint_url, headers=config.headers, auth=None)
                    )
                    # It returns three values; we only need the first two.
                    read_stream, write_stream, _ = http_transport
                    transport_name = "Streamable HTTP"

                elif config.transport == TransportType.SSE:
                    # This branch maintains backward compatibility
                    sse_url = str(config.url).rstrip('/') + "/sse"
                    print(f"{Fore.YELLOW}Attempting to connect to MCP server '{server_name}' at {sse_url} using legacy SSE transport...{Fore.RESET}")

                    # Use the old sse_client
                    sse_transport = await exit_stack.enter_async_context(
                        sse_client(url=sse_url, headers=config.headers)
                    )
                    read_stream, write_stream = sse_transport
                    transport_name = "legacy SSE"

                else:
                    raise ValueError(f"Unsupported transport type configured for server '{server_name}': {config.transport}")

                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                await session.initialize()

                async with self._manager_lock:
                    self.sessions[server_name] = session
                    self.exit_stacks[server_name] = exit_stack
                print(f"{Fore.GREEN}Successfully connected to MCP server '{server_name}' via {transport_name}{Fore.RESET}")

            except Exception as e:
                await exit_stack.aclose()
                print(f"{Fore.RED}Failed to connect to MCP server '{server_name}': {e}{Fore.RESET}")
                raise RuntimeError(f"Connection to '{server_name}' failed.") from e

    async def disconnect(self, server_name: str):
        """
        Disconnects from a specific server and cleans up resources.
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
        Lists tools available on a specific connected server.
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
        Calls a tool on a specific connected server.
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