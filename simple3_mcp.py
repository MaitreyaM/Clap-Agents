from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import uvicorn
import contextlib

# Create MCP server with name and stateless_http flag
mcp = FastMCP("multiply_server", stateless_http=True)

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies integer a by integer b."""
    print(f"[Local Tool] Multiplying: {a} * {b}")
    return a * b

if __name__ == "__main__":
    print("Starting streamable http MCP server")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    # Create FastAPI app with lifespan and mount MCP server
    app = FastAPI(lifespan=lifespan)
    # Mount at root so the internal '/mcp' path is available at '/mcp'
    app.mount("/", mcp.streamable_http_app())

    # Run with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)