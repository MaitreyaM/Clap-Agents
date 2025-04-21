

# Multi-agent pattern
from .multiagent_pattern.agent import Agent
from .multiagent_pattern.team import Team

# ReAct pattern
from .react_pattern.react_agent import ReactAgent

# Tool pattern
from .tool_pattern.tool import tool, Tool
from .tool_pattern.tool_agent import ToolAgent

# LLM Services (Interface and implementations)
from .llm_services.base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall
from .llm_services.groq_service import GroqService
from .llm_services.google_openai_compat_service import GoogleOpenAICompatService

from .mcp_client.client import MCPClientManager, SseServerConfig 


from .tools.web_search import duckduckgo_search
from .tools.web_crawler import scrape_url, extract_text_by_query
from .tools.email_tools import send_email, fetch_recent_emails

__all__ = [
    # Core classes
    "Agent",
    "Team",
    "ReactAgent",
    "ToolAgent",
    "Tool",
    "tool", # The decorator

    # LLM Services
    "LLMServiceInterface",
    "StandardizedLLMResponse",
    "LLMToolCall",
    "GroqService",
    "GoogleOpenAICompatService",

    # MCP Client
    "MCPClientManager",
    "SseServerConfig", # Expose config type

    "duckduckgo_search",
]


# __version__ = "0.1.0"

