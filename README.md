<p align="center">
  <img src="[https://your-image-url.com/logo.png](https://i.imgur.com/7Dv4psB.png)" alt="CLAP Logo" width="200"/>
</p>

# CLAP - Cognitive Layer Agents Package

[![PyPI version](https://img.shields.io/pypi/v/CLAP.svg)](https://pypi.org/project/CLAP/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/CLAP.svg)](https://pypi.org/project/CLAP/)
<!-- Add other badges as desired, e.g., build status, coverage -->

**CLAP (Cognitive Layer Agent Package)** is a Python framework providing building blocks for creating sophisticated AI agents based on modern agentic patterns. It enables developers to easily construct agents capable of reasoning, planning, and interacting with external tools and systems.

Built with an asynchronous core (`asyncio`), CLAP offers flexibility and performance for complex agentic workflows.

<p align="center">
  <img src="[https://your-image-url.com/logo.png](https://i.imgur.com/wfw1yBy.png)" alt="CLAP Logo" width="200"/>
</p>


# CLAP - Cognitive Layer Agents Package
## Key Features

*   **Modular Agent Patterns:**
    *   **ReAct Agent:** Implements the Reason-Act loop with robust thought-prompting and native tool calling.
    *   **Multi-Agent Teams:** Define teams of specialized agents with dependencies, enabling collaborative task execution (sequential or parallel).
    *   **Simple Tool Agent:** A straightforward agent for single-step tool usage.
*   **Advanced Tool Integration:**
    *   **Native LLM Tool Calling:** Leverages modern LLM APIs for reliable tool execution.
    *   **Local Tools:** Easily define and use local Python functions (both synchronous and asynchronous) as tools using the `@tool` decorator.
    *   **Remote Tools (MCP):** Integrates with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers via the included `MCPClientManager`, allowing agents to discover and use tools exposed by external systems (currently supports SSE transport).
    *   **Robust Validation:** Uses `jsonschema` for strict validation of tool arguments provided by the LLM.
*   **Pluggable LLM Backends:**
    *   Uses a **Strategy Pattern** (`LLMServiceInterface`) to abstract LLM interactions.
    *   Includes ready-to-use service implementations for:
        *   **Groq:** (`GroqService`)
        *   **Google Generative AI (Gemini):** (`GoogleOpenAICompatService` via OpenAI compatibility layer)
    *   Easily extensible to support other LLM providers.
*   **Asynchronous Core:** Built entirely on `asyncio` for efficient I/O operations and potential concurrency.
*   **Structured Context Passing:** Enables clear and organized information flow between agents in a team using Python dictionaries.
*   **Built-in Tools:** Includes helpers for web search, web scraping, and email interaction (optional dependencies may apply).

## Installation

Ensure you have Python 3.10 or later installed.

```bash
pip install clap-agents
```
Depending on the tools or LLM backends you intend to use, you might need additional dependencies listed in the pyproject.toml (e.g., groq, openai, mcp, jsonschema, requests, duckduckgo-search, graphviz). Check the [project.dependencies] and [project.optional-dependencies] sections.


## Quick Start: Simple Tool calling Agent with a Local Tool
This example demonstrates creating a Tool calling agent using the Groq backend and a local tool

```
from dotenv import load_dotenv
from clap import ToolAgent
from clap import duckduckgo_search

load_dotenv()

async def main():
    agent = ToolAgent(tools=duckduckgo_search, model="meta-llama/llama-4-scout-17b-16e-instruct")
    user_query = "Search the web for recent news about AI advancements."
    response = await agent.run(user_msg=user_query)
    print(f"Response:\n{response}")

asyncio.run(main())
```


## Quick Start: Simple ReAct Agent with a Local Tool
This example demonstrates creating a ReAct agent using the Groq backend and a local tool.

```
import asyncio
import os
from dotenv import load_dotenv
from clap import ReactAgent, tool, GroqService

# --- Setup ---
load_dotenv() 
@tool
def get_word_length(word: str) -> int:
    """Calculates the length of a word."""
    print(f"[Local Tool] Calculating length of: {word}")
    return len(word)

async def main():
    groq_service = GroqService() # Your service of choice (either groq or Google)
    agent = ReactAgent(
        llm_service=groq_service,
        model="llama-3.3-70b-versatile", # Or another Groq model
        tools=[get_word_length], # Provide the local tool
        # system_prompt="You are a helpful assistant." # Optional base prompt
    )

    user_query = "How many letters are in the word 'framework'?"
    response = await agent.run(user_msg=user_query)
    
    print(response)
    
asyncio.run(main())
```

## Exploring Further
Multi-Agent Teams: See examples/team_agent.py for setting up sequential or parallel agent workflows.

## MCP Integration: 
Check examples/minimal_react_mcp_test.py and examples/test_tool_agent_mcp.py (ensure the corresponding MCP server like examples/minimal_mcp_server.py is running).
Google GenAI: Modify the Quick Start to use GoogleOpenAICompatService instead of GroqService (ensure GOOGLE_API_KEY is set and openai library is installed).
Built-in Tools: Explore the tools provided in clap.tools (like duckduckgo_search, scrape_url, etc.).


License
This project is licensed under the terms of the Apache License 2.0. See the LICENSE file for details.
