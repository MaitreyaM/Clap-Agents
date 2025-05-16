import asyncio
import os
from dotenv import load_dotenv
from clap import Agent, Team, ToolAgent # Core CLAP
from clap.tool_pattern.tool import tool # For defining local tools

# --- LLM Services ---
from clap.llm_services.ollama_llm_service import OllamaOpenAICompatService
from clap.llm_services.groq_service import GroqService
from clap.llm_services.google_openai_compat_service import GoogleOpenAICompatService

# --- Embedding Functions ---
from clap.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddings
from clap.embedding.ollama_embedding import OllamaEmbeddings
from clap.embedding.fastembed_embedding import FastEmbedEmbeddings # If you keep this

# --- Vector Stores ---
from clap.vector_stores.chroma_store import ChromaStore
from clap.vector_stores.qdrant_store import QdrantStore

# --- Pre-built Tools ---
from clap.tools.web_search import duckduckgo_search
from clap.tools.web_crawler import scrape_url 

load_dotenv()

OLLAMA_LLM_MODEL = "llama3.2:latest" # Or your preferred Ollama model
OLLAMA_EMBED_MODEL = "nomic-embed-text" # Or your preferred Ollama embedding model
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
OLLAMA_HOST = "http://localhost:11434"

@tool
def get_capital(country: str) -> str:
    """Returns the capital of a given country."""
    capitals = {"france": "Paris", "germany": "Berlin", "japan": "Tokyo"}
    return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

@tool
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    if city.lower() == "london": return "Weather in London is 15°C and cloudy."
    return f"Weather for {city} is sunny."

# async def main():
#     ollama_service = OllamaOpenAICompatService(default_model=OLLAMA_LLM_MODEL, base_url=f"{OLLAMA_HOST}/v1")
    
#     agent = ToolAgent(llm_service=ollama_service, model=OLLAMA_LLM_MODEL, tools=[])
    
#     query = "Hello, how are you today?"
#     response = await agent.run(user_msg=query)

#     await ollama_service.close()

async def main():
    print(f"Using Ollama model: {OLLAMA_LLM_MODEL} on {OLLAMA_HOST}")
    ollama_service = OllamaOpenAICompatService(default_model=OLLAMA_LLM_MODEL, base_url=f"{OLLAMA_HOST}/v1")
    
    # ToolAgent initialized with the get_capital tool
    agent = ToolAgent(llm_service=ollama_service, model=OLLAMA_LLM_MODEL, tools=[get_capital])
    
    query = "What is the capital of France?"
    response = await agent.run(user_msg=query)
    print("Ollama ToolAgent (Local Tool)", query, response)

    query_unknown = "What is the capital of Wonderland?"
    response_unknown = await agent.run(user_msg=query_unknown)
    print("Ollama ToolAgent (Local Tool - Unknown)", query_unknown, response_unknown)

    await ollama_service.close()

asyncio.run(main())