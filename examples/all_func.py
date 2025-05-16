# --- START OF FILE examples/test_ollama_suite.py ---
import asyncio
import os
import shutil
import time
import json
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import uuid 

# --- CLAP & Dependencies ---
from clap import Agent, Team, ToolAgent # Core CLAP
from clap.tool_pattern.tool import tool # For defining local tools
from clap.llm_services.ollama_service import OllamaOpenAICompatService
from clap.embedding.ollama_embedding import OllamaEmbeddings, KNOWN_OLLAMA_EMBEDDING_DIMENSIONS
from clap.vector_stores.qdrant_store import QdrantStore # Using Qdrant for RAG
from clap.utils.rag_utils import chunk_text_by_fixed_size # Simple chunker
from qdrant_client import models as qdrant_models # For Qdrant config

load_dotenv()

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_LLM_FOR_AGENT = "llama3.2:latest"  # Model for agent reasoning (ensure pulled)
OLLAMA_MODEL_FOR_EMBEDDINGS = "nomic-embed-text" # Model for embeddings (ensure pulled)

RAG_DB_PATH = "./ollama_suite_qdrant_db"
RAG_COLLECTION_NAME = "ollama_suite_rag_collection"
RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP = 200, 20

SAMPLE_DOCS_FOR_RAG = [
    "Ollama allows running large language models locally, such as Llama 3, Mistral, and Gemma.",
    "Embedding models like nomic-embed-text can be used with Ollama to generate vector representations of text.",
    "The CLAP framework supports integration with local Ollama instances for both LLM and embedding tasks."
]

# --- Helper Functions ---
def print_section_header(title: str):
    print(f"\n\n{'='*20} {title.upper()} {'='*20}")

def print_test_case(name: str, query: str, response: Any):
    print(f"\n--- Test Case: {name} ---")
    print(f"Query/Task: {query}")
    final_output = response.get("output") if isinstance(response, dict) else response
    print(f"Response:\n{final_output}")
    print("--------------------")

# --- Local Tools for Testing ---
@tool
def get_city_population(city_name: str) -> str:
    """Returns a fictional population for a given city."""
    populations = {"paris": "2.1 million", "tokyo": "13.9 million", "berlin": "3.6 million"}
    return populations.get(city_name.lower(), f"Population data for {city_name} unknown.")

@tool
def square_number(number: float) -> float:
    """Calculates the square of a number."""
    return number * number

# --- Test Functions ---

async def test_tool_agent_direct_answer(ollama_service: OllamaOpenAICompatService):
    print_section_header("ToolAgent - Direct Answer (No Tools Used)")
    agent = ToolAgent(
        llm_service=ollama_service,
        model=OLLAMA_LLM_FOR_AGENT,
        tools=[] # No tools provided
    )
    query = "What is the main purpose of the Ollama software?"
    response = await agent.run(user_msg=query)
    print_test_case("ToolAgent Direct Answer", query, response)

async def test_tool_agent_local_tool(ollama_service: OllamaOpenAICompatService):
    print_section_header("ToolAgent - Local Tool Usage")
    agent = ToolAgent(
        llm_service=ollama_service,
        model=OLLAMA_LLM_FOR_AGENT,
        tools=[get_city_population]
    )
    query = "What is the population of Paris?"
    response = await agent.run(user_msg=query)
    print_test_case("ToolAgent Local Tool (Paris)", query, response)

    query_unknown = "What is the population of Atlantis?"
    response_unknown = await agent.run(user_msg=query_unknown)
    print_test_case("ToolAgent Local Tool (Atlantis)", query_unknown, response_unknown)


async def setup_rag_store_for_ollama() -> Optional[QdrantStore]:
    print_section_header("Setting up RAG Vector Store with Ollama Embeddings")
    if OLLAMA_MODEL_FOR_EMBEDDINGS not in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS:
        print(f"ERROR: Dimension for Ollama embed model '{OLLAMA_MODEL_FOR_EMBEDDINGS}' unknown.")
        return None
    try:
        ollama_ef = OllamaEmbeddings(model_name=OLLAMA_MODEL_FOR_EMBEDDINGS, ollama_host=OLLAMA_HOST)
    except Exception as e:
        print(f"Failed to init OllamaEmbeddings: {e}"); return None

    if os.path.exists(RAG_DB_PATH): shutil.rmtree(RAG_DB_PATH)
    try:
        vector_store = await QdrantStore.create(
            collection_name=RAG_COLLECTION_NAME, embedding_function=ollama_ef,
            path=RAG_DB_PATH, recreate_collection_if_exists=True,
            distance_metric=qdrant_models.Distance.COSINE
        )
    except Exception as e:
        print(f"Failed to create QdrantStore: {e}"); return None

    chunks = []
    for doc in SAMPLE_DOCS_FOR_RAG:
        chunks.extend(chunk_text_by_fixed_size(doc, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP))
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": "clap_ollama_test_data"} for _ in range(len(chunks))]

    if chunks: await vector_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks into '{RAG_COLLECTION_NAME}'.")
    return vector_store


async def test_agent_rag_only(ollama_service: OllamaOpenAICompatService, vector_store: QdrantStore):
    print_section_header("Agent (ReAct) - RAG Only")
    agent = Agent(
        name="OllamaRAGer", backstory="You answer questions based on provided Ollama documentation.",
        task_description="What models can Ollama run locally?", # Will be set by query
        llm_service=ollama_service, model=OLLAMA_LLM_FOR_AGENT, vector_store=vector_store
    )
    agent.task_description = "What models can Ollama run locally?" # The actual query
    response = await agent.run()
    print_test_case("Agent RAG Only", agent.task_description, response)


async def test_agent_rag_with_local_tool(ollama_service: OllamaOpenAICompatService, vector_store: QdrantStore):
    print_section_header("Agent (ReAct) - RAG with Local Tool")
    agent = Agent(
        name="OllamaRAGToolUser",
        backstory="You use info from Ollama docs and can square numbers.",
        task_description="Based on Ollama documentation, what kind of models does it support? Then, what is 5 squared?",
        llm_service=ollama_service, model=OLLAMA_LLM_FOR_AGENT,
        vector_store=vector_store, tools=[square_number]
    )
    # The task_description is the multi-part query for this agent
    response = await agent.run()
    print_test_case("Agent RAG + Local Tool", agent.task_description, response)


async def main():
    overall_start_time = time.time()
    print("Ensure Ollama server is running and models are pulled:")
    print(f"  LLM for Agent: ollama pull {OLLAMA_LLM_FOR_AGENT}")
    print(f"  Model for Embeddings: ollama pull {OLLAMA_MODEL_FOR_EMBEDDINGS}")
    print("-" * 60)

    # Initialize Ollama LLM Service once
    ollama_llm_service = OllamaOpenAICompatService(
        default_model=OLLAMA_LLM_FOR_AGENT,
        base_url=f"{OLLAMA_HOST}/v1"
    )

    # --- Run ToolAgent Tests ---
    await test_tool_agent_direct_answer(ollama_llm_service)
    await test_tool_agent_local_tool(ollama_llm_service)

    # --- Setup RAG and Run Agent (ReAct) Tests ---
    rag_vector_store = await setup_rag_store_for_ollama()
    if rag_vector_store:
        await test_agent_rag_only(ollama_llm_service, rag_vector_store)
        await test_agent_rag_with_local_tool(ollama_llm_service, rag_vector_store)
        await rag_vector_store.close() # Close vector store connection
    else:
        print("Skipping RAG tests due to vector store setup failure.")

    await ollama_llm_service.close() # Close LLM service client

    print(f"\n\nTotal Ollama Test Suite Took: {time.time() - overall_start_time:.2f} seconds.")
    if os.path.exists(RAG_DB_PATH):
        print(f"Cleaning up test database: {RAG_DB_PATH}")
        shutil.rmtree(RAG_DB_PATH)

if __name__ == "__main__":
    asyncio.run(main())
