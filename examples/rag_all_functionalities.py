# --- START OF FILE examples/complex_rag_mcp_team_test.py ---

import asyncio
import os
import shutil
import json
from dotenv import load_dotenv
from pydantic import HttpUrl # For MCP ServerConfig

# --- CLAP Imports ---
from clap import Agent, Team
from clap.vector_stores.chroma_store import ChromaStore
from clap.utils.rag_utils import chunk_text_by_separator
from clap.llm_services.groq_service import GroqService # Or GoogleOpenAICompatService
from clap.tool_pattern.tool import tool # For local tools
from clap.mcp_client.client import MCPClientManager, SseServerConfig # MCP Client
from clap.tools.web_search import duckduckgo_search # Pre-built local tool

# --- Embedding Function Imports ---
try:
    # Requires: pip install sentence-transformers
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    DEFAULT_EF = SentenceTransformerEmbeddingFunction()
except ImportError:
    print("Warning: sentence-transformers not installed. ChromaDB might use its default EF.")
    print("Install with: pip install sentence-transformers")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    DEFAULT_EF = DefaultEmbeddingFunction()

# --- Config ---
load_dotenv()
CHROMA_DB_PATH = "./complex_test_chroma_db"
COLLECTION_NAME = "clap_complex_demo"

# --- Sample Data for RAG ---
SAMPLE_RAG_DOCUMENTS = [
    """Agent performance metrics are crucial for evaluation. Key factors include task completion rate, latency, and cost per task. Tool usage accuracy is also vital, especially in complex workflows involving external APIs or databases.""",
    """Integrating multiple AI systems often presents challenges. Ensuring reliable data flow, handling asynchronous operations gracefully, and maintaining context across different agents requires careful architectural design. Frameworks aim to simplify this.""",
    """User experience in agentic systems depends heavily on responsiveness and accuracy. Long delays caused by inefficient tool calls or complex reasoning loops can frustrate users. Optimizing the ReAct cycle is important.""",
    """Security considerations for AI agents involve managing API keys securely, validating tool inputs and outputs, and preventing prompt injection attacks. Access control for tools and data sources (like vector stores or MCP endpoints) is necessary.""",
]

# --- Local Tool Definition ---
@tool
def multiply(a: int, b: int) -> int:
    """Calculates the product of two integers."""
    print(f"[Local Tool Executing] multiply({a}, {b})")
    return a * b

# --- Vector Store Setup ---
async def setup_vector_store():
    """Sets up the ChromaDB store and adds sample data."""
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    vector_store = ChromaStore(
        path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=DEFAULT_EF
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []
    doc_id_counter = 1
    chunk_id_counter = 1
    for doc in SAMPLE_RAG_DOCUMENTS:
        chunks = chunk_text_by_separator(doc, separator=". ")
        for chunk in chunks:
            if not chunk.strip(): continue
            chunk_id = f"rag_doc{doc_id_counter}_chunk{chunk_id_counter}"
            all_chunks.append(chunk.strip() + ".")
            all_ids.append(chunk_id)
            all_metadatas.append({"source_doc": f"rag_doc{doc_id_counter}", "type": "internal_note"})
            chunk_id_counter += 1
        doc_id_counter += 1
        chunk_id_counter = 1

    print(f"Adding {len(all_chunks)} chunks to ChromaDB...")
    if all_chunks:
        await vector_store.add_documents(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadatas
        )
    else:
        print("No chunks to add.")

    print("Vector store setup complete.")
    return vector_store

# --- Main Test Function ---
async def run_complex_team():
    """Initializes and runs the complex agent team."""
    print("\n--- Running Complex RAG + MCP + Tools Team ---")

    # Initialize LLM Service (Choose one)
    llm_service = GroqService()
    # llm_service = GoogleOpenAICompatService() # Uncomment if using Google

    # Configure MCP Client
    mcp_server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
        "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001"))
    }
    mcp_manager = MCPClientManager(mcp_server_configs)

    # Setup Vector Store
    vector_store = await setup_vector_store()

    # Define the team
    try:
        with Team() as team:
            # Agent 1: Research + RAG + MCP Add
            researcher = Agent(
                name="Topic_Researcher",
                backstory="Expert researcher skilled in web search, internal knowledge retrieval (vector search), and basic calculations.",
                task_description="1. Search the web for 'challenges in multi-agent system design'. "
                                 "2. Query the internal vector store for information related to 'agent performance'. "
                                 "3. Use the remote 'add' tool to calculate 25 + 17.",
                task_expected_output="A combined summary of web search results, vector store findings on performance, and the result of the addition.",
                llm_service=llm_service,
                model="llama-3.3-70b-versatile", # Or your preferred model
                tools=[duckduckgo_search],       # Local tool
                vector_store=vector_store,       # RAG capability
                mcp_manager=mcp_manager,         # MCP capability
                mcp_server_names=["adder_server"] # Specific server for 'add'
            )

            # Agent 2: Calculations (Local + MCP Sub)
            calculator = Agent(
                name="Number_Cruncher",
                backstory="Specialist in performing calculations based on provided inputs.",
                task_description="1. Take the addition result from the Researcher (should be 42). "
                                 "2. Use the remote 'sub' tool to calculate Result - 12. "
                                 "3. Use the local 'multiply' tool to calculate the original numbers from the addition task (25 * 17).",
                task_expected_output="The result of the subtraction and the result of the multiplication.",
                llm_service=llm_service,
                model="llama-3.3-70b-versatile",
                tools=[multiply],                  # Local tool
                mcp_manager=mcp_manager,           # MCP capability
                mcp_server_names=["subtract_server"] # Specific server for 'sub'
            )

            # Agent 3: Reporting
            reporter = Agent(
                name="Report_Synthesizer",
                backstory="Skilled writer adept at summarizing complex information from multiple sources.",
                task_description="Compile all the information provided by the Researcher (web search, RAG findings, addition result) and the Calculator (subtraction result, multiplication result) into a single, concise final report paragraph.",
                task_expected_output="One paragraph summarizing all findings and calculations.",
                llm_service=llm_service,
                model="llama-3.3-70b-versatile"
                # No tools needed for this agent
            )

            # Define Dependencies
            researcher >> calculator >> reporter

            # Run the team
            await team.run()

            # Print final results from the team object
            print(json.dumps(team.results, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n--- An error occurred during team execution: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure MCP connections are closed
        print("\n--- Cleaning up MCP connections ---")
        await mcp_manager.disconnect_all()
        # Clean up DB after run
        if os.path.exists(CHROMA_DB_PATH):
            print(f"\nCleaning up test database: {CHROMA_DB_PATH}")
            # shutil.rmtree(CHROMA_DB_PATH) # Uncomment to auto-delete DB

if __name__ == "__main__":
   
    asyncio.run(run_complex_team())

