import asyncio
import os
import shutil
import time
import json
import uuid
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

from clap import (
    Agent, Team, ToolAgent, tool,
    LLMServiceInterface, GroqService, GoogleOpenAICompatService, 
    EmbeddingFunctionInterface, 
    VectorStoreInterface, QueryResult, 
    duckduckgo_search
)


from clap.llm_services.ollama_service import OllamaOpenAICompatService as OllamaService

from clap.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddings
from clap.embedding.ollama_embedding import OllamaEmbeddings, KNOWN_OLLAMA_EMBEDDING_DIMENSIONS
from clap.embedding.fastembed_embedding import FastEmbedEmbeddings, KNOWN_FASTEMBED_DIMENSIONS as FE_KNOWN_DIMS

from clap.vector_stores.chroma_store import ChromaStore
from clap.vector_stores.qdrant_store import QdrantStore


from qdrant_client import models as qdrant_models
QDRANT_CLIENT_INSTALLED = True


try:
    CHROMA_CLIENT_INSTALLED = True 
    import chromadb 
except ImportError:
    CHROMA_CLIENT_INSTALLED = False



from clap import MCPClientManager, SseServerConfig
from pydantic import HttpUrl 

from clap.utils.rag_utils import load_pdf_file, chunk_text_by_fixed_size

load_dotenv()


PDF_PATH = "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/handsonml.pdf"
DB_BASE_PATH = "./clap_suite_dbs"

GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
OLLAMA_LLM_MODEL = "llama3.2:latest" 
GOOGLE_LLM_MODEL = "gemini-2.5-flash-preview-04-17" 


ST_EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest" 
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"



OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") 


CHUNK_SIZE, CHUNK_OVERLAP = 400, 40


PDF_CONTENT_CACHE: Optional[str] = None
MCP_MANAGER_INSTANCE: Optional[MCPClientManager] = None
MCP_ENABLED = False


def print_section_header(title: str): print(f"\n\n{'='*25} {title.upper()} {'='*25}")
def print_test_case_header(name: str): print(f"\n--- Test Case: {name} ---")
def print_query(query:str): print(f"User Query/Task: {query}")
def print_agent_response(response: Any):
    final_output = response.get("output") if isinstance(response, dict) else response
    print(f"Agent Response:\n{final_output}")
    print("-" * 50)

async def get_pdf_content() -> str:
    global PDF_CONTENT_CACHE
    if PDF_CONTENT_CACHE is None:
        if not os.path.exists(PDF_PATH):
            print(f"ERROR: PDF file not found: '{PDF_PATH}'. RAG tests limited."); PDF_CONTENT_CACHE = ""; return ""
        print(f"Loading PDF '{PDF_PATH}' (once)..."); PDF_CONTENT_CACHE = load_pdf_file(PDF_PATH)
        if not PDF_CONTENT_CACHE: print(f"Failed to load content from '{PDF_PATH}'.")
    return PDF_CONTENT_CACHE

def get_mcp_manager() -> Optional[MCPClientManager]:
    global MCP_MANAGER_INSTANCE
    if MCP_MANAGER_INSTANCE is None:
        try:
            mcp_server_configs = {
                "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
                "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001"))
            }
            MCP_MANAGER_INSTANCE = MCPClientManager(mcp_server_configs); print("MCP Manager Initialized.")
        except Exception as e: print(f"Failed to init MCP Manager: {e}"); MCP_MANAGER_INSTANCE = None
    return MCP_MANAGER_INSTANCE

@tool
def simple_math(a: int, b: int, operation: str = "add") -> str:
    """Performs 'add' or 'subtract'. Args: a (int), b (int), operation (str)."""
    if operation == "add": return f"{a} + {b} = {a + b}"
    if operation == "subtract": return f"{a} - {b} = {a - b}"
    return "Unknown math operation."

async def setup_vector_store(
    store_type: str, db_name: str, collection_name: str, ef: Optional[EmbeddingFunctionInterface]
) -> Optional[VectorStoreInterface]:
    db_path = os.path.join(DB_BASE_PATH, db_name)
    if os.path.exists(db_path): shutil.rmtree(db_path)
    else: os.makedirs(db_path, exist_ok=True)
    print(f"Setting up {store_type} at '{db_path}' for collection '{collection_name}'...")
    store: Optional[VectorStoreInterface] = None
    try:
        if store_type == "ChromaStore" and CHROMA_CLIENT_INSTALLED:
            store = ChromaStore(path=db_path, collection_name=collection_name, embedding_function=ef)
        elif store_type == "QdrantStore" and QDRANT_CLIENT_INSTALLED:
            if ef is None:
                print(f"ERROR: QdrantStore requires an explicit embedding function for setup. Skipping {collection_name}.")
                return None
            store = await QdrantStore.create(
                collection_name=collection_name, embedding_function=ef, path=db_path,
                recreate_collection_if_exists=True, distance_metric=qdrant_models.Distance.COSINE )
        else: print(f"Unsupported store_type '{store_type}' or client not installed."); return None
        pdf_content = await get_pdf_content()
        if pdf_content and store:
            chunks = chunk_text_by_fixed_size(pdf_content, CHUNK_SIZE, CHUNK_OVERLAP)
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            metadatas = [{"source": os.path.basename(PDF_PATH), "chunk_index": i} for i, _ in enumerate(chunks)]
            if chunks: await store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
            print(f"Ingested {len(chunks)} chunks into {collection_name}.")
        elif not pdf_content: print("No PDF content to ingest for RAG.")
        return store
    except Exception as e: print(f"ERROR setting up {store_type} '{collection_name}': {e}"); return None


async def cleanup_vector_store(store: Optional[VectorStoreInterface], db_name: str):
    if store and hasattr(store, 'close') and callable(store.close): await store.close()

async def run_llm_service_tests(llm_service: LLMServiceInterface, llm_model: str, service_name: str):
    print_section_header(f"{service_name} LLM Tests (ToolAgent & ReactAgent Basic)")
    agent1 = ToolAgent(llm_service=llm_service, model=llm_model, tools=[])
    query1 = "What is the capital of Japan?"
    print_test_case_header(f"{service_name} ToolAgent DirectQ"); print_query(query1); response1 = await agent1.run(user_msg=query1); print_agent_response(response1)
    agent2 = ToolAgent(llm_service=llm_service, model=llm_model, tools=[simple_math])
    query2 = "What is 150 plus 75?"
    print_test_case_header(f"{service_name} ToolAgent LocalTool"); print_query(query2); response2 = await agent2.run(user_msg=query2); print_agent_response(response2)
    agent3 = Agent(name=f"{service_name}Thinker", backstory="I break problems down.",
                   task_description="Explain pros/cons of remote work, 3 points each.",
                   llm_service=llm_service, model=llm_model)
    print_test_case_header(f"{service_name} ReactAgent Thought"); print_query(agent3.task_description); response3 = await agent3.run(); print_agent_response(response3)

async def run_rag_tests_for_service(
    llm_service: LLMServiceInterface, llm_model: str, service_name: str,
    ef: EmbeddingFunctionInterface, ef_name: str,
    store_type: str, store_db_name_suffix: str ):
    collection_name = f"rag_{service_name.lower()}_{ef_name.lower()}_{store_db_name_suffix}"
    db_name = f"{service_name.lower()}_{ef_name.lower()}_{store_db_name_suffix}"
    print_section_header(f"{service_name} RAG ({ef_name} on {store_type})")
    vector_store = await setup_vector_store(store_type, db_name, collection_name, ef)
    if not vector_store: print(f"Skipping RAG for {service_name}/{ef_name}/{store_type} due to VS setup error."); return
    agent_rag = Agent( name=f"{service_name}{ef_name.replace('_', '')}RAGer", backstory="I answer from the ML book.",
        task_description="Explain what a confusion matrix is, based on the book.",
        llm_service=llm_service, model=llm_model, vector_store=vector_store )
    print_test_case_header(f"{service_name} ReactAgent RAG"); print_query(agent_rag.task_description); response_rag = await agent_rag.run(); print_agent_response(response_rag)
    agent_tool_rag = ToolAgent( llm_service=llm_service, model=llm_model, vector_store=vector_store )
    query_tool_rag = "Typical steps in an ML project according to the book?"
    print_test_case_header(f"{service_name} ToolAgent RAG"); print_query(query_tool_rag); response_tool_rag = await agent_tool_rag.run(user_msg=query_tool_rag); print_agent_response(response_tool_rag)
    await cleanup_vector_store(vector_store, db_name)

async def run_rag_tests_chroma_default_ef(llm_service: LLMServiceInterface, llm_model: str, service_name: str):
    collection_name = f"rag_{service_name.lower()}_chroma_default"
    db_name = f"{service_name.lower()}_chroma_default_db"
    print_section_header(f"{service_name} RAG (ChromaDB with Default EF)")
    vector_store = await setup_vector_store("ChromaStore", db_name, collection_name, ef=None)
    if not vector_store: print(f"Skipping RAG for {service_name}/ChromaDefault due to VS error."); return
    agent_rag = Agent(name=f"{service_name}ChromaDefRAG", backstory="ML book expert via Chroma default EF.",
        task_description="What does the book say about model evaluation techniques?",
        llm_service=llm_service, model=llm_model, vector_store=vector_store)
    print_test_case_header(f"{service_name} ReactAgent RAG (Chroma Default)"); print_query(agent_rag.task_description); response_rag = await agent_rag.run(); print_agent_response(response_rag)
    await cleanup_vector_store(vector_store, db_name)

async def run_team_tests_for_service(llm_service: LLMServiceInterface, llm_model: str, service_name: str, mcp_manager: Optional[MCPClientManager]):
    print_section_header(f"{service_name} Team Test (RAG + Local + MCP)")
    st_ef = SentenceTransformerEmbeddings(model_name=ST_EMBED_MODEL)
    rag_db_name = f"team_rag_{service_name.lower()}"; rag_collection_name = "team_rag_coll"
    vector_store = await setup_vector_store("ChromaStore", rag_db_name, rag_collection_name, st_ef)
    if not vector_store: print(f"Skipping Team RAG for {service_name} due to VS setup error."); return
    with Team() as team:
        researcher = Agent( name=f"{service_name}Researcher", backstory="I find info.",
            task_description="Define 'validation set' using the book and web search.",
            llm_service=llm_service, model=llm_model, tools=[duckduckgo_search], vector_store=vector_store)
        calculator = Agent( name=f"{service_name}Calculator", backstory="I do math.",
            task_description="If researcher found 3 points, add 5 to it using 'add' tool, then multiply the sum by 2 using 'simple_math'.",
            llm_service=llm_service, model=llm_model, tools=[simple_math],
            mcp_manager=mcp_manager if MCP_ENABLED else None, mcp_server_names=["adder_server"] if MCP_ENABLED else None)
        reporter = Agent( name=f"{service_name}Reporter", backstory="I summarize.",
            task_description="Combine researcher and calculator outputs into a report.",
            llm_service=llm_service, model=llm_model)
        researcher >> calculator >> reporter
        await team.run()
    print_test_case_header(f"{service_name} Team Report"); print_query("Team Task: Validation sets, calc, report.")
    print_agent_response(team.results.get(f"{service_name}Reporter", {}))
    await cleanup_vector_store(vector_store, rag_db_name)

async def main():
    global MCP_ENABLED
    overall_start_time = time.time()
    if not os.path.exists(DB_BASE_PATH): os.makedirs(DB_BASE_PATH)
    await get_pdf_content()

    
    ollama_service: Optional[OllamaService] = None
    if os.getenv("RUN_OLLAMA_TESTS", "true").lower() == "true":
        try:
            ollama_service = OllamaService(
                default_model=OLLAMA_LLM_MODEL,
                base_url=f"{OLLAMA_HOST}/v1" # Use the global OLLAMA_HOST
            )
            print("Ollama Service Initialized.")
        except Exception as e: print(f"Failed to init Ollama Service: {e}")

    # ... (GroqService and GoogleOpenAICompatService initialization remains the same) ...
    groq_service: Optional[GroqService] = None
    if os.getenv("GROQ_API_KEY"):
        try: groq_service = GroqService(); print("Groq Service Initialized.")
        except Exception as e: print(f"Failed to init Groq Service: {e}")

    google_service: Optional[GoogleOpenAICompatService] = None
    if os.getenv("GOOGLE_API_KEY"):
        try: google_service = GoogleOpenAICompatService(); print("Google Service Initialized.")
        except Exception as e: print(f"Failed to init Google Service: {e}")


    st_ef = SentenceTransformerEmbeddings(model_name=ST_EMBED_MODEL)
    
    ollama_ef: Optional[OllamaEmbeddings] = None
    if ollama_service:
       
        if OLLAMA_EMBED_MODEL in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS:
            try:
                ollama_ef = OllamaEmbeddings(
                    model_name=OLLAMA_EMBED_MODEL,
                    ollama_host=OLLAMA_HOST 
                )
            except Exception as e: print(f"Could not initialize OllamaEmbeddings: {e}")
        else:
            print(f"Warning: Ollama embed model '{OLLAMA_EMBED_MODEL}' not in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS. Ollama RAG tests will be skipped.")
    
    fast_ef: Optional[FastEmbedEmbeddings] = None
    # FE_KNOWN_DIMS is imported at the top
    try:
        if FASTEMBED_MODEL in FE_KNOWN_DIMS:
            fast_ef = FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)
        else:
            print(f"Warning: FastEmbed model '{FASTEMBED_MODEL}' not in FE_KNOWN_DIMS. FastEmbed RAG will be skipped.")
    except NameError: # If FastEmbedEmbeddings or FE_KNOWN_DIMS not defined due to import error
        print("FastEmbedEmbeddings or its KNOWN_DIMS not available.")
    except Exception as e:
        print(f"Could not initialize FastEmbedEmbeddings: {e}")
    if fast_ef: print("FastEmbedEmbeddings initialized (RAG tests may be commented out).")
    # print("INFO: FastEmbed RAG tests are currently commented out to save time.")


    mcp_manager = get_mcp_manager()
    MCP_ENABLED = mcp_manager is not None
    if not MCP_ENABLED: print("MCP Manager not available. MCP tool tests in Team will be skipped.")

    services_to_test = []
    if ollama_service: services_to_test.append({"service": ollama_service, "model": OLLAMA_LLM_MODEL, "name": "Ollama"})
    if groq_service: services_to_test.append({"service": groq_service, "model": GROQ_LLM_MODEL, "name": "Groq"})
    if google_service: services_to_test.append({"service": google_service, "model": GOOGLE_LLM_MODEL, "name": "Google"})

    for test_config in services_to_test:
        svc, model, name = test_config["service"], test_config["model"], test_config["name"]
        await run_llm_service_tests(svc, model, name)

        if name == "Ollama":
            if ollama_ef:
                if CHROMA_CLIENT_INSTALLED: await run_rag_tests_for_service(svc, model, name, ollama_ef, "OllamaEF", "ChromaStore", f"{name.lower()}_chroma_ollama")
                if QDRANT_CLIENT_INSTALLED: await run_rag_tests_for_service(svc, model, name, ollama_ef, "OllamaEF", "QdrantStore", f"{name.lower()}_qdrant_ollama")
        else: # For Groq and Google, use ST_EF as the primary EF for RAG
            if CHROMA_CLIENT_INSTALLED: await run_rag_tests_for_service(svc, model, name, st_ef, "ST_EF", "ChromaStore", f"{name.lower()}_chroma_st")
            if QDRANT_CLIENT_INSTALLED: await run_rag_tests_for_service(svc, model, name, st_ef, "ST_EF", "QdrantStore", f"{name.lower()}_qdrant_st")
            # FastEmbed RAG tests are commented out here
            # if fast_ef:
            #     if CHROMA_CLIENT_INSTALLED: await run_rag_tests_for_service(svc, model, name, fast_ef, "FastEF", "ChromaStore", f"{name.lower()}_chroma_fe")
            #     if QDRANT_CLIENT_INSTALLED: await run_rag_tests_for_service(svc, model, name, fast_ef, "FastEF", "QdrantStore", f"{name.lower()}_qdrant_fe")

        # Chroma Default EF test for all services
        if CHROMA_CLIENT_INSTALLED: await run_rag_tests_chroma_default_ef(svc, model, name)
        
        await run_team_tests_for_service(svc, model, name, mcp_manager)
        if hasattr(svc, 'close') and callable(svc.close): await svc.close()

    if mcp_manager: await mcp_manager.disconnect_all()
    print(f"\n\nTotal Comprehensive Test Suite Took: {time.time() - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
