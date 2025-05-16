# --- START OF MODIFIED examples/tool_agent_rag.py ---
import asyncio
import os
import shutil
import time
import json
import uuid # <--- IMPORT THE uuid MODULE
from dotenv import load_dotenv

# ... (other imports like ToolAgent, OllamaOpenAICompatService, OllamaEmbeddings, QdrantStore, etc.)
from clap import ToolAgent
from clap.llm_services.ollama_service import OllamaOpenAICompatService
from clap.embedding.ollama_embedding import OllamaEmbeddings, KNOWN_OLLAMA_EMBEDDING_DIMENSIONS
from clap.vector_stores.qdrant_store import QdrantStore
from clap.utils.rag_utils import chunk_text_by_fixed_size
from qdrant_client import models as qdrant_models


load_dotenv()

# ... (Configuration constants remain the same) ...
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_LLM_FOR_AGENT = "llama3.2:latest"
OLLAMA_MODEL_FOR_EMBEDDINGS = "nomic-embed-text"
RAG_DB_PATH = "./ollama_toolagent_rag_db"
RAG_COLLECTION_NAME = "toolagent_rag_docs"
RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP = 200, 20
SAMPLE_DOCS = [
    "The ToolAgent can now perform RAG by querying a vector store.",
    "Vector query results are passed as observations back to the LLM for a final answer.",
    "This allows simpler agents to access knowledge bases without a full ReAct loop."
]



async def main():
    print_section_header("ToolAgent - RAG Test")
    if OLLAMA_MODEL_FOR_EMBEDDINGS not in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS:
        print(f"ERROR: Dimension for Ollama embed model '{OLLAMA_MODEL_FOR_EMBEDDINGS}' unknown."); return

    ollama_ef = OllamaEmbeddings(model_name=OLLAMA_MODEL_FOR_EMBEDDINGS, ollama_host=OLLAMA_HOST)
    if os.path.exists(RAG_DB_PATH): shutil.rmtree(RAG_DB_PATH)
    vector_store = await QdrantStore.create(
        collection_name=RAG_COLLECTION_NAME, embedding_function=ollama_ef,
        path=RAG_DB_PATH, recreate_collection_if_exists=True,
        distance_metric=qdrant_models.Distance.COSINE
    )
    chunks = []; # ids = []; # No longer needed here if generating UUIDs per chunk
    all_ids = [] # Use a new list for UUIDs
    metadatas = []

    doc_counter = 0
    for i, doc_text_content in enumerate(SAMPLE_DOCS): # Renamed 'doc' to avoid conflict if you use 'doc' later
        doc_chunks = chunk_text_by_fixed_size(doc_text_content, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
        for j, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            # --- MODIFIED ID GENERATION ---
            all_ids.append(str(uuid.uuid4())) # Generate a new UUID for each chunk
            # --- END MODIFICATION ---
            metadatas.append({"source": f"sample_doc_{i}", "original_chunk_id": f"doc{i}_chunk{j}"})
        doc_counter +=1

    if chunks: await vector_store.add_documents(documents=chunks, ids=all_ids, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks.")

    ollama_llm_service = OllamaOpenAICompatService(default_model=OLLAMA_LLM_FOR_AGENT, base_url=f"{OLLAMA_HOST}/v1")
    
    agent = ToolAgent(
        llm_service=ollama_llm_service,
        model=OLLAMA_LLM_FOR_AGENT,
        vector_store=vector_store
    )
    
    query = "How can ToolAgent perform RAG?"
    response = await agent.run(user_msg=query)
    print("ToolAgent with RAG", query, response)

    await vector_store.close()
    await ollama_llm_service.close()
    if os.path.exists(RAG_DB_PATH): shutil.rmtree(RAG_DB_PATH)

def print_section_header(title): print(f"\n{'='*20} {title.upper()} {'='*20}")

if __name__ == "__main__":
    print(f"Ensure Ollama (models: {OLLAMA_LLM_FOR_AGENT}, {OLLAMA_MODEL_FOR_EMBEDDINGS}) is running.")
    asyncio.run(main())

# --- END OF MODIFIED examples/tool_agent_rag.py ---