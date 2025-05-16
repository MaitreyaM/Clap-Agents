# --- START OF FILE examples/test_rag_ollama_minimal.py ---
import asyncio
import os
import shutil
import time
from dotenv import load_dotenv
import uuid
from clap import Agent
from clap.vector_stores.qdrant_store import QdrantStore
from clap.utils.rag_utils import load_pdf_file, chunk_text_by_fixed_size
from clap.llm_services.ollama_llm_service import OllamaOpenAICompatService
from clap.embedding.ollama_embedding import OllamaEmbeddings, KNOWN_OLLAMA_EMBEDDING_DIMENSIONS
from qdrant_client import models as qdrant_models

load_dotenv()

PDF_PATH = "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/Hands_On_ML.pdf" # Ensure this PDF exists
DB_PATH = "./ollama_rag_qdrant_minimal_db"
COLLECTION_NAME = "ml_book_ollama_minimal"
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3.2:latest" # Ensure pulled: ollama pull llama3
OLLAMA_EMBED_MODEL = "nomic-embed-text" # Ensure pulled: ollama pull nomic-embed-text

async def run_minimal_ollama_rag():
    start_time = time.time()
    if OLLAMA_EMBED_MODEL not in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS:
        print(f"ERROR: Dimension for '{OLLAMA_EMBED_MODEL}' unknown."); return

    print(f"Minimal Ollama RAG Test (LLM: {OLLAMA_LLM_MODEL}, Embed: {OLLAMA_EMBED_MODEL})")

    if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)

    ollama_ef = OllamaEmbeddings(model_name=OLLAMA_EMBED_MODEL, ollama_host=OLLAMA_HOST)
    vector_store = await QdrantStore.create(
        collection_name=COLLECTION_NAME, embedding_function=ollama_ef, path=DB_PATH,
        recreate_collection_if_exists=True, distance_metric=qdrant_models.Distance.COSINE
    )

    print("Loading & Chunking PDF...")
    pdf_content = load_pdf_file(PDF_PATH)
    if not pdf_content: await vector_store.close(); return
    chunks = chunk_text_by_fixed_size(pdf_content, CHUNK_SIZE, CHUNK_OVERLAP)
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))] 
    metadatas = [{"src": os.path.basename(PDF_PATH)} for _ in range(len(chunks))]

    print(f"Ingesting {len(chunks)} chunks...")
    ingest_start = time.time()
    if chunks: await vector_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"Ingestion Took: {time.time() - ingest_start:.2f}s")

    ollama_llm_service = OllamaOpenAICompatService(default_model=OLLAMA_LLM_MODEL, base_url=f"{OLLAMA_HOST}/v1")
    rag_agent = Agent(
        name="OllamaMinimalExpert", backstory="Answer from book context.",
        task_description="What are Generative Adversarial Networks?", # Will be overwritten
        llm_service=ollama_llm_service, model=OLLAMA_LLM_MODEL, vector_store=vector_store
    )

    user_query = "Explain Generative Adversarial Networks (GANs) based on the book."
    print(f"\nUser Query: {user_query}")
    rag_agent.task_description = user_query
    
    query_start = time.time()
    result = await rag_agent.run()
    print(f"Query Took: {time.time() - query_start:.2f}s")
    print("\n--- Agent Answer ---")
    print(result.get("output", "No answer."))

    await vector_store.close()
    await ollama_llm_service.close()
    print(f"\nTotal Test Took: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    print(f"Ensure Ollama server (models: {OLLAMA_LLM_MODEL}, {OLLAMA_EMBED_MODEL}) is running.")
    asyncio.run(run_minimal_ollama_rag())
# --- END OF FILE examples/test_rag_ollama_minimal.py ---