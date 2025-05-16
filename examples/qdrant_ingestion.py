# --- START OF FILE examples/qdrant_ingestion_test.py ---
import asyncio
import os
import shutil
import time
import json
import uuid
from dotenv import load_dotenv
from typing import Optional

from clap import Agent
from clap.vector_stores.qdrant_store import QdrantStore
from clap.utils.rag_utils import (
    load_pdf_file,
    chunk_text_by_fixed_size
)
from clap.llm_services.groq_service import GroqService 
from clap.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddings
from clap.embedding.fastembed_embedding import FastEmbedEmbeddings


# Qdrant models for distance
try:
    from qdrant_client import models as qdrant_models
except ImportError:
    print("ERROR: qdrant-client not found. Run: pip install 'qdrant-client[fastembed]'")
    exit(1)


# --- Configuration ---
load_dotenv()
# !!! IMPORTANT: Update this path to your large PDF file !!!
PDF_PATH = "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/Hands_On_ML.pdf" # Or the actual path to your PDF
# ---
DB_BASE_PATH = "./qdrant_test_dbs" # Base directory for Qdrant DBs
COLLECTION_NAME_CUSTOM_EF = "ml_book_custom_ef"
COLLECTION_NAME_FASTEMBED = "ml_book_fastembed"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LLM_MODEL = "llama-3.3-70b-versatile" # Or your preferred model

# Qdrant's default fastembed model and its known dimension
# (Ensure this matches the actual default or the one you want to test)
QDRANT_INTERNAL_FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


async def perform_rag_cycle(
    db_path: str,
    collection_name: str,
    embedding_function_instance: Optional[SentenceTransformerEmbeddings], # Explicitly typed for this test
    use_internal_fastembed_flag: bool,
    fastembed_model_to_use: Optional[str]
):
    """Performs a full RAG cycle: setup, ingest, query."""
    cycle_start_time = time.time()
    if embedding_function_instance:
        print(f"Embedding: Custom SentenceTransformerEmbeddings")
    elif use_internal_fastembed_flag and fastembed_model_to_use:
        print(f"Embedding: Qdrant Internal FastEmbed (model: {fastembed_model_to_use})")
    else:
        print("Error: Invalid embedding configuration for cycle.")
        return

        # --- Inside perform_rag_cycle in the example script ---
    # Replace the old initialization block with this:
    try:
        qdrant_store = await QdrantStore.create(
            collection_name=collection_name,
            embedding_function=embedding_function_instance,
            use_internal_fastembed=use_internal_fastembed_flag,
            fastembed_model_name=fastembed_model_to_use, # type: ignore
            path=db_path,
            recreate_collection_if_exists=True,
            distance_metric=qdrant_models.Distance.COSINE
        )
    except Exception as e:
        print(f"FATAL: Failed to create QdrantStore: {e}")
        return # Stop this cycle if store creation fails
    # --- Continue with the rest of the cycle ---


    # --- 2. Load and Chunk PDF ---
    pdf_content = load_pdf_file(PDF_PATH)
    if not pdf_content:
        print(f"Failed to load PDF content from {PDF_PATH}.")
        await qdrant_store.close()
        return

    print("Chunking PDF content...")
    chunks = chunk_text_by_fixed_size(pdf_content, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Generated {len(chunks)} chunks.")

    # --- 3. Prepare and Add Data ---
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": os.path.basename(PDF_PATH), "chunk_idx": i} for i in range(len(chunks))]

    print("Adding chunks to vector store (embedding process)...")
    ingestion_start_time = time.time()
    if chunks:
        await qdrant_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
        print("Ingestion complete.")
    else:
        print("No chunks generated to add.")
    ingestion_duration = time.time() - ingestion_start_time
    print(f"Data Ingestion Took: {ingestion_duration:.2f} seconds.")

    llm_service = GroqService()
    rag_agent = Agent(
        name=f"BookExpert_{collection_name}",
        backstory="Assistant answering questions based *only* on the provided Machine Learning book context.",
        task_description="Placeholder - will be set by query",
        llm_service=llm_service,
        model=LLM_MODEL,
        vector_store=qdrant_store
    )

    # --- 5. Ask a Question ---
    user_query = "Compare and contrast decision trees and support vector machines as explained in the book."
    print(f"\n--- RAG Query for '{collection_name}' ---")
    print(f"User Query: {user_query}")

    rag_agent.task_description = user_query # Set the actual query
    query_start_time = time.time()
    result = await rag_agent.run()
    query_duration = time.time() - query_start_time
    print(f"RAG Query Took: {query_duration:.2f} seconds.")

    # --- 6. Display Result ---
    print("\n--- Final Agent Answer ---")
    print(result.get("output", "Agent failed to produce an answer."))

    # --- 7. Cleanup ---
    await qdrant_store.close() # Important to close Qdrant, especially file-based
    cycle_duration = time.time() - cycle_start_time
    print(f"--- RAG Cycle for {collection_name} Took: {cycle_duration:.2f} seconds ---")


async def main():
    overall_start_time = time.time()

    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF file not found at '{PDF_PATH}'. Please update the PDF_PATH variable.")
        return
    if not os.getenv("GROQ_API_KEY"): # Add check for GOOGLE_API_KEY if using Google LLM
         print("Warning: LLM API Key (e.g., GROQ_API_KEY) not found in environment variables.")
         # return # Optionally exit if key is strictly needed

    # --- Test Cycle 1: Custom EmbeddingFunctionInterface ---
    print("\n\n========== TEST 1: QDRANT WITH CUSTOM SentenceTransformerEmbeddings ==========")
    st_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # Default model
    db_path_custom_ef = os.path.join(DB_BASE_PATH, "custom_ef_db")
    await perform_rag_cycle(
        db_path=db_path_custom_ef,
        collection_name=COLLECTION_NAME_CUSTOM_EF,
        embedding_function_instance=st_embeddings,
        use_internal_fastembed_flag=False,
        fastembed_model_to_use=None
    )

    await asyncio.sleep(2) # Small pause

    # --- Test Cycle 2: Qdrant WITH FastEmbed Wrapper ---
    print("\n\n========== TEST 2: QDRANT WITH FastEmbedEmbeddings WRAPPER ==========")
    try:
        # Instantiate the wrapper
        fast_embed_ef = FastEmbedEmbeddings(model_name=QDRANT_INTERNAL_FASTEMBED_MODEL_NAME)

        db_path_fastembed = os.path.join(DB_BASE_PATH, "fastembed_via_wrapper_db")
        await perform_rag_cycle(
            db_path=db_path_fastembed,
            collection_name=COLLECTION_NAME_FASTEMBED, # Can reuse name or make unique
            # Pass the wrapper instance as the embedding_function
            embedding_function_instance=fast_embed_ef,
            # These are now irrelevant as we provide the EF
            use_internal_fastembed_flag=False,
            fastembed_model_to_use=None
        )
    except ImportError as e:
        print(f"\nSkipping FastEmbed test cycle: {e}")
    except Exception as e:
         print(f"\nError during FastEmbed test cycle: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    asyncio.run(main())

# --- END OF FILE examples/qdrant_ingestion_test.py ---