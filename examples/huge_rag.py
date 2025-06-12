import asyncio
import os
import shutil
import time
from dotenv import load_dotenv
from clap import Agent                                  
from clap.vector_stores.chroma_store import ChromaStore 
from clap.utils.rag_utils import (
    load_pdf_file,
    chunk_text_by_fixed_size
)                                                       
from clap.llm_services.google_openai_compat_service import GoogleOpenAICompatService

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
DEFAULT_EF = SentenceTransformerEmbeddingFunction()

load_dotenv()
PDF_PATH = "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/handsonml.pdf"
CHROMA_DB_PATH = "./large_pdf_chroma_db" 
COLLECTION_NAME = "ml_book_rag"
CHUNK_SIZE = 500    
CHUNK_OVERLAP = 50    
LLM_MODEL = "gemini-2.5-flash-preview-04-17" 

async def run_minimal_rag():
    start_time = time.time()

    if os.path.exists(CHROMA_DB_PATH):
        print(f"Removing existing DB at {CHROMA_DB_PATH}...")
        shutil.rmtree(CHROMA_DB_PATH)

    vector_store = ChromaStore(
        path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=DEFAULT_EF
    )

    pdf_content = load_pdf_file(PDF_PATH)

    chunks = chunk_text_by_fixed_size(pdf_content, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Generated {len(chunks)} chunks.")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": PDF_PATH, "chunk_index": i} for i in range(len(chunks))]

    if chunks:
        await vector_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
        print("Ingestion complete.")

    ingestion_time = time.time() - start_time
    print(f"Ingestion took {ingestion_time:.2f} seconds.")

    llm_service = GoogleOpenAICompatService() 
    rag_agent = Agent(
        name="Book_Expert",
        backstory="Assistant answering questions based *only* on the provided Machine Learning book context.",
        task_description="Placeholder Query",
        task_expected_output="A concise answer derived solely from the retrieved book context.",
        llm_service=llm_service,
        model=LLM_MODEL,
        vector_store=vector_store 
    )

    queries = [
    "Compare Random Forests and Gradient Boosting machines, highlighting the key differences in how they build ensembles of decision trees.",
    "Describe the concept of the 'kernel trick' as used in Support Vector Machines (SVMs) and explain its primary benefit.",]

    for q in queries:
        rag_agent.task_description = q

        result = await rag_agent.run()

        print(result.get("output", "Agent failed to produce an answer."))

        end_time = time.time()
        print(f"\nTotal process took {(end_time - start_time):.2f} seconds.")


asyncio.run(run_minimal_rag())