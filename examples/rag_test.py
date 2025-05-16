
import asyncio
import os
import shutil
from dotenv import load_dotenv

# --- CLAP Imports ---
from clap import Agent, Team # Assuming Agent is modified as above
from clap.vector_stores.chroma_store import ChromaStore
from clap.utils.rag_utils import chunk_text_by_separator # Example chunker
from clap.llm_services.groq_service import GroqService 
from clap import GoogleOpenAICompatService


# Requires: pip install sentence-transformers
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
DEFAULT_EF = SentenceTransformerEmbeddingFunction()



# --- Config ---
load_dotenv()
CHROMA_DB_PATH = "./rag_test_chroma_db"
COLLECTION_NAME = "clap_rag_demo"

# --- Sample Data ---
SAMPLE_DOCUMENTS = [
    """Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water – hence the name photosynthesis, from the Greek φῶς, phos, "light", and σύνθεσις, synthesis, "putting together".""",
    """The Formula One World Championship, commonly known as Formula 1 or F1, is the highest class of international racing for open-wheel single-seater formula racing cars sanctioned by the Fédération Internationale de l'Automobile (FIA). The World Drivers' Championship, which became the FIA Formula One World Championship in 1981, has been one of the premier forms of racing around the world since its inaugural season in 1950.""",
    """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Locally nicknamed "La dame de fer" (French for "Iron Lady"), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."""
]

async def setup_vector_store():
    """Sets up the ChromaDB store and adds sample data."""
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")
    # Clean up previous run if exists
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # Initialize ChromaStore
    vector_store = ChromaStore(
        path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=DEFAULT_EF
    )

    # Prepare data (Chunking is simple here, could use rag_utils)
    all_chunks = []
    all_ids = []
    all_metadatas = []
    doc_id_counter = 1
    chunk_id_counter = 1

    for doc in SAMPLE_DOCUMENTS:
        # Example: Simple chunking (could use rag_utils.chunk_text_...)
        chunks = chunk_text_by_separator(doc, separator=". ") # Split by sentence
        for chunk in chunks:
            if not chunk.strip(): continue
            chunk_id = f"doc{doc_id_counter}_chunk{chunk_id_counter}"
            all_chunks.append(chunk.strip() + ".") # Add back period for context
            all_ids.append(chunk_id)
            all_metadatas.append({"source_doc_id": f"doc{doc_id_counter}"})
            chunk_id_counter += 1
        doc_id_counter += 1
        chunk_id_counter = 1 # Reset chunk counter for next doc

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

async def run_rag_agent(vector_store: ChromaStore):
    # ... (agent initialization remains the same, BUT the initial task_description doesn't matter much now)
    llm_service = GroqService() # Use Gemini
    rag_agent = Agent(
        name="RAG_Expert",
        backstory="You are an AI assistant that answers questions based on provided context...",
        task_description="Placeholder - will be overwritten in loop", # Initial value doesn't matter
        task_expected_output="A concise answer based on retrieved context...",
        llm_service=llm_service,
        model="llama-3.3-70b-versatile", # Use a highly capable model
        vector_store=vector_store
    )

    queries = [
        "What is photosynthesis?",
        "Who designed the Eiffel Tower?",
        "What is the capital of Germany?",
        "Tell me about Formula 1 racing."
    ]

    # Using Team context manager even for a single agent is fine
    with Team() as team:
        for query in queries:
            print(f"\n--- User Query: {query} ---")
            rag_agent.received_context = {} # Reset context

            # --- THE FIX ---
            # Update the agent's task description to the *current* query
            rag_agent.task_description = query
            # --- END FIX ---

            # Now, when agent.run() calls create_prompt(), self.task_description will be correct
            result = await rag_agent.run()
            output_content = result.get("output", "Agent did not produce 'output' key.")
            print(output_content)
            await asyncio.sleep(1)

async def main():
    vector_store = await setup_vector_store()
    await run_rag_agent(vector_store)

    # Clean up DB after run
    if os.path.exists(CHROMA_DB_PATH):
         print(f"\nCleaning up test database: {CHROMA_DB_PATH}")
         shutil.rmtree(CHROMA_DB_PATH) # Uncomment to auto-delete DB

asyncio.run(main())

