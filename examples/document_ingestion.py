# --- START OF FILE examples/document_ingestion_test.py ---

import asyncio
import os
import shutil
import json
import uuid # For generating unique IDs
from dotenv import load_dotenv

# --- CLAP Imports ---
from clap.vector_stores.chroma_store import ChromaStore
from clap.utils.rag_utils import (
    load_text_file,
    load_pdf_file,
    load_csv_file,
    chunk_text_by_fixed_size # Use the fixed size chunker
)

# --- Embedding Function Imports ---
try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    DEFAULT_EF = SentenceTransformerEmbeddingFunction()
except ImportError:
    print("Warning: sentence-transformers not installed.")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    DEFAULT_EF = DefaultEmbeddingFunction()

# --- Config ---
load_dotenv()
CHROMA_DB_PATH = "./ingestion_test_chroma_db"
COLLECTION_NAME = "clap_ingestion_demo"

# --- File Paths ---
TXT_FILE = "examples/sample.txt"
PDF_FILE = "examples/sample.pdf"
CSV_FILE = "examples/sample.csv"

# --- Chunking Config ---
CHUNK_SIZE = 150 # User-defined chunk size
CHUNK_OVERLAP = 20 # User-defined overlap

async def ingest_and_query():
    """Loads, chunks, ingests, and queries documents."""
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    vector_store = ChromaStore(
        path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=DEFAULT_EF
    )

    all_docs_to_add: List[str] = []
    all_ids_to_add: List[str] = []
    all_metadatas_to_add: List[Dict[str, Any]] = []

    # --- 1. Process TXT File ---
    print(f"\n--- Processing TXT: {TXT_FILE} ---")
    txt_content = load_text_file(TXT_FILE)
    if txt_content:
        txt_chunks = chunk_text_by_fixed_size(
            txt_content,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print(f"Chunked TXT into {len(txt_chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
        for i, chunk in enumerate(txt_chunks):
            chunk_id = f"txt_{os.path.basename(TXT_FILE)}_{i}"
            all_docs_to_add.append(chunk)
            all_ids_to_add.append(chunk_id)
            all_metadatas_to_add.append({"source": TXT_FILE, "chunk_num": i})

    # --- 2. Process PDF File ---
    print(f"\n--- Processing PDF: {PDF_FILE} ---")
    pdf_content = load_pdf_file(PDF_FILE)
    if pdf_content:
        pdf_chunks = chunk_text_by_fixed_size(
            pdf_content,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print(f"Chunked PDF into {len(pdf_chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
        for i, chunk in enumerate(pdf_chunks):
            chunk_id = f"pdf_{os.path.basename(PDF_FILE)}_{i}"
            all_docs_to_add.append(chunk)
            all_ids_to_add.append(chunk_id)
            all_metadatas_to_add.append({"source": PDF_FILE, "chunk_num": i})

    # --- 3. Process CSV File ---
    print(f"\n--- Processing CSV: {CSV_FILE} ---")
    # Treat each row's 'Content' as a document, use 'Title', 'Category', 'Year' as metadata
    csv_data = load_csv_file(
        CSV_FILE,
        content_column="Content",
        metadata_columns=["Title", "Category", "Year"] # Specify metadata columns by name
        # Alternatively use indices: content_column=2, metadata_columns=[1, 3, 4]
    )
    if csv_data:
        print(f"Loaded {len(csv_data)} rows from CSV.")
        for i, (content, metadata) in enumerate(csv_data):
            # Option 1: Add each row as one "document" (no further chunking here)
            row_id = f"csv_{os.path.basename(CSV_FILE)}_row{metadata.get('source_row', i)}"
            all_docs_to_add.append(content)
            all_ids_to_add.append(row_id)
            # Add original source file to metadata
            metadata["source"] = CSV_FILE
            all_metadatas_to_add.append(metadata)

            # Option 2 (Alternative): Chunk the content of *each* CSV row if needed
            # csv_chunks = chunk_text_by_fixed_size(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            # for j, chunk in enumerate(csv_chunks):
            #     chunk_id = f"csv_{os.path.basename(CSV_FILE)}_row{metadata.get('source_row', i)}_chunk{j}"
            #     all_docs_to_add.append(chunk)
            #     all_ids_to_add.append(chunk_id)
            #     chunk_meta = metadata.copy()
            #     chunk_meta["source"] = CSV_FILE
            #     chunk_meta["chunk_num"] = j
            #     all_metadatas_to_add.append(chunk_meta)


    # --- 4. Add all prepared documents to Chroma ---
    print(f"\n--- Adding {len(all_docs_to_add)} total processed chunks/rows to ChromaDB ---")
    if all_docs_to_add:
        # Consider batching adds if the list becomes very large
        await vector_store.aadd_documents(
            documents=all_docs_to_add,
            ids=all_ids_to_add,
            metadatas=all_metadatas_to_add
        )
        print("Ingestion complete.")
    else:
        print("No documents processed to add.")

    # --- 5. Query the ingested data ---
    print("\n--- Querying Ingested Data ---")
    queries = [
        "What is ChromaDB?",
        "Tell me about Machine Learning",
        "What is the first sentence about?" # Should match TXT
    ]
    for query in queries:
        print(f"\nQuerying for: '{query}' (Top 2 results)")
        results = await vector_store.aquery(
            query_texts=[query],
            n_results=2,
            include=["metadatas", "documents", "distances"]
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))

    # --- Cleanup ---
    if os.path.exists(CHROMA_DB_PATH):
         print(f"\nCleaning up test database: {CHROMA_DB_PATH}")
         # shutil.rmtree(CHROMA_DB_PATH) # Uncomment to auto-delete DB

# --- Main Execution ---
if __name__ == "__main__":
    asyncio.run(ingest_and_query())