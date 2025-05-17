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
from clap.llm_services.groq_service import GroqService  

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
DEFAULT_EF = SentenceTransformerEmbeddingFunction()

load_dotenv()
PDF_PATH = "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/handsonml.pdf"
CHROMA_DB_PATH = "./large_pdf_chroma_db" 
COLLECTION_NAME = "ml_book_rag"
CHUNK_SIZE = 500    
CHUNK_OVERLAP = 50    
LLM_MODEL = "llama-3.3-70b-versatile" 

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

    print("Adding chunks to vector store (embedding process)...")
    if chunks:
        await vector_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
        print("Ingestion complete.")
    else:
        print("No chunks generated to add.")

    ingestion_time = time.time() - start_time
    print(f"Ingestion took {ingestion_time:.2f} seconds.")

    llm_service = GroqService() # Or GoogleOpenAICompatService() etc.
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
    # Core Concepts & Comparisons
    "Compare Random Forests and Gradient Boosting machines, highlighting the key differences in how they build ensembles of decision trees.",
    "Describe the concept of the 'kernel trick' as used in Support Vector Machines (SVMs) and explain its primary benefit.",

    # Algorithms & Techniques
    "Explain the objective of Principal Component Analysis (PCA) and outline the main steps involved in its calculation according to the text.",
    "What is the fundamental purpose of the backpropagation algorithm in training artificial neural networks?",
    "Discuss the purpose of regularization in linear models, mentioning techniques like Ridge or Lasso regression as explained in the book.",
    "What are some limitations or disadvantages of the K-Means clustering algorithm mentioned in the text?",

    # Deep Learning Specifics
    # "Explain the role of activation functions in neural networks, perhaps using the ReLU function as an example described in the text.",
    # "What is a convolutional layer and what specific type of patterns or features is it designed to detect in Convolutional Neural Networks (CNNs)?",
    # "Describe the concept of transfer learning in the context of deep learning models and its main advantage.",
    # "What are Recurrent Neural Networks (RNNs) typically used for, according to the book, and what is a common challenge when training them?",

    # # Data & Evaluation
    # "According to the book, why is feature scaling often a necessary preprocessing step for many machine learning algorithms?",
    # "Explain the purpose of cross-validation in model evaluation and briefly describe the K-Fold strategy.",
    "What are precision and recall, and why might you prioritize one over the other in different classification scenarios discussed in the text?"]


    for q in queries:
        rag_agent.task_description = q

        result = await rag_agent.run()

        print(result.get("output", "Agent failed to produce an answer."))

        end_time = time.time()
        print(f"\nTotal process took {(end_time - start_time):.2f} seconds.")


asyncio.run(run_minimal_rag())