
import json
import functools
from typing import Any, Dict, List, Optional

import anyio

try:
    import chromadb
    from chromadb import Collection
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import (
        EmbeddingFunction,
        SentenceTransformerEmbeddingFunction, 
        DefaultEmbeddingFunction 
    )
except ImportError:
    raise ImportError(
        "ChromaDB not found. Please install it using: pip install chromadb sentence-transformers"
    )


# Import the CLAP base interface and types
from .base import (
    Document,
    Embedding,
    ID,
    Metadata,
    QueryResult,
    VectorStoreInterface,
)


class ChromaStore(VectorStoreInterface):
    """Implementation of VectorStoreInterface for ChromaDB."""

    def __init__(
        self,
        path: Optional[str] = None,  
        host: Optional[str] = None,  
        port: Optional[int] = None,  
        collection_name: str = "clap_collection",
        embedding_function: Optional[EmbeddingFunction] = None,
        client_settings: Optional[Settings] = None,
        
    ):
        """
        Initializes the ChromaDB vector store.

        Connects using PersistentClient if 'path' is provided, otherwise
        uses HttpClient if 'host' and 'port' are provided. Falls back to
        EphemeralClient if neither is specified.

        Args:
            path: Path for PersistentClient data.
            host: Hostname for HttpClient.
            port: Port for HttpClient.
            collection_name: Name of the collection to use/create.
            embedding_function: An instance of a ChromaDB-compatible embedding function.
                                Defaults to Chroma's DefaultEmbeddingFunction if None.
            client_settings: Optional ChromaDB client settings.
        """
        if path:
            self._client = chromadb.PersistentClient(path=path, settings=client_settings)
            print(f"ChromaStore: Initialized PersistentClient at path '{path}'")
        elif host and port:
            self._client = chromadb.HttpClient(host=host, port=port, settings=client_settings)
            print(f"ChromaStore: Initialized HttpClient connecting to '{host}:{port}'")
        else:
            self._client = chromadb.EphemeralClient(settings=client_settings)
            print("ChromaStore: Initialized EphemeralClient (in-memory)")

        self._embedding_function = embedding_function or DefaultEmbeddingFunction()
        self.collection_name = collection_name

        
        print(f"ChromaStore: Getting or creating collection '{self.collection_name}'...")
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            
        )
        print(f"ChromaStore: Collection '{self.collection_name}' ready.")


    async def _run_sync(self, func, *args, **kwargs):
        """Runs a synchronous function in a thread pool."""
        bound_func = functools.partial(func, *args, **kwargs)
        return await anyio.to_thread.run_sync(bound_func)


    async def add_documents(
        self,
        documents: List[Document],
        ids: List[ID],
        metadatas: Optional[List[Metadata]] = None,
        embeddings: Optional[List[Embedding]] = None,
    ) -> None:
        """Adds documents/embeddings to the Chroma collection."""
        await self._run_sync(
            self._collection.add,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"ChromaStore: Added/Updated {len(ids)} documents.")


    async def aquery(
        self,
        query_texts: Optional[List[Document]] = None,
        query_embeddings: Optional[List[Embedding]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["metadatas", "documents", "distances"],
    ) -> QueryResult:
        """Queries the Chroma collection."""
        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided.")
        if query_texts and query_embeddings:
            raise ValueError("Provide either query_texts or query_embeddings, not both.")

        results = await self._run_sync(
            self._collection.query,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )
        print(f"ChromaStore: Query returned {len(results.get('ids', [[]])[0])} results for {len(query_texts or query_embeddings)} queries.")
        
        return QueryResult(
            ids=results.get("ids", []),
            embeddings=results.get("embeddings"),
            documents=results.get("documents"),
            metadatas=results.get("metadatas"),
            distances=results.get("distances"),
        )

    async def adelete(
        self,
        ids: Optional[List[ID]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Deletes documents from the Chroma collection."""
        await self._run_sync(
            self._collection.delete,
            ids=ids,
            where=where,
            where_document=where_document
        )
        deleted_count = len(ids) if ids else "matching documents"
        print(f"ChromaStore: Deleted {deleted_count}.")


async def _test():
    print("Testing ChromaStore...")
    try:
        ef = SentenceTransformerEmbeddingFunction() # Example specific EF
        store = ChromaStore(path="./chroma_test_db", embedding_function=ef, collection_name="test_rag")

        docs = ["This is document one about apples.", "This is document two about oranges."]
        ids = ["doc1", "doc2"]
        metas = [{"source": "file1"}, {"source": "file2"}]
        await store.add_documents(documents=docs, ids=ids, metadatas=metas)

        docs3 = ["This is document three about bananas."]
        ids3 = ["doc3"]
        metas3 = [{"source": "file3"}]
        await store.add_documents(documents=docs3, ids=ids3, metadatas=metas3)


        query = "What fruit is mentioned?"
        results = await store.aquery(query_texts=[query], n_results=2, include=["metadatas", "documents", "distances"])
        print("\nQuery Results:")
        print(json.dumps(results, indent=2))

        await store.adelete(ids=["doc1"])
        print("\nDeleted doc1.")

        results_after_delete = await store.aquery(query_texts=[query], n_results=3)
        print("\nQuery Results after delete:")
        print(json.dumps(results_after_delete, indent=2))

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        import shutil
        try:
            shutil.rmtree("./chroma_test_db")
            print("Cleaned up test ChromaDB.")
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(_test())

