import json
import functools
from typing import Any, Dict, List, Optional, cast, Callable, Coroutine
import asyncio 
import anyio

_CHROMADB_LIB_AVAILABLE = False
_ChromaDB_Client_Placeholder_Type = Any
_ChromaDB_Collection_Placeholder_Type = Any
_ChromaDB_Settings_Placeholder_Type = Any
_ChromaDB_EmbeddingFunction_Placeholder_Type = Any
_ChromaDB_DefaultEmbeddingFunction_Placeholder_Type = Any

try:
    import chromadb
    from chromadb import Collection as ImportedChromaCollection
    from chromadb.config import Settings as ImportedChromaSettings
    from chromadb.utils.embedding_functions import (
        EmbeddingFunction as ImportedChromaEF, 
        DefaultEmbeddingFunction as ImportedChromaDefaultEF
    )
    from chromadb.api.types import Documents, Embeddings 
    _CHROMADB_LIB_AVAILABLE = True
    _ChromaDB_Client_Placeholder_Type = chromadb.Client
    _ChromaDB_Collection_Placeholder_Type = ImportedChromaCollection
    _ChromaDB_Settings_Placeholder_Type = ImportedChromaSettings
    _ChromaDB_EmbeddingFunction_Placeholder_Type = ImportedChromaEF
    _ChromaDB_DefaultEmbeddingFunction_Placeholder_Type = ImportedChromaDefaultEF
except ImportError:
    class Documents: pass 
    class Embeddings: pass 
    pass


from .base import ( Document as ClapDocument, Embedding as ClapEmbedding, ID, Metadata, QueryResult, VectorStoreInterface,) # Aliased Document
from clap.embedding.base_embedding import EmbeddingFunctionInterface as ClapEFInterface


class _AsyncEFWrapperForChroma:
    """Wraps an async CLAP EmbeddingFunctionInterface to be callable synchronously by Chroma."""
    def __init__(self, async_ef_call: Callable[..., Coroutine[Any, Any, Embeddings]], loop: asyncio.AbstractEventLoop):
        self._async_ef_call = async_ef_call
        self._loop = loop

    def __call__(self, input: Documents) -> Embeddings: 
        
        if self._loop.is_running():
            
            future = asyncio.run_coroutine_threadsafe(self._async_ef_call(input), self._loop)
            return future.result() 
        else:
            return self._loop.run_until_complete(self._async_ef_call(input))


class ChromaStore(VectorStoreInterface):
    _client: _ChromaDB_Client_Placeholder_Type
    _collection: _ChromaDB_Collection_Placeholder_Type
    _clap_ef_is_async: bool = False 

    def __init__(
        self,
        collection_name: str,
        embedding_function: Optional[ClapEFInterface] = None, # CLAP's interface
        path: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None,
        client_settings: Optional[_ChromaDB_Settings_Placeholder_Type] = None,
    ):
        if not _CHROMADB_LIB_AVAILABLE:
            raise ImportError("The 'chromadb' library is required to use ChromaStore.")

        self.collection_name = collection_name
        
        if path: self._client = chromadb.PersistentClient(path=path, settings=client_settings)
        elif host and port: self._client = chromadb.HttpClient(host=host, port=port, settings=client_settings)
        else: self._client = chromadb.EphemeralClient(settings=client_settings)
       

        chroma_ef_to_pass_to_chroma: Optional[_ChromaDB_EmbeddingFunction_Placeholder_Type]

        if embedding_function is None:
            chroma_ef_to_pass_to_chroma = _ChromaDB_DefaultEmbeddingFunction_Placeholder_Type()
            self._clap_ef_is_async = False 
            print(f"ChromaStore: Using ChromaDB's DefaultEmbeddingFunction for '{self.collection_name}'.")
        
        elif isinstance(embedding_function, _ChromaDB_DefaultEmbeddingFunction_Placeholder_Type): 
            chroma_ef_to_pass_to_chroma = embedding_function
            self._clap_ef_is_async = False 
            print(f"ChromaStore: Using provided native Chroma EmbeddingFunction for '{self.collection_name}'.")
        else:
            
            ef_call_method = getattr(embedding_function, "__call__", None)
            if ef_call_method and asyncio.iscoroutinefunction(ef_call_method):
                self._clap_ef_is_async = True
                print(f"ChromaStore: Wrapping async CLAP EmbeddingFunction for Chroma compatibility for '{self.collection_name}'.")
                
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError: 
                    
                    print("ChromaStore WARNING: No running asyncio loop to wrap async EF for Chroma. This might fail.")
                    
                    loop = asyncio.new_event_loop() 
                                                  

                chroma_ef_to_pass_to_chroma = _AsyncEFWrapperForChroma(ef_call_method, loop) 
            else: 
                self._clap_ef_is_async = False
                print(f"ChromaStore: Using synchronous CLAP EmbeddingFunction for '{self.collection_name}'.")
                chroma_ef_to_pass_to_chroma = embedding_function # type: ignore

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=chroma_ef_to_pass_to_chroma
        )
        # print(f"ChromaStore: Collection '{self.collection_name}' ready.")

    
    async def _run_sync(self, func, *args, **kwargs):
        bound_func = functools.partial(func, *args, **kwargs)
        return await anyio.to_thread.run_sync(bound_func)

    async def add_documents(self, documents: List[ClapDocument], ids: List[ID], metadatas: Optional[List[Metadata]] = None, embeddings: Optional[List[ClapEmbedding]] = None) -> None:
        
        # print(f"ChromaStore: Adding {len(ids)} documents to '{self.collection_name}'...")
        await self._run_sync(self._collection.add, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        # print(f"ChromaStore: Add/Update completed for {len(ids)} documents.")

    async def aquery(self, query_texts: Optional[List[ClapDocument]] = None, query_embeddings: Optional[List[ClapEmbedding]] = None, n_results: int = 5, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None, include: List[str] = ["metadatas", "documents", "distances"]) -> QueryResult:
        if not query_texts and not query_embeddings: raise ValueError("Requires query_texts or query_embeddings.")
        if query_texts and query_embeddings: query_texts = None
        
       
        # print(f"ChromaStore: Querying collection '{self.collection_name}'...")
        results = await self._run_sync(self._collection.query, query_embeddings=query_embeddings, query_texts=query_texts, n_results=n_results, where=where, where_document=where_document, include=include)
        
        num_queries = len(query_texts or query_embeddings or [])
       
        return QueryResult(
            ids=results.get("ids") or ([[]] * num_queries), embeddings=results.get("embeddings"),
            documents=results.get("documents"), metadatas=results.get("metadatas"), distances=results.get("distances") )

    async def adelete(self, ids: Optional[List[ID]] = None, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> None:
        await self._run_sync(self._collection.delete, ids=ids, where=where, where_document=where_document)
