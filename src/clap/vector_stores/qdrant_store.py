# --- START OF FILE src/clap/vector_stores/qdrant_store.py (PURE ASYNC VERSION) ---

import asyncio
import json
import functools
import os
from typing import Any, Dict, List, Optional, cast, Type

import anyio

try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client import QdrantClient 
    QDRANT_INSTALLED = True
except ImportError:
    raise ImportError(
        "Qdrant client not found. Please install it using: pip install qdrant-client"
    )

from .base import (
    Document,
    Embedding,
    ID,
    Metadata,
    QueryResult,
    VectorStoreInterface,
)
# This interface is now MANDATORY for QdrantStore
from clap.embedding.base_embedding import EmbeddingFunctionInterface


class QdrantStore(VectorStoreInterface):
    """
    Pure Asynchronous implementation of VectorStoreInterface for Qdrant (local modes).
    Requires a mandatory EmbeddingFunctionInterface for all embedding tasks.

    Use the `QdrantStore.create(...)` classmethod factory to instantiate.
    """
    _async_client: AsyncQdrantClient

    def __init__(self):
        """Private constructor. Use the `create` classmethod instead."""
        if not hasattr(self, "_initialized_via_factory"):
             raise RuntimeError("Use QdrantStore.create(...) async factory method.")

    @classmethod
    async def create(
        cls: Type['QdrantStore'],
        collection_name: str,
        embedding_function: EmbeddingFunctionInterface, # Now MANDATORY
        path: Optional[str] = None, # Path for local file persistence
        distance_metric: models.Distance = models.Distance.COSINE,
        recreate_collection_if_exists: bool = False,
        **qdrant_async_client_kwargs: Any
    ) -> 'QdrantStore':
        """
        Asynchronously creates and initializes the QdrantStore.

        Args:
            collection_name: Name of the Qdrant collection.
            embedding_function: A mandatory object adhering to EmbeddingFunctionInterface.
            path: Local path for Qdrant data. None or ":memory:" for in-memory.
            distance_metric: Distance metric for vector comparison.
            recreate_collection_if_exists: Deletes and recreates collection if True.
            **qdrant_async_client_kwargs: Additional kwargs passed to AsyncQdrantClient.

        Returns:
            An initialized instance of QdrantStore.
        """
        if not QDRANT_INSTALLED: raise ImportError("Qdrant client not installed.")
        if not embedding_function: raise ValueError("embedding_function is required for QdrantStore.")

        instance = cls.__new__(cls)
        instance._initialized_via_factory = True

        instance.collection_name = collection_name
        instance.models = models
        instance._embedding_function = embedding_function # Store the mandatory EF
        instance.distance_metric = distance_metric
        instance.vector_size = instance._embedding_function.get_embedding_dimension() # Get size from EF

        client_location_for_log = path if path else ":memory:"
        print(f"QdrantStore (Async): Initializing client for collection '{instance.collection_name}' at '{client_location_for_log}'.")

        if path:
            instance._async_client = AsyncQdrantClient(path=path, **qdrant_async_client_kwargs)
        else:
            instance._async_client = AsyncQdrantClient(location=":memory:", **qdrant_async_client_kwargs)

        await instance._setup_collection_async(recreate_collection_if_exists)

        return instance

    

    async def _setup_collection_async(self, recreate_if_exists: bool):
        """Ensures collection exists using the async client."""
        # (This method remains the same as the previous version's async setup)
        try:
            if recreate_if_exists:
                print(f"QdrantStore: Recreating collection '{self.collection_name}'...")
                await self._async_client.delete_collection(collection_name=self.collection_name)
                print(f"QdrantStore: Delete request sent for '{self.collection_name}'. Waiting for creation...")
                await self._async_client.create_collection(
                     collection_name=self.collection_name,
                     vectors_config=self.models.VectorParams(size=self.vector_size, distance=self.distance_metric)
                )
                print(f"QdrantStore: Collection '{self.collection_name}' recreated.")
                return

            try:
                collection_info = await self._async_client.get_collection(collection_name=self.collection_name)
                print(f"QdrantStore: Collection '{self.collection_name}' exists. Verifying...")
                current_config = collection_info.config.params
                if current_config.size != self.vector_size or current_config.distance.lower() != self.distance_metric.lower():
                    raise ValueError(f"Collection '{self.collection_name}' exists with incompatible config.")
                print(f"QdrantStore: Existing collection parameters compatible.")
            except (UnexpectedResponse, ValueError) as e:
                 if isinstance(e, UnexpectedResponse) and e.status_code == 404 or "not found" in str(e).lower():
                      print(f"QdrantStore: Collection '{self.collection_name}' not found, creating...")
                      await self._async_client.create_collection(
                          collection_name=self.collection_name,
                          vectors_config=self.models.VectorParams(size=self.vector_size, distance=self.distance_metric)
                      )
                      print(f"QdrantStore: Collection '{self.collection_name}' created.")
                 else: raise
        except Exception as e:
            print(f"QdrantStore: Error during collection setup: {e}")
            raise

    # Inside src/clap/vector_stores/qdrant_store.py

    async def _embed_texts_via_interface(self, texts: List[Document]) -> List[Embedding]:
        """Helper to embed texts using the provided EmbeddingFunctionInterface."""
        if not self._embedding_function:
            raise RuntimeError("QdrantStore: EmbeddingFunctionInterface not available for embedding.")

        print(f"QdrantStore: Embedding {len(texts)} texts using provided EmbeddingFunctionInterface...")

        ef_call = self._embedding_function.__call__ # Get the method

        if asyncio.iscoroutinefunction(ef_call):
            # If the __call__ method of the embedding function is async, await it directly.
            # Pass 'texts' as a keyword argument if that's how the async __call__ expects it,
            # or as a positional argument. Assuming positional based on interface.
            print("QdrantStore: EF __call__ is async, awaiting directly.")
            embeddings = await ef_call(texts)
        else:
            # If the __call__ method is synchronous, run it in a thread.
            print("QdrantStore: EF __call__ is sync, running in thread.")
            bound_embed_func = functools.partial(ef_call, texts)
            embeddings = await anyio.to_thread.run_sync(bound_embed_func)
        
        return cast(List[Embedding], embeddings)

    async def add_documents(
        self,
        documents: List[Document],
        ids: List[ID],
        metadatas: Optional[List[Metadata]] = None,
        embeddings: Optional[List[Embedding]] = None,
    ) -> None:
        """Adds documents to the Qdrant collection using the async client."""
        if not documents and not embeddings: raise ValueError("Requires 'documents' or 'embeddings'.")
        num_items = len(documents) if documents else (len(embeddings) if embeddings else 0)
        if num_items == 0: return
        if len(ids) != num_items: raise ValueError("'ids' length mismatch.")
        if metadatas and len(metadatas) != num_items: raise ValueError("'metadatas' length mismatch.")

        if not embeddings and documents:
            embeddings = await self._embed_texts_via_interface(documents)
        if not embeddings:
            print("QdrantStore: No embeddings available to add.")
            return

        points_to_upsert: List[models.PointStruct] = []
        for i, item_id in enumerate(ids):
            payload = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            if documents and i < len(documents):
                payload["_clap_document_content_"] = documents[i] 

            points_to_upsert.append(
                self.models.PointStruct(id=str(item_id), vector=embeddings[i], payload=payload)
            )

        if points_to_upsert:
            print(f"QdrantStore: Upserting {len(points_to_upsert)} points to '{self.collection_name}'...")
            await self._async_client.upsert(
                collection_name=self.collection_name, points=points_to_upsert, wait=True
            )
            print(f"QdrantStore: Upsert operation completed.")

    async def aquery(
        self,
        query_texts: Optional[List[Document]] = None,
        query_embeddings: Optional[List[Embedding]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = ["metadatas", "documents", "distances"],
        **kwargs
    ) -> QueryResult:
        """Queries the Qdrant collection using the async client."""
        if not query_texts and not query_embeddings: raise ValueError("Requires query_texts or query_embeddings.")
        if query_texts and query_embeddings: query_texts = None 

        qdrant_filter_model: Optional[models.Filter] = self._translate_clap_filter(where)
        search_results_raw: List[List[models.ScoredPoint]] = []

        query_vectors_to_search: List[Embedding]
        if query_texts:
            query_vectors_to_search = await self._embed_texts_via_interface(query_texts)
        elif query_embeddings:
            query_vectors_to_search = query_embeddings 
        else: # Should not happen
            return QueryResult(ids=[[]], embeddings=None, documents=None, metadatas=None, distances=None)

        print(f"QdrantStore: Searching '{self.collection_name}' for {len(query_vectors_to_search)} query vectors...")
        for q_vector in query_vectors_to_search:
            hits = await self._async_client.search(
                collection_name=self.collection_name,
                query_vector=q_vector,
                query_filter=qdrant_filter_model,
                limit=n_results,
                with_payload="documents" in include or "metadatas" in include,
                with_vectors="embeddings" in include,
                **kwargs
            )
            search_results_raw.append(hits)

        return self._format_qdrant_results(search_results_raw, include)

    def _translate_clap_filter(self, clap_where_filter: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not clap_where_filter: return None
        must_conditions = []
        for key, value in clap_where_filter.items():
            try:
                if isinstance(value, dict) and "$eq" in value: must_conditions.append(self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value["$eq"])))
                elif isinstance(value, (str, int, float, bool)): must_conditions.append(self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value)))
                else: print(f"QdrantStore: Unsupported 'where' filter condition (key '{key}'): {value}. Skipping.")
            except Exception as filter_ex: print(f"QdrantStore: Error translating filter condition (key '{key}'): {filter_ex}. Skipping.")
        if must_conditions: return self.models.Filter(must=must_conditions)
        elif clap_where_filter: print(f"QdrantStore: Could not translate 'where' filter: {clap_where_filter}")
        return None

    def _format_qdrant_results(self, raw_results: List[List[models.ScoredPoint]], include: List[str]) -> QueryResult:
        final_ids, final_embeddings, final_documents, final_metadatas, final_distances = [], [], [], [], []
        inc_emb = "embeddings" in include; inc_doc = "documents" in include; inc_meta = "metadatas" in include; inc_dist = "distances" in include
        for hits_list in raw_results:
            ids, embs, docs, metas, dists = [], [], [], [], []
            for hit in hits_list:
                ids.append(str(hit.id)); payload = hit.payload if hit.payload else {}
                if inc_dist and hit.score is not None: dists.append(hit.score)
                if inc_emb and hasattr(hit, 'vector') and hit.vector: embs.append(cast(List[float], hit.vector))
                if inc_doc: docs.append(payload.get("_clap_document_content_", ""))
                if inc_meta: metas.append({k: v for k, v in payload.items() if k != "_clap_document_content_"})
            final_ids.append(ids)
            if inc_emb: final_embeddings.append(embs)
            if inc_doc: final_documents.append(docs)
            if inc_meta: final_metadatas.append(metas)
            if inc_dist: final_distances.append(dists)
        print(f"QdrantStore: Query result formatting completed.")
        return QueryResult(ids=final_ids, embeddings=final_embeddings if inc_emb else None, documents=final_documents if inc_doc else None, metadatas=final_metadatas if inc_meta else None, distances=final_distances if inc_dist else None,)


    async def adelete(self, ids: Optional[List[ID]] = None, where: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if not ids and not where: print("QdrantStore: Delete requires 'ids' or 'where'."); return
        qdrant_filter_model = self._translate_clap_filter(where)
        points_selector_for_delete: Any = None
        op_desc = "criteria"
        if ids and qdrant_filter_model:
            print("QdrantStore Info: Deleting points matching BOTH IDs AND filter.")
            id_filter = self.models.Filter(must=[self.models.HasIdCondition(has_id=[str(i) for i in ids])]) 
            combined_filter = self.models.Filter(must=[qdrant_filter_model, id_filter])
            points_selector_for_delete = self.models.FilterSelector(filter=combined_filter)
            op_desc = "by matching IDs within filter"
        elif ids:
            points_selector_for_delete = self.models.PointIdsList(points=[str(pid) for pid in ids])
            op_desc = "by ID"
        elif qdrant_filter_model:
            points_selector_for_delete = self.models.FilterSelector(filter=qdrant_filter_model)
            op_desc = "by filter"
        if points_selector_for_delete:
            print(f"QdrantStore: Deleting points {op_desc} from '{self.collection_name}'...")
            await self._async_client.delete(collection_name=self.collection_name, points_selector=points_selector_for_delete, wait=True)
            print(f"QdrantStore: Deletion request completed for points {op_desc}.")
        else: print("QdrantStore: No valid criteria for deletion.")


    async def close(self):
        """Closes the Qdrant async client."""
        print("QdrantStore: Closing Qdrant async client...")
        if hasattr(self, '_async_client') and self._async_client:
            try: await self._async_client.close(timeout=5)
            except Exception as e: print(f"QdrantStore: Error closing async client: {e}")
        print("QdrantStore: Async client closed.")

