import asyncio
import json
import functools
import os
from typing import Any, Dict, List, Optional, cast, Type

import anyio

_QDRANT_LIB_AVAILABLE = False
_AsyncQdrantClient_Placeholder_Type = Any
_QdrantClient_Placeholder_Type = Any 
_qdrant_models_Placeholder_Type = Any
_QdrantUnexpectedResponse_Placeholder_Type = type(Exception)

try:
    from qdrant_client import AsyncQdrantClient as ImportedAsyncQC, QdrantClient as ImportedSyncQC, models as ImportedModels
    from qdrant_client.http.exceptions import UnexpectedResponse as ImportedQdrantUnexpectedResponse
    _QDRANT_LIB_AVAILABLE = True
    _AsyncQdrantClient_Placeholder_Type = ImportedAsyncQC
    _QdrantClient_Placeholder_Type = ImportedSyncQC 
    _qdrant_models_Placeholder_Type = ImportedModels
    _QdrantUnexpectedResponse_Placeholder_Type = ImportedQdrantUnexpectedResponse
except ImportError:
    pass


from .base import ( Document, Embedding, ID, Metadata, QueryResult, VectorStoreInterface,)
from clap.embedding.base_embedding import EmbeddingFunctionInterface


class QdrantStore(VectorStoreInterface):
    _async_client: _AsyncQdrantClient_Placeholder_Type
    models: Any # For qdrant_client.models

    def __init__(self):
        if not hasattr(self, "_initialized_via_factory"):
             raise RuntimeError("Use QdrantStore.create(...) async factory method.")

    @classmethod
    async def create(
        cls: Type['QdrantStore'],
        collection_name: str, embedding_function: EmbeddingFunctionInterface,
        path: Optional[str] = None, distance_metric: Any = None, 
        recreate_collection_if_exists: bool = False, **qdrant_client_kwargs: Any
    ) -> 'QdrantStore':
        if not _QDRANT_LIB_AVAILABLE:
            raise ImportError("The 'qdrant-client' library is required. Install with 'pip install \"clap-agents[qdrant]\"'")
        if not embedding_function: raise ValueError("embedding_function is required.")

        instance = cls.__new__(cls)
        instance._initialized_via_factory = True
        instance.collection_name = collection_name
        instance.models = _qdrant_models_Placeholder_Type 
        instance._embedding_function = embedding_function
        instance.distance_metric = distance_metric if distance_metric else instance.models.Distance.COSINE
        instance.vector_size = instance._embedding_function.get_embedding_dimension()

        client_location_for_log = path if path else ":memory:"
        # print(f"QdrantStore (Async): Initializing client for '{instance.collection_name}' at '{client_location_for_log}'.")
        if path: instance._async_client = _AsyncQdrantClient_Placeholder_Type(path=path, **qdrant_client_kwargs)
        else: instance._async_client = _AsyncQdrantClient_Placeholder_Type(location=":memory:", **qdrant_client_kwargs)
        
        await instance._setup_collection_async(recreate_collection_if_exists)
        return instance

    async def _setup_collection_async(self, recreate_if_exists: bool):
        try:
            if recreate_if_exists:
                await self._async_client.delete_collection(collection_name=self.collection_name)
                await self._async_client.create_collection(
                     collection_name=self.collection_name,
                     vectors_config=self.models.VectorParams(size=self.vector_size, distance=self.distance_metric))
                return
            try:
                info = await self._async_client.get_collection(collection_name=self.collection_name)
                if info.config.params.size != self.vector_size or info.config.params.distance.lower() != self.distance_metric.lower(): # type: ignore
                    raise ValueError("Existing Qdrant collection has incompatible config.")
            except (_QdrantUnexpectedResponse_Placeholder_Type, ValueError) as e: # type: ignore
                 if isinstance(e, _QdrantUnexpectedResponse_Placeholder_Type) and e.status_code == 404 or "not found" in str(e).lower(): # type: ignore
                      await self._async_client.create_collection(
                          collection_name=self.collection_name,
                          vectors_config=self.models.VectorParams(size=self.vector_size, distance=self.distance_metric))
                 else: raise
        except Exception as e: print(f"QdrantStore: Error during collection setup: {e}"); raise


    async def _embed_texts_via_interface(self, texts: List[Document]) -> List[Embedding]:
        if not self._embedding_function: raise RuntimeError("EF missing.")
        ef_call = self._embedding_function.__call__
        if asyncio.iscoroutinefunction(ef_call): return await ef_call(texts) 
        return await anyio.to_thread.run_sync(functools.partial(ef_call, texts)) 

    async def add_documents(self, documents: List[Document], ids: List[ID], metadatas: Optional[List[Metadata]] = None, embeddings: Optional[List[Embedding]] = None) -> None:
        if not documents and not embeddings: raise ValueError("Requires docs or embeddings.")
        num_items = len(documents) if documents else (len(embeddings) if embeddings else 0)
        if num_items == 0: return
        if len(ids) != num_items: raise ValueError("'ids' length mismatch.")
        if metadatas and len(metadatas) != num_items: raise ValueError("'metadatas' length mismatch.")
        if not embeddings and documents: embeddings = await self._embed_texts_via_interface(documents)
        if not embeddings: return
        points: List[Any] = [] # Use Any for models.PointStruct if models is Any
        for i, item_id in enumerate(ids):
            payload = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            if documents and i < len(documents): payload["_clap_document_content_"] = documents[i]
            points.append(self.models.PointStruct(id=str(item_id), vector=embeddings[i], payload=payload))
        if points: await self._async_client.upsert(collection_name=self.collection_name, points=points, wait=True)

    async def aquery(self, query_texts: Optional[List[Document]] = None, query_embeddings: Optional[List[Embedding]] = None, n_results: int = 5, where: Optional[Dict[str, Any]] = None, include: List[str] = ["metadatas", "documents", "distances"], **kwargs) -> QueryResult:
        if not query_texts and not query_embeddings: raise ValueError("Requires query_texts or query_embeddings.")
        if query_texts and query_embeddings: query_texts = None
        q_filter = self._translate_clap_filter(where) 
        if query_texts: q_vectors = await self._embed_texts_via_interface(query_texts)
        elif query_embeddings: q_vectors = query_embeddings
        else: return QueryResult(ids=[[]], embeddings=None, documents=None, metadatas=None, distances=None)
        raw_results: List[List[Any]] = [] 
        for qv in q_vectors:
            hits = await self._async_client.search(collection_name=self.collection_name, query_vector=qv, query_filter=q_filter, limit=n_results, with_payload=True, with_vectors="embeddings" in include, **kwargs)
            raw_results.append(hits)
        return self._format_qdrant_results(raw_results, include) # Keep using self.models

    def _translate_clap_filter(self, clap_where_filter: Optional[Dict[str, Any]]) -> Optional[Any]: # Return Any
        if not clap_where_filter or not _QDRANT_LIB_AVAILABLE: return None
        must = []
        for k, v in clap_where_filter.items():
            if isinstance(v, dict) and "$eq" in v: must.append(self.models.FieldCondition(key=k, match=self.models.MatchValue(value=v["$eq"])))
            elif isinstance(v, (str,int,float,bool)): must.append(self.models.FieldCondition(key=k, match=self.models.MatchValue(value=v)))
        return self.models.Filter(must=must) if must else None

    def _format_qdrant_results(self, raw_results: List[List[Any]], include: List[str]) -> QueryResult: 
        ids, embs, docs, metas, dists_final = [], [], [], [], []
        for hits_list in raw_results:
            current_ids, current_embs, current_docs, current_metas, current_dists = [],[],[],[],[]
            for hit in hits_list: 
                current_ids.append(str(hit.id))
                payload = hit.payload if hit.payload else {}
                if "distances" in include and hit.score is not None: current_dists.append(hit.score)
                if "embeddings" in include and hit.vector : current_embs.append(cast(List[float], hit.vector))
                if "documents" in include : current_docs.append(payload.get("_clap_document_content_", ""))
                if "metadatas" in include : current_metas.append({k:v for k,v in payload.items() if k != "_clap_document_content_"})
            ids.append(current_ids); embs.append(current_embs); docs.append(current_docs); metas.append(current_metas); dists_final.append(current_dists)
        return QueryResult(ids=ids, embeddings=embs if "embeddings" in include else None, documents=docs if "documents" in include else None, metadatas=metas if "metadatas" in include else None, distances=dists_final if "distances" in include else None)


    async def adelete(self, ids: Optional[List[ID]] = None, where: Optional[Dict[str, Any]] = None, **kwargs ) -> None:
        if not ids and not where: return
        q_filter = self._translate_clap_filter(where)
        selector: Any = None
        if ids and q_filter: selector = self.models.FilterSelector(filter=self.models.Filter(must=[q_filter, self.models.HasIdCondition(has_id=[str(i) for i in ids])]))
        elif ids: selector = self.models.PointIdsList(points=[str(i) for i in ids])
        elif q_filter: selector = self.models.FilterSelector(filter=q_filter)
        if selector: await self._async_client.delete(collection_name=self.collection_name, points_selector=selector, wait=True)

    async def close(self):
        if hasattr(self, '_async_client') and self._async_client:
            await self._async_client.close(timeout=5)
