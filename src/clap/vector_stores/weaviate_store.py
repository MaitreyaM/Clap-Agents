# --- START OF FILE src/clap/vector_stores/weaviate_store.py (REFINED V4 ASYNC) ---
import asyncio
import json
import functools
import uuid
from typing import Any, Dict, List, Optional, cast, Type

import anyio

# --- Weaviate Client Imports (v4 style) ---
try:
    import weaviate
    import weaviate.classes as wvc # For new filter syntax, properties etc.
    # We will primarily use the WeaviateAsyncClient
    # from weaviate import WeaviateClient # Sync client, might be needed for some setup if async not available
    from weaviate.exceptions import WeaviateBaseError, WeaviateQueryError, WeaviateInsertError
    WEAVIATE_INSTALLED = True
except ImportError:
    WEAVIATE_INSTALLED = False
    class wvc: # Dummy for type hints if not installed
        class query:
            class MetadataQuery: pass
        class config:
            class Configure:
                class Vectorizer:
                    @staticmethod
                    def none(): pass
            class Property: pass
            class DataType:
                TEXT = "text"
    class WeaviateBaseError(Exception): pass
    class WeaviateQueryError(WeaviateBaseError): pass
    class WeaviateInsertError(WeaviateBaseError): pass


# --- CLAP Imports ---
from .base import (
    Document,
    Embedding,
    ID,
    Metadata,
    QueryResult,
    VectorStoreInterface,
)
from clap.embedding.base_embeddings import EmbeddingFunctionInterface

def capitalize_classname(name: str) -> str:
    """Weaviate class names must be capitalized."""
    if not name: return "DefaultClapClass"
    return name[0].upper() + name[1:]


class WeaviateStore(VectorStoreInterface):
    _client: weaviate.WeaviateClient # Store the async client
    _class_name: str
    _vector_size: int
    _text_key: str = "clap_document_content" # Weaviate property name for original text

    def __init__(self): # Private constructor
        if not hasattr(self, "_initialized_via_factory"):
             raise RuntimeError("Use WeaviateStore.create(...) async factory method.")

    @classmethod
    async def create(
        cls: Type['WeaviateStore'],
        class_name_prefix: str,
        embedding_function: EmbeddingFunctionInterface,
        weaviate_url: str = "http://localhost:8080", # For connect_to_custom or local
        # For WCD:
        # cluster_url: Optional[str] = None,
        # auth_credentials: Optional[wvc.init.Auth] = None,
        # For embedded:
        # embedded_options: Optional[wvc.init.EmbeddedOptions] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        recreate_class_if_exists: bool = False,
        **connection_kwargs: Any # For other connection helper params
    ) -> 'WeaviateStore':
        if not WEAVIATE_INSTALLED:
            raise ImportError("Weaviate client v4+ not found. Run: pip install weaviate-client")
        if not embedding_function:
            raise ValueError("embedding_function is required for WeaviateStore.")

        instance = cls.__new__(cls)
        instance._initialized_via_factory = True

        instance._class_name = capitalize_classname(class_name_prefix)
        instance._embedding_function = embedding_function
        instance._vector_size = instance._embedding_function.get_embedding_dimension()

        print(f"WeaviateStore: Attempting to connect to Weaviate for class '{instance._class_name}' at '{weaviate_url}'.")
        try:
            # Using connect_to_custom for flexibility, can be adapted for connect_to_local, connect_to_wcd
            # The new client requires connection params to be more structured
            # For a simple local connection:
            if weaviate_url.startswith("http://localhost") or weaviate_url.startswith("http://127.0.0.1"):
                 # Assuming default gRPC port for local unless specified in connection_kwargs
                 grpc_port = connection_kwargs.pop("grpc_port", 50051)
                 instance._client = await anyio.to_thread.run_sync(
                     weaviate.connect_to_local,
                     host=weaviate_url.split(":")[1].replace("//",""), # extract host
                     port=int(weaviate_url.split(":")[2]),       # extract port
                     grpc_port=grpc_port,
                     headers=additional_headers,
                     **connection_kwargs
                 )
            else: # For custom URL (potentially cloud or remote self-hosted)
                 instance._client = await anyio.to_thread.run_sync(
                     weaviate.connect_to_custom,
                     http_host=weaviate_url.split("://")[1].split(":")[0], # http(s)://host:port
                     http_port=int(weaviate_url.split(":")[-1]),
                     http_secure=weaviate_url.startswith("https"),
                     # Assuming gRPC details might be in connection_kwargs or need separate params
                     grpc_host=connection_kwargs.pop("grpc_host", weaviate_url.split("://")[1].split(":")[0]),
                     grpc_port=connection_kwargs.pop("grpc_port", 50051), # Default gRPC port
                     grpc_secure=connection_kwargs.pop("grpc_secure", weaviate_url.startswith("https")),
                     auth_client_secret=connection_kwargs.pop("auth_client_secret", None),
                     headers=additional_headers,
                     **connection_kwargs
                 )

            if not instance._client.is_connected(): # v4 uses is_connected()
                await anyio.to_thread.run_sync(instance._client.connect) # Manually connect if not auto
            if not instance._client.is_ready():
                 raise ConnectionError("Weaviate instance connected but not ready.")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to Weaviate: {e}")

        await instance._setup_weaviate_class_async(recreate_class_if_exists)
        print(f"WeaviateStore for class '{instance._class_name}' initialized and ready.")
        return instance

    async def _setup_weaviate_class_async(self, recreate_if_exists: bool):
        """Ensures the Weaviate class (schema) exists using async client methods if available."""
        collection_exists = await anyio.to_thread.run_sync(self._client.collections.exists, self._class_name)

        if recreate_if_exists and collection_exists:
            print(f"WeaviateStore: Deleting existing class '{self._class_name}'...")
            await anyio.to_thread.run_sync(self._client.collections.delete, self._class_name)
            collection_exists = False # Mark as not existing for creation step

        if not collection_exists:
            print(f"WeaviateStore: Class '{self._class_name}' does not exist. Creating schema...")
            class_properties = [
                wvc.config.Property(name=self._text_key, data_type=wvc.config.DataType.TEXT)
                # Add other fixed metadata properties here if desired, e.g.:
                # wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                # wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
            ]
            # Ensure all properties have unique names
            prop_names = set()
            unique_properties = []
            for prop in class_properties:
                if prop.name not in prop_names:
                    unique_properties.append(prop)
                    prop_names.add(prop.name)

            await anyio.to_thread.run_sync(
                self._client.collections.create,
                name=self._class_name,
                description=f"CLAP managed class for {self._class_name}",
                vectorizer_config=wvc.config.Configure.Vectorizer.none(), # BYOV
                # vector_index_config can be added here if needed
                properties=unique_properties
            )
            print(f"WeaviateStore: Class '{self._class_name}' created.")
        else:
            print(f"WeaviateStore: Class '{self._class_name}' already exists.")
            # TODO: Optionally verify existing schema for vectorizer and essential properties

    async def _embed_texts_via_interface(self, texts: List[Document]) -> List[Embedding]:
        # (Same as QdrantStore's implementation)
        if not self._embedding_function: raise RuntimeError("EmbeddingFunctionInterface missing.")
        ef_call = self._embedding_function.__call__
        if asyncio.iscoroutinefunction(ef_call): return await ef_call(texts)
        return await anyio.to_thread.run_sync(functools.partial(ef_call, texts)) # type: ignore

    async def add_documents(
        self,
        documents: List[Document],
        ids: List[ID], # Weaviate generates UUIDs if not provided; we'll use provided IDs as UUIDs if valid
        metadatas: Optional[List[Metadata]] = None,
        embeddings: Optional[List[Embedding]] = None,
    ) -> None:
        # (Input validation similar to QdrantStore)
        if not documents and not embeddings: raise ValueError("Requires 'documents' or 'embeddings'.")
        num_items = len(documents) if documents else (len(embeddings) if embeddings else 0)
        if num_items == 0: return
        if len(ids) != num_items: raise ValueError("'ids' length mismatch.")
        if metadatas and len(metadatas) != num_items: raise ValueError("'metadatas' length mismatch.")

        if not embeddings and documents:
            embeddings = await self._embed_texts_via_interface(documents)
        if not embeddings: return

        data_objects: List[wvc.data.DataObject[Dict[str, Any], List[float]]] = []
        for i, item_id_str in enumerate(ids):
            properties: Dict[str, Any] = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            if documents and i < len(documents):
                properties[self._text_key] = documents[i]

            # Validate and use provided ID as UUID, or generate one
            try:
                object_uuid = uuid.UUID(item_id_str)
            except ValueError:
                # If ID is not a UUID, generate one based on it (deterministically)
                # or just generate a new one and store original ID in metadata.
                # For now, let's ensure we pass a valid UUID string.
                # Using generate_uuid5 for deterministic UUIDs from original ID.
                object_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, item_id_str)
                properties["_clap_original_id_"] = item_id_str # Store original

            data_objects.append(
                wvc.data.DataObject(
                    properties=properties,
                    vector=embeddings[i],
                    uuid=object_uuid # Pass the UUID object
                )
            )

        if data_objects:
            print(f"WeaviateStore: Batch inserting {len(data_objects)} objects into '{self._class_name}'...")
            collection = self._client.collections.get(self._class_name) # Get collection object
            # The batch manager methods are synchronous
            def _batch_insert():
                with collection.batch.fixed_size(batch_size=100, concurrent_requests=2) as batch:
                    for obj in data_objects:
                        batch.add_object(
                            properties=obj.properties,
                            vector=obj.vector,
                            uuid=obj.uuid
                        )
                if batch.number_errors > 0:
                    raise WeaviateInsertError(f"Failed to insert {batch.number_errors} objects: {batch.failed_objects}")

            await anyio.to_thread.run_sync(_batch_insert)
            print(f"WeaviateStore: Batch insert completed.")


    async def aquery(
        self,
        query_texts: Optional[List[Document]] = None,
        query_embeddings: Optional[List[Embedding]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None, # Weaviate filter object (wvc.query.Filter)
        include: List[str] = ["metadatas", "documents", "distances"],
        **kwargs # For other Weaviate query params like certainty, distance, hybrid settings
    ) -> QueryResult:
        # (Input validation similar to QdrantStore)
        if not query_texts and not query_embeddings: raise ValueError("Requires query_texts or query_embeddings.")
        if query_texts and query_embeddings: query_texts = None # Prioritize embeddings

        query_vectors_to_search: List[Embedding]
        if query_texts: query_vectors_to_search = await self._embed_texts_via_interface(query_texts)
        elif query_embeddings: query_vectors_to_search = query_embeddings
        else: return QueryResult(ids=[[]], embeddings=None, documents=None, metadatas=None, distances=None)

        weaviate_filter_obj: Optional[wvc.query.Filter] = self._translate_clap_filter_to_weaviate(where)

        print(f"WeaviateStore: Querying class '{self._class_name}' for {len(query_vectors_to_search)} queries...")

        # Determine return_properties and return_metadata
        return_props_list: List[str] = []
        if "documents" in include: return_props_list.append(self._text_key)
        # If "metadatas" is in include, we fetch all non-text_key properties by default.
        # Weaviate's `return_properties` can be a list of names or a TypedDict model.
        # `return_metadata=wvc.query.MetadataQuery(distance=True, vector=True)` controls _additional fields

        include_vector_weaviate = "embeddings" in include
        # Weaviate returns distance or certainty in _additional metadata
        # Always request distance for now, can adjust based on `include`
        return_metadata_query = wvc.query.MetadataQuery(distance=True, vector=include_vector_weaviate)


        all_query_results_raw = []
        collection = self._client.collections.get(self._class_name)

        for q_vector in query_vectors_to_search:
            try:
                # collection.query.near_vector is synchronous
                def _perform_query():
                    return collection.query.near_vector(
                        near_vector=q_vector,
                        limit=n_results,
                        filters=weaviate_filter_obj,
                        return_metadata=return_metadata_query,
                        return_properties=return_props_list if return_props_list else None, # None fetches all
                        **kwargs # Pass e.g. distance or certainty thresholds
                    )
                response = await anyio.to_thread.run_sync(_perform_query)
                all_query_results_raw.append(response) # response is a GenerativeReturn or QueryReturn
            except WeaviateQueryError as e:
                raise RuntimeError(f"Weaviate query error: {e.message}") from e # Access message attribute
            except Exception as e:
                raise RuntimeError(f"Error during Weaviate query: {e}") from e

        return self._format_weaviate_results(all_query_results_raw, include)

    def _translate_clap_filter_to_weaviate(self, clap_where_filter: Optional[Dict[str, Any]]) -> Optional[wvc.query.Filter]:
        """Translates a simple CLAP 'where' dict to a Weaviate Filter object."""
        if not clap_where_filter: return None
        # Weaviate v4 Filter: e.g., Filter.by_property("name").equal("value")
        # This needs robust implementation. For a simple pass-through example:
        # if isinstance(clap_where_filter, wvc.query.Filter): return clap_where_filter
        print(f"WeaviateStore: Basic filter translation for {clap_where_filter}. Advanced filters require direct Weaviate Filter objects.")
        # Very basic example for a single equality:
        # {"property_name": "value_to_match"}
        # Or {"property_name": {"$eq": "value_to_match"}}
        conditions = []
        for key, value_or_op_dict in clap_where_filter.items():
            if isinstance(value_or_op_dict, dict) and "$eq" in value_or_op_dict:
                conditions.append(wvc.query.Filter.by_property(key).equal(value_or_op_dict["$eq"]))
            elif isinstance(value_or_op_dict, (str, int, float, bool)):
                 conditions.append(wvc.query.Filter.by_property(key).equal(value_or_op_dict))
            # TODO: Add more operators: $ne, $gt, $in, $and, $or, etc.
            # $and: return wvc.query.Filter.all_of([cond1, cond2])
            # $or:  return wvc.query.Filter.any_of([cond1, cond2])
            else:
                print(f"WeaviateStore: Skipping unsupported filter condition: {key}={value_or_op_dict}")
        
        if not conditions: return None
        if len(conditions) == 1: return conditions[0]
        return wvc.query.Filter.all_of(conditions) # Default to AND for multiple simple conditions


    def _format_weaviate_results(self, raw_responses: List[Any], include: List[str]) -> QueryResult:
        """Formats raw Weaviate query responses (List of QueryReturn) into CLAP's QueryResult."""
        # (Similar formatting logic as Qdrant, adjusted for Weaviate _Object structure)
        final_ids, final_embeddings, final_documents, final_metadatas, final_distances = [], [], [], [], []
        inc_emb = "embeddings" in include; inc_doc = "documents" in include
        inc_meta = "metadatas" in include; inc_dist = "distances" in include

        for response in raw_responses: # response is a QueryReturn or GenerativeReturn object
            current_q_ids, current_q_embs, current_q_docs, current_q_metas, current_q_dists = [], [], [], [], []
            for obj in response.objects: # obj is an _Object type
                current_q_ids.append(str(obj.uuid)) # UUID object to string
                
                if obj.metadata and inc_dist and obj.metadata.distance is not None:
                    current_q_dists.append(obj.metadata.distance)
                
                if inc_emb and obj.vector: # obj.vector is Dict[str, List[float]] or List[float]
                    # Assuming default unnamed vector
                    if isinstance(obj.vector, list):
                        current_q_embs.append(obj.vector)
                    # Handle named vectors if necessary, though our schema uses default

                if inc_doc and obj.properties:
                    current_q_docs.append(obj.properties.get(self._text_key, ""))
                
                if inc_meta and obj.properties:
                    meta_to_return = {k:v for k,v in obj.properties.items() if k != self._text_key}
                    if "_clap_original_id_" in meta_to_return: # Restore original CLAP ID if present
                         meta_to_return["original_clap_id"] = meta_to_return.pop("_clap_original_id_")
                    current_q_metadatas.append(meta_to_return)

            final_ids.append(current_q_ids)
            if inc_emb: final_embeddings.append(current_q_embs)
            if inc_doc: final_documents.append(current_q_docs)
            if inc_meta: final_metadatas.append(current_q_metas)
            if inc_dist: final_distances.append(current_q_dists)

        print(f"WeaviateStore: Query result formatting completed.")
        return QueryResult(
            ids=final_ids, embeddings=final_embeddings if inc_emb else None,
            documents=final_documents if inc_doc else None,
            metadatas=final_metadatas if inc_meta else None,
            distances=final_distances if inc_dist else None,
        )


    async def adelete(
        self,
        ids: Optional[List[ID]] = None,
        where: Optional[Dict[str, Any]] = None, # Weaviate filter object
        where_document: Optional[Dict[str, Any]] = None # Not directly supported by Weaviate delete by filter
    ) -> None:
        """Deletes objects from Weaviate by UUIDs or a filter."""
        collection = self._client.collections.get(self._class_name)
        if ids:
            # Convert all IDs to UUID objects or valid UUID strings for Weaviate
            uuids_to_delete: List[uuid.UUID] = []
            print_warning = False
            for item_id_str in ids:
                try:
                    uuids_to_delete.append(uuid.UUID(item_id_str))
                except ValueError:
                    # If not a UUID, try to find by _clap_original_id_ (more complex, needs query then delete)
                    # For now, we'll just warn and skip non-UUIDs if deleting by ID list.
                    print(f"WeaviateStore Warning: ID '{item_id_str}' is not a valid UUID. Cannot delete by this ID directly.")
                    print_warning = True
            if print_warning:
                 print("To delete by non-UUID original CLAP IDs, use a 'where' filter targeting '_clap_original_id_'.")


            if uuids_to_delete:
                print(f"WeaviateStore: Batch deleting {len(uuids_to_delete)} objects by UUID from '{self._class_name}'...")
                # collection.data.delete_many is synchronous
                def _batch_delete_by_uuid():
                    # where_filter = wvc.query.Filter.by_id().any_of(uuids_to_delete) # Filter by list of UUIDs
                    # return collection.data.delete_many(where=where_filter)
                    # Simpler: delete one by one if batch delete by list of UUIDs is not direct
                    # Or use client.batch.delete_objects with a filter for IDs
                    failed_ids = []
                    with collection.batch.fixed_size(batch_size=100) as batch:
                        for u in uuids_to_delete:
                            batch.delete_object(uuid=u) # Uses UUID object
                    if batch.number_errors > 0:
                         failed_ids = [f.message for f in batch.failed_objects] # Just get messages
                         raise WeaviateBaseError(f"Failed to delete some objects by UUID: {failed_ids}")
                    return len(uuids_to_delete) - len(failed_ids)

                num_deleted = await anyio.to_thread.run_sync(_batch_delete_by_uuid)
                print(f"WeaviateStore: Batch delete by UUIDs completed. Affected: {num_deleted}")


        elif where:
            weaviate_filter_obj = self._translate_clap_filter_to_weaviate(where)
            if not weaviate_filter_obj:
                print("WeaviateStore: No valid filter provided for deletion.")
                return

            print(f"WeaviateStore: Deleting objects by filter from '{self._class_name}'...")
            # collection.data.delete_many is synchronous
            def _perform_delete_by_filter():
                # Returns a BatchDeleteReturn object
                result = collection.data.delete_many(where=weaviate_filter_obj)
                return result.successes # Number of successful deletions

            successful_deletes = await anyio.to_thread.run_sync(_perform_delete_by_filter)
            print(f"WeaviateStore: Deletion by filter completed. Objects deleted: {successful_deletes}")
        else:
            print("WeaviateStore: Delete operation requires 'ids' or a 'where' filter.")


    async def close(self):
        """Closes the Weaviate client connection."""
        print("WeaviateStore: Closing Weaviate client...")
        if hasattr(self, '_client') and self._client:
            try:
                # client.close() is synchronous
                await anyio.to_thread.run_sync(self._client.close)
                print("WeaviateStore: Client closed.")
            except Exception as e:
                print(f"WeaviateStore: Error closing client: {e}")

# --- END OF FILE src/clap/vector_stores/weaviate_store.py (REFINED V4 ASYNC) ---