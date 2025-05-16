
import abc
from typing import Any, Dict, List, Optional, TypedDict, Union

Document = str
Embedding = List[float]
ID = str
Metadata = Dict[str, Any]

class QueryResult(TypedDict):
    ids: List[List[ID]]
    embeddings: Optional[List[List[Embedding]]]
    documents: Optional[List[List[Document]]]
    metadatas: Optional[List[List[Metadata]]]
    distances: Optional[List[List[float]]]

class VectorStoreInterface(abc.ABC):
    """Abstract Base Class for Vector Store interactions."""

    @abc.abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        ids: List[ID],
        metadatas: Optional[List[Metadata]] = None,
        embeddings: Optional[List[Embedding]] = None,
    ) -> None:
        """
        Add documents and their embeddings to the store.
        If embeddings are not provided, the implementation should handle embedding generation.

        Args:
            documents: List of document texts.
            ids: List of unique IDs for each document.
            metadatas: Optional list of metadata dictionaries for each document.
            embeddings: Optional list of pre-computed embeddings.
        """
        pass

    @abc.abstractmethod
    async def aquery(
        self,
        query_texts: Optional[List[Document]] = None,
        query_embeddings: Optional[List[Embedding]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["metadatas", "documents", "distances"],
    ) -> QueryResult:
        """
        Query the vector store for similar documents.
        Provide either query_texts or query_embeddings.

        Args:
            query_texts: List of query texts. Embeddings will be generated.
            query_embeddings: List of query embeddings.
            n_results: Number of results to return for each query.
            where: Optional metadata filter (syntax depends on implementation).
            where_document: Optional document content filter (syntax depends on implementation).
            include: List of fields to include in the results (e.g., "documents", "metadatas", "distances", "embeddings").

        Returns:
            A QueryResult dictionary containing the search results.
        """
        pass

    @abc.abstractmethod
    async def adelete(
        self,
        ids: Optional[List[ID]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete documents from the store by ID or filter.

        Args:
            ids: Optional list of IDs to delete.
            where: Optional metadata filter for deletion.
            where_document: Optional document content filter for deletion.
        """
        pass

   

