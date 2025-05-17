import asyncio
import functools
from typing import List, Optional, Any, cast

import anyio

from .base_embedding import EmbeddingFunctionInterface

_OLLAMA_LIB_AVAILABLE = False
_OllamaAsyncClient_Placeholder_Type = Any
_OllamaResponseError_Placeholder_Type = type(Exception)

try:
    from ollama import AsyncClient as ImportedOllamaAsyncClient
    from ollama import ResponseError as ImportedOllamaResponseError
    _OllamaAsyncClient_Placeholder_Type = ImportedOllamaAsyncClient
    _OllamaResponseError_Placeholder_Type = ImportedOllamaResponseError
    _OLLAMA_LIB_AVAILABLE = True
except ImportError:
    pass

KNOWN_OLLAMA_EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768, "mxbai-embed-large": 1024, "all-minilm": 384,
    "llama3": 4096, "llama3.2:latest": 4096, "nomic-embed-text:latest": 768,
}
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"

class OllamaEmbeddings(EmbeddingFunctionInterface):
    _client: _OllamaAsyncClient_Placeholder_Type
    _model_name: str
    _dimension: int

    def __init__(self,
                 model_name: str = DEFAULT_OLLAMA_EMBED_MODEL,
                 dimension: Optional[int] = None,
                 ollama_host: str = "http://localhost:11434",
                 **kwargs: Any):
        if not _OLLAMA_LIB_AVAILABLE:
            raise ImportError("The 'ollama' Python library is required. Install with: pip install 'clap-agents[ollama]'")

        self.model_name = model_name
        self._client = _OllamaAsyncClient_Placeholder_Type(host=ollama_host, **kwargs)

        if dimension is not None: self._dimension = dimension
        elif model_name in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS:
            self._dimension = KNOWN_OLLAMA_EMBEDDING_DIMENSIONS[model_name]
        else:
            raise ValueError(f"Dimension for Ollama model '{model_name}' unknown. Provide 'dimension' or update KNOWN_OLLAMA_EMBEDDING_DIMENSIONS.")
        print(f"Initialized OllamaEmbeddings for model '{self.model_name}' (dim: {self._dimension}).")

    async def __call__(self, input: List[str]) -> List[List[float]]: 
        if not input: return []
        if not _OLLAMA_LIB_AVAILABLE: raise RuntimeError("Ollama library not available.")
        try:
            response = await self._client.embed(model=self.model_name, input=input) 
            embeddings_data = response.get("embeddings")
            if embeddings_data is None and len(input) == 1 and response.get("embedding"):
                single_embedding = response.get("embedding")
                if isinstance(single_embedding, list) and all(isinstance(x, (int, float)) for x in single_embedding):
                    embeddings_data = [single_embedding]
            if not isinstance(embeddings_data, list) or not all(isinstance(e, list) for e in embeddings_data):
                raise TypeError(f"Ollama embed returned unexpected format. Expected List[List[float]]. Resp: {response}")
            return cast(List[List[float]], embeddings_data)
        except _OllamaResponseError_Placeholder_Type as e:
            print(f"Ollama API error: {getattr(e, 'error', str(e))} (Status: {getattr(e, 'status_code', 'N/A')})")
            raise
        except Exception as e: print(f"Unexpected Ollama embedding error: {e}"); raise

    def get_embedding_dimension(self) -> int: return self._dimension

    async def close(self):
        if _OLLAMA_LIB_AVAILABLE:
            if hasattr(self._client, "_client") and hasattr(self._client._client, "is_closed"):
                if not self._client._client.is_closed: await self._client._client.aclose() 
            elif hasattr(self._client, 'aclose'): await self._client.aclose() 
        print(f"OllamaEmbeddings: Closed client for {self.model_name}.")
