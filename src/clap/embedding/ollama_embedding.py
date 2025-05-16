# --- START OF REVISED src/clap/embedding/ollama_embeddings.py ---
import asyncio
import functools
from typing import List, Optional, Any, cast
import sys 
import os 
import anyio

from .base_embedding import EmbeddingFunctionInterface

try:
    from ollama import AsyncClient as OllamaAsyncClient
    from ollama import ResponseError as OllamaResponseError
    OLLAMA_PYTHON_INSTALLED = True
except ImportError as e:
    print(f"DEBUG: ImportError for 'ollama': {e}")
    OLLAMA_PYTHON_INSTALLED = False
    class OllamaAsyncClient: pass 
    class OllamaResponseError(Exception): pass 

KNOWN_OLLAMA_EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384, 
    "llama3": 4096, 
    "llama3.2:latest": 4096, 
}
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text" 

class OllamaEmbeddings(EmbeddingFunctionInterface):
    _client: OllamaAsyncClient
    _model_name: str
    _dimension: int

    def __init__(self,
                 model_name: str = DEFAULT_OLLAMA_EMBED_MODEL,
                 dimension: Optional[int] = None, 
                 ollama_host: str = "http://localhost:11434",
                 **kwargs: Any
                ):
        if not OLLAMA_PYTHON_INSTALLED:
            raise ImportError("The 'ollama' Python library is required. Install with: pip install ollama")

        self.model_name = model_name
        self._client = OllamaAsyncClient(host=ollama_host, **kwargs)

        if dimension is not None:
            self._dimension = dimension
            print(f"OllamaEmbeddings: Using user-provided dimension {self._dimension} for model '{self.model_name}'.")
        elif model_name in KNOWN_OLLAMA_EMBEDDING_DIMENSIONS:
            self._dimension = KNOWN_OLLAMA_EMBEDDING_DIMENSIONS[model_name]
            print(f"OllamaEmbeddings: Using known dimension {self._dimension} for model '{self.model_name}'.")
        else:
            
            raise ValueError(
                f"Dimension for Ollama model '{self.model_name}' is unknown and not provided. "
                "Please provide the 'dimension' parameter during initialization or "
                f"add the model and its dimension to KNOWN_OLLAMA_EMBEDDING_DIMENSIONS in ollama_embeddings.py."
            )
        
        print(f"Initialized OllamaEmbeddings for model '{self.model_name}' (dim: {self._dimension}) targeting {ollama_host}.")

    async def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        
        try:
            response = await self._client.embed(model=self.model_name, input=input)
            
            embeddings_data = response.get("embeddings") 
            if embeddings_data is None and len(input) == 1 and response.get("embedding"):
                single_embedding = response.get("embedding")
                if isinstance(single_embedding, list) and all(isinstance(x, (int, float)) for x in single_embedding):
                    embeddings_data = [single_embedding] 
            
            if not isinstance(embeddings_data, list) or \
               not all(isinstance(e, list) for e in embeddings_data):
                raise TypeError(
                    f"Ollama embed returned unexpected format: {type(embeddings_data)}. "
                    f"Expected List[List[float]]. Raw response: {response}"
                )

            # print(f"OllamaEmbeddings: Embedding completed for {len(texts)} texts.")
            return cast(List[List[float]], embeddings_data)
        except OllamaResponseError as e:
            print(f"Ollama API error during embedding with model '{self.model_name}': {e.error} (Status: {e.status_code})")
            raise
        except Exception as e:
            print(f"Unexpected error during Ollama embedding: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        return self._dimension

    async def close(self):
        if hasattr(self._client, "_client") and hasattr(self._client._client, "is_closed"):
             if not self._client._client.is_closed: 
                await self._client._client.aclose() 
        elif hasattr(self._client, 'aclose'): 
            await self._client.aclose() 
        print(f"OllamaEmbeddings: Closed client for {self.model_name}.")
