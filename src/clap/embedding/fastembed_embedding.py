import asyncio
import functools
from typing import List, Optional, Any, cast
import anyio

from .base_embedding import EmbeddingFunctionInterface

try:
    from fastembed import TextEmbedding 
    _FASTEMBED_LIB_AVAILABLE = True
except ImportError:
    _FASTEMBED_LIB_AVAILABLE = False

KNOWN_FASTEMBED_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384, 
    # Add other known model_name: dimension pairs
}
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5" 

class FastEmbedEmbeddings(EmbeddingFunctionInterface):
    _model: TextEmbedding
    _dimension: int
    DEFAULT_EMBED_BATCH_SIZE = 256 

    def __init__(self,
                 model_name: str = DEFAULT_FASTEMBED_MODEL,
                 dimension: Optional[int] = None, # User can override or provide for unknown models
                 embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
                 **kwargs: Any # Passed to TextEmbedding constructor
                ):
        if not _FASTEMBED_LIB_AVAILABLE:
            raise ImportError("The 'fastembed' library is required. Install via 'pip install fastembed'")

        self.model_name = model_name
        self.embed_batch_size = embed_batch_size

        if dimension is not None:
            self._dimension = dimension
            print(f"FastEmbedEmbeddings: Using user-provided dimension {self._dimension} for model '{self.model_name}'.")
        elif model_name in KNOWN_FASTEMBED_DIMENSIONS:
            self._dimension = KNOWN_FASTEMBED_DIMENSIONS[model_name]
            print(f"FastEmbedEmbeddings: Using known dimension {self._dimension} for model '{self.model_name}'.")
        else:
            raise ValueError(
                f"Dimension for fastembed model '{self.model_name}' is unknown and not provided. "
                "Please provide the 'dimension' parameter or "
                f"add the model and its dimension to KNOWN_FASTEMBED_DIMENSIONS in fastembed_embeddings.py."
            )
        
        try:
            # print(f"Initializing fastembed model '{self.model_name}' (dim: {self._dimension})...")
            self._model = TextEmbedding(model_name=self.model_name, **kwargs)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize fastembed model '{self.model_name}': {e}")

    async def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        
        all_embeddings_list: List[List[float]] = []
        # print(f"FastEmbedEmbeddings: Embedding {len(input)} input in batches of {self.embed_batch_size}...")
        for i in range(0, len(input), self.embed_batch_size):
            batch_texts = input[i:i + self.embed_batch_size]
            if not batch_texts: continue

            try:
                embeddings_iterable = await anyio.to_thread.run_sync(
                    self._model.embed, list(batch_texts)
                )
                for emb_np in embeddings_iterable:
                    all_embeddings_list.append(emb_np.tolist())
            except Exception as e:
                print(f"Error embedding batch with fastembed: {e}")
                raise
        
        # print(f"FastEmbedEmbeddings: Embedding completed for {len(texts)} texts.")
        return all_embeddings_list

    def get_embedding_dimension(self) -> int:
        return self._dimension
