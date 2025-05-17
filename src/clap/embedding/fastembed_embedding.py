import asyncio
import functools
from typing import List, Optional, Any, cast

import anyio

from .base_embedding import EmbeddingFunctionInterface

_FASTEMBED_LIB_AVAILABLE = False
_FastEmbed_TextEmbedding_Placeholder_Type = Any 

try:
    from fastembed import TextEmbedding as ActualTextEmbedding
    _FastEmbed_TextEmbedding_Placeholder_Type = ActualTextEmbedding
    _FASTEMBED_LIB_AVAILABLE = True
except ImportError:
    pass

KNOWN_FASTEMBED_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

class FastEmbedEmbeddings(EmbeddingFunctionInterface):
    _model: _FastEmbed_TextEmbedding_Placeholder_Type
    _dimension: int
    DEFAULT_EMBED_BATCH_SIZE = 256

    def __init__(self,
                 model_name: str = DEFAULT_FASTEMBED_MODEL,
                 dimension: Optional[int] = None,
                 embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
                 **kwargs: Any
                ):
        if not _FASTEMBED_LIB_AVAILABLE:
            raise ImportError(
                "The 'fastembed' library is required to use FastEmbedEmbeddings. "
                "Install with 'pip install fastembed' or 'pip install \"clap-agents[qdrant]\"' (if qdrant includes it as an extra)."
            )

        self.model_name = model_name
        self.embed_batch_size = embed_batch_size

        if dimension is not None:
            self._dimension = dimension
        elif model_name in KNOWN_FASTEMBED_DIMENSIONS:
            self._dimension = KNOWN_FASTEMBED_DIMENSIONS[model_name]
        else:
            raise ValueError(
                f"Dimension for fastembed model '{self.model_name}' is unknown. "
                "Provide 'dimension' parameter or update KNOWN_FASTEMBED_DIMENSIONS."
            )
        
        try:
            self._model = _FastEmbed_TextEmbedding_Placeholder_Type(model_name=self.model_name, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize fastembed model '{self.model_name}': {e}")

    async def __call__(self, input: List[str]) -> List[List[float]]: # Changed to 'input'
        if not input: return []
        if not _FASTEMBED_LIB_AVAILABLE: raise RuntimeError("FastEmbed library not available.")
        
        all_embeddings_list: List[List[float]] = []
        for i in range(0, len(input), self.embed_batch_size):
            batch_texts = input[i:i + self.embed_batch_size]
            if not batch_texts: continue
            try:
                embeddings_iterable = await anyio.to_thread.run_sync(self._model.embed, list(batch_texts))
                for emb_np in embeddings_iterable: all_embeddings_list.append(emb_np.tolist())
            except Exception as e: print(f"Error embedding batch with fastembed: {e}"); raise
        return all_embeddings_list

    def get_embedding_dimension(self) -> int:
        return self._dimension
