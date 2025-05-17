from typing import List, Optional, Any 
from .base_embedding import EmbeddingFunctionInterface

_ST_LIB_AVAILABLE = False
_SentenceTransformer_Placeholder_Type = Any 

try:
    from sentence_transformers import SentenceTransformer as ImportedSentenceTransformer
    _SentenceTransformer_Placeholder_Type = ImportedSentenceTransformer
    _ST_LIB_AVAILABLE = True
except ImportError:
    pass

class SentenceTransformerEmbeddings(EmbeddingFunctionInterface):
    model: _SentenceTransformer_Placeholder_Type
    _dimension: int

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if not _ST_LIB_AVAILABLE:
            raise ImportError(
                "The 'sentence-transformers' library is required to use SentenceTransformerEmbeddings. "
                "Install with 'pip install sentence-transformers' or 'pip install \"clap-agents[sentence-transformers]\"'."
            )
        
        try:
            self.model = _SentenceTransformer_Placeholder_Type(model_name, device=device)
            dim = self.model.get_sentence_embedding_dimension() 
            if dim is None: 
                dummy_embedding = self.model.encode("test") 
                dim = len(dummy_embedding)
            self._dimension = dim
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer model '{model_name}': {e}")



    def __call__(self, input: List[str]) -> List[List[float]]: 
        if not _ST_LIB_AVAILABLE: 
            raise RuntimeError("SentenceTransformers library not available for embedding operation.")
        embeddings_np = self.model.encode(input, convert_to_numpy=True) 
        return embeddings_np.tolist()

    def get_embedding_dimension(self) -> int:
        return self._dimension
