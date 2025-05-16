from typing import List, Optional
from .base_embedding import EmbeddingFunctionInterface

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers library not found. Please install with `pip install sentence-transformers`"
    )

class SentenceTransformerEmbeddings(EmbeddingFunctionInterface):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        if self._dimension is None:
            dummy_embedding = self.model.encode("test")
            self._dimension = len(dummy_embedding)
        print(f"Initialized SentenceTransformerEmbeddings with model '{model_name}', dimension: {self._dimension}")


    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist() 

    def get_embedding_dimension(self) -> int:
        return self._dimension