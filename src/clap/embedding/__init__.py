from .base_embedding import EmbeddingFunctionInterface

__all__ = ["EmbeddingFunctionInterface"]

try:
    from .sentence_transformer_embedding import SentenceTransformerEmbeddings
    __all__.append("SentenceTransformerEmbeddings")
except ImportError:
    pass

try:
    from .fastembed_embedding import FastEmbedEmbeddings
    __all__.append("FastEmbedEmbeddings")
except ImportError:
    pass

try:
    from .ollama_embedding import OllamaEmbeddings
    __all__.append("OllamaEmbeddings")
except ImportError:
    pass