# src/clap/embedding/__init__.py
from .base_embedding import EmbeddingFunctionInterface
from .sentence_transformer_embedding import SentenceTransformerEmbeddings
from .fastembed_embedding import FastEmbedEmbeddings # Assuming this was kept

try:
    from .ollama_embeddings import OllamaEmbeddings
    _OLLAMA_EMBED_AVAILABLE = True
except ImportError:
    _OLLAMA_EMBED_AVAILABLE = False

__all__ = [
    "EmbeddingFunctionInterface",
    "SentenceTransformerEmbeddings",
    "FastEmbedEmbeddings", # Keep if you retained it
]
if _OLLAMA_EMBED_AVAILABLE:
    __all__.append("OllamaEmbeddings")