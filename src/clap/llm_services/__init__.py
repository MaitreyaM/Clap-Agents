# src/clap/llm_services/__init__.py
from .base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall
from .groq_service import GroqService
from .google_openai_compat_service import GoogleOpenAICompatService

__all__ = [
    "LLMServiceInterface", "StandardizedLLMResponse", "LLMToolCall",
    "GroqService", "GoogleOpenAICompatService",
]

try:
    from .ollama_service import OllamaOpenAICompatService as OllamaService # Assuming file is ollama_service.py
    __all__.append("OllamaService")
except ImportError:
    pass