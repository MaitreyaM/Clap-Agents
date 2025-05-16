# src/clap/llm_services/__init__.py
from .base import LLMServiceInterface, StandardizedLLMResponse, LLMToolCall
from .groq_service import GroqService
from .google_openai_compat_service import GoogleOpenAICompatService
# Remove the old custom OllamaLLMService if you had it
try:
    from .ollama_service import OllamaOpenAICompatService # New one
    _OLLAMA_LLM_AVAILABLE = True
except ImportError:
    _OLLAMA_LLM_AVAILABLE = False


__all__ = [
    "LLMServiceInterface", "StandardizedLLMResponse", "LLMToolCall",
    "GroqService", "GoogleOpenAICompatService"
]
if _OLLAMA_LLM_AVAILABLE:
    __all__.append("OllamaOpenAICompatService") # Add the new one