# CHANGELOG.md

## [0.2.0] - 2025-05-16
### Added
- RAG (Retrieval Augmented Generation) capabilities for `Agent` and `ToolAgent`.
- `VectorStoreInterface` for abstracting vector database interactions.
- `ChromaStore` implementation for ChromaDB.
- `QdrantStore` implementation for Qdrant (local mode, supporting custom EFs and fastembed wrapper).
- `EmbeddingFunctionInterface` and concrete implementations:
    - `SentenceTransformerEmbeddings`
    - `OllamaEmbeddings`
    - `FastEmbedEmbeddings`
- `Ollama Compatibility` Now run Ollama models locally (both embedding and chat).
- PDF and CSV loading utilities in `rag_utils`.
- Argument type coercion in `Tool.run()` for robustness.

### Changed
- `ToolAgent` and `Agent` now require `llm_service` and `model` to be explicitly passed during initialization.
- Refined prompts for `ReactAgent` for better tool usage and RAG.
- Improved error handling and logging in various components.

### Fixed
- Resolved various import errors and `NameError` issues in examples and core files.
- Fixed argument type mismatches for LLM tool calls.

## [0.1.1] - 2025-04-19
- Initial release with core agent patterns.