(cognitive-layer) maitreyamishra@Maitreyas-MacBook-Air examples % python qdrant_ingestion.py


========== TEST 1: QDRANT WITH CUSTOM SentenceTransformerEmbeddings ==========
Initialized SentenceTransformerEmbeddings with model 'all-MiniLM-L6-v2', dimension: 384

--- Starting RAG Cycle ---
DB Path: ./qdrant_test_dbs/custom_ef_db
Collection: ml_book_custom_ef
Embedding: Custom SentenceTransformerEmbeddings
QdrantStore: Using provided EmbeddingFunctionInterface (dimension: 384).
QdrantStore: Initializing Qdrant client for collection 'ml_book_custom_ef' at './qdrant_test_dbs/custom_ef_db'.
Traceback (most recent call last):
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/site-packages/portalocker/portalocker.py", line 118, in lock
    LOCKER(file_, flags)
BlockingIOError: [Errno 35] Resource temporarily unavailable

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/site-packages/qdrant_client/local/async_qdrant_local.py", line 125, in _load
    portalocker.lock(
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/site-packages/portalocker/portalocker.py", line 131, in lock
    raise exceptions.AlreadyLocked(
portalocker.exceptions.AlreadyLocked: [Errno 35] Resource temporarily unavailable

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/qdrant_ingestion.py", line 190, in <module>
    asyncio.run(main())
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/qdrant_ingestion.py", line 164, in main
    await perform_rag_cycle(
  File "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/examples/qdrant_ingestion.py", line 81, in perform_rag_cycle
    qdrant_store = QdrantStore(
  File "/Users/maitreyamishra/PROJECTS/Cognitive-Layer/src/clap/vector_stores/qdrant_store.py", line 117, in __init__
    self._async_client = AsyncQdrantClient(path=path)
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/site-packages/qdrant_client/async_qdrant_client.py", line 121, in __init__
    self._client = AsyncQdrantLocal(
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/site-packages/qdrant_client/local/async_qdrant_local.py", line 66, in __init__
    self._load()
  File "/Applications/anaconda3/envs/cognitive-layer/lib/python3.10/site-packages/qdrant_client/local/async_qdrant_local.py", line 130, in _load
    raise RuntimeError(
RuntimeError: Storage folder ./qdrant_test_dbs/custom_ef_db is already accessed by another instance of Qdrant client. If you require concurrent access, use Qdrant server instead.
(cognitive-layer) maitreyamishra@Maitreyas-MacBook-Air examples % 

