# RAG Architecture Components: Building Blocks vs. Streamlined Components

This document provides a detailed overview of the core components in the Multi-modal Enterprise KM Agent's RAG (Retrieval-Augmented Generation) system. Components are categorized into **Building Blocks** (low-level, reusable utilities for specific tasks) and **Streamlined Components** (higher-level orchestrators that combine building blocks into end-to-end workflows).

The architecture follows a modular design where building blocks handle atomic operations (e.g., parsing, embedding), while streamlined components orchestrate these into complete pipelines (e.g., full RAG workflows or model management).

## Building Blocks

These are the foundational "Lego pieces" for specific, reusable tasks. They focus on one responsibility and can be used independently or extended.

### 1. `src/rag/managers/quantization_manager.py`
- **Purpose**: Applies PyTorch quantization (dynamic/static) to models for memory reduction.
- **Key Features**: Dynamic/static quantization, M1 Pro optimization (qnnpack engine), compatibility checks.
- **Use Cases**:
  - Memory optimization for local inference (e.g., reduce DialoGPT from 2GB to ~500MB).
  - Pre-deployment model compression.
- **Example**: `quantized_model = quantization_manager.apply_pytorch_quantization(model, "dynamic")`.

### 2. `src/rag/vector_database.py`
- **Purpose**: Wrapper for ChromaDB vector storage and similarity queries.
- **Key Features**: Collection management, document addition with embeddings, metadata support, batch operations.
- **Use Cases**:
  - Knowledge base storage post-indexing.
  - Retrieval queries in RAG pipelines.
- **Example**: `db = get_vector_db(); db.add_documents(ids, embeddings, metadatas, docs)`.

### 3. `src/rag/text_chunker.py`
- **Purpose**: Splits text into chunks using strategies like tokens, words, or sentences.
- **Key Features**: Multiple strategies (spaCy/jieba), overlap support, lazy-loading dependencies.
- **Use Cases**:
  - Document preprocessing before embedding/indexing.
  - Multilingual text handling (e.g., Chinese with jieba).
- **Example**: `chunker = TextChunker(strategy="sentences"); chunks = chunker.chunk_text(long_text)`.

### 4. `src/rag/prompt_template.py`
- **Purpose**: Formats RAG prompts with questions and context.
- **Key Features**: System role customization, context truncation, simple templating.
- **Use Cases**:
  - Prompt engineering for LLMs.
  - Input formatting before generation.
- **Example**: `template = get_prompt_template(system_role="Be helpful"); prompt = template.format_prompt(question, context_docs)`.

### 5. `src/rag/embedding.py`
- **Purpose**: Generates vector embeddings using sentence-transformers.
- **Key Features**: Model loading (e.g., all-MiniLM-L6-v2), batch encoding, device handling.
- **Use Cases**:
  - Convert text to vectors for retrieval.
  - Switch embedding models for experimentation.
- **Example**: `model = get_embedding_model(); vec = model.encode("Hello")`.

### 6. `src/rag/document_parser.py`
- **Purpose**: Extracts text from PDFs and TXT files.
- **Key Features**: pypdf for PDFs, basic I/O for TXT, file validation and error handling.
- **Use Cases**:
  - Document ingestion from user uploads.
  - Preprocessing before chunking.
- **Example**: `text = parse_pdf("file.pdf")`.

## Streamlined Components

These are higher-level orchestrators that combine building blocks into complete workflows. They handle complexity, error recovery, and integration (e.g., with tracking or caching).

### 1. `src/rag/managers/experiment_manager.py`
- **Purpose**: Manages A/B testing, benchmarking, and performance tracking with MLflow.
- **Key Features**: Experiment logging, system resource monitoring, winner determination.
- **Use Cases**:
  - Production monitoring (e.g., latency tracking).
  - A/B testing models (e.g., base vs. fine-tuned).
  - Benchmarking before deployment.
- **Example**: `experiment_manager.start_ab_test(ABTestConfig(test_name="model_v1_vs_v2", model_a="DialoGPT-medium", model_b="Phi-2"))`.

### 2. `src/rag/managers/lora_manager.py`
- **Purpose**: Handles LoRA fine-tuning, adapter management, and M1 Pro optimization.
- **Key Features**: Model loading with LoRA, adapter saving/loading, memory-efficient training.
- **Use Cases**:
  - Fine-tuning models (e.g., adapt DialoGPT to dialogue tasks).
  - Memory-constrained adaptation (reduce trainable params by 90%+).
  - Iterative experimentation with adapters.
- **Example**: `lora_manager = LoRAConfigManager(base_model_path="microsoft/DialoGPT-medium"); model = lora_manager.apply_lora()`.

### 3. `src/rag/managers/model_manager.py`
- **Purpose**: Central hub for model loading, caching, optimization, and selection.
- **Key Features**: Quantization, device mapping, dynamic switching, A/B testing integration.
- **Use Cases**:
  - Dynamic model selection (e.g., based on latency constraints).
  - Production deployment with caching.
  - Experimentation and benchmarking.
- **Example**: `model, tokenizer = model_manager.load_model("microsoft/DialoGPT-medium", use_quantization=True)`.

### 4. `src/rag/retriever.py`
- **Purpose**: Performs semantic search by embedding queries and querying the vector DB.
- **Key Features**: Top-k retrieval, metadata inclusion, experiment tracking integration.
- **Use Cases**:
  - RAG retrieval step (get relevant context).
  - Standalone knowledge base queries.
- **Example**: `retriever = get_retriever(); results = retriever.retrieve("What is RAG?", top_k=5)`.

### 5. `src/rag/rag_pipeline.py`
- **Purpose**: End-to-end RAG pipeline (retrieve → prompt → generate) with tracking.
- **Key Features**: Context truncation, generation, MLflow logging.
- **Use Cases**:
  - Complete Q&A workflows.
  - Powering production APIs (e.g., FastAPI endpoints).
- **Example**: `pipeline = get_rag_pipeline(); answer = pipeline.answer_question("What is AI?")`.

### 6. `src/rag/indexing_pipeline.py`
- **Purpose**: Indexes documents (parse → chunk → embed → store).
- **Key Features**: Batch processing, error handling, integration with parsers and DB.
- **Use Cases**:
  - Knowledge base construction (e.g., index PDFs/TXTs).
  - Adding new documents to the system.
- **Example**: `pipeline = get_indexing_pipeline(); pipeline.index_document("doc.pdf")`.

### 7. `src/rag/llm_client.py`
- **Purpose**: Handles LLM inference (local Transformers or cloud vLLM) with model selection.
- **Key Features**: Generation, dynamic loading, A/B testing, performance tracking.
- **Use Cases**:
  - Text generation from prompts.
  - Model comparison and benchmarking.
- **Example**: `client = get_llm_client(); response = client.generate("Prompt")`.

## Usage Guidance

- **Prefer Streamlined Components for Production**: Use them for end-to-end tasks to avoid reinventing orchestration (e.g., use `rag_pipeline.py` instead of manually chaining `retriever.py` + `llm_client.py`).
- **Extend Building Blocks for Customization**: Modify or subclass them for specific needs (e.g., add a new chunking strategy in `text_chunker.py`).
- **Managers as Controllers**: Use managers (e.g., `model_manager.py`) for cross-cutting concerns like caching or optimization.
- **Integration in Workflows**: In your RAG notebook/script, start with streamlined components (e.g., `indexing_pipeline.py` for Phase 1) and drill down to building blocks only when customizing.
- **Avoid Overlap**: Don't duplicate logic—building blocks feed into streamlined components.

This structure ensures scalability: as your project grows, add new building blocks and compose them into streamlined components without disrupting existing workflows.
