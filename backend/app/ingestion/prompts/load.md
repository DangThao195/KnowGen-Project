# Load Stage

## Overview

The Load stage is the final step in the data ingestion pipeline. It takes the structured `TransformedDocument` items (output from the Transform stage), converts their text chunks into mathematical vectors (Embeddings), and stores them into a **FAISS (Facebook AI Similarity Search)** vector database for efficient downstream retrieval in the RAG system.

---

## Input

A list of `TransformedDocument` objects produced by `transform_documents()`:

```python
class TransformedDocument(BaseModel):
    doc_id: str
    title: str
    source_type: str
    chunks: List[Chunk]       # Includes both the summary_chunk and content chunks.
    metadata: Dict[str, Any]  # Passes down metadata like file path or URLs.
```

---

## Pipeline Steps

### Step 1 — Initialize Embedding Model

Set up the embedding model that will convert text into fixed-size dense vectors. Since the project frequently contains cross-lingual vocabulary (e.g., mixing Vietnamese grammar with English terminology like "convolutional neural network"), the embedding model must be highly capable of multilingual semantic alignment.

- **Model (Local - Recommended for Cross-lingual):** `intfloat/multilingual-e5-base`. This HuggingFace model maps 100+ languages into a shared vector space, performing extremely well when sentences contain mixed languages. *(Note: E5 models typically require prefixing documents with `passage: ` and user queries with `query: `).*
- **Output:** A configured `Embeddings` object compatible with LangChain.

### Step 2 — Flatten Chunks into Documents

LangChain's FAISS implementation expects a list of standard `langchain_core.documents.Document` objects. We need to unpack our custom Pydantic `Chunk` models.

- Iterate through the `.chunks` list of each `TransformedDocument`.
- For every `Chunk`, create a standard LangChain `Document`:
  - `page_content`: `chunk.content` (Note: The Local-Global Context technique has already prepended the summary to this content in the Transform stage).
  - `metadata`: `chunk.metadata` (Includes `doc_id`, `role`, `header_path`, etc.).

### Step 3 — Embed and Store in FAISS

Pass the flattened documents to FAISS to compute embeddings and build the retrieval index.

- **Action:** `FAISS.from_documents(documents, embedding_model)`
- **Behavior:** This step automatically batches the text chunks, calls the Embedding model to generate vector representations, and maps them in the local RAM.

### Step 4 — Save to Disk

Persist the FAISS index locally so the Recommendation or QA Agents can load it instantaneously later without needing to re-embed the text.

- **Action:** `vectorstore.save_local("vector_store/faiss_index")`
- **Output artifacts:** 
  - `index.faiss` (The raw dense vectors optimized for similarity search).
  - `index.pkl` (The pickled metadata mapping between the vectors and the original text).

---

## Output

This function does not necessarily need to return a value (it can return a boolean success flag or a status string), but the actual product of this stage is the **local FAISS database folder** saved to the disk.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Empty input | If the `documents` list is empty, log a warning and return early to avoid memory errors. |
| API Rate Limit (if using Cloud Embeddings) | LangChain's embedding wrappers typically have built-in retry mechanisms. However, wrap the embedding call in a Try/Catch block to log exceptions and prevent corrupting the existing local FAISS index on disk. |
| Missing FAISS dependencies | Ensure `faiss-cpu` (or `faiss-gpu`) is installed. Catch `ImportError` explicitly at the top of the file. |


