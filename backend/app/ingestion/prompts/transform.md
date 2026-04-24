# Transform Stage

## Overview

The Transform stage takes raw `Document` objects (output from the Extract stage) and prepares them for embedding and storage. It runs a **3-step pipeline**: Clean → dual sub-pipeline (Summarize + Split) → Combine.

---

## Input

A list of `Document` objects produced by `extract.py`:

```python
class Document(BaseModel):
    id: str                        # e.g. "report.pdf"
    source_type: str               # "pdf" | "docx" | "notion"
    title: str
    content: str                   # Raw Markdown text
    metadata: Dict[str, Any]
```

---

## Pipeline Steps

### Step 1 — Clean (`clean_document`)

Normalize the raw Markdown content before any further processing.

**Rules (applied in order):**

| # | Rule | Detail |
|---|------|--------|
| 1 | Remove emojis | Strip all Unicode emoji characters (blocks U+1F300–U+1FAFF and related) |
| 2 | Remove URLs | Strip `http(s)://...` and bare `www.` links |
| 3 | Remove excessive whitespace | Collapse 3+ consecutive blank lines → 2 blank lines; strip trailing spaces per line |
| 4 | Lowercase text | Convert all text to lowercase **except** Markdown heading markers (`#`) and code fences (` ``` `) |
| 5 | Normalize Unicode | Apply NFC normalization (`unicodedata.normalize("NFC", text)`) |
| 6 | Strip control characters | Remove non-printable characters (except `\n`, `\t`) |

**Output:** A new `Document` with cleaned `content` and `metadata["cleaned"] = True`.

---

### Step 2 — Dual Sub-Pipeline

Run **both** sub-pipelines in parallel on the cleaned document.

#### 2a. Summarize (`summarize_document`)

Generate a concise summary of the full document.

- **Model:** Use the project's LLM client (`llm_client.py`)
- **Prompt template:** `prompts/summarize.md` (create if not exists)
- **Input:** `document.content` (full cleaned Markdown)
- **Output:** A plain-text string — `summary`
- **Max tokens for summary:** ~300 tokens

**Prompt guidelines:**
```
You are a helpful assistant. Summarize the following document concisely.
Focus on key topics, main arguments, and important facts.
Respond in the same language as the document.
```

#### 2b. Chunk (`chunk_document`)

Split the cleaned Markdown into semantic chunks using `MarkdownHeaderTextSplitter`.

- **Splitter:** `langchain_text_splitters.MarkdownHeaderTextSplitter`
- **Headers to split on:**
  ```python
  headers_to_split_on = [
      ("#",  "H1"),
      ("##", "H2"),
      ("###","H3"),
  ]
  ```
- **Fallback:** If the document has no Markdown headers, fall back to `RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=50`.
- **Output:** A list of `Chunk` objects:

```python
class Chunk(BaseModel):
    doc_id: str           # parent Document.id
    chunk_index: int      # 0-based position in the chunk list
    content: str          # chunk text
    metadata: Dict        # inherits parent metadata + header path info
```

---

### Step 3 — Combine (`combine_results`)

Merge the summary and chunks into a list of `TransformedDocument` objects ready for embedding.

**Design principle — Separate retrieval content from generation context:**

The summary and chunk content serve two different consumers:
- The **embedding model** (retrieval stage) needs clean, precise chunk text to produce accurate similarity scores. Mixing the summary into `page_content` causes all chunks to share nearly identical embeddings, making retrieval unreliable.
- The **LLM** (generation stage) benefits from the summary as global context when synthesizing an answer.

Therefore: the summary is stored in `metadata["doc_summary"]` only. The Generator agent injects it into the prompt at generation time — it is never embedded into `page_content`.

**Structure per document:**

```
[Summary Chunk]  ← index 0, special role — indexed for retrieval like any other chunk
  doc_id     : document.id
  chunk_index: 0
  content    : "Summary of <title>:\n<summary text>"   ← embedded as its own chunk
  metadata   : { ..., "role": "summary", "doc_summary": "<summary text>" }

[Content Chunks] ← index 1, 2, 3, ...
  doc_id     : document.id
  chunk_index: 1, 2, 3, ...
  content    : <chunk text only — no summary prefix>
  metadata   : { ..., "role": "chunk", "header_path": "H1 > H2 > ...",
                 "doc_summary": "<summary text>" }   ← summary stored here for Generator use
```

**Key rules:**
- The `doc_summary` field is populated on **every** chunk (both summary chunk and content chunks) so the Generator always has access to the document's global context regardless of which chunks are retrieved.
- The summary text is **never prepended** to content chunk `content`. Embedding the summary inline would cause every chunk to produce nearly identical vectors, collapsing similarity scores and making ranked retrieval meaningless.
- The summary chunk (index 0) is the only place the summary appears in `content`, and it is indexed as a standalone retrieval unit — useful when the user query targets high-level document topics.

**Example output for `report.pdf`:**
```
chunk 0 → content   : "Summary of report.pdf:\nThis document covers..."
           metadata  : { "role": "summary", "doc_summary": "This document covers..." }

chunk 1 → content   : "# introduction\nThis report aims to..."
           metadata  : { "role": "chunk", "header_path": "H1", 
                         "doc_summary": "This document covers..." }

chunk 2 → content   : "## methodology\nWe used a mixed-methods..."
           metadata  : { "role": "chunk", "header_path": "H1 > H2",
                         "doc_summary": "This document covers..." }
```

**How the Generator uses `doc_summary` at inference time:**
```python
# Generator agent — build LLM prompt context from retrieved chunks
for doc in retrieved_docs:
    summary = doc.metadata.get("doc_summary", "")
    chunk_text = doc.page_content
    full_context = f"[Document Summary]: {summary}\n[Relevant Chunk]: {chunk_text}"
```

**Output type:**

```python
class TransformedDocument(BaseModel):
    doc_id: str
    title: str
    source_type: str
    chunks: List[Chunk]       # summary chunk first (index 0), then content chunks
    metadata: Dict[str, Any]
```

---

## Output

A list of `TransformedDocument` objects, one per input `Document`, ready to be passed to the **Load** stage for embedding and vector store insertion.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| LLM summarization fails | Log warning; use `summary = ""` and mark `metadata["summary_failed"] = True` |
| Document has no content after cleaning | Skip document; log warning |
| Chunking produces 0 chunks | Use the full cleaned content as a single chunk |

---

## File Layout

```
backend/app/ingestion/
├── extract.py          # Stage 1 — Extract
├── transform.py        # Stage 2 — Transform  ← implement here
├── transform.md        # This spec
└── prompts/
    └── summarize.md    # Summarization prompt template
```