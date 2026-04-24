# KnowGen - Multi-Agent Pedagogic Assistant

## Project Overview

KnowGen is a **multi-agent RAG (Retrieval-Augmented Generation) system** that serves as a pedagogic assistant. It ingests documents (PDF, DOCX, Notion), processes them into a vector store, and uses a LangGraph-based multi-agent workflow to answer questions and generate quizzes. The system is **bilingual** (English/Vietnamese).

## Architecture

The system has two main workflows:

### 1. Offline Ingestion (ETL Pipeline)

Raw documents go through Extract -> Transform -> Load:

- **Extract** (`backend/app/ingestion/extract.py`): Converts PDF (pymupdf4llm), DOCX (python-docx), and Notion (API) into a canonical `Document` schema with Markdown-formatted content.
- **Transform** (`backend/app/ingestion/transform.py`): Cleans text (emoji/URL removal, Unicode normalization, lowercasing), generates LLM summaries, chunks via `MarkdownHeaderTextSplitter` (fallback: `RecursiveCharacterTextSplitter` at 500 chars/50 overlap), and applies **Global-Local Context** strategy (prepend summary to each chunk).
- **Load** (`backend/app/ingestion/load.py`): Embeds chunks using `intfloat/multilingual-e5-base` (with `passage:` prefix) and stores in FAISS index at `vector_store/faiss_index/`.
- **Runner** (`backend/app/ingestion/etl.py`): CLI orchestrator with `--files`, `--notion`, `--sample` args.

### 2. Online Workflow (Multi-Agent LangGraph)

User queries flow through a LangGraph state machine defined in `backend/app/agents/multi_agent.py`:

```
START -> SUPERVISOR -> RETRIEVER -> [GENERATOR -> CRITIC (loop)] -> END
```

**Shared State** (`backend/app/agents/agent_state.py`): `AgentState` TypedDict with fields for planning, retrieval, generation, and reflection.

#### Agents:

- **Supervisor** (`backend/app/agents/supervisor_agent.py`) - COMPLETE
  - First node. Classifies task_type (qa/quiz/unknown), detects language (en/vi/mixed), summarizes intent, generates execution plan.
  - Prompt: `backend/app/agents/prompts/supervisor_agent.md`

- **Retriever** (`backend/app/agents/retriever_agent.py`) - COMPLETE
  - Rewrites queries for E5 embeddings, executes FAISS vector search, ranks results using multi-signal scoring (70% similarity, 20% source type, 10% chunk position), deduplicates, extracts evidence.
  - Confidence threshold: 0.30 (testing), top-k: 5.
  - Prompt: `backend/app/agents/prompts/retriever_agent.md`

- **Generator** (`backend/app/agents/generator_agent.py`) - STUB
  - Will draft answers for QA or generate quiz questions based on retrieved evidence.
  - Prompt: `backend/app/agents/prompts/generator_agent.md`

- **Critic** (`backend/app/agents/critic_agent.py`) - STUB
  - Will evaluate drafts for grounding, completeness, and format. Passes or loops back with revision instructions.
  - Prompt: `backend/app/agents/prompts/critic_agent.md`

## Tech Stack

- **Orchestration**: LangGraph, LangChain
- **LLM**: Google Gemini 2.5-flash via `langchain_google_genai` (`backend/app/llm/llm_client.py`)
- **Embeddings**: `intfloat/multilingual-e5-base` (HuggingFace, multilingual)
- **Vector Store**: FAISS (faiss-cpu)
- **Document Processing**: pymupdf4llm, python-docx, Notion API
- **Data Validation**: Pydantic

## Environment Variables

- `GOOGLE_API_KEY` - Gemini API access
- `NOTION_TOKEN` - Notion API authentication
- `NOTION_TEST_PAGE_ID` - Test page for Notion extraction (optional)

## Directory Structure

```
backend/app/
  agents/           # Multi-agent orchestration (LangGraph)
    prompts/        # Agent system prompts (.md files)
  ingestion/        # ETL pipeline (extract, transform, load)
    prompts/        # Ingestion prompt templates
  llm/              # LLM client wrapper (Gemini)
  api/              # REST API endpoints (STUB)
  config/           # Configuration management (STUB)
  retrieval/        # Retrieval abstractions (STUB - logic in agents/)
  storage/          # Vector store abstractions (STUB - logic in ingestion/)
frontend/           # Frontend UI (TBD)
docker/             # Docker configuration
```

## Implementation Status

| Component | Status |
|-----------|--------|
| ETL Pipeline (Extract/Transform/Load) | Complete |
| Supervisor Agent | Complete |
| Retriever Agent | Complete |
| Generator Agent | Stub |
| Critic Agent | Stub |
| API Routes (chat, upload, notion) | Stub |
| Config Module | Stub |
| Frontend | TBD |

## Key Design Decisions

- **Global-Local Context**: Each chunk has the document summary prepended for better retrieval context.
- **Markdown-Native**: All documents stored as Markdown with header hierarchies preserved for semantic chunking.
- **Bilingual**: Language detection at supervisor level; E5 multilingual embeddings for cross-lingual retrieval.
- **Multi-Signal Ranking**: Composite scoring (similarity + source type + chunk position) instead of pure vector similarity.
- **Deterministic State**: Structured `AgentState` TypedDict instead of raw conversation history for agent communication.
