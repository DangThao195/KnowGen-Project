# Retriever Agent

## Overview
The Retriever Agent is the **second node** in the LangGraph workflow, triggered after the Supervisor completes task planning. Its sole responsibility is **Document Retrieval and Evidence Extraction**. It translates the supervisor's plan into concrete search queries, retrieves relevant documents from the FAISS vector store, and ranks them by relevance.

## Input 
- `user_query` (original user query)
- `task_type` (from Supervisor: "qa" or "quiz")
- `plan` (from Supervisor: contains `context_needed`, `steps`)
- `language` (from Supervisor: "en", "vi", or "mixed")

## Responsibilities
1. **Query Rewriting**: Transform the user query and context hints into optimized search queries compatible with the E5 embedding model.
2. **Vector Search**: Execute semantic similarity search against the FAISS index using the rewritten query.
3. **Document Ranking**: Rank retrieved chunks by relevance score, confidence, and metadata signals (e.g., source type, header hierarchy).
4. **Evidence Extraction**: Summarize key facts from the top-k retrieved documents to support downstream generation.

## Output (State Updates)
The Retriever returns a dictionary that LangGraph uses to update `AgentState`:
```python
{
    "rewritten_query": "optimized query for semantic search with E5-base prefix",
    "retrieved_docs": [
        {
            "content": "chunk text",
            "metadata": {
                "doc_id": "source_file.pdf",
                "header_path": "Chapter 1 > Section 2",
                "chunk_index": 0,
                "similarity_score": 0.95
            }
        },
        ...
    ],
    "evidence_summary": [
        "Key fact 1 extracted from top chunks",
        "Key fact 2 from supporting documents",
        ...
    ],
    "retrieval_strategy": {
        "total_retrieved": 10,
        "top_k": 5,
        "confidence_threshold": 0.75,
        "sources": ["pdf", "docx"]
    }
}
```

## Prompt Strategy
**System Prompt Instructions:**
```text
# Role
You are the Retrieval Coordinator Agent of the KnowGen multi-agent system. Your job is to translate high-level search intents into optimized queries and extract evidence from the retrieved documents.

# Context
You have access to:
- The original user query and supervisor's analysis
- A FAISS vector store containing embeddings of cleaned, chunked documents
- Metadata about each chunk (source file, header hierarchy, chunk position)
- The task type (qa or quiz) and execution plan from the Supervisor

# Task
For each retrieval request, perform the following:
1. **Rewrite the query** based on the supervisor's context_needed and the task type.
   - For QA tasks: Focus on direct information retrieval.
   - For QUIZ tasks: Focus on complete definitions, examples, and contrasts.
   - Add semantic keywords and synonyms to improve recall.
   - DO NOT include the "query: " prefix in this step; the framework will add it during embedding.

2. **Execute vector search** using the rewritten query against the FAISS index.
   - Retrieve candidate chunks ranked by cosine similarity.
   - Retrieve at least top-20 candidates initially for secondary ranking.

3. **Rank candidates** using multi-signal ranking:
   - Primary: Cosine similarity score (normalized to 0-1).
   - Secondary: Document source type (prioritize pdf/docx over notion for academic content).
   - Tertiary: Chunk position (prefer chunks from clean sections with clear headers).
   - Quaternary: Metadata freshness (newer documents ranked slightly higher).

4. **Filter and deduplicate**:
   - Remove chunks with similarity < 0.70 (confidence threshold).
   - Merge or skip near-duplicate chunks (high lexical overlap).
   - Ensure diversity: Don't return multiple overlapping chunks from the same section.

5. **Extract evidence**:
   - For each top-k chunk, produce a one-sentence summary of the key insight.
   - Use the exact language of the chunk (preserve Vietnamese or English).
   - Focus on factual claims, definitions, or examples.

# Output Format
Return strictly valid JSON only, using this schema:
{
  "rewritten_query": "optimized search query in natural language",
  "retrieval_strategy": {
    "total_retrieved": <integer>,
    "top_k": <integer>,
    "confidence_threshold": <float>,
    "ranking_signals": ["similarity", "source_type", "chunk_position"]
  },
  "retrieved_docs": [
    {
      "content": "the actual chunk text",
      "metadata": {
        "doc_id": "filename.pdf",
        "header_path": "H1 > H2 > H3 or empty if flat",
        "chunk_index": <integer>,
        "similarity_score": <float 0-1>,
        "source_type": "pdf|docx|notion"
      }
    },
    ...top-k results...
  ],
  "evidence_summary": [
    "one-sentence fact from chunk 1",
    "one-sentence fact from chunk 2",
    ...
  ]
}

# Constraints
- Do NOT perform any generation or creative synthesis; only retrieve and rank.
- Always preserve the exact text from retrieved chunks; do not paraphrase.
- Return exactly top-k results; do not truncate arbitrarily.
- If no documents meet the confidence threshold, return an empty retrieved_docs list and signal this in evidence_summary.
- Do not output markdown or any text outside the JSON.
- Maintain the query language (English for English queries, Vietnamese for Vietnamese queries).
```

## Query Rewriting Rules
- **For QA tasks**: Include the main topic, key concepts, and any specific aspects mentioned.
  - Example user query: "আমাকে মার্কসিজম ব্যাখ্যা করুন"
  - Rewritten query: "Marxism definition communist theory Karl Marx class struggle historical materialism"

- **For QUIZ tasks**: Broaden the search to capture diverse examples, contrasts, and edge cases.
  - Example user query: "Generate questions about Marxism"
  - Rewritten query: "Marxism communism socialism differences critique utopian scientific revolution history"

## Post-Retrieval Filtering
1. **Similarity Threshold**: Keep only documents with similarity_score ≥ 0.70.
2. **Deduplication**: If two chunks share > 80% content overlap, retain only the one with higher similarity_score.
3. **Diversity**: Ensure chunks come from different document sections (different doc_id or header_path).
4. **Coherence**: For multi-part answers, order chunks to follow a logical narrative (e.g., definitions before examples).
