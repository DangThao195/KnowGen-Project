# Multi-Agent Orchestration & Shared State

## Overview
This document defines the shared state and the high-level orchestration logic for the internal KnowGen multi-agent system using **LangGraph**. 

Instead of passing raw conversation histories around, the system relies on a **Deterministic Structured State** designed specifically for RAG workloads. This allows specialized agents (Supervisor, Retriever, Generator, Reviewer) to clearly read inputs and write outputs without corrupting the context.

---

## 1. Global Graph Schema (`AgentState`)

The proposed state tracks the entire lifecycle of a request from understanding to final delivery. This lives globally across the LangGraph workflow.

```python
from typing import TypedDict, List, Dict, Any, Literal
from langchain_core.documents import Document

class AgentState(TypedDict):
    # --- Input & Planning ---
    user_query: str                    # Original raw query from user
    task_type: Literal["qa", "quiz"]  # Categorized intent
    plan: Dict[str, Any]               # Supervisor's breakdown of the execution
    
    # --- Retrieval ---
    retrieval_strategy: Dict[str, Any] # FAISS k-nearest parameters, filters, etc.
    rewritten_query: str               # Optimized query matching document vocabulary
    retrieved_docs: List[Document]     # Found chunks from FAISS
    evidence_summary: List[str]        # (Optional) Pre-synthesis of long chunks
    
    # --- Generation & Reflection ---
    draft: str                         # Initial generated answer
    critique: Dict[str, Any]           # Evaluator's feedback (e.g. {"pass": False, "reason": "..."})
    revision_count: int                # Reflection loop counter (failsafe)
    final_answer: str                  # Output validated for the user
```

---

## 2. Agent Graph Roster

1. **Supervisor Node**: Focuses exclusively on reading `user_query` to classify the `task_type` and generate a `plan`. (Does zero retrieval).
2. **Retriever Node (Agent)**: Uses the `plan` to shape `rewritten_query` and pulls `retrieved_docs` from FAISS using `intfloat/multilingual-e5-base`.
3. **Generator Node (Agent)**: Takes the plan and documents to write the `draft`.
4. **Evaluator Node (Agent)**: Generates a `critique` on the draft. If it fails, increments `revision_count` and passes back to Generator or Retriever. If it passes, updates `final_answer`.

---

## 3. Workflow Diagram (Graph)

```text
               +-----------------+
               |   user_query    |
               +--------+--------+
                        |
                        v
               [ Supervisor Node ]
                 -> output: task_type, plan
                        |
                        v
               [ Retriever Node ]
                 -> output: retrieved_docs
                        |
                        v
               [ Generator Node ]
                 -> output: draft
                        |
                        v
               [ Evaluator Node ]
                 -> output: critique
                        |
                (Fail?) +----+ (Pass?)
                   |             |
                   v             v
            (Loop Back)     [ Output ] -> final_answer
```
