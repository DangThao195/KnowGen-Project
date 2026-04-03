"""
AgentState Definition - Shared State for Multi-Agent Workflow

This module defines the AgentState TypedDict that is used by all agents
in the KnowGen multi-agent system. By separating this definition into its own
module, we avoid circular import issues when different agents import this state.
"""

from typing import TypedDict, List, Dict, Any, Literal
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    Shared state dictionary for all agents in the LangGraph workflow.
    
    Supervisor Agent fields:
    - user_query: Original user input
    - task_type: Classified task type ("qa", "quiz", "unknown")
    - language: Detected language ("en", "vi", "mixed")
    - intent_summary: High-level description of user intent
    - plan: Execution plan with steps and context
    
    Retriever Agent fields:
    - rewritten_query: Optimized query for vector search
    - retrieved_docs: List of relevant documents from FAISS
    - evidence_summary: Key facts extracted from documents
    - retrieval_strategy: Metadata about retrieval process
    
    Future Agent fields (Generator, Critic):
    - draft: Initial generated answer
    - critique: Review and scoring of draft
    - revision_count: Number of refinement iterations
    - final_answer: Final output after all processing
    """
    # Supervisor inputs
    user_query: str
    
    # Supervisor outputs
    task_type: Literal["qa", "quiz", "unknown"]
    language: Literal["en", "vi", "mixed"]
    intent_summary: str
    plan: Dict[str, Any]
    
    # Retriever inputs (from Supervisor)
    retrieval_focus: Dict[str, Any]
    
    # Retriever outputs
    retrieval_strategy: Dict[str, Any]
    rewritten_query: str
    retrieved_docs: List[Document]
    evidence_summary: List[str]
    
    # Future: Generator Agent outputs
    draft: str
    
    # Future: Critic Agent outputs
    critique: Dict[str, Any]
    revision_count: int
    
    # Final output
    final_answer: str
