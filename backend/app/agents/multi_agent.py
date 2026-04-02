import json
from typing import TypedDict, List, Dict, Any, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate

# Prevent import errors if running as a script
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.llm.llm_client import LLMClient


# 1. SHARED GRAPH STATE (TYPED DICT)
class AgentState(TypedDict, total=False):
    user_query: str
    task_type: Literal["qa", "quiz", "unknown"]
    language: Literal["en", "vi", "mixed"]
    intent_summary: str
    plan: Dict[str, Any]

    retrieval_focus: Dict[str, Any]
    retrieval_strategy: Dict[str, Any]
    rewritten_query: str
    retrieved_docs: List[Document]
    evidence_summary: List[str]

    draft: str
    critique: Dict[str, Any]
    revision_count: int
    final_answer: str
