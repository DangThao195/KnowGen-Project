import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Import local modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.llm.llm_client import LLMClient
from app.agents.multi_agent import AgentState

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VECTOR_STORE_DIR = Path(os.getcwd()) / "vector_store" / "faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CONFIDENCE_THRESHOLD = 0.70
TOP_K = 5
INITIAL_SEARCH_K = 20
DEDUP_THRESHOLD = 0.80  # 80% overlap = duplicate


class RetrieverAgent:
    """
    Retrieves and ranks documents from FAISS vector store.
    Handles query rewriting, semantic search, ranking, deduplication, and evidence extraction.
    """
    
    def __init__(self, vector_store_dir: str = None):
        """
        Initialize the Retriever Agent.
        
        Args:
            vector_store_dir: Path to FAISS index directory. Defaults to VECTOR_STORE_DIR.
        """
        self.vector_store_dir = vector_store_dir or str(VECTOR_STORE_DIR)
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.llm_client = LLMClient()
        logger.info("RetrieverAgent initialized successfully")
    
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load the E5 embedding model."""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def _load_vectorstore(self) -> Optional[FAISS]:
        """Load FAISS index from disk."""
        try:
            if not os.path.exists(self.vector_store_dir):
                logger.warning(f"Vector store directory not found: {self.vector_store_dir}")
                return None
            
            logger.info(f"Loading FAISS index from: {self.vector_store_dir}")
            vectorstore = FAISS.load_local(self.vector_store_dir, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    
    def rewrite_query(self, 
                     user_query: str, 
                     context_needed: str, 
                     task_type: str,
                     language: str) -> str:
        """
        Rewrite the user query using LLM to optimize for vector search.
        Includes semantic keywords, synonyms, and context-specific terms.
        
        Args:
            user_query: Original user query
            context_needed: Key concepts from supervisor's plan
            task_type: "qa" or "quiz"
            language: "en", "vi", or "mixed"
        
        Returns:
            Optimized search query string
        """
        task_instruction = {
            "qa": "Focus on direct information retrieval. Include main topic, key concepts, and specific aspects.",
            "quiz": "Broaden the search to capture diverse examples, contrasts, edge cases, and nuances.",
            "unknown": "Standard information retrieval with emphasis on clarity."
        }.get(task_type, "Standard retrieval")
        
        prompt = f"""You are a query optimization expert. Rewrite the following user query into an optimized search query for semantic vector search.

Original Query: {user_query}
Context Needed: {context_needed}
Task Type: {task_type} - {task_instruction}
Query Language: {language}

Optimization Rules:
1. Expand abbreviations and implicit concepts
2. Include both the main topic and related keywords (synonyms, broader terms, specific aspects)
3. Use natural language without special operators
4. Preserve the query language (English for English queries, Vietnamese for Vietnamese queries)
5. Output ONLY the rewritten query, no explanation

Optimized Query:"""
        
        try:
            rewritten = self.llm_client.generate_response(prompt)
            rewritten_query = rewritten.strip()
            logger.info(f"Query rewritten: {user_query[:50]}... → {rewritten_query[:50]}...")
            return rewritten_query
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            return user_query
    
    def search_and_rank(self, 
                       query: str, 
                       initial_k: int = INITIAL_SEARCH_K,
                       top_k: int = TOP_K) -> List[Tuple[Document, float]]:
        """
        Execute vector search and rank candidates by multiple signals.
        
        Args:
            query: Search query (will be prefixed with "query: " for E5-base)
            initial_k: Number of initial candidates to retrieve
            top_k: Final number of results to return
        
        Returns:
            List of (Document, similarity_score) tuples ranked by composite score
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        # Add E5-base query prefix
        query_with_prefix = f"query: {query}"
        
        try:
            # Search for candidates
            logger.info(f"Searching for {initial_k} candidates with query: {query}")
            # FAISS similarity_search_with_scores returns (Document, score) tuples
            results = self.vectorstore.similarity_search_with_scores(
                query_with_prefix, 
                k=initial_k
            )
            logger.info(f"Found {len(results)} candidates")
            
            # Rank using multi-signal approach
            ranked = self._multi_signal_rank(results, top_k)
            logger.info(f"After ranking and filtering: {len(ranked)} documents retained")
            
            return ranked
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _multi_signal_rank(self, 
                          results: List[Tuple[Document, float]], 
                          top_k: int) -> List[Tuple[Document, float]]:
        """
        Apply multi-signal ranking to candidates.
        Signals: similarity (primary) → source_type (secondary) → chunk_position (tertiary).
        Also applies confidence threshold and deduplication.
        
        Args:
            results: List of (Document, similarity_score) from FAISS
            top_k: Number of final results
        
        Returns:
            Filtered and ranked list of (Document, similarity_score)
        """
        scored_docs = []
        
        for doc, similarity_score in results:
            # Normalize similarity to 0-1 range (FAISS returns distances, closer = lower score sometimes)
            # Assuming FAISS already returns normalized similarity in [0, 1]
            norm_similarity = max(0.0, min(1.0, similarity_score))
            
            # Apply confidence threshold
            if norm_similarity < CONFIDENCE_THRESHOLD:
                continue
            
            # Extract metadata signals
            metadata = doc.metadata or {}
            source_type = metadata.get("source_type", "unknown")
            chunk_index = metadata.get("chunk_index", 999)
            header_path = metadata.get("header_path", "")
            
            # Secondary score: source type priority
            # Prioritize pdf/docx over notion for academic content
            source_priorities = {"pdf": 1.0, "docx": 0.95, "notion": 0.8, "unknown": 0.5}
            source_score = source_priorities.get(source_type, 0.5)
            
            # Tertiary score: chunk position (prefer chunks from clear sections)
            # Chunks with header_path get slight boost
            position_score = 1.0 if header_path else 0.85
            
            # Composite score
            composite_score = (
                0.7 * norm_similarity +      # 70% weight on similarity
                0.2 * source_score +          # 20% weight on source type
                0.1 * position_score          # 10% weight on position/headers
            )
            
            scored_docs.append({
                "doc": doc,
                "similarity_score": norm_similarity,
                "composite_score": composite_score,
                "source_type": source_type
            })
        
        # Sort by composite score (descending)
        scored_docs.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Deduplication: remove near-duplicates
        deduplicated = self._deduplicate_chunks(scored_docs)
        
        # Return top-k as (Document, similarity_score)
        final_results = []
        for item in deduplicated[:top_k]:
            final_results.append((item["doc"], item["similarity_score"]))
        
        return final_results
    
    def _deduplicate_chunks(self, scored_docs: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate chunks based on content overlap.
        If two chunks overlap > DEDUP_THRESHOLD (80%), keep only the higher-scoring one.
        Also ensure diversity: prefer chunks from different sections.
        
        Args:
            scored_docs: List of scored document dictionaries
        
        Returns:
            Deduplicated list of scored documents
        """
        deduplicated = []
        
        for candidate in scored_docs:
            is_duplicate = False
            
            for kept in deduplicated:
                # Check content overlap
                overlap_ratio = self._calculate_overlap(
                    candidate["doc"].page_content,
                    kept["doc"].page_content
                )
                
                if overlap_ratio > DEDUP_THRESHOLD:
                    # Found a near-duplicate; keep the higher-scoring one
                    if candidate["composite_score"] > kept["composite_score"]:
                        deduplicated.remove(kept)
                        deduplicated.append(candidate)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(candidate)
        
        return deduplicated
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate content overlap between two texts using SequenceMatcher.
        Returns a ratio in range [0, 1].
        """
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def extract_evidence(self, 
                        ranked_docs: List[Tuple[Document, float]],
                        llm_extraction: bool = False) -> List[str]:
        """
        Extract one-sentence evidence summaries from top ranked documents.
        
        Args:
            ranked_docs: List of (Document, similarity_score) tuples
            llm_extraction: If True, use LLM to extract; else use heuristics
        
        Returns:
            List of evidence strings (one per document)
        """
        evidence_list = []
        
        for doc, score in ranked_docs:
            try:
                if llm_extraction:
                    # Use LLM to summarize each chunk to one sentence
                    evidence = self._extract_with_llm(doc.page_content)
                else:
                    # Use heuristic: first meaningful sentence
                    evidence = self._extract_heuristic(doc.page_content)
                
                if evidence:
                    evidence_list.append(evidence)
            except Exception as e:
                logger.warning(f"Evidence extraction failed for doc: {e}")
                continue
        
        return evidence_list
    
    def _extract_with_llm(self, content: str, max_length: int = 200) -> str:
        """Extract evidence using LLM - one sentence summary."""
        prompt = f"""Extract the main factual claim or key insight from this text in ONE sentence. 
        Preserve the original language (English or Vietnamese).
        Focus on definitions, facts, or important examples.
        Output ONLY the sentence, nothing else.

Text: {content[:500]}

Key Fact:"""
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response.strip()[:max_length]
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return ""
    
    def _extract_heuristic(self, content: str, max_length: int = 200) -> str:
        """Extract evidence using heuristics - first meaningful sentence."""
        # Split by sentence delimiters
        sentences = [s.strip() for s in content.replace("。", ".").split(".") if s.strip()]
        
        if sentences:
            # Return first sentence, truncated to max_length
            return sentences[0][:max_length]
        
        return content[:max_length]
    
    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Main retriever execution function.
        Called by LangGraph to update AgentState with retrieved documents and evidence.
        
        Args:
            state: Current AgentState from LangGraph
        
        Returns:
            Dictionary with keys to update AgentState:
            - rewritten_query
            - retrieved_docs
            - retrieval_strategy
            - evidence_summary
        """
        logger.info("=== RETRIEVER AGENT RUNNING ===")
        
        # Extract inputs from state
        user_query = state.get("user_query", "")
        task_type = state.get("task_type", "qa")
        language = state.get("language", "en")
        plan = state.get("plan", {})
        context_needed = plan.get("context_needed", "")
        
        if not user_query:
            logger.error("No user query provided")
            return {
                "rewritten_query": "",
                "retrieved_docs": [],
                "evidence_summary": [],
                "retrieval_strategy": {}
            }
        
        # Step 1: Rewrite query
        rewritten_query = self.rewrite_query(user_query, context_needed, task_type, language)
        
        # Step 2: Search and rank
        ranked_docs = self.search_and_rank(rewritten_query, initial_k=INITIAL_SEARCH_K, top_k=TOP_K)
        
        if not ranked_docs:
            logger.warning("No documents retrieved with sufficient confidence")
            return {
                "rewritten_query": rewritten_query,
                "retrieved_docs": [],
                "evidence_summary": ["No relevant documents found."],
                "retrieval_strategy": {
                    "total_retrieved": 0,
                    "top_k": TOP_K,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "sources": []
                }
            }
        
        # Step 3: Extract evidence
        evidence_summary = self.extract_evidence(ranked_docs, llm_extraction=False)
        
        # Step 4: Format retrieved_docs for output
        retrieved_docs_formatted = []
        sources_set = set()
        
        for doc, similarity_score in ranked_docs:
            metadata = doc.metadata or {}
            sources_set.add(metadata.get("source_type", "unknown"))
            
            retrieved_docs_formatted.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **metadata,
                        "similarity_score": float(similarity_score)
                    }
                )
            )
        
        # Step 5: Build retrieval strategy metadata
        retrieval_strategy = {
            "total_retrieved": len(ranked_docs),
            "top_k": TOP_K,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "ranking_signals": ["similarity", "source_type", "chunk_position"],
            "sources": list(sources_set)
        }
        
        logger.info(f"Retrieval complete: {len(ranked_docs)} documents, {len(evidence_summary)} evidence items")
        
        # Return state updates
        return {
            "rewritten_query": rewritten_query,
            "retrieved_docs": retrieved_docs_formatted,
            "evidence_summary": evidence_summary,
            "retrieval_strategy": retrieval_strategy
        }


# Standalone function for LangGraph integration
def retriever_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node function that wraps the RetrieverAgent.
    Called by the LangGraph workflow.
    """
    try:
        agent = RetrieverAgent()
        result = agent.run(state)
        return result
    except Exception as e:
        logger.error(f"Retriever agent failed: {e}", exc_info=True)
        return {
            "rewritten_query": state.get("user_query", ""),
            "retrieved_docs": [],
            "evidence_summary": [f"Retrieval failed: {str(e)}"],
            "retrieval_strategy": {}
        }


if __name__ == "__main__":
    # Demo: Test the retriever agent
    logger.info("Starting Retriever Agent Demo...")
    
    # Create sample state
    sample_state = AgentState(
        user_query="What is Marxism?",
        task_type="qa",
        language="en",
        intent_summary="User asking for definition of Marxism",
        plan={
            "steps": ["Retrieve Marxism docs", "Extract definitions", "Compile answer"],
            "context_needed": "Marxism, communist theory, Karl Marx"
        }
    )
    
    # Run retriever
    agent = RetrieverAgent()
    result = agent.run(sample_state)
    
    # Display results
    print("\n" + "="*80)
    print("RETRIEVER AGENT RESULTS")
    print("="*80)
    print(f"\nRewritten Query: {result.get('rewritten_query')}")
    print(f"\nDocuments Retrieved: {len(result.get('retrieved_docs', []))}")
    
    if result.get('retrieved_docs'):
        print("\nTop Documents:")
        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
            print(f"\n[{i}] Score: {doc.metadata.get('similarity_score', 'N/A'):.2f}")
            print(f"    Header: {doc.metadata.get('header_path', 'N/A')}")
            print(f"    Preview: {doc.page_content[:100]}...")
    
    print(f"\nEvidence Summary:")
    for i, evidence in enumerate(result.get('evidence_summary', []), 1):
        print(f"  {i}. {evidence}")
    
    print(f"\nRetrieval Strategy: {json.dumps(result.get('retrieval_strategy', {}), indent=2)}")
