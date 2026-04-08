import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.agents.agent_state import AgentState
from app.agents.base_agent import BaseAgent
from app.llm.llm_client import LLMClient

load_dotenv()

# --------------- Configuration ---------------
VECTOR_STORE_DIR = os.path.join(os.getcwd(), "vector_store", "faiss_index")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CONFIDENCE_THRESHOLD = 0.55   # Per-chunk minimum similarity
DOC_RELEVANCE_THRESHOLD = 0.70 # Document-level gate: if the summary chunk of a document
                                # scores below this against the query, ALL chunks from
                                # that document are discarded — prevents vocabulary-overlap
                                # false positives for off-topic queries.
TOP_K = 5
INITIAL_SEARCH_K = 20
DEDUP_THRESHOLD = 0.80

TASK_INSTRUCTIONS = {
    "qa": "Focus on direct information retrieval. Include main topic, key concepts, and specific aspects.",
    "quiz": "Broaden the search to capture diverse examples, contrasts, edge cases, and nuances.",
    "unknown": "Standard information retrieval with emphasis on clarity.",
}


class RetrieverAgent(BaseAgent):
    """
    Retriever Agent — Node 2 of 4 in the KnowGen LangGraph workflow.

    Position in graph: Supervisor → **[RETRIEVER]** → Generator → Critic → END

    Mission:
        Translate the Supervisor's high-level plan into concrete vector-search
        operations, retrieve the most relevant document chunks from the FAISS
        index, and prepare ranked evidence for the Generator to synthesize.
        This node does NOT generate answers — it only retrieves and ranks.

    Responsibilities:
        1. Query Rewriting — Use the LLM to expand the original user query with
           synonyms, related concepts, and abbreviation expansions so the
           embedding search has maximum recall. The rewritten query is adapted
           based on task_type:
             - "qa"  : focused, precise keyword expansion.
             - "quiz" : broader expansion covering contrasts, edge cases, examples.
        2. Vector Search — Run the rewritten query (prefixed with "query: " for
           E5-base compatibility) against the FAISS index, retrieving an initial
           pool of ``INITIAL_SEARCH_K`` (20) candidates.
        3. Multi-Signal Ranking — Score each candidate using a composite of:
             - 70 % cosine similarity (normalized 0-1)
             - 20 % source type priority (pdf > docx > notion)
             - 10 % chunk position quality (chunks with header paths preferred)
           Candidates below ``CONFIDENCE_THRESHOLD`` (0.30) are discarded.
        4. Deduplication — Remove near-duplicate chunks (> 80 % content overlap
           via SequenceMatcher), keeping the higher-scoring variant.
        5. Evidence Extraction — For each of the final ``TOP_K`` (5) chunks,
           extract a one-sentence factual summary (heuristic by default, or
           LLM-based when ``use_llm=True``).

    Reads from AgentState:
        - user_query (str): Original raw query.
        - task_type (str): "qa" | "quiz" | "unknown" (from Supervisor).
        - language (str): "en" | "vi" | "mixed" (from Supervisor).
        - plan (dict): Must contain "context_needed" — key concepts to search for.

    Writes to AgentState:
        - rewritten_query (str): The LLM-expanded search query.
        - retrieved_docs (List[Document]): Top-k LangChain Documents with
          ``similarity_score`` injected into metadata.
        - evidence_summary (List[str]): One-sentence evidence per retrieved doc.
        - retrieval_strategy (dict): Metadata about the retrieval run
          (total_retrieved, top_k, confidence_threshold, ranking_signals, sources).

    Failure behaviour:
        - If the FAISS index is missing or fails to load, search returns empty.
        - If query rewriting fails, the original user_query is used as fallback.
        - If no documents pass the confidence threshold, evidence_summary reports
          "No relevant documents found." so the Generator can handle gracefully.
    """

    name = "retriever"

    QUERY_REWRITE_PROMPT = """
    # Role
    You are a query optimization expert for Vietnamese/English academic documents.
    Your job is to rewrite the user query so that it retrieves the most relevant chunks from a FAISS vector store using E5 multilingual embeddings.

    # Context
    - Original Query: {user_query}
    - Key Concepts Needed: {context_needed}
    - Task Type: {task_type}
    - Task Instruction: {task_instruction}
    - Query Language: {language}

    # Rewriting Rules by Task Type
    - **QA**: Focus on direct information retrieval. Include main topic, key concepts, and specific aspects mentioned.
    - **QUIZ**: Broaden the search to capture diverse examples, contrasts, edge cases, and nuances.

    # Examples

    ## Vietnamese
    - "Chủ nghĩa xã hội là gì?" → "Chủ nghĩa xã hội khoa học định nghĩa khái niệm lý thuyết Marx Engels đặc điểm"
    - "Những đặc điểm chính của CNXH là gì?" → "Đặc điểm chính CNXH khoa học tính chất nguyên tắc cơ bản"

    ## English
    - "What is socialism?" → "socialism scientific socialism theory characteristics definition features Marx"
    - "What are main features?" → "main features characteristics attributes key properties aspects"

    # Instructions
    1. Expand ALL abbreviations (CNXH → Chủ nghĩa Xã hội, etc.)
    2. Add related synonyms and concept variations
    3. Include broader terms and specific aspects
    4. Make the query LONGER with more keywords than the original
    5. Preserve the original language

    # Output
    Return ONLY the expanded query string. No explanations, no labels, no formatting."""

    EVIDENCE_PROMPT = """Extract the main factual claim or key insight from this text in ONE sentence.
    Preserve the original language (English or Vietnamese).
    Focus on definitions, facts, or important examples.
    Output ONLY the sentence, nothing else.

    Text: {content}

    Key Fact:"""

    def __init__(self, vector_store_dir: str | None = None):
        super().__init__()
        self.llm_client = LLMClient()
        self.vector_store_dir = vector_store_dir or str(VECTOR_STORE_DIR)
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        self.logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_vectorstore(self) -> FAISS | None:
        try:
            if not os.path.exists(self.vector_store_dir):
                self.logger.warning(f"Vector store not found: {self.vector_store_dir}")
                return None
            self.logger.info(f"Loading FAISS index from: {self.vector_store_dir}")
            vs = FAISS.load_local(
                self.vector_store_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("FAISS index loaded successfully")
            return vs
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            return None


    # 1. Query rewriting
    def rewrite_query(
        self,
        user_query: str,
        context_needed: str,
        task_type: str,
        language: str,
    ) -> str:
        task_instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["unknown"])
        prompt = self.QUERY_REWRITE_PROMPT.format(
            user_query=user_query,
            context_needed=context_needed,
            task_type=task_type,
            task_instruction=task_instruction,
            language=language,
        )

        try:
            rewritten = self.llm_client.generate_response(prompt)
            rewritten_text = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
            rewritten_query = rewritten_text.strip()

            # Fallback if LLM didn't actually expand
            if len(rewritten_query) <= len(user_query) and context_needed:
                rewritten_query = f"{user_query} {context_needed}"

            self.logger.info(
                f"Query rewritten: '{user_query[:40]}…' ({len(user_query)} chars) "
                f"→ '{rewritten_query[:40]}…' ({len(rewritten_query)} chars)"
            )
            return rewritten_query
        except Exception as e:
            self.logger.warning(f"Query rewriting failed, using original: {e}")
            return user_query

    # 2. Search & rank
    def search_and_rank(
        self,
        query: str,
        initial_k: int = INITIAL_SEARCH_K,
        top_k: int = TOP_K,
    ) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            self.logger.error("Vector store not initialised")
            return []

        query_with_prefix = f"query: {query}"
        try:
            self.logger.info(f"Searching {initial_k} candidates…")
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query_with_prefix, k=initial_k
            )
            self.logger.info(f"Found {len(results)} candidates")

            # Document-level relevance gate — prune documents whose summary
            # chunk scores below DOC_RELEVANCE_THRESHOLD before chunk ranking.
            gated = self._gate_by_doc_summary(results)
            if not gated:
                self.logger.warning(
                    "All documents failed the doc-relevance gate "
                    f"(threshold={DOC_RELEVANCE_THRESHOLD}). Query is off-topic."
                )
                return []

            ranked = self._multi_signal_rank(gated, top_k)
            self.logger.info(f"After ranking & filtering: {len(ranked)} documents")
            return ranked
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def _gate_by_doc_summary(
        self,
        results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Document-level relevance gate.

        For each unique doc_id in the candidate pool, find its summary chunk's
        similarity score.  If the best summary score for a document is below
        DOC_RELEVANCE_THRESHOLD, ALL chunks from that document are removed.

        Rationale: the summary chunk encodes the full document topic.  When a
        query is off-topic, the summary similarity will be low even though
        individual chunks may share generic academic vocabulary with the query
        and pass the per-chunk CONFIDENCE_THRESHOLD.

        Fallback: if a document has no summary chunk in the candidate pool,
        the highest chunk similarity for that document is used as a proxy.
        """
        # Step 1 — collect the best summary score per doc_id
        doc_best_summary: Dict[str, float] = {}
        for doc, score in results:
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id", "__unknown__")
            role   = meta.get("role", "chunk")
            if role == "summary":
                doc_best_summary[doc_id] = max(
                    doc_best_summary.get(doc_id, 0.0), score
                )

        # Step 2 — for docs with no summary chunk in the pool, use max chunk score
        doc_best_chunk: Dict[str, float] = {}
        for doc, score in results:
            meta  = doc.metadata or {}
            doc_id = meta.get("doc_id", "__unknown__")
            doc_best_chunk[doc_id] = max(doc_best_chunk.get(doc_id, 0.0), score)

        # Step 3 — decide which doc_ids pass the gate
        passed_docs: set = set()
        for doc_id in doc_best_chunk:
            relevance = doc_best_summary.get(doc_id, doc_best_chunk[doc_id])
            if relevance >= DOC_RELEVANCE_THRESHOLD:
                passed_docs.add(doc_id)
            else:
                self.logger.info(
                    f"Doc '{doc_id}' pruned by relevance gate "
                    f"(summary_sim={relevance:.4f} < {DOC_RELEVANCE_THRESHOLD})"
                )

        # Step 4 — keep only chunks from passing documents
        gated = [
            (doc, score)
            for doc, score in results
            if (doc.metadata or {}).get("doc_id", "__unknown__") in passed_docs
        ]
        self.logger.info(
            f"Doc-relevance gate: {len(passed_docs)}/{len(doc_best_chunk)} doc(s) passed, "
            f"{len(gated)}/{len(results)} chunks retained."
        )
        return gated

    def _multi_signal_rank(
        self,
        results: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        source_priorities = {"pdf": 1.0, "docx": 0.95, "notion": 0.8, "unknown": 0.5}
        scored_docs: List[Dict[str, Any]] = []

        for doc, similarity_score in results:
            norm_sim = max(0.0, min(1.0, similarity_score))
            if norm_sim < CONFIDENCE_THRESHOLD:
                continue

            metadata = doc.metadata or {}
            source_score = source_priorities.get(metadata.get("source_type", "unknown"), 0.5)
            position_score = 1.0 if metadata.get("header_path") else 0.85

            composite = 0.7 * norm_sim + 0.2 * source_score + 0.1 * position_score
            scored_docs.append({
                "doc": doc,
                "similarity_score": norm_sim,
                "composite_score": composite,
            })

        scored_docs.sort(key=lambda x: x["composite_score"], reverse=True)
        deduped = self._deduplicate(scored_docs)
        return [(d["doc"], d["similarity_score"]) for d in deduped[:top_k]]

    # 3. Deduplication
    def _deduplicate(self, scored_docs: List[Dict]) -> List[Dict]:
        kept: List[Dict] = []
        for candidate in scored_docs:
            is_dup = False
            for i, existing in enumerate(kept):
                overlap = SequenceMatcher(
                    None, candidate["doc"].page_content, existing["doc"].page_content
                ).ratio()
                if overlap > DEDUP_THRESHOLD:
                    if candidate["composite_score"] > existing["composite_score"]:
                        kept[i] = candidate
                    is_dup = True
                    break
            if not is_dup:
                kept.append(candidate)
        return kept

    # 4. Evidence extraction
    def extract_evidence(
        self,
        ranked_docs: List[Tuple[Document, float]],
        use_llm: bool = False,
    ) -> List[str]:
        evidence: List[str] = []
        for doc, _ in ranked_docs:
            try:
                if use_llm:
                    fact = self._extract_with_llm(doc.page_content)
                else:
                    fact = self._extract_heuristic(doc.page_content)
                if fact:
                    evidence.append(fact)
            except Exception as e:
                self.logger.warning(f"Evidence extraction failed: {e}")
        return evidence

    def _extract_with_llm(self, content: str, max_length: int = 200) -> str:
        prompt = self.EVIDENCE_PROMPT.format(content=content[:500])
        try:
            resp = self.llm_client.generate_response(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            return text.strip()[:max_length]
        except Exception as e:
            self.logger.warning(f"LLM evidence extraction failed: {e}")
            return ""

    @staticmethod
    def _extract_heuristic(content: str, max_length: int = 200) -> str:
        sentences = [s.strip() for s in content.replace("\u3002", ".").split(".") if s.strip()]
        return sentences[0][:max_length] if sentences else content[:max_length]

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute the full retrieval pipeline on the current AgentState.

        Steps:
            1. Extract ``user_query``, ``task_type``, ``language``, and
               ``plan.context_needed`` from state.
            2. Rewrite the query via LLM for better embedding recall.
            3. Run FAISS similarity search → multi-signal rank → deduplicate.
            4. Extract one-sentence evidence from each top-k chunk.
            5. Return a dict that LangGraph merges into AgentState:
               ``rewritten_query``, ``retrieved_docs``, ``evidence_summary``,
               ``retrieval_strategy``.

        Returns an empty result set (with diagnostic message) when no documents
        meet the confidence threshold, allowing the Generator to handle the
        "no context found" case.
        """
        user_query = state.get("user_query", "")
        task_type = state.get("task_type", "qa")
        language = state.get("language", "en")
        plan = state.get("plan", {})
        context_needed = plan.get("context_needed", "")

        if not user_query:
            self.logger.error("No user query provided")
            return self._empty_result()

        # 1 — Rewrite query
        rewritten_query = self.rewrite_query(user_query, context_needed, task_type, language)

        # 2 — Search & rank
        ranked_docs = self.search_and_rank(rewritten_query)
        if not ranked_docs:
            self.logger.warning("No documents above confidence threshold")
            return {
                "rewritten_query": rewritten_query,
                "retrieved_docs": [],
                "evidence_summary": ["No relevant documents found."],
                "retrieval_strategy": self._build_strategy(0, []),
            }

        # 3 — Extract evidence
        evidence_summary = self.extract_evidence(ranked_docs, use_llm=False)

        # 4 — Format output
        formatted_docs: List[Document] = []
        sources: set[str] = set()
        for doc, sim_score in ranked_docs:
            meta = doc.metadata or {}
            sources.add(meta.get("source_type", "unknown"))
            formatted_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**meta, "similarity_score": float(sim_score)},
                )
            )

        self.logger.info(
            f"Retrieval complete: {len(formatted_docs)} docs, {len(evidence_summary)} evidence items"
        )
        return {
            "rewritten_query": rewritten_query,
            "retrieved_docs": formatted_docs,
            "evidence_summary": evidence_summary,
            "retrieval_strategy": self._build_strategy(len(formatted_docs), list(sources)),
        }

    # Helpers
    @staticmethod
    def _build_strategy(total: int, sources: List[str]) -> Dict[str, Any]:
        return {
            "total_retrieved": total,
            "top_k": TOP_K,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "ranking_signals": ["similarity", "source_type", "chunk_position"],
            "sources": sources,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "rewritten_query": "",
            "retrieved_docs": [],
            "evidence_summary": [],
            "retrieval_strategy": {},
        }

_agent = RetrieverAgent()


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for the Retriever Agent.

    This is the function registered with ``workflow.add_node("retriever", retriever_node)``.
    It delegates to a module-level ``RetrieverAgent`` singleton so the FAISS index
    and embedding model are loaded once and reused across invocations.

    Input (from AgentState):
        - user_query (str)
        - task_type (str): from Supervisor
        - language (str): from Supervisor
        - plan (dict): from Supervisor — must contain "context_needed"

    Output (merged into AgentState):
        - rewritten_query (str)
        - retrieved_docs (List[Document])
        - evidence_summary (List[str])
        - retrieval_strategy (dict)
    """
    return _agent(state)
