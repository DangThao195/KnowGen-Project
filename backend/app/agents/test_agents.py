"""
Test script for Supervisor Agent & Retriever Agent.

Usage (from backend/):
    python -m app.agents.test_agents
"""

import json
import logging
import os
import sys
from pathlib import Path
import time

# --------------- Path setup ---------------
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # agents -> app -> backend
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv()

# --------------- Logging (console + file) ---------------
LOG_FILE = BACKEND_DIR / "app" / "agents" / "test_agents.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_agents")

# --------------- Configuration ---------------
PDF_PATH = r"C:\Users\huuda\Downloads\a.thuvienvatly.com.2160e.47524.pdf"
VECTOR_STORE_DIR = BACKEND_DIR / "vector_store" / "faiss_index"
JSON_FILE = BACKEND_DIR / "app" / "agents" / "test_agents_output.json"


# ============================================================================
# STEP 1 — Ingest PDF into FAISS via existing ETL pipeline
# ============================================================================
def ingest_pdf(pdf_path: str) -> bool:
    """Run the ETL pipeline from etl.py to ingest the PDF into FAISS."""
    logger.info("=" * 70)
    logger.info("STEP 1: INGESTING PDF INTO FAISS")
    logger.info("=" * 70)

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return False

    from app.ingestion.etl import run_etl_pipeline

    try:
        run_etl_pipeline(file_paths=[pdf_path])
        logger.info(f"  -> FAISS index saved to: {VECTOR_STORE_DIR}")
        return True
    except Exception as e:
        logger.error(f"  -> ETL pipeline failed: {e}")
        return False


# ============================================================================
# STEP 2 — Test Supervisor Agent
# ============================================================================
def test_supervisor(queries: list[str]) -> list[dict]:
    """Run each query through SupervisorAgent, log and return results."""
    logger.info("=" * 70)
    logger.info("STEP 2: TESTING SUPERVISOR AGENT")
    logger.info("=" * 70)

    from app.agents.supervisor_agent import supervisor_node

    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"\n--- Query {i}/{len(queries)}: {query}")

        state = {"user_query": query}
        output = supervisor_node(state)
        time.sleep(15)
        results.append(output)

        logger.info(f"  task_type      : {output.get('task_type')}")
        logger.info(f"  language       : {output.get('language')}")
        logger.info(f"  intent_summary : {output.get('intent_summary')}")
        plan = output.get("plan", {})
        logger.info(f"  context_needed : {plan.get('context_needed', '')}")
        for step in plan.get("steps", []):
            logger.info(f"    step: {step}")

    return results


# ============================================================================
# STEP 3 — Test Retriever Agent
# ============================================================================
def test_retriever(queries: list[str], supervisor_outputs: list[dict]) -> list[dict]:
    """Run each query through RetrieverAgent using supervisor output, log results."""
    logger.info("=" * 70)
    logger.info("STEP 3: TESTING RETRIEVER AGENT")
    logger.info("=" * 70)

    from app.agents.retriever_agent import RetrieverAgent

    retriever = RetrieverAgent(vector_store_dir=str(VECTOR_STORE_DIR))
    results = []

    for i, (query, sup) in enumerate(zip(queries, supervisor_outputs), 1):
        logger.info(f"\n--- Query {i}/{len(queries)}: {query}")

        # Build state from supervisor output
        state = {
            "user_query": query,
            "task_type": sup.get("task_type", "qa"),
            "language": sup.get("language", "en"),
            "intent_summary": sup.get("intent_summary", ""),
            "plan": sup.get("plan", {}),
        }

        output = retriever.run(state)
        results.append(output)

        # --- 1. Rewritten Query (full) ---
        logger.info(f"  [REWRITTEN QUERY]")
        logger.info(f"  {output.get('rewritten_query', '')}")

        # --- 2. Retrieval Strategy (full) ---
        strategy = output.get("retrieval_strategy", {})
        logger.info(f"  [RETRIEVAL STRATEGY]")
        logger.info(f"  total_retrieved    : {strategy.get('total_retrieved')}")
        logger.info(f"  top_k              : {strategy.get('top_k')}")
        logger.info(f"  confidence_threshold: {strategy.get('confidence_threshold')}")
        logger.info(f"  ranking_signals    : {strategy.get('ranking_signals')}")
        logger.info(f"  sources            : {strategy.get('sources')}")

        # --- 3. Retrieved Documents (full content + all metadata) ---
        docs = output.get("retrieved_docs", [])
        logger.info(f"  [RETRIEVED DOCS] — {len(docs)} document(s)")
        for j, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            logger.info(f"  ---- Doc {j}/{len(docs)} ----")
            logger.info(f"  similarity_score : {meta.get('similarity_score', 0):.4f}")
            logger.info(f"  header_path      : {meta.get('header_path', '—')}")
            logger.info(f"  doc_id           : {meta.get('doc_id', '—')}")
            logger.info(f"  role             : {meta.get('role', '—')}")
            logger.info(f"  source_type      : {meta.get('source_type', '—')}")
            logger.info(f"  chunk_index      : {meta.get('chunk_index', '—')}")
            # Full page_content — no truncation
            logger.info(f"  page_content     :")
            for line in doc.page_content.split("\n"):
                logger.info(f"    | {line}")

        # --- 4. Evidence Summary (full) ---
        evidence = output.get("evidence_summary", [])
        logger.info(f"  [EVIDENCE SUMMARY] — {len(evidence)} item(s)")
        for j, ev in enumerate(evidence, 1):
            logger.info(f"  {j}. {ev}")

    return results


# ============================================================================
# STEP 4 — Summary table
# ============================================================================
def print_summary(queries, sup_results, ret_results):
    logger.info("=" * 70)
    logger.info("STEP 4: SUMMARY")
    logger.info("=" * 70)

    for i, (q, s, r) in enumerate(zip(queries, sup_results, ret_results), 1):
        n_docs = len(r.get("retrieved_docs", []))
        scores = [
            d.metadata.get("similarity_score", 0)
            for d in r.get("retrieved_docs", [])
            if d.metadata
        ]
        avg_score = sum(scores) / len(scores) if scores else 0

        logger.info(
            f"  Q{i} | type={s.get('task_type'):7s} | lang={s.get('language'):5s} "
            f"| docs={n_docs} | avg_sim={avg_score:.3f} | {q[:50]}"
        )


# ============================================================================
# STEP 5 — Dump full outputs to JSON
# ============================================================================
def dump_results(queries, sup_results, ret_results):
    """Write all agent outputs to a JSON file for inspection."""
    logger.info("=" * 70)
    logger.info("STEP 5: WRITING FULL OUTPUTS TO JSON")
    logger.info("=" * 70)

    output = []
    for i, (q, s, r) in enumerate(zip(queries, sup_results, ret_results), 1):
        # Convert Document objects to serialisable dicts
        docs_serialised = []
        for doc in r.get("retrieved_docs", []):
            docs_serialised.append({
                "page_content": doc.page_content[:500],
                "metadata": doc.metadata,
            })

        output.append({
            "query": q,
            "supervisor": s,
            "retriever": {
                "rewritten_query": r.get("rewritten_query", ""),
                "evidence_summary": r.get("evidence_summary", []),
                "retrieval_strategy": r.get("retrieval_strategy", {}),
                "retrieved_docs": docs_serialised,
            },
        })

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"  -> Saved {len(output)} query results to {JSON_FILE}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # --- Test queries (mix of Vietnamese physics + CNXHKH + quiz) ---
    test_queries = [
        # Vietnamese physics (from the PDF)
        "Sóng cơ học là gì? Phân loại sóng ngang và sóng dọc.",
        "Công thức tính bước sóng và tần số là gì?",
        # Vietnamese CNXHKH (from existing FAISS content)
        "Những đặc điểm chính của CNXH khoa học là gì?",
        # Quiz request
        "Tạo 5 câu hỏi trắc nghiệm về sóng cơ học.",
    ]

    # Step 1 — Ingest the physics PDF
    ingest_pdf(PDF_PATH)

    # Step 2 — Supervisor
    sup_results = test_supervisor(test_queries)

    # Step 3 — Retriever
    ret_results = test_retriever(test_queries, sup_results)

    # Step 4 — Summary
    print_summary(test_queries, sup_results, ret_results)

    # Step 5 — Dump full JSON outputs to file
    dump_results(test_queries, sup_results, ret_results)

    logger.info("=" * 70)
    logger.info("ALL TESTS COMPLETE")
    logger.info(f"Log file  : {LOG_FILE}")
    logger.info(f"JSON file : {JSON_FILE}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
