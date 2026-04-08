import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.agents.agent_state import AgentState
from app.agents.supervisor_agent import supervisor_node
from app.agents.retriever_agent import retriever_node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Agent state is imported from agent_state.py to avoid circular imports


# ============================================================================
# 2. WORKFLOW BUILDER FUNCTION
# ============================================================================
def build_knowgen_workflow() -> StateGraph:
    """
    Construct the complete KnowGen multi-agent LangGraph workflow.
    
    Workflow:
        START → SUPERVISOR → RETRIEVER → (GENERATOR → CRITIC)* → END
    
    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    logger.info("Building KnowGen Multi-Agent Workflow...")
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # ========================================================================
    # Add nodes to the graph
    # ========================================================================
    logger.info("Adding agent nodes...")
    
    # Supervisor: Task understanding and planning
    workflow.add_node("supervisor", supervisor_node)
    logger.info("  ✓ Supervisor node added")
    
    # Retriever: Document retrieval and ranking
    workflow.add_node("retriever", retriever_node)
    logger.info("  ✓ Retriever node added")
    
    # Future nodes can be added here:
    # workflow.add_node("generator", generator_agent)
    # workflow.add_node("critic", critic_agent)
    
    # ========================================================================
    # Define graph structure (edges)
    # ========================================================================
    logger.info("Adding edges...")
    
    # Entry point
    workflow.set_entry_point("supervisor")
    logger.info("  ✓ Entry point: supervisor")
    
    # Linear flow for now: supervisor → retriever
    workflow.add_edge("supervisor", "retriever")
    logger.info("  ✓ supervisor → retriever")
    
    # Exit point
    workflow.add_edge("retriever", END)
    logger.info("  ✓ retriever → END")
    
    # Future: Add conditional routing from retriever
    # workflow.add_conditional_edges(
    #     "retriever",
    #     route_to_generator,
    #     {"generate": "generator", "skip": END}
    # )
    
    logger.info("Workflow structure complete")
    return workflow


# ============================================================================
# 3. CONDITIONAL ROUTING FUNCTION (for future use)
# ============================================================================
def route_after_retrieval(state: AgentState) -> str:
    """
    Determine next node after retrieval.
    Future: Route to Generator only if documents were found.
    
    Args:
        state: Current AgentState
    
    Returns:
        str: Next node name ("generator" or "end")
    """
    retrieved_docs = state.get("retrieved_docs", [])
    
    if len(retrieved_docs) > 0:
        logger.info("Documents retrieved successfully → proceeding to generator")
        return "generator"
    else:
        logger.warning("No documents retrieved → skipping generator")
        return "end"


# ============================================================================
# 4. MAIN WORKFLOW EXECUTION FUNCTION
# ============================================================================
def run_knowgen_pipeline(user_query: str) -> Dict[str, Any]:
    """
    Execute the complete KnowGen pipeline for a user query.
    
    Args:
        user_query: User's question or request
    
    Returns:
        Dict[str, Any]: Final state with all agent outputs
    """
    logger.info(f"Starting KnowGen Pipeline for query: {user_query[:50]}...")
    
    # Build workflow
    workflow = build_knowgen_workflow()
    
    # Compile the graph
    try:
        app = workflow.compile()
        logger.info("Workflow compiled successfully")
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")
        return {"error": f"Compilation failed: {str(e)}"}
    
    # Create initial state
    initial_state = {"user_query": user_query}
    
    # Execute the pipeline
    try:
        logger.info("Invoking workflow...")
        final_state = app.invoke(initial_state)
        logger.info("Workflow completed successfully")
        return final_state
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return {"error": f"Execution failed: {str(e)}", "user_query": user_query}


# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================
def print_state_summary(state: Dict[str, Any]) -> None:
    """
    Pretty print a summary of the agent state.
    
    Args:
        state: AgentState dictionary
    """
    print("\n" + "=" * 80)
    print("AGENT STATE SUMMARY")
    print("=" * 80)
    
    print("\n📝 INPUT:")
    print(f"  User Query: {state.get('user_query', 'N/A')[:60]}...")
    
    if "task_type" in state:
        print("\n🧠 SUPERVISOR OUTPUT:")
        print(f"  Task Type: {state.get('task_type')}")
        print(f"  Language: {state.get('language')}")
        print(f"  Intent: {state.get('intent_summary', 'N/A')[:60]}...")
    
    if "rewritten_query" in state:
        print("\n🔍 RETRIEVER OUTPUT:")
        print(f"  Rewritten Query: {state.get('rewritten_query', 'N/A')[:60]}...")
        print(f"  Documents Retrieved: {len(state.get('retrieved_docs', []))}")
        print(f"  Evidence Items: {len(state.get('evidence_summary', []))}")
    
    if "final_answer" in state:
        print("\n✨ FINAL OUTPUT:")
        print(f"  Answer: {state.get('final_answer', 'N/A')[:100]}...")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Demo: Test the complete workflow
    print("\n" + "=" * 80)
    print("KNOWGEN MULTI-AGENT SYSTEM - DEMO")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "What is Marxism?",
        "Hãy giải thích những nguyên tắc cơ bản của chủ nghĩa xã hội khoa học",
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: {query}")
        result = run_knowgen_pipeline(query)
        print_state_summary(result)
    
    print("\n✅ Demo completed!")
