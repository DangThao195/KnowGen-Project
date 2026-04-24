import json
import re
from typing import Any, Dict  # Dict/Any still used by _parse_json

from app.agents.agent_state import AgentState
from app.agents.base_agent import BaseAgent
from app.llm.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent — Node 1 of 4 in the KnowGen LangGraph workflow.

    Position in graph: START → **[SUPERVISOR]** → Retriever → Generator → Critic → END

    Mission:
        Act as the entry-point coordinator that receives the raw user query and
        produces a structured execution plan BEFORE any retrieval or generation
        happens. This node does NOT retrieve documents or generate answers — it
        only analyzes, classifies, and plans.

    Responsibilities:
        1. Language Detection — Determine whether the query is in English ("en"),
           Vietnamese ("vi"), or a mix of both ("mixed").
        2. Intent Classification — Understand what the user is asking for and
           classify it into a task type:
             - "qa"   : a factual question or information request about the documents.
             - "quiz" : a request to generate practice questions, MCQs, or exercises.
             - "unknown" : ambiguous or unsupported request requiring clarification.
        3. Intent Summarization — Produce a short natural-language summary of the
           user's intent so downstream agents can quickly grasp the goal.
        4. Execution Planning — Draft a step-by-step plan including:
             - Ordered steps for the Retriever, Generator, and Critic to follow.
             - Key concepts / topics / keywords ("context_needed") that the
               Retriever must search for in the FAISS vector store.

    Reads from AgentState:
        - user_query (str): The raw user input.

    Writes to AgentState:
        - task_type (str): "qa" | "quiz" | "unknown"
        - language (str): "en" | "vi" | "mixed"
        - intent_summary (str): One-line description of the user's intent.
        - plan (dict): {"steps": List[str], "context_needed": str}

    Failure behaviour:
        If the LLM response cannot be parsed as valid JSON, the node returns a
        safe fallback with task_type="unknown" so the pipeline can still proceed
        or gracefully terminate.
    """

    name = "supervisor"

    SYSTEM_PROMPT = """
        # Role
        You are the Chief Coordinator (Supervisor) Agent of the KnowGen multi-agent system. You are fully bilingual and can natively understand both English and Vietnamese queries.

        # Context
        The system contains specialized sub-agents that handle document retrieval, question answering (QA), and quiz generation based on a local knowledge base. Before any sub-agent can act, you must diagnose the user's request and devise an execution plan.

        # Task
        For each user query, do the following internally:
        1. Detect the query language.
        2. Infer the user's intent.
        3. Classify the request into a task type.
        4. Identify the core concepts, topics, and retrieval keywords.
        5. Draft a step-by-step plan for downstream agents.

        # Task Types
        - qa: The user is asking a specific question or requesting information about the documents.
        - quiz: The user wants multiple-choice questions, flashcards, or an exercise to test their knowledge.
        - unknown: The request is ambiguous, unsupported, or lacks enough detail.

        # Output
        Return strictly valid JSON only, using this schema:
        {
        "task_type": "qa | quiz | unknown",
        "language": "en | vi | mixed",
        "intent_summary": "Short summary of the user's request",
        "plan": {
            "steps": ["Step 1: ...", "Step 2: ..."],
            "context_needed": "Specific topics or keywords the retriever must find"
        }
        }

        # Constraints
        - Think step by step internally, but do not reveal internal reasoning.
        - Do not output markdown.
        - Do not output any text outside the JSON.
        - If the request is ambiguous, use task_type = "unknown" and set the plan to clarify the missing information.
    """

    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the Supervisor logic on the current AgentState.

        Steps:
            1. Read ``user_query`` from state.
            2. Send the query + SYSTEM_PROMPT to the LLM requesting a JSON
               classification response.
            3. Parse the JSON (stripping markdown fences if the LLM wraps them).
            4. Return an AgentState partial that LangGraph merges into the shared state:
               ``task_type``, ``language``, ``intent_summary``, ``plan``.

        On parse failure, returns ``_fallback_state()`` so the pipeline does not crash.
        """
        user_query = state.get("user_query", "")

        prompt = f"{self.SYSTEM_PROMPT}\n\nUser Query: {user_query}\n\nReturn strictly valid JSON only."
        response = self.llm_client.generate_response(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        parsed = self._parse_json(response_text)
        if parsed:
            return AgentState(
                task_type=parsed.get("task_type", "unknown"),
                language=parsed.get("language", "en"),
                intent_summary=parsed.get("intent_summary", "Unknown intent"),
                plan=parsed.get("plan", {"steps": [], "context_needed": ""}),
            )

        self.logger.error(f"Failed to parse supervisor JSON. Raw: {response_text}")
        return self._fallback_state()

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any] | None:
        """Extract JSON from text, stripping markdown fences if present."""
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fallback_state() -> AgentState:
        return AgentState(
            task_type="unknown",
            language="en",
            intent_summary="Failed to parse supervisor response.",
            plan={"steps": ["Clarify user intent"], "context_needed": ""},
        )


_agent = SupervisorAgent()


def supervisor_node(state: AgentState) -> AgentState:
    """
    LangGraph node wrapper for the Supervisor Agent.

    This is the function registered with ``workflow.add_node("supervisor", supervisor_node)``.
    It delegates to a module-level ``SupervisorAgent`` singleton so the LLM client
    is initialized once and reused across invocations.

    Input (from AgentState):
        - user_query (str)

    Output (partial AgentState merged by LangGraph):
        - task_type (str): "qa" | "quiz" | "unknown"
        - language (str): "en" | "vi" | "mixed"
        - intent_summary (str)
        - plan (dict): {"steps": [...], "context_needed": "..."}
    """
    return _agent(state)
