import os
import re
import json
from typing import Dict, Any

from langchain_core.messages import AIMessage

try:
    from app.agents.multi_agent import AgentState
except ImportError:
    pass

# Import the LLM client
from app.llm.llm_client import LLMClient

def supervisor_node(state: dict) -> Dict[str, Any]:
    """
    Analyze the user's query and generate a structured execution plan for the multi-agent system.

    This node acts as the Supervisor (Coordinator) agent in the KnowGen architecture. It is responsible
    for interpreting the user's request, classifying the task type, and producing a high-level plan
    that downstream agents (retriever, generator, critic) will follow.

    Responsibilities:
    - Detect the input language (English, Vietnamese, or mixed)
    - Infer the user's intent at a semantic level
    - Classify the request into one of the supported task types:
        - "qa": question answering / information request
        - "quiz": quiz or assessment generation
        - "unknown": ambiguous or unsupported request
    - Identify key concepts and context needed for retrieval
    - Generate a step-by-step execution plan for downstream agents

    Input:
        state (dict):
            A shared state dictionary containing at least:
            - "user_query" (str): The raw user input

    Output:
        Dict[str, Any]: Updated state fields including:
            - "task_type" (str): Classified task type ("qa", "quiz", or "unknown")
            - "language" (str): Detected language ("en", "vi", or "mixed")
            - "intent_summary" (str): High-level description of user intent
            - "plan" (dict):
                - "steps" (List[str]): Ordered execution steps for downstream agents
                - "context_needed" (str): Topics/keywords required for retrieval
    """
    user_query = state.get("user_query", "")
    system_prompt = """
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
            

    prompt = f"{system_prompt}\n\nUser Query: {user_query}\n\nReturn strictly valid JSON only."
    
    llm_client = LLMClient()
    response = llm_client.generate_response(prompt)
    

    response_text = response.content if hasattr(response, "content") else str(response)
    
    try:

        json_match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
            
        parsed_result = json.loads(response_text.strip())
        
        return {
            "task_type": parsed_result.get("task_type", "unknown"),
            "language": parsed_result.get("language", "en"),
            "intent_summary": parsed_result.get("intent_summary", "Unknown intent"),
            "plan": parsed_result.get("plan", {"steps": [], "context_needed": ""})
        }
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from supervisor response: {e}\nRaw Output: {response_text}")
        return {
            "task_type": "unknown",
            "language": "en",
            "intent_summary": "Failed to parse supervisor response.",
            "plan": {"steps": ["Clarify user intent"], "context_needed": "None"}
        }
