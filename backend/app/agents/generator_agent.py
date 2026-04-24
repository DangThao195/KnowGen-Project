import json
import re
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.agents.agent_state import AgentState
from app.agents.base_agent import BaseAgent
from app.llm.llm_client import LLMClient


# --------------- Configuration ---------------
DEFAULT_NUM_QUESTIONS = 5
TARGET_DIFFICULTY_MIX = {"easy": 0.30, "medium": 0.40, "hard": 0.30}
INSUFFICIENT_CONTEXT_MESSAGES = {
    "en": "I could not find enough relevant information in the knowledge base to answer this question reliably. Please try rephrasing the question or provide more context.",
    "vi": "Tôi không tìm thấy đủ thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi này một cách đáng tin cậy. Vui lòng diễn đạt lại câu hỏi hoặc cung cấp thêm ngữ cảnh.",
    "mixed": "I could not find enough relevant information in the knowledge base to answer this question reliably. / Tôi không tìm thấy đủ thông tin liên quan để trả lời.",
}
CLARIFICATION_MESSAGES = {
    "en": "I could not confidently determine whether you are asking a question (QA) or requesting a quiz. Please rephrase your request — for example, 'Explain ...' for a question, or 'Generate 5 quiz questions about ...' for a quiz.",
    "vi": "Tôi chưa xác định được bạn đang muốn hỏi (QA) hay yêu cầu tạo bộ câu hỏi (quiz). Vui lòng diễn đạt lại — ví dụ: 'Giải thích ...' cho câu hỏi, hoặc 'Tạo 5 câu trắc nghiệm về ...' cho bài quiz.",
    "mixed": "Please clarify whether you want a question answered or a quiz generated. / Vui lòng làm rõ bạn muốn hỏi hay muốn tạo quiz.",
}


class GeneratorAgent(BaseAgent):
    """
    Generator Agent — Node 3 of 4 in the KnowGen LangGraph workflow.

    Position in graph: Supervisor → Retriever → **[GENERATOR]** → Critic → END

    Mission:
        Synthesize grounded content from the Retriever's ranked evidence. Produces
        either a direct answer (``task_type == "qa"``) or a pedagogically sound
        multiple-choice quiz (``task_type == "quiz"``). Never retrieves, never
        self-evaluates — the Critic is responsible for evaluation and may trigger
        a revision loop back to this node.

    Responsibilities:
        1. Task Routing — dispatch on ``task_type`` to QA / Quiz / clarification.
        2. Empty-Context Handling — if the Retriever returned nothing, return an
           "insufficient context" draft in the user's language. Never fabricate.
        3. QA Generation — grounded RAG answer with inline ``[Doc N]`` citations
           in the user's language, concise (3-6 sentences by default).
        4. Quiz Generation — MCQ set obeying Bloom's-taxonomy difficulty mix,
           distractor quality rules, and per-question source-chunk traceability.
        5. Revision Handling — if ``critique`` is present in state, apply the
           Critic's suggestions against the same retrieved context.

    Reads from AgentState:
        - user_query (str)
        - task_type (str): "qa" | "quiz" | "unknown"
        - language (str): "en" | "vi" | "mixed"
        - intent_summary (str)
        - plan (dict)
        - retrieved_docs (List[Document])
        - evidence_summary (List[str])
        - revision_count (int, optional)
        - critique (dict, optional): present only on revision loops.

    Writes to AgentState:
        - draft (str): plain text for QA, JSON-serialised payload for quiz.
        - generation_metadata (dict): task_type, language, used_chunks, and
          quiz-only fields (num_questions, difficulty_distribution).
        - revision_count (int): incremented when a critique was applied.
    """

    name = "generator"

    QA_PROMPT = """
    # Role
    You are the Generator Agent of the KnowGen system. You synthesize grounded answers from retrieved academic documents.

    # Context
    - User Query: {user_query}
    - Language: {language}
    - Intent: {intent_summary}
    - Retrieved Context:
    {numbered_context_blocks}

    {revision_block}

    # Rules
    1. Answer ONLY using the retrieved context above. If the answer is not present in the context, explicitly state that in {language}.
    2. Do NOT introduce facts that are not in the context.
    3. Cite source chunks inline using [Doc N] notation, where N is the context block index.
    4. Preserve the original language ({language}) of the user query.
    5. Be concise (3-6 sentences) unless the user explicitly asked for depth.
    6. Do NOT use markdown formatting, headings, or bullet lists. Plain prose only.

    # Output
    Return the answer as plain text only. No markdown, no JSON, no preamble."""

    QUIZ_PROMPT = """
    # Role
    You are a pedagogy expert writing multiple-choice questions for university students, based strictly on the provided academic context.

    # Context
    - Topic: {intent_summary}
    - User Query: {user_query}
    - Language: {language}
    - Retrieved Context (numbered):
    {numbered_context_blocks}

    {revision_block}

    # Task
    Generate {num_questions} multiple-choice questions following these non-negotiable rules.

    ## Grounding
    - Every fact in every question stem, every correct answer, and every distractor MUST appear in the retrieved context above.
    - If the context is insufficient to produce {num_questions} grounded questions, generate fewer — do NOT hallucinate.
    - Each question must reference the index of the context block it was drawn from via the `source_chunk_index` field (0-based; matches [Doc N] where N = source_chunk_index + 1).

    ## Difficulty Mix (Bloom's taxonomy)
    Aim for roughly:
    - ~30% easy (remember — direct recall of definitions or facts).
    - ~40% medium (understand — explain, compare, classify concepts).
    - ~30% hard (apply / analyze — apply the concept to a scenario or edge case).

    Each question must carry a `difficulty` field ("easy" | "medium" | "hard") and a `bloom_level` field ("remember" | "understand" | "apply" | "analyze").

    ## Distractor Quality
    - Distractors must be plausible misconceptions, closely-related concepts drawn from the SAME context, or partially-correct statements that miss a key qualifier.
    - Forbidden: absurd options, distractors grammatically inconsistent with the stem, distractors with length dramatically different from the correct answer, lazy fillers like "None of the above" or "All of the above".
    - All 4 choices must be roughly equal length (±30% characters) and use parallel grammatical structure.
    - The correct answer must NOT be systematically longer or shorter than the distractors.

    ## Coverage
    - Spread questions across DIFFERENT context blocks when possible. Avoid generating multiple questions from the same block unless the block is uniquely rich.

    ## Language
    - The stem, choices, and explanation must all be in {language}.

    # Output
    Return STRICTLY valid JSON matching this schema. No markdown fences, no commentary outside the JSON:
    {{
      "questions": [
        {{
          "id": "q1",
          "question": "...",
          "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
          "correct_answer": "A",
          "explanation": "Brief explanation citing [Doc N].",
          "difficulty": "easy",
          "bloom_level": "remember",
          "source_chunk_index": 0
        }}
      ]
    }}"""

    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        task_type = state.get("task_type", "unknown")
        retrieved_docs = state.get("retrieved_docs") or []
        evidence_summary = state.get("evidence_summary") or []

        if task_type == "unknown":
            return self._clarification_response(state)

        no_docs = not retrieved_docs
        no_evidence = evidence_summary == ["No relevant documents found."]
        if no_docs or no_evidence:
            return self._empty_context_response(state)

        if task_type == "qa":
            return self._generate_qa(state)
        if task_type == "quiz":
            return self._generate_quiz(state)

        return self._clarification_response(state)

    # ------------------------------------------------------------------
    # QA generation
    # ------------------------------------------------------------------
    def _generate_qa(self, state: AgentState) -> AgentState:
        retrieved_docs: List[Document] = state.get("retrieved_docs") or []
        language = state.get("language", "en")
        user_query = state.get("user_query", "")
        intent_summary = state.get("intent_summary", "")

        numbered_blocks = self._format_context_blocks(retrieved_docs)
        revision_block = self._build_revision_block(state)

        prompt = self.QA_PROMPT.format(
            user_query=user_query,
            language=language,
            intent_summary=intent_summary,
            numbered_context_blocks=numbered_blocks,
            revision_block=revision_block,
        )

        draft = self._invoke_llm(prompt).strip()
        if not draft:
            self.logger.warning("LLM returned empty QA draft; using insufficient-context fallback")
            return self._empty_context_response(state)

        used_chunks = self._detect_cited_chunks(draft, len(retrieved_docs))

        return AgentState(
            draft=draft,
            generation_metadata={
                "task_type": "qa",
                "language": language,
                "used_chunks": used_chunks,
                "revision_count": state.get("revision_count", 0),
            },
            revision_count=state.get("revision_count", 0),
        )

    # ------------------------------------------------------------------
    # Quiz generation
    # ------------------------------------------------------------------
    def _generate_quiz(self, state: AgentState) -> AgentState:
        retrieved_docs: List[Document] = state.get("retrieved_docs") or []
        language = state.get("language", "en")
        user_query = state.get("user_query", "")
        intent_summary = state.get("intent_summary", "")
        plan = state.get("plan") or {}
        num_questions = int(plan.get("num_questions", DEFAULT_NUM_QUESTIONS))

        numbered_blocks = self._format_context_blocks(retrieved_docs)
        revision_block = self._build_revision_block(state)

        prompt = self.QUIZ_PROMPT.format(
            user_query=user_query,
            intent_summary=intent_summary,
            language=language,
            numbered_context_blocks=numbered_blocks,
            revision_block=revision_block,
            num_questions=num_questions,
        )

        raw = self._invoke_llm(prompt)
        quiz = self._parse_json(raw)

        if quiz is None:
            # One repair pass — ask the LLM to fix its own output to valid JSON.
            self.logger.warning("Quiz JSON invalid; attempting one repair pass")
            repair_prompt = (
                "The following text was supposed to be valid JSON matching the KnowGen quiz schema, "
                "but it is malformed. Return ONLY the corrected, strictly valid JSON — no markdown, "
                "no commentary.\n\n"
                f"---\n{raw}\n---"
            )
            repaired = self._invoke_llm(repair_prompt)
            quiz = self._parse_json(repaired)

        if quiz is None or not isinstance(quiz, dict) or "questions" not in quiz:
            self.logger.error("Quiz generation produced invalid JSON after repair attempt")
            return AgentState(
                draft=json.dumps(
                    {
                        "error": "quiz_malformed",
                        "message": "Generator could not produce a valid quiz JSON payload.",
                        "raw_output": raw[:1000],
                    },
                    ensure_ascii=False,
                ),
                generation_metadata={
                    "task_type": "unknown",
                    "language": language,
                    "used_chunks": [],
                    "num_questions": 0,
                    "difficulty_distribution": {},
                    "revision_count": state.get("revision_count", 0),
                    "error": "quiz_malformed",
                },
                revision_count=state.get("revision_count", 0),
            )

        validated = self._validate_quiz(quiz, retrieved_docs)

        difficulty_distribution = self._compute_difficulty_distribution(validated["questions"])
        used_chunks = sorted({q.get("source_chunk_index") for q in validated["questions"]
                              if isinstance(q.get("source_chunk_index"), int)})

        return AgentState(
            draft=json.dumps(validated, ensure_ascii=False),
            generation_metadata={
                "task_type": "quiz",
                "language": language,
                "used_chunks": used_chunks,
                "num_questions": len(validated["questions"]),
                "difficulty_distribution": difficulty_distribution,
                "revision_count": state.get("revision_count", 0),
            },
            revision_count=state.get("revision_count", 0),
        )

    # ------------------------------------------------------------------
    # Quiz validation (post-processing)
    # ------------------------------------------------------------------
    def _validate_quiz(self, quiz: Dict[str, Any], retrieved_docs: List[Document]) -> Dict[str, Any]:
        """
        Light post-processing pass. Drops questions with clearly malformed
        structure and clamps ``source_chunk_index`` into valid range. Heavy
        grounding / distractor-quality judgement is the Critic's job.
        """
        valid_questions: List[Dict[str, Any]] = []
        max_idx = len(retrieved_docs) - 1 if retrieved_docs else -1

        for i, q in enumerate(quiz.get("questions", [])):
            if not isinstance(q, dict):
                continue
            choices = q.get("choices")
            if not isinstance(choices, dict) or set(choices.keys()) != {"A", "B", "C", "D"}:
                self.logger.debug(f"Dropping question {i}: malformed choices")
                continue
            if not all(isinstance(v, str) and v.strip() for v in choices.values()):
                self.logger.debug(f"Dropping question {i}: empty choice text")
                continue
            if q.get("correct_answer") not in {"A", "B", "C", "D"}:
                self.logger.debug(f"Dropping question {i}: invalid correct_answer")
                continue
            if not isinstance(q.get("question"), str) or not q["question"].strip():
                self.logger.debug(f"Dropping question {i}: empty stem")
                continue

            # Clamp source_chunk_index into valid range; default to 0 if absent.
            idx = q.get("source_chunk_index")
            if not isinstance(idx, int) or idx < 0 or idx > max_idx:
                q["source_chunk_index"] = max(0, min(idx if isinstance(idx, int) else 0, max_idx)) if max_idx >= 0 else 0

            # Fill missing metadata with safe defaults so downstream consumers don't KeyError.
            q.setdefault("id", f"q{len(valid_questions) + 1}")
            q.setdefault("difficulty", "medium")
            q.setdefault("bloom_level", "understand")
            q.setdefault("explanation", "")

            valid_questions.append(q)

        return {"questions": valid_questions}

    # ------------------------------------------------------------------
    # Fallback responses
    # ------------------------------------------------------------------
    def _empty_context_response(self, state: AgentState) -> AgentState:
        language = state.get("language", "en")
        message = INSUFFICIENT_CONTEXT_MESSAGES.get(language, INSUFFICIENT_CONTEXT_MESSAGES["en"])
        return AgentState(
            draft=message,
            generation_metadata={
                "task_type": state.get("task_type", "unknown"),
                "language": language,
                "used_chunks": [],
                "revision_count": state.get("revision_count", 0),
                "reason": "insufficient_context",
            },
            revision_count=state.get("revision_count", 0),
        )

    def _clarification_response(self, state: AgentState) -> AgentState:
        language = state.get("language", "en")
        message = CLARIFICATION_MESSAGES.get(language, CLARIFICATION_MESSAGES["en"])
        return AgentState(
            draft=message,
            generation_metadata={
                "task_type": "unknown",
                "language": language,
                "used_chunks": [],
                "revision_count": state.get("revision_count", 0),
                "reason": "clarification_required",
            },
            revision_count=state.get("revision_count", 0),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_context_blocks(docs: List[Document]) -> str:
        if not docs:
            return "(no context)"
        blocks = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            title = meta.get("title", "")
            header_path = meta.get("header_path", "")
            header_line = f" — {title}" if title else ""
            if header_path:
                header_line += f" / {header_path}"
            blocks.append(f"[Doc {i}]{header_line}\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def _build_revision_block(state: AgentState) -> str:
        critique = state.get("critique")
        if not critique:
            return ""
        issues = critique.get("issues") or []
        suggestions = critique.get("suggestions") or []
        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "(none listed)"
        sugg_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "(none listed)"
        return (
            "# Revision Instructions\n"
            "Your previous draft was reviewed by the Critic and needs revision. "
            "Keep the same retrieved context, but address these points:\n"
            f"## Issues\n{issues_text}\n"
            f"## Suggestions\n{sugg_text}\n"
        )

    @staticmethod
    def _detect_cited_chunks(draft: str, num_docs: int) -> List[int]:
        """Parse inline [Doc N] citations and return 0-based indices within range."""
        if num_docs <= 0:
            return []
        cited = set()
        for match in re.finditer(r"\[Doc\s+(\d+)\]", draft):
            idx = int(match.group(1)) - 1  # 1-based in prompt, 0-based internally
            if 0 <= idx < num_docs:
                cited.add(idx)
        return sorted(cited)

    @staticmethod
    def _compute_difficulty_distribution(questions: List[Dict[str, Any]]) -> Dict[str, float]:
        if not questions:
            return {}
        counts: Dict[str, int] = {}
        for q in questions:
            diff = q.get("difficulty", "medium")
            counts[diff] = counts.get(diff, 0) + 1
        total = len(questions)
        return {k: round(v / total, 2) for k, v in counts.items()}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any] | None:
        """Extract JSON from LLM output, stripping markdown fences if present."""
        if not isinstance(text, str):
            return None
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        try:
            parsed = json.loads(text.strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _invoke_llm(self, prompt: str) -> str:
        try:
            resp = self.llm_client.generate_response(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            return ""


_agent = GeneratorAgent()


def generator_node(state: AgentState) -> AgentState:
    """
    LangGraph node wrapper for the Generator Agent.

    Registered with ``workflow.add_node("generator", generator_node)``. Delegates
    to a module-level ``GeneratorAgent`` singleton so the LLM client is
    initialised once and reused across invocations.

    Input (from AgentState):
        - user_query, task_type, language, intent_summary, plan
        - retrieved_docs, evidence_summary
        - critique, revision_count (optional, present on revision loops)

    Output (merged into AgentState):
        - draft (str): plain text for QA, JSON string for quiz
        - generation_metadata (dict)
        - revision_count (int)
    """
    return _agent(state)
