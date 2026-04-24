# Generator Agent

## Overview
The Generator Agent is the **third node** in the KnowGen LangGraph workflow. It runs **after** the Retriever has produced ranked evidence and **before** the Critic evaluates the draft.

Position in graph: `Supervisor → Retriever → [GENERATOR] → Critic → END`

Its sole responsibility is **grounded content synthesis**: turning retrieved document chunks into either a direct answer (`qa`) or a pedagogically sound quiz (`quiz`). It does **not** retrieve, re-rank, or self-evaluate — the Critic handles evaluation and may trigger a revision loop back to this agent.

## Input (from `AgentState`)
- `user_query` (str): original raw user input.
- `task_type` (str): `"qa" | "quiz" | "unknown"` — routes to the correct generation strategy.
- `language` (str): `"en" | "vi" | "mixed"` — output must preserve this language.
- `intent_summary` (str): high-level intent from Supervisor.
- `plan` (dict): supervisor's plan, especially `context_needed`.
- `retrieved_docs` (List[Document]): ranked chunks from the Retriever, each with `similarity_score` and source metadata.
- `evidence_summary` (List[str]): one-sentence factual claims per chunk.
- `revision_count` (int, optional): if the Critic sent the draft back, this indicates iteration number; the Generator may use `critique` (if present) to refine.
- `critique` (dict, optional): Critic feedback on a previous draft (only present on revision loops).

## Responsibilities

### 1. Task Routing
Read `task_type` and dispatch to the correct internal generator:
- `qa` → `_generate_qa()`
- `quiz` → `_generate_quiz()`
- `unknown` → return a clarification request (no hallucinated content).

### 2. Empty-Context Handling
If `retrieved_docs` is empty **or** `evidence_summary == ["No relevant documents found."]`, the Generator MUST NOT fabricate content. It returns a polite "insufficient context" message in the detected language, and sets `draft` accordingly so the Critic can decide to terminate.

### 3. QA Generation (standard grounded RAG)
For `task_type == "qa"`:
- Build a prompt that concatenates `retrieved_docs` page_content as numbered context blocks (e.g., `[Doc 1]`, `[Doc 2]` …).
- Instruct the LLM to answer **only** from the provided context and to cite source indices inline.
- Preserve `language`.
- Target length: concise (3–6 sentences unless the user explicitly asked for detail).

### 4. Quiz Generation (pedagogically sound MCQs)
For `task_type == "quiz"`, the Generator must follow these pedagogical rules — enforced via prompt and validated in post-processing:

#### 4.1 Coverage & Grounding
- Every question, every correct answer, and every distractor MUST be traceable to at least one chunk in `retrieved_docs`. **No question is allowed to introduce facts that do not appear in the retrieved context.**
- Spread questions across **distinct chunks** when possible — avoid generating 5 questions all from the same paragraph.
- Include the `source_chunk_index` in each question's metadata for Critic validation.

#### 4.2 Difficulty Diversity
Generate a balanced mix across Bloom's taxonomy levels:
- **Remember** (~30%): direct recall of definitions or facts.
- **Understand** (~40%): explain, compare, classify concepts.
- **Apply / Analyze** (~30%): apply the concept to a scenario, distinguish edge cases, identify exceptions.

Each question is tagged with a `difficulty` field: `"easy" | "medium" | "hard"`.

#### 4.3 Distractor Quality (the hardest part)
Distractors (wrong choices) must be **plausible**, not absurd. The prompt must explicitly forbid:
- Obviously wrong answers (e.g., "The sky is green").
- Distractors that are grammatically inconsistent with the question stem (a common tell that reveals the correct answer).
- Distractors whose length is dramatically shorter/longer than the correct answer (another tell).
- Joke / filler options ("None of the above" used lazily, "All of the above" when only one option is actually correct).

Positive distractor rules:
- Each distractor should represent a **realistic misconception**, a **closely related but distinct concept from the same document**, or a **partially-correct statement that misses a key qualifier**.
- All 4 choices should be roughly the same length (±30% character count).
- All 4 choices should use parallel grammatical structure.

#### 4.4 Output Schema for Quiz
```json
{
  "questions": [
    {
      "id": "q1",
      "question": "…",
      "choices": {"A": "…", "B": "…", "C": "…", "D": "…"},
      "correct_answer": "B",
      "explanation": "Brief explanation citing the source chunk.",
      "difficulty": "easy | medium | hard",
      "bloom_level": "remember | understand | apply | analyze",
      "source_chunk_index": 2
    }
  ]
}
```

### 5. Revision Handling
If `critique` is present in state (the Critic sent the draft back):
- Read `critique.issues` and `critique.suggestions`.
- Re-generate with those corrections applied, keeping the same retrieved context.
- Increment `revision_count`. The LangGraph wiring (not this agent) enforces a max revision cap.

## Output (State Updates)
```python
{
    "draft": "...",                       # str for qa; JSON-serialised quiz for quiz
    "generation_metadata": {
        "task_type": "qa | quiz",
        "language": "en | vi | mixed",
        "used_chunks": [0, 2, 3],         # indices of retrieved_docs actually cited
        "num_questions": 5,               # quiz only
        "difficulty_distribution": {...}, # quiz only
        "revision_count": 0
    }
}
```

Note: `draft` is intentionally the Critic's input. The Critic may mutate state to trigger another loop or promote the draft to `final_answer`.

## Prompt Strategy

### QA System Prompt (sketch)
```text
# Role
You are the Generator Agent of the KnowGen system. You synthesize grounded answers from retrieved academic documents.

# Context
- User Query: {user_query}
- Language: {language}
- Intent: {intent_summary}
- Retrieved Context:
{numbered_context_blocks}

# Rules
1. Answer ONLY using the retrieved context. If the answer is not present, explicitly say so in {language}.
2. Do NOT introduce facts that are not in the context.
3. Cite source chunks inline using [Doc N] notation.
4. Preserve the original language ({language}).
5. Be concise (3–6 sentences) unless the user asked for depth.

# Output
Return the answer as plain text. No markdown, no JSON.
```

### Quiz System Prompt (sketch)
```text
# Role
You are a pedagogy expert writing multiple-choice questions for university students, based strictly on the provided academic context.

# Context
- Topic: {intent_summary}
- Language: {language}
- Retrieved Context (numbered):
{numbered_context_blocks}

# Task
Generate {num_questions} MCQs following these non-negotiable rules:

## Grounding
- Every fact in every question and every choice MUST appear in the retrieved context.
- If the context is insufficient for N questions, generate fewer — do not hallucinate.

## Difficulty Mix
- ~30% easy (recall), ~40% medium (understand/compare), ~30% hard (apply/analyze).

## Distractor Quality
- Distractors must be plausible misconceptions or closely related concepts drawn from the SAME context.
- Forbidden: absurd options, grammar mismatches between stem and choices, length mismatches, "None of the above" fillers.
- All 4 choices must be roughly equal length and parallel in grammatical structure.
- The correct answer must NOT be systematically longer/shorter than distractors.

## Coverage
- Spread questions across different chunks when possible.
- Tag each question with source_chunk_index referencing the context block it came from.

# Output
Return strictly valid JSON matching the schema provided. No markdown, no prose outside JSON.
```

## Failure Behaviour
- **Empty retrieval** → return a clarification/insufficient-context draft in `language`; do not hallucinate.
- **LLM returns malformed JSON for quiz** → attempt one repair pass (re-prompt with the raw output + "fix to valid JSON"); if still invalid, return a draft with `task_type="unknown"` and an error note for the Critic.
- **`task_type == "unknown"`** → return a clarification message asking the user to reformulate. Do not call the LLM for content generation.

## Class Skeleton (for implementation)
Inherits `BaseAgent` like `SupervisorAgent` and `RetrieverAgent`.

```python
class GeneratorAgent(BaseAgent):
    name = "generator"

    QA_PROMPT = "..."
    QUIZ_PROMPT = "..."

    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()

    def run(self, state: AgentState) -> AgentState:
        task_type = state.get("task_type", "unknown")
        if not state.get("retrieved_docs"):
            return self._empty_context_response(state)
        if task_type == "qa":
            return self._generate_qa(state)
        if task_type == "quiz":
            return self._generate_quiz(state)
        return self._clarification_response(state)

    def _generate_qa(self, state: AgentState) -> AgentState: ...
    def _generate_quiz(self, state: AgentState) -> AgentState: ...
    def _validate_quiz(self, quiz: dict, retrieved_docs) -> dict: ...
    def _empty_context_response(self, state): ...
    def _clarification_response(self, state): ...


_agent = GeneratorAgent()


def generator_node(state: AgentState) -> AgentState:
    return _agent(state)
```