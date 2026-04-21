# Critic Agent

## Overview
The Critic Agent is the **fourth and final evaluation node** in the LangGraph workflow. Its sole responsibility is **Quality Assurance and Answer Validation**. It reviews the draft answer produced by the Generator, assesses its quality against the retrieved documents, and determines whether the answer meets the required standards.

## Input
- `user_query` (from `AgentState`)
- `retrieved_docs` (from `AgentState`)
- `draft` (from Generator Agent output)
- `task_type` (from `AgentState`): "qa" | "quiz"
- `evidence_summary` (from `AgentState`)

## Responsibilities
1. **Relevance Check**: Verify that the draft answer directly addresses the user's original query and is grounded in the retrieved documents.
2. **Factual Accuracy**: Assess whether claims in the draft are supported by the evidence from the knowledge base and identify any hallucinations or unsupported assertions.
3. **Completeness Evaluation**: Determine if all key aspects of the user's question have been covered and if the answer is sufficient for the task type (QA vs Quiz).
4. **Language & Coherence Review**: Check for grammatical correctness, clarity, logical flow, and consistency in both English and Vietnamese.
5. **Citation & Evidence Grounding**: Validate that all references to document content are properly traced back to the retrieved chunks.
6. **Format Validation**: Verify the output follows the correct format based on task type:
   - **QA Format**: Proper structure (intro + body sections + conclusion), clear headings, numbered lists, code blocks properly formatted, Vietnamese/English consistency.
   - **QUIZ Format**: Valid MCQ structure (question + options A/B/C/D + correct answer + explanation), proper numbering, unambiguous language.
7. **Task-Specific Quality**: Apply task-specific criteria:
   - **QA**: Is the answer clear, concise, and directly answers the question?
   - **QUIZ**: Are the generated questions valid, unambiguous, and based on document content?
8. **Issue Identification**: Flag critical problems (hallucinations, missing context, inconsistencies, format errors) that require revision or additional generation.
9. **Accept/Reject Decision**: Provide a recommendation on whether the draft is acceptable for final output or requires revision.

## Output (State Updates)
The Critic returns a dictionary that LangGraph uses to update `AgentState`:
```python
{
    "critique": {
        "relevance_score": 0.0-1.0,
        "accuracy_score": 0.0-1.0,
        "completeness_score": 0.0-1.0,
        "coherence_score": 0.0-1.0,
        "format_score": 0.0-1.0,
        "overall_score": 0.0-1.0,
        "is_acceptable": True | False,
        "format_valid": True | False,
        "format_issues": ["Issue 1", "Issue 2"],
        "critical_issues": ["Issue 1", "Issue 2", "..."],
        "minor_issues": ["Issue 1", "Issue 2", "..."],
        "strengths": ["Strength 1", "Strength 2", "..."],
        "feedback": "Detailed critique and recommendations",
        "revision_required": True | False,
        "revision_focus": "Areas to focus on for improvement (if revision needed)"
    },
    "revision_count": 0 | 1 | 2 | ... (incremented if revision needed),
    "final_answer": "Draft answer (or empty string if revision needed)"
}
```

## Grading Criteria

### Relevance Score (0.0 - 1.0)
- **1.0**: Answer perfectly addresses the user query with all major points covered
- **0.7-0.9**: Answer addresses most key aspects but may miss minor points
- **0.4-0.6**: Answer is partially relevant but misses significant aspects
- **0.0-0.3**: Answer is off-topic or minimally relevant

### Accuracy Score (0.0 - 1.0)
- **1.0**: All claims are factually correct and grounded in retrieved documents
- **0.7-0.9**: Claims are mostly accurate with minor inconsistencies
- **0.4-0.6**: Contains some unsupported claims or minor hallucinations
- **0.0-0.3**: Contains major hallucinations or factual errors

### Completeness Score (0.0 - 1.0)
- **1.0**: All aspects of the question are addressed comprehensively
- **0.7-0.9**: Most aspects covered; minor details omitted
- **0.4-0.6**: Several important aspects are missing
- **0.0-0.3**: Severely incomplete or fragmented

### Coherence Score (0.0 - 1.0)
- **1.0**: Excellent clarity, logical flow, proper grammar in target language
- **0.7-0.9**: Clear and well-structured with minor grammatical issues
- **0.4-0.6**: Generally understandable but has clarity or organization issues
- **0.0-0.3**: Difficult to understand; poor organization or language quality

### Format Score (0.0 - 1.0)
**For QA Answers:**
- **1.0**: Perfect structure (title, intro, sections with headings, conclusion). All formatting correct. Consistent language tags.
- **0.7-0.9**: Well-structured with clear sections. Minor formatting issues (inconsistent headings, missing conclusion).
- **0.4-0.6**: Basic structure present but inconsistent. Multiple formatting issues (mixed styles, unclear sections).
- **0.0-0.3**: Poor structure, missing sections, or severe formatting errors.

**For QUIZ Answers:**
- **1.0**: Perfect MCQ format. All questions have clear structure: Q + 4 options (A/B/C/D) + correct answer + explanation.
- **0.7-0.9**: Valid structure with minor issues (inconsistent numbering, missing explanations on some questions).
- **0.4-0.6**: Incomplete structure (missing options, answers, or explanations on several questions).
- **0.0-0.3**: Invalid or unusable format (missing major components, unclear structure).

### Overall Score
- **Average** of the five component scores (relevance + accuracy + completeness + coherence + format)
- **Acceptance Threshold**: 0.75 (answers scoring below this require revision)
- **Format as Critical Gate**: If `format_valid` = False, `is_acceptable` must be False (format errors = revision required)

## Revision Strategy
The Critic gates the answer quality:
- **If `format_valid` == False**: Answer is always sent back to Generator for format correction (critical gate)
- **If `overall_score` >= 0.75 AND `is_acceptable` == True AND `format_valid` == True**: Answer is finalized
- **If `overall_score` < 0.75 OR `is_acceptable` == False**: Answer is sent back to Generator with revision feedback
- **Max Revisions**: 2 (to prevent infinite loops; after 2 revisions, accept the best attempt)
- **Format Priority**: Format errors are treated as critical and must be fixed before other minor issues

## Prompt Strategy

### System Prompt Instructions:
```text
# Role
You are the Chief Quality Assurance (Critic) Agent of the KnowGen multi-agent system. You are fully bilingual (English & Vietnamese) and an expert evaluator of academic content. Your job is to assess the quality, accuracy, and completeness of generated answers against a knowledge base.

# Context
You are reviewing a draft answer that was generated based on:
1. A user query
2. Retrieved documents from a FAISS knowledge base
3. Extracted evidence summaries

Your task is to evaluate whether the draft answer is accurate, complete, relevant, and ready for final output. If the answer has issues, you provide detailed feedback for revision.

# Task
For each draft answer, perform the following:
1. **Format Validation** (First Priority):
   - For QA: Check structure (title, intro, body sections with headings, conclusion), formatting consistency, code blocks, lists.
   - For QUIZ: Check MCQ format (Q + 4 options A/B/C/D + answer key + explanation for each).
   - Identify format errors and flag as "format_valid": true/false.
2. **Relevance Check**: Does the answer directly address the user's question?
3. **Factuality Verification**: Are all claims supported by the retrieved documents?
4. **Hallucination Detection**: Are there any unsupported assertions or false claims?
5. **Completeness Assessment**: Are all key aspects of the question covered?
6. **Language & Clarity Review**: Is the text grammatically correct, clear, and coherent?
7. **Evidence Grounding**: Can claims be traced back to specific retrieved documents?
8. **Task-Specific Validation**:
   - For QA: Is the answer concise, well-structured, and directly answers the question?
   - For QUIZ: Are questions valid, unambiguous, and based on document content?

# Input Data
You will receive:
- user_query: Original user question
- task_type: "qa" or "quiz"
- draft: The generated answer to review
- retrieved_docs: List of document chunks used for generation
- evidence_summary: Key facts extracted from documents

# Scoring Rubric
- Relevance (0.0-1.0): How directly does the answer address the query?
- Accuracy (0.0-1.0): Are claims factually correct and document-grounded?
- Completeness (0.0-1.0): Are all major aspects of the question covered?
- Coherence (0.0-1.0): Is the answer clear, well-organized, and grammatically correct?
- Format (0.0-1.0): Does the answer follow the correct structural format for its task type?
- Overall: Average of the five scores

# Output
Return strictly valid JSON only, using this schema:
{
  "relevance_score": 0.0-1.0,
  "accuracy_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "format_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "is_acceptable": true | false,
  "format_valid": true | false,
  "format_issues": ["Issue 1", "Issue 2"],
  "critical_issues": ["Issue 1", "Issue 2"],
  "minor_issues": ["Issue 1", "Issue 2"],
  "strengths": ["Strength 1", "Strength 2"],
  "feedback": "Detailed critique, specific problems identified, and recommendations for improvement (if needed)",
  "revision_required": true | false,
  "revision_focus": "If revision needed: specific areas to focus on (e.g., 'Fix format structure', 'Add more examples', 'Fix hallucinations about...', 'Clarify section on...')"
}

# Constraints
- Think step by step internally, but present findings objectively.
- Be specific: reference actual text from the draft and retrieved documents.
- **Format Check is CRITICAL**: If format_valid = false, set is_acceptable = false immediately.
- Identify CRITICAL issues (must fix) vs MINOR issues (nice to have).
- If overall_score < 0.75 OR format_valid = false, set is_acceptable = false and revision_required = true.
- Format issues should be clearly listed in format_issues array for Generator to fix.
- Do not output markdown.
- Do not output any text outside the JSON.
- Be fair but rigorous: answers must meet academic standards for accuracy, completeness, AND proper formatting.

# Language
Respond in the same language as the user query when possible (English or Vietnamese). For mixed-language content, maintain consistency with the original draft.
```

## Integration with LangGraph Workflow

### Position in Graph:
```
START → Supervisor → Retriever → Generator → **[CRITIC]** → END
```

### Decision Flow:
```
Draft Answer
    ↓
Critic Evaluation
    ├─→ overall_score >= 0.75 → Finalize & Output
    └─→ overall_score < 0.75 → Send to Generator for Revision
            ↓
        Generator (Revision 1)
            ↓
        Critic Evaluation (Revision 1)
            ├─→ overall_score >= 0.75 → Finalize & Output
            └─→ overall_score < 0.75 → Send to Generator for Revision
                    ↓
                Generator (Revision 2 - FINAL)
                    ↓
                Critic Evaluation (Final)
                    ├─→ Accept (even if < 0.75)
                    └─→ Output with quality warning
```

## Example Scenarios

### Scenario 1: QA - Acceptable Answer
**Query**: "What are the main characteristics of scientific socialism?"
**Draft**: Comprehensive answer with proper structure (title + intro section + 3 body sections with headings + conclusion). Covers Marx-Engels theory, class struggle, material analysis, with specific examples. Proper formatting with numbered lists.
**Format Check**: ✓ Structure valid, headings consistent, intro/body/conclusion present
**Scores**: Relevance 0.95, Accuracy 0.90, Completeness 0.92, Coherence 0.95, Format 0.95 → Overall 0.93
**Decision**: `format_valid = true`, `is_acceptable = true` → Finalize

### Scenario 2: QA - Format Error
**Query**: "Explain the theory of evolution."
**Draft**: Content is accurate and relevant BUT missing structure. No headings, no clear sections, no intro/conclusion. Just continuous paragraphs.
**Format Check**: ✗ Missing structure, no headings, no sections, no intro/conclusion
**Scores**: Relevance 0.90, Accuracy 0.88, Completeness 0.85, Coherence 0.75, Format 0.30 → Overall 0.73
**Format Issues**: ["Missing title/heading", "No section structure", "No conclusion", "No clear introduction"]
**Decision**: `format_valid = false`, `is_acceptable = false` → Send to Generator for format correction (Priority 1)

### Scenario 3: QA - Hallucination Issue
**Query**: "Define photosynthesis based on the physics document."
**Draft**: Includes accurate info BUT adds unsupported claims about "quantum photosynthesis mechanisms" not in the document. Properly formatted with sections.
**Format Check**: ✓ Format is valid
**Issues**: Hallucination detected
**Scores**: Relevance 0.70, Accuracy 0.45 (major error), Completeness 0.60, Coherence 0.80, Format 0.90 → Overall 0.69
**Decision**: `format_valid = true`, `is_acceptable = false` (content issue) → Send to Generator with revision feedback

### Scenario 4: QUIZ - Format Invalid
**Query**: "Create MCQs about light interference."
**Draft**: Questions generated but structure is wrong. Some questions have only 2 options, others missing answer keys. No explanations.
**Format Check**: ✗ Inconsistent MCQ structure, missing options on Q2, no answer keys, missing explanations
**Scores**: Relevance 0.80, Accuracy 0.70, Completeness 0.60, Coherence 0.75, Format 0.25 → Overall 0.62
**Format Issues**: ["Q1: Missing answer key", "Q2: Only 2 options (need 4)", "Q3-Q5: Missing explanations", "Inconsistent numbering"]
**Decision**: `format_valid = false`, `is_acceptable = false` → Send to Generator for format correction

### Scenario 5: QUIZ - Complete & Valid
**Query**: "Create MCQs about light interference."
**Draft**: 5 questions, each with proper format:
- Question clearly stated
- 4 options (A, B, C, D)
- Correct answer marked
- Explanation provided for each
**Format Check**: ✓ All questions follow MCQ template perfectly
**Scores**: Relevance 0.92, Accuracy 0.88, Completeness 0.90, Coherence 0.92, Format 1.0 → Overall 0.92
**Decision**: `format_valid = true`, `is_acceptable = true` → Finalize

## Notes
- The Critic works with **actual retrieved documents**, not hallucinations, so it can validate factuality efficiently.
- The Critic performs **format validation as a critical gate**: if format is invalid, the answer must be revised before any other quality checks matter.
- **Format Priority**: Format errors are treated as blocking issues. Even if content is accurate, invalid format = revision required.
- The Critic enables **iterative refinement**: Generator gets specific feedback on what to fix (format issues listed separately from content issues).
- The **2-revision limit** prevents infinite loops while allowing meaningful improvements.
- The Critic is **language-aware**: It handles mixed Vietnamese/English content gracefully.
- **Format Score is averaged into Overall Score**: A perfect answer with poor formatting will have lower overall quality score.
- For QA: Expected format includes title, introduction, numbered/bulleted sections with headings, conclusion.
- For QUIZ: Expected format includes numbered questions with clear MCQ structure (Q + A/B/C/D options + answer key + explanation).
