import os
import re
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.ingestion.extract import Document
from app.llm.llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output schemas
class Chunk(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransformedDocument(BaseModel):
    doc_id: str
    title: str
    source_type: str
    chunks: List[Chunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Prompt loader
_PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str, **kwargs) -> str:
    """Load a markdown prompt template and inject variables via str.format()."""
    prompt_path = _PROMPT_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    return template.format(**kwargs)


# STEP 1 — Clean
# Emoji regex: covers the principal Unicode emoji ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # Misc symbols, emoticons, transport, etc.
    "\U00002600-\U000027BF"   # Misc symbols
    "\U0001F000-\U0001F02F"   # Mahjong / domino tiles
    "\U0001F0A0-\U0001F0FF"   # Playing cards
    "\U00002702-\U000027B0"   # Dingbats subset
    "\uFE00-\uFE0F"           # Variation selectors
    "\u200d"                  # Zero-width joiner
    "]+",
    flags=re.UNICODE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def _lowercase_body(text: str) -> str:
    """Lowercase all text while preserving Markdown heading markers and code fences."""
    lines = text.split("\n")
    result = []
    in_code_block = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)          # keep code fence as-is
        elif in_code_block:
            result.append(line)          # keep code content as-is
        elif stripped.startswith("#"):
            # Preserve heading markers; lowercase only the heading text
            match = re.match(r"(#+\s*)(.*)", line)
            if match:
                result.append(match.group(1) + match.group(2).lower())
            else:
                result.append(line.lower())
        else:
            result.append(line.lower())
    return "\n".join(result)


def clean_document(doc: Document) -> Document:
    """
    Step 1: Normalize raw Markdown content.
    Rules applied in order:
      1. Remove emojis
      2. Remove URLs
      3. Strip trailing whitespace per line
      4. Collapse 3+ blank lines → 2 blank lines
      5. Lowercase (preserve heading markers & code fences)
      6. NFC Unicode normalization
      7. Strip non-printable control characters
    """
    text = doc.content

    text = _EMOJI_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _TRAILING_SPACES_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _lowercase_body(text)
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.strip()

    if not text:
        logger.warning("Document '%s' is empty after cleaning — skipping.", doc.id)
        return None  # Signal to skip

    new_meta = {**doc.metadata, "cleaned": True}
    return doc.model_copy(update={"content": text, "metadata": new_meta})


# STEP 2a — Summarize
def summarize_document(doc: Document, llm: LLMClient) -> str:
    """
    Step 2a: Generate a concise summary via the LLM.
    Uses prompts/summarize.md as the prompt template.
    Returns an empty string on failure.
    """
    try:
        prompt = load_prompt("summarize.md", title=doc.title, content=doc.content)
        summary = llm.generate_response(prompt)
        return summary.strip()
    except Exception as exc:
        logger.warning("Summarization failed for '%s': %s", doc.id, exc)
        return ""


# STEP 2b — Chunk
_HEADERS_TO_SPLIT = [
    ("#",   "H1"),
    ("##",  "H2"),
    ("###", "H3"),
]


def chunk_document(doc: Document) -> List[Chunk]:
    """
    Step 2b: Split cleaned Markdown into semantic chunks.
    Primary:  MarkdownHeaderTextSplitter on #/##/###.
    Fallback: RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
              when no Markdown headers are found.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    raw_chunks = splitter.split_text(doc.content)

    # Fallback: no headers found → single large chunk → use recursive splitter
    if len(raw_chunks) <= 1 and not any(
        line.startswith("#") for line in doc.content.splitlines()
    ):
        logger.info(
            "Document '%s' has no Markdown headers — using RecursiveCharacterTextSplitter.",
            doc.id,
        )
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        raw_chunks = fallback_splitter.create_documents([doc.content])

    chunks: List[Chunk] = []
    for idx, lc_chunk in enumerate(raw_chunks):
        # Build a breadcrumb from metadata keys H1/H2/H3
        header_path = " > ".join(
            v
            for k in ("H1", "H2", "H3")
            if (v := lc_chunk.metadata.get(k))
        )
        chunk_meta = {
            **doc.metadata,
            "role": "chunk",
            "header_path": header_path or "",
        }
        chunks.append(
            Chunk(
                doc_id=doc.id,
                chunk_index=idx,          # will be re-indexed in combine step
                content=lc_chunk.page_content.strip(),
                metadata=chunk_meta,
            )
        )

    # Safety: if still nothing, wrap the whole content as one chunk
    if not chunks:
        logger.warning(
            "Document '%s' produced 0 chunks — using full content as single chunk.", doc.id
        )
        chunks = [
            Chunk(
                doc_id=doc.id,
                chunk_index=0,
                content=doc.content,
                metadata={**doc.metadata, "role": "chunk", "header_path": ""},
            )
        ]

    return chunks


# STEP 3 — Combine
def combine_results(
    doc: Document,
    summary: str,
    content_chunks: List[Chunk],
) -> TransformedDocument:
    """
    Step 3: Merge summary + content chunks into a TransformedDocument.

    Layout:
      chunk 0  → summary  (role: "summary")
      chunk 1+ → content  (role: "chunk")
    """
    summary_text = f"Summary of {doc.title}:\n{summary}" if summary else ""
    summary_chunk = Chunk(
        doc_id=doc.id,
        chunk_index=0,
        content=summary_text,
        metadata={
            **doc.metadata,
            "role": "summary",
            "summary_failed": summary == "",
        },
    )

    # Re-index content chunks starting at 1
    # Global-Local Context: Prepend summary to the beginning of each content chunk
    reindexed = []
    for i, chunk in enumerate(content_chunks):
        enhanced_content = f"{summary_text}\n\n---\n\n{chunk.content}" if summary_text else chunk.content
        reindexed.append(
            chunk.model_copy(update={
                "chunk_index": i + 1,
                "content": enhanced_content
            })
        )

    all_chunks = [summary_chunk] + reindexed

    return TransformedDocument(
        doc_id=doc.id,
        title=doc.title,
        source_type=doc.source_type,
        chunks=all_chunks,
        metadata=doc.metadata,
    )


# Main pipeline entry point
def transform_documents(
    documents: List[Document],
    llm: Optional[LLMClient] = None,
) -> List[TransformedDocument]:
    """
    Full Transform pipeline:
      For each Document →
        1. clean_document
        2a. summarize_document (LLM)
        2b. chunk_document
        3. combine_results
    Returns a list of TransformedDocument ready for the Load stage.
    """
    if llm is None:
        llm = LLMClient()

    results: List[TransformedDocument] = []

    for doc in documents:
        logger.info("Transforming document: %s", doc.id)

        # Step 1 — Clean
        cleaned = clean_document(doc)
        if cleaned is None:
            continue   # empty after cleaning

        # Step 2 — Parallel sub-pipelines
        summary = summarize_document(cleaned, llm)      # 2a
        chunks  = chunk_document(cleaned)               # 2b

        # Step 3 — Combine
        transformed = combine_results(cleaned, summary, chunks)
        results.append(transformed)
        logger.info(
            "Done '%s': summary=%s, chunks=%d",
            doc.id,
            "ok" if summary else "failed",
            len(transformed.chunks),
        )

    return results
