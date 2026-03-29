# Extract Stage

## Overview

The Extract stage handles gathering data from various raw data sources (PDF, DOCX, Notion) and converting them into a unified, canonical `Document` schema containing Markdown-formatted text. This readies the structured information for the downstream Transform stage.

---

## Input

Depending on the adapter being used, the input can be:
- **Local files:** Path to a `.pdf` or `.docx` file.
- **Notion:** A Notion Page ID (`notion_page_id`) and API Token (`notion_token`).

---

## Pipeline Steps

The extraction process handles sources individually via specific **adapters**, converting each into a standardized `Document` object:

### 1. Canonical Schema (`Document`)

Every extracted source must follow this canonical schema:
```python
class Document(BaseModel):
    id: str                        # Filename or Notion Page ID
    source_type: str               # "pdf", "docx", or "notion"
    title: str                     # Extracted title or filename
    content: str                   # Converted Markdown text
    metadata: Dict[str, Any]       # Extra fields (e.g., file path, url)
```

### 2. PDF Adapter (`extract_pdf_to_md`)

Extracts textual content from a PDF document and transforms it into Markdown.

- **Library:** `pymupdf4llm`
- **Process:** Directly consumes the PDF path and converts the document into raw Markdown using `pymupdf4llm.to_markdown()`.
- **Output details:** 
  - `id`: filename
  - `source_type`: "pdf"
  - `metadata`: `{"path": file_path}`

### 3. DOCX Adapter (`extract_docx_to_md`)

Extracts text and styles from a DOCX Word document and maps formatting to Markdown.

- **Library:** `python-docx` (`docx.Document`)
- **Process:** Iterates over document paragraphs and checks their Word style (`para.style.name.lower()`).
- **Style Mappings:**
  - `Heading 1` → `# Text`
  - `Heading 2` → `## Text`
  - `Heading 3` → `### Text`
  - `Heading` → `#### Text`
  - `List Bullet` → `- Text`
  - `List Number` → `1. Text`
  - Everything else is treated as plain text paragraphs.
- **Output details:**
  - `id`: filename
  - `source_type`: "docx"
  - `metadata`: `{"path": file_path}`

### 4. Notion Adapter (`extract_notion_to_md`)

Extracts content from a public or private Notion page via the Notion API.

- **Library:** `requests`
- **Process:** 
  1. Requests the **Page API** to retrieve the actual page Title.
  2. Requests the **Blocks API** to retrieve the page content block by block.
- **Block Mappings:**
  - `heading_1`, `heading_2`, `heading_3` → `#`, `##`, `###` Markdown headers
  - `bulleted_list_item` → `- Text`
  - `numbered_list_item` → `1. Text`
  - `quote` → `> Text`
  - `code` → ````language\nText\n```` (with formatting depending on language)
  - `divider` → `---`
- **Output details:**
  - `id`: page_id
  - `source_type`: "notion"
  - `metadata`: `{"url": page_data.get("url")}`

---

## Output

A list of `Document` objects (defined in the canonical schema above), which are passed directly to the **Transform** stage for cleaning, summarization, and chunking.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Failing to extract a file or page | The exception is caught, a print/log error message is generated (e.g., `Error extracting...`), and the item is skipped. Flow continues. |
| Missing Notion `.env` variables | A message (`Skip Notion:...`) is printed, and the Notion extraction is skipped. |
| Empty Word paragraph | The parser ignores it and moves to the next block. |


