import os
import sys
from pathlib import Path
from typing import List

workspace_root = Path(__file__).resolve().parent.parent
backend_dir = workspace_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.ingestion.extract import extract_pdf_to_md, extract_docx_to_md
from app.ingestion.transform import transform_documents
from app.ingestion.load import load_documents_to_faiss

SOURCE_FILES: List[str] = [
    r"H:\ĐH 1,2\NNHT.pdf",
    r"H:\ĐH 1,2\Vật lí 1\Chương 10 - Giao thoa ánh sáng.pdf",
    r"H:\ĐH 1,2\BT CNXHKH.docx",
]
VECTOR_STORE_DIR = backend_dir / "vector_store" / "faiss_index"


def run_etl(files: List[str]):
    documents = []
    errors = []

    for file_path in files:
        print(f"Processing: {file_path}")
        if not os.path.exists(file_path):
            print(f"  ❌ File missing: {file_path}")
            errors.append((file_path, "missing"))
            continue

        try:
            if file_path.lower().endswith(".pdf"):
                doc = extract_pdf_to_md(file_path)
            elif file_path.lower().endswith(".docx"):
                doc = extract_docx_to_md(file_path)
            else:
                raise ValueError("Unsupported file type")

            print(f"  ✓ Extracted {doc.id} ({doc.source_type})")
            documents.append(doc)
        except Exception as exc:
            print(f"  ❌ Extract failed: {exc}")
            errors.append((file_path, str(exc)))

    if not documents:
        print("No documents extracted. Aborting ETL.")
        return

    transformed = transform_documents(documents)
    print(f"Transformed {len(transformed)} document(s)")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = load_documents_to_faiss(transformed, save_dir=str(VECTOR_STORE_DIR))
    if vectorstore:
        print(f"FAISS index saved to: {VECTOR_STORE_DIR}")
    else:
        print("Failed to create FAISS index.")


if __name__ == "__main__":
    run_etl(SOURCE_FILES)
