import os
import argparse
import logging
from typing import List

from dotenv import load_dotenv

from app.ingestion.extract import Document, extract_pdf_to_md, extract_docx_to_md, extract_notion_to_md
from app.ingestion.transform import transform_documents
from app.ingestion.load import load_documents_to_faiss

# Load environment variables (adjust path if needed)
load_dotenv(os.path.join(os.path.dirname(os.getcwd()), '.env'))
load_dotenv() # Fallback to current directory .env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_etl_pipeline(file_paths: List[str] = None, run_notion: bool = False):
    """
    Run the complete Extract -> Transform -> Load pipeline.
    """
    # 1. EXTRACT STAGE
    documents: List[Document] = []
    
    # Extract Local Files
    if file_paths:
        logger.info("EXTRACTING LOCAL FILES")
        for path in file_paths:
            if not os.path.exists(path):
                logger.error("File not found: %s", path)
                continue
                
            filename = os.path.basename(path).lower()
            try:
                if filename.endswith(".pdf"):
                    documents.append(extract_pdf_to_md(path))
                elif filename.endswith(".docx"):
                    documents.append(extract_docx_to_md(path))
                else:
                    logger.warning("Unsupported file type: %s", path)
            except Exception as e:
                logger.error("Failed to extract %s: %s", path, e)

    # Extract Notion
    if run_notion:
        logger.info("EXTRACTING NOTION")
        notion_token = os.getenv("NOTION_TOKEN")
        notion_page_id = os.getenv("NOTION_TEST_PAGE_ID")
        
        if notion_token and notion_page_id:
            try:
                documents.append(extract_notion_to_md(notion_page_id, notion_token))
            except Exception as e:
                logger.error("Failed to extract Notion page: %s", e)
        else:
            logger.warning("NOTION_TOKEN or NOTION_TEST_PAGE_ID is missing in .env")

    if not documents:
        logger.error("No documents extracted. Aborting pipeline.")
        return

    logger.info("Extracted %d document(s) successfully.", len(documents))

    # 2. TRANSFORM STAGE
    logger.info("TRANSFORMING DOCUMENTS")
    transformed_docs = transform_documents(documents)
    
    if not transformed_docs:
        logger.error("No documents transformed successfully. Aborting pipeline.")
        return
        
    logger.info("Transformed %d document(s) successfully.", len(transformed_docs))

    # 3. LOAD STAGE
    logger.info("LOADING TO VECTOR STORE (FAISS)")
    vectorstore = load_documents_to_faiss(transformed_docs)
    
    if vectorstore:
        logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logger.error("ETL PIPELINE FAILED DURING LOAD STAGE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Document ETL Pipeline")
    parser.add_argument("files", nargs="+", help="List of PDF or DOCX files to process")
    parser.add_argument("notion", action="store_true", help="Extract from Notion using .env credentials")
    parser.add_argument("sample", action="store_true", help="Run with a hardcoded sample file")
    
    args = parser.parse_args()
    
    files_to_process = []
    
    if args.files:
        files_to_process.extend(args.files)
        
    if args.sample:
        # Default testing sample
        sample_path = "C:\\Users\\huuda\\Downloads\\a.thuvienvatly.com.2160e.47524.pdf"
        files_to_process.append(sample_path)
        
    if not files_to_process and not args.notion:
        logger.warning("No inputs provided.")
        print("\nUsage Examples:")
        print("  python etl.py --sample")
        print("  python etl.py --files \"path/to/doc1.pdf\" \"path/to/doc2.docx\"")
        print("  python etl.py --notion")
        print("  python etl.py --notion --files \"path/to/file.pdf\"")
    else:
        run_etl_pipeline(file_paths=files_to_process, run_notion=args.notion)
