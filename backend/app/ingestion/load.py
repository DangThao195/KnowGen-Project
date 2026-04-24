import os
import logging
from typing import List, Optional
from pathlib import Path

# LangChain and Vectorstore imports
from langchain_core.documents import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from app.ingestion.transform import TransformedDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default saving directory
VECTOR_STORE_DIR = Path(os.getcwd()) / "vector_store" / "faiss_index"


def get_huggingface_embeddings(model_name: str = "intfloat/multilingual-e5-base") -> HuggingFaceEmbeddings:
    """
    Step 1: Initialize the HuggingFace embedding model.
    E5-base requires prefixing for queries and passages.
    We handle the passage prefix during the flattening step.
    """
    logger.info("Loading Embedding: %s", model_name)
    
    # E5 models typically perform best with normalized embeddings
    encode_kwargs = {'normalize_embeddings': True} 
    
    # model_kwargs={'device': 'cpu'} can be set, but HuggingFaceEmbeddings auto-detects GPU/CPU.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


def flatten_transformed_documents(transformed_docs: List[TransformedDocument]) -> List[LangchainDocument]:
    """
    Step 2: Flatten custom Chunks into standard Langchain Documents.
    Important: The intfloat/multilingual-e5-base model requires the 'passage: ' prefix 
    for proper clustering of document texts.
    """
    lc_docs = []
    
    for t_doc in transformed_docs:
        for chunk in t_doc.chunks:
            # Prepend 'passage: ' string as strictly required by E5-base
            content_with_prefix = f"passage: {chunk.content}"
            
            doc = LangchainDocument(
                page_content=content_with_prefix,
                metadata=chunk.metadata
            )
            lc_docs.append(doc)
            
    logger.info(
        "Flattened %d TransformedDocument(s) into %d Langchain Document(s) (added 'passage: ' prefix)", 
        len(transformed_docs), len(lc_docs)
    )
    return lc_docs


def load_documents_to_faiss(
    transformed_docs: List[TransformedDocument], 
    save_dir: str = str(VECTOR_STORE_DIR),
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    merge: bool = True,
) -> Optional[FAISS]:
    """
    Full Load Pipeline:
        1. Initialize Embeddings (intfloat/multilingual-e5-base)
        2. Flatten documents
        3. Embed new chunks and index into FAISS
        4. Merge with existing FAISS index (if ``merge=True`` and one exists),
           or replace it (``merge=False``).
        5. Save the final index locally.

    Args:
        transformed_docs: Chunks produced by the Transform stage.
        save_dir: Directory where the FAISS index is persisted.
        embeddings: Pre-initialised embeddings (optional; created if None).
        merge: When True (default), new chunks are merged into the existing
               index so that documents from previous ingestion runs are kept.
               Set to False to wipe the index and start fresh.
    """
    if not transformed_docs:
        logger.warning("TransformedDocument list is empty. Skipping Load step.")
        return None
        
    try:
        # Step 1: Initialize Embeddings
        if embeddings is None:
            embeddings = get_huggingface_embeddings()
            
        # Step 2: Flatten chunks
        lc_docs = flatten_transformed_documents(transformed_docs)
        
        if not lc_docs:
            logger.warning("No text chunks found after flattening. Skipping FAISS.")
            return None
            
        # Step 3: Embed new chunks into a fresh in-RAM index
        logger.info(
            "Creating Vector Embeddings for %d chunks and indexing into FAISS... (This process may take time depending on your machine)", 
            len(lc_docs)
        )
        new_index = FAISS.from_documents(lc_docs, embeddings)
        logger.info("Successfully created new FAISS index in RAM with %d vectors.", len(lc_docs))

        # Step 4: Merge with existing index or start fresh
        os.makedirs(save_dir, exist_ok=True)
        index_files_exist = os.path.isfile(os.path.join(save_dir, "index.faiss"))

        if merge and index_files_exist:
            logger.info("Existing FAISS index found — merging new vectors into it...")
            existing_index = FAISS.load_local(
                save_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            existing_index.merge_from(new_index)
            vectorstore = existing_index
            logger.info("Merge complete.")
        else:
            if not index_files_exist:
                logger.info("No existing index found — creating a fresh one.")
            else:
                logger.info("merge=False — replacing existing FAISS index.")
            vectorstore = new_index
        
        # Step 5: Persist to disk
        vectorstore.save_local(save_dir)
        logger.info("Successfully saved FAISS Index database to directory: %s", save_dir)
        
        return vectorstore
        
    except ImportError as e:
        logger.error(
            "Missing required libraries for FAISS or HuggingFace: %s.\n"
            "Please run: pip install faiss-cpu langchain-huggingface sentence-transformers", 
            e
        )
    except Exception as e:
        logger.error("Error occurred during loading into FAISS: %s", e)
        
    return None
