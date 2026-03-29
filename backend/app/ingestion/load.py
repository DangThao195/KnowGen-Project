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
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> Optional[FAISS]:
    """
    Full Load Pipeline:
        1. Initialize Embeddings (intfloat/multilingual-e5-base)
        2. Flatten documents
        3. Embed and store in FAISS
        4. Save FAISS index locally
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
            
        # Step 3: Embed and Store in FAISS
        logger.info(
            "Creating Vector Embeddings for %d chunks and indexing into FAISS... (This process may take time depending on your machine)", 
            len(lc_docs)
        )
        vectorstore = FAISS.from_documents(lc_docs, embeddings)
        logger.info("Successfully created FAISS index in RAM!")
        
        # Step 4: Save to Disk
        os.makedirs(save_dir, exist_ok=True)
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
