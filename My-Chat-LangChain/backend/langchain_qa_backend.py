# backend/langchain_qa_backend.py

import os
import asyncio
import logging
from urllib.parse import urlparse
import hashlib
from typing import Optional, List

# Document Loaders
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader, PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Text Splitting & Embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Rerank & Retrieval
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# --- Configuration ---
# Vercel (Serverless) only allows writing to /tmp
# Render allows persistent storage if configured
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/data")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Singleton Global Vector Store
_vector_store_instance = None

def get_vector_store():
    """
    Get or initialize the singleton Chroma vector store.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        logging.info(f"Initializing Global Vector Store at {VECTOR_STORE_PATH}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
        
        # Ensure directory exists
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        _vector_store_instance = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings,
            collection_name="unified_knowledge_base" 
        )
    return _vector_store_instance

async def ingest_url(url: str) -> bool:
    """
    Ingest a URL into the unified vector store.
    Removes existing documents from this URL before adding new ones.
    """
    logging.info(f"Starting ingestion for URL: {url}")
    try:
        # 1. Load Documents
        parsed_url = urlparse(url)
        base_domain_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = f"{base_domain_url}/sitemap.xml"
        
        # Try Sitemap first
        loader = SitemapLoader(sitemap_url, filter_urls=[url], continue_on_failure=True)
        documents = await asyncio.to_thread(loader.load)
        
        # Fallback to RecursiveUrlLoader
        if not documents:
            logging.info("Sitemap load failed/empty, falling back to RecursiveUrlLoader")
            loader_fallback = RecursiveUrlLoader(url, max_depth=1)
            documents = await asyncio.to_thread(loader_fallback.load)
            
        if not documents:
            logging.error(f"Failed to load any documents from {url}")
            return False

        # 2. Clean HTML
        bs_transformer = BeautifulSoupTransformer()
        cleaned_documents = bs_transformer.transform_documents(documents, unwanted_tags=["script", "style"])

        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(cleaned_documents)
        
        # 4. Add Metadata
        for split in splits:
            split.metadata["source"] = url
            split.metadata["type"] = "url"

        # 5. Update Vector Store (Delete old -> Add new)
        vector_store = get_vector_store()
        try:
            # Access underlying collection to delete by metadata
            vector_store._collection.delete(where={"source": url})
            logging.info(f"Removed existing documents for {url}")
        except Exception as e:
            logging.warning(f"Could not delete existing docs (might be empty): {e}")

        await asyncio.to_thread(vector_store.add_documents, splits)
        
        logging.info(f"Successfully ingested {len(splits)} chunks from {url}")
        return True
        
    except Exception as e:
        logging.error(f"Error ingesting URL {url}: {e}", exc_info=True)
        return False

async def ingest_file(filepath: str, original_filename: str) -> bool:
    """
    Ingest a local file into the unified vector store.
    Removes existing documents for this filename before adding new ones.
    """
    logging.info(f"Starting ingestion for file: {original_filename}")
    try:
        # 1. Load Documents
        if filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            logging.error(f"Unsupported file type: {filepath}")
            return False
            
        documents = await asyncio.to_thread(loader.load)
        if not documents:
            return False

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 3. Add Metadata
        for split in splits:
            split.metadata["source"] = original_filename
            split.metadata["type"] = "file"
            
        # 4. Update Vector Store (Delete old -> Add new)
        vector_store = get_vector_store()
        try:
            vector_store._collection.delete(where={"source": original_filename})
            logging.info(f"Removed existing documents for {original_filename}")
        except Exception as e:
            logging.warning(f"Could not delete existing docs (might be empty): {e}")

        await asyncio.to_thread(vector_store.add_documents, splits)
        
        logging.info(f"Successfully ingested {len(splits)} chunks from {original_filename}")
        return True
    except Exception as e:
        logging.error(f"Error ingesting file {original_filename}: {e}", exc_info=True)
        return False

def get_retrieval_chain(source_filter: str = None):
    """
    Create a RAG chain using the global vector store.
    
    Args:
        source_filter: If provided, only retrieve documents where metadata['source'] == source_filter.
                       If None, retrieve from the entire knowledge base.
    """
    vector_store = get_vector_store()
    
    # Define Search Kwargs
    search_kwargs = {"k": 20}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}
    
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # Rerank
    logging.info("Initializing FlashrankRerank...")
    try:
        reranker = FlashrankRerank(top_n=10)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        final_retriever = compression_retriever
    except Exception as e:
        logging.warning(f"FlashrankRerank init failed ({e}), falling back to base retriever")
        final_retriever = base_retriever

    # LLM & Chain
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) 
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(final_retriever, combine_docs_chain)
    
    return retrieval_chain
