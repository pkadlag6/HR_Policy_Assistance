import os
from langchain_community.vectorstores import FAISS

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model

from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# build a vector store

def build_vector_store(chunks):
    logger.info("Embedding %d chunk(s) and building faiss index")
    embeddings_model = get_embeddings_model()
    logger.info("Faiss build in memory")
    return FAISS.from_documents(chunks,embeddings_model)


# save vector store
def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
    vector_store.save_local(path)
    logger.info("SAVED faiss index to '%s'",path)


# load vectore store

def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    logger.info("Loading faiss index from '%s'",path)
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(path, embeddings_model,allow_dangerous_deserialization=True)


# 

def vector_store_exists(path:str = config.VECTOR_STORE_PATH)-> bool:
    return os.path.exists(os.path.join(path,"index.faiss"))

# retriever
def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever
    that returns the top-k matching chunks."""
    logger.info("Creating retrievers with top_k=%d",k)
    return vector_store.as_retriever(search_kwargs={"k": k})