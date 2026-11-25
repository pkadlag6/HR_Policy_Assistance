from hr_assistant import config
from langchain_community.embeddings import JinaEmbeddings
import os
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_embeddings_model():
    logger.info("Initializing embeddings model '%s'", config.EMBEDDING_MODEL_NAME)
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)