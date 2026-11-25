import os
from dotenv import load_dotenv

load_dotenv

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# load data
DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")

# vector store
VECTOR_STORE_PATH = os.path.join("data","faiss.index")

# models
LLM_MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "jina-embeddings-v3"


## 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# retrieval process
TOP_K_RESULTS = 3


## system instruct
SYSTEM_PROMPT = ("You are HR assistant. Always search in policy documents do not add anything beyond the hr policy")


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")