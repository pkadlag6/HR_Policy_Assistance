# step 1: read data
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def load_document(dir_path: str = config.DATA_DIR):
    logger.info("locument loading from data folder",config.DATA_DIR)
    loader = DirectoryLoader(
        dir_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    logger.info("Loaded %d documents(s)",len(documents))
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {dir_path}")
    return documents