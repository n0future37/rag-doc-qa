import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"

FAISS_INDEX_PATH = STORAGE_DIR / "faiss.index"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"


RIGHT_CODE_API_KEY = os.getenv("RIGHT_CODE_API_KEY", "")
RIGHT_CODE_MODEL = os.getenv("RIGHT_CODE_MODEL", "gpt-5.2")
RIGHT_CODE_BASE_URL = os.getenv(
    "RIGHT_CODE_BASE_URL",
    "https://www.right.codes/codex/v1/responses",
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    STORAGE_DIR.mkdir(exist_ok=True)