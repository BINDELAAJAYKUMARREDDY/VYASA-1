import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DB_DIR = ROOT_DIR / "db"

CORPUS_PATH = PROCESSED_DIR / "vyasa_corpus.jsonl"

CHROMA_DIR = DB_DIR / "chroma"
CHROMA_COLLECTION = "vyasa_corpus"

EMBED_MODEL_NAME = os.environ.get("VYASA_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

MAX_CONTEXT_CHARS = int(os.environ.get("VYASA_MAX_CONTEXT_CHARS", "12000"))
TOP_K_PER_CHUNK = int(os.environ.get("VYASA_TOP_K_PER_CHUNK", "4"))

CHUNK_WORDS = int(os.environ.get("VYASA_CHUNK_WORDS", "800"))

# v4 reliability controls
MAX_CONCURRENT_REQUESTS = int(os.environ.get("VYASA_MAX_CONCURRENT_REQUESTS", "3"))
ASK_TIMEOUT_S = int(os.environ.get("VYASA_ASK_TIMEOUT_S", "60"))
RETRIEVE_TIMEOUT_S = int(os.environ.get("VYASA_RETRIEVE_TIMEOUT_S", "8"))
LLM_TIMEOUT_S = int(os.environ.get("VYASA_LLM_TIMEOUT_S", "35"))
