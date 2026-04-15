import os
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

DATA_DIR = Path("data")
CORPUS_FILE = DATA_DIR / "vyasa_corpus.jsonl"
CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("VYASA-1 Phase 2: Building RAG Vector Index")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection("vyasa_corpus")
except:
    pass

collection = client.create_collection(
    name="vyasa_corpus",
    metadata={"hnsw:space": "cosine"}
)

print("\nReading corpus...")
entries = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            e = json.loads(line.strip())
            if e.get("english", "").strip():
                entries.append(e)
        except:
            pass

print(f"Total entries to index: {len(entries)}")

BATCH = 256
added = 0

for start in range(0, len(entries), BATCH):
    batch = entries[start:start + BATCH]

    texts = []
    ids = []
    metas = []

    for e in batch:
        text = e.get("english", "")
        if e.get("sanskrit"):
            text = e["sanskrit"][:200] + " " + text
        texts.append(text[:512])
        ids.append(str(e.get("id", f"entry_{start}")))
        metas.append({
            "source": str(e.get("source", ""))[:200],
            "category": str(e.get("category", "general")),
            "keywords": ",".join(e.get("keywords", [])),
            "sanskrit": str(e.get("sanskrit", ""))[:300],
            "english": str(e.get("english", ""))[:500],
        })

    vecs = embed_model.encode(texts, batch_size=64, show_progress_bar=False, device=device)
    vecs_list = vecs.tolist()

    collection.add(
        ids=ids,
        embeddings=vecs_list,
        metadatas=metas,
        documents=texts
    )

    added += len(batch)
    print(f"   Indexed {added}/{len(entries)}", end="\r")

print(f"\n\nTotal indexed: {added}")
print(f"ChromaDB saved to: {CHROMA_DIR}")
print("\nPHASE 2 COMPLETE – Ready for next phase")
