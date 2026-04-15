import json
import os
from pathlib import Path

import chromadb

import sys
from pathlib import Path as _Path

# Make `import app...` work when running from scripts/
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import CORPUS_PATH, CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL_NAME
from app.services.embedder import LocalTextEmbedder


def main():
    chunk_path = CORPUS_PATH.parent / "vyasa_corpus_chunks.jsonl"
    use_path = chunk_path if chunk_path.exists() else CORPUS_PATH

    if not use_path.exists():
        raise SystemExit(f"Missing corpus: {CORPUS_PATH}. Run: python scripts\\build_corpus.py")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    col = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    embed_mode, emb, device = _load_embedder()
    print(f"Embedding mode: {embed_mode}")
    print(f"Embedding device: {device}")

    entries = []
    with open(use_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            txt_check = (e.get("english") or e.get("text") or "").strip()
            if not txt_check:
                continue
            entries.append(e)

    batch_size = 64
    done = 0

    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        texts = []
        ids = []
        metas = []

        for e in batch:
            s = (e.get("sanskrit") or "")[:200]
            en = (e.get("english") or e.get("text") or "")[:450]
            text = (s + " " + en).strip()[:650]
            texts.append(text)
            ids.append(str(e.get("id") or f"row_{start}"))
            metas.append(
                {
                    "source": (e.get("source") or "")[:250],
                    "category": (e.get("category") or "general"),
                    "keywords": ",".join(e.get("keywords") or []),
                    "sanskrit": (e.get("sanskrit") or "")[:900],
                    "english": (e.get("english") or "")[:1200],
                    "canto": str(e.get("canto") or ""),
                    "chapter": str(e.get("chapter") or ""),
                    "verse": str(e.get("verse") or ""),
                }
            )

        vecs = _encode_texts(emb, texts, device=device, mode=embed_mode)
        col.add(ids=ids, embeddings=vecs, metadatas=metas, documents=texts)

        done += len(batch)
        print(f"Indexed {done}/{len(entries)}", end="\r")

    print(f"\nDone. Chroma saved at: {CHROMA_DIR}")


def _load_embedder():
    # Try text-only transformer embedding path (avoids sentence-transformers CLIP import issues).
    import torch

    force = (os.environ.get("VYASA_FORCE_DEVICE", "auto") or "auto").lower().strip()
    has_cuda = torch.cuda.is_available()
    if force == "cuda" and not has_cuda:
        raise RuntimeError(
            "VYASA_FORCE_DEVICE=cuda is set, but CUDA is not available in this Python. "
            "Install a CUDA-enabled PyTorch build."
        )
    device = "cuda" if (has_cuda and force in ["auto", "cuda"]) else "cpu"
    model_candidates = [EMBED_MODEL_NAME, "sentence-transformers/all-MiniLM-L6-v2"]
    last_error = None
    for model_name in model_candidates:
        try:
            emb = LocalTextEmbedder(model_name, device=device)
            return f"transformers:{model_name}", emb, device
        except Exception as ex:
            print(f"Warning: failed to load {model_name} ({type(ex).__name__})")
            last_error = ex

    raise RuntimeError(
        "Could not load embedding model with transformers backend. "
        "Please verify internet/model download and torch installation."
    ) from last_error


def _encode_texts(emb, texts, device: str, mode: str):
    return emb.encode(texts, batch_size=64, normalize_embeddings=True)


if __name__ == "__main__":
    main()

