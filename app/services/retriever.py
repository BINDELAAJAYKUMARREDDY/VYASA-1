from typing import List, Dict, Any

import os
import chromadb

from app.core.config import CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL_NAME
from app.core.config import CORPUS_PATH
from app.core.logger import get_logger
from app.services.embedder import LocalTextEmbedder
from app.services.source_mapper import get_clean_source_name


_embed_model = None
_client = None
_collection = None
_embed_cache = {}
_retrieval_cache = {}
_reranker = None
_embed_mode = "unknown"
log = get_logger("vyasa.retriever")


def _get_device():
    try:
        import torch
        force = (os.environ.get("VYASA_FORCE_DEVICE", "auto") or "auto").lower().strip()
        has_cuda = torch.cuda.is_available()
        if force == "cuda" and not has_cuda:
            raise RuntimeError("VYASA_FORCE_DEVICE=cuda but CUDA is unavailable in current Python environment.")
        if has_cuda and force in ["auto", "cuda"]:
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def _get_embed_model():
    global _embed_model, _embed_mode
    if _embed_model is None:
        import torch

        device = _get_device()
        models = [EMBED_MODEL_NAME, "sentence-transformers/all-MiniLM-L6-v2"]
        last_error = None
        for model_name in models:
            try:
                _embed_model = LocalTextEmbedder(model_name, device=device)
                _embed_mode = f"transformers:{model_name}"
                log.info(f"embedding_mode={_embed_mode} device={device}")
                break
            except Exception as ex:
                last_error = ex
                log.warning(f"embedding_load_failed model={model_name} err={type(ex).__name__}")
        if _embed_model is None:
            raise RuntimeError(
                "No sentence-transformers embedding model could be loaded. "
                "Install sentence-transformers and fix torch/torchvision compatibility."
            ) from last_error
    return _embed_model


def _get_collection():
    global _client, _collection
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if _collection is None:
        try:
            _collection = _client.get_collection(CHROMA_COLLECTION)
        except Exception:
            _collection = _client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


def retrieve_many(chunks: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    try:
        col = _get_collection()
        emb = _get_embed_model()
        device = _get_device()
    except Exception:
        log.warning("embedding_unavailable -> keyword_scan_fallback")
        return _fallback_keyword_scan(chunks, top_k=top_k)

    results_all = []
    for chunk in chunks:
        ckey = (chunk or "").strip()
        if ckey in _retrieval_cache:
            results_all.extend(_retrieval_cache[ckey][:])
            continue

        vec = _encode_cached(emb, device, ckey)
        res = col.query(
            query_embeddings=[vec],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        metas = (res or {}).get("metadatas") or []
        if metas:
            for m in metas[0]:
                dist = None
                try:
                    dist = ((res or {}).get("distances") or [[None]])[0][metas[0].index(m)]
                except Exception:
                    dist = None
                results_all.append(
                    {
                        "source": (m.get("source") or "").strip(),
                        "sanskrit": (m.get("sanskrit") or "").strip(),
                        "english": (m.get("english") or "").strip(),
                        "category": (m.get("category") or "").strip(),
                        "keywords": (m.get("keywords") or "").strip(),
                        "canto": (m.get("canto") or ""),
                        "chapter": (m.get("chapter") or ""),
                        "verse": (m.get("verse") or ""),
                        "retrieval_distance": dist,
                    }
                )
        _retrieval_cache[ckey] = results_all[-top_k:]
        _trim_cache(_retrieval_cache, max_items=80)
    return results_all


def retrieve_and_rerank(query: str, chunks: List[str], retrieve_k: int = 20, final_k: int = 5):
    # 0) rewrite query (small fix for better retrieval)
    q2 = rewrite_query(query)

    # 1) first retrieval
    docs = retrieve_many(chunks + [q2], top_k=min(10, retrieve_k))
    if not docs:
        return []

    # 1.5) multi-hop retrieval with expanded terms
    hop_q = expand_query(q2)
    docs2 = retrieve_many([hop_q], top_k=max(10, retrieve_k))
    docs = docs + docs2

    # 2) rerank using CrossEncoder (fallback to our lightweight rank_results)
    scored = _rerank_cross_encoder(q2, docs)
    if not scored:
        scored = rank_results(query, docs)
    log.info(f"retrieval docs={len(docs)} reranked={len(scored)}")

    # 3) enforce source diversity (avoid 5 from same book)
    diverse = diversify_sources(scored, final_k=final_k, max_per_source=2)
    if diverse:
        preview = ",".join([str(round(float(d.get("score", 0.0)), 3)) for d in diverse[:5]])
        log.info(f"retrieval_top_scores={preview}")
    return diverse


def rewrite_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q
    low = q.lower()
    if "brahmaloka" in low and "time" in low:
        return q + " Srimad Bhagavatam time difference devas brahma loka verse"
    if "dharma" in low:
        return q + " dharma duty truth ahimsa teaching verse"
    if "which verse" in low or "exact reference" in low or "specific line" in low:
        return q + " exact verse source chapter canto"
    if "sun chariot" in low and "seven horses" in low:
        return "Vishnu Purana Surya chariot seven horses meaning verse"
    return q


def expand_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q
    extras = ["Bhagavad Gita", "Mahabharata", "Bhagavatam", "Upanishads", "Vishnu Purana"]
    return q + " " + " ".join(extras)


def diversify_sources(docs: List[Dict[str, Any]], final_k: int = 5, max_per_source: int = 2):
    if not docs:
        return []

    picked = []
    counts = {}

    for d in docs:
        src = (d.get("source") or "").strip()
        name = get_clean_source_name(src)
        counts[name] = counts.get(name, 0)

        if counts[name] >= max_per_source:
            continue

        picked.append(d)
        counts[name] += 1

        if len(picked) >= final_k:
            break

    # if still short, fill with remaining (even if repeated)
    if len(picked) < final_k:
        for d in docs:
            if d in picked:
                continue
            picked.append(d)
            if len(picked) >= final_k:
                break

    return picked[:final_k]


def _rerank_cross_encoder(query: str, docs: List[Dict[str, Any]]):
    global _reranker
    q = (query or "").strip()
    if not q or not docs:
        return []

    try:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = []
        for d in docs:
            txt = ((d.get("english") or "") + " " + (d.get("sanskrit") or "") + " " + (d.get("source") or "")).strip()
            pairs.append((q[:4000], txt[:4000]))

        scores = _reranker.predict(pairs)
        out = []
        for d, s in zip(docs, scores):
            d2 = dict(d)
            d2["score"] = float(s)
            out.append(d2)

        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out
    except Exception:
        return []


def _encode_cached(emb, device: str, text: str):
    key = (text or "").strip()
    if not key:
        return emb.encode([""], batch_size=1, normalize_embeddings=True)[0]

    if key in _embed_cache:
        return _embed_cache[key]

    vec = emb.encode([key], batch_size=1, normalize_embeddings=True)[0]
    _embed_cache[key] = vec
    _trim_cache(_embed_cache, max_items=120)
    return vec


def _trim_cache(d: dict, max_items: int = 100):
    if len(d) <= max_items:
        return
    # remove oldest inserted keys (insertion-ordered dict in py3.7+)
    for k in list(d.keys())[: max(1, len(d) - max_items)]:
        d.pop(k, None)


def _fallback_keyword_scan(chunks: List[str], top_k: int = 4):
    if not CORPUS_PATH.exists():
        return []

    # Very simple + safe: count overlaps of top words with each entry's english.
    import json
    import re

    wanted = []
    for ch in chunks:
        words = re.findall(r"[a-z]{4,}", (ch or "").lower())
        wanted.extend(words[:40])
    wanted = list(dict.fromkeys(wanted))[:120]
    if not wanted:
        return []

    scored = []
    with open(CORPUS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 20000:
                break
            try:
                e = json.loads(line)
            except Exception:
                continue
            en = (e.get("english") or "").lower()
            if not en:
                continue
            score = 0
            for w in wanted:
                if w in en:
                    score += 1
            if score > 0:
                scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, e in scored[: max(5, top_k * 4)]:
        out.append(
            {
                "source": (e.get("source") or "").strip(),
                "sanskrit": (e.get("sanskrit") or "").strip(),
                "english": (e.get("english") or "").strip(),
                "category": (e.get("category") or "").strip(),
                "keywords": ",".join(e.get("keywords") or []),
                "canto": (e.get("canto") or ""),
                "chapter": (e.get("chapter") or ""),
                "verse": (e.get("verse") or ""),
            }
        )
    return out


def find_exact_verse(query: str, docs: List[Dict[str, Any]]):
    # Used when user asks for specific line / exact verse.
    t = (query or "").lower()
    wants_exact = any(k in t for k in ["specific line", "which verse", "exact reference", "where exactly", "exact verse"])
    if not wants_exact:
        return None

    for d in docs:
        v = str(d.get("verse") or "").strip()
        ch = str(d.get("chapter") or "").strip()
        can = str(d.get("canto") or "").strip()
        src = (d.get("source") or "").strip()
        if v and (ch or can or src):
            return d
    return None


def merge_context(items: List[Dict[str, Any]], max_chars: int = 12000):
    seen = set()
    merged = []

    def key_of(x):
        return (x.get("source", "")[:200], x.get("english", "")[:300])

    for it in items:
        # context filtering: drop low-score entries if present
        if isinstance(it.get("score"), (int, float)) and it["score"] < 0.2:
            continue
        k = key_of(it)
        if k in seen:
            continue
        seen.add(k)
        merged.append(it)

    # keep simple: prefer entries that have both sanskrit+english and a source
    def score(x):
        s = float(x.get("score") or 0.0)
        if x.get("source"):
            s += 2
        if x.get("english"):
            s += 2
        if x.get("sanskrit"):
            s += 1
        return s

    merged.sort(key=score, reverse=True)

    blocks = []
    total = 0
    for i, x in enumerate(merged, 1):
        src = x.get("source", "")
        sa = x.get("sanskrit", "")
        en = x.get("english", "")

        block = []
        block.append(f"[Source {i}] {src}".strip())
        if sa:
            block.append("Sanskrit: " + sa[:800])
        if en:
            block.append("English: " + en[:900])
        block_text = "\n".join(block).strip() + "\n\n"

        if total + len(block_text) > max_chars:
            break
        blocks.append(block_text)
        total += len(block_text)

    context = "".join(blocks).strip()
    sources = [x.get("source", "") for x in merged if x.get("source")]
    return context, sources


def rank_results(query: str, items: List[Dict[str, Any]]):
    q = (query or "").lower()
    q_words = _words(q)
    q_set = set(q_words)

    ranked = []
    for it in items:
        src = (it.get("source") or "")
        text = ((it.get("english") or "") + " " + (it.get("sanskrit") or "") + " " + src).lower()

        overlap = 0
        if q_set:
            t_words = set(_words(text))
            overlap = len(q_set & t_words)

        length = len(text)
        length_penalty = min(1.5, length / 3000.0)

        base = overlap / max(3, len(q_set))
        src_bonus = 0.2 if src else 0.0
        en_bonus = 0.1 if (it.get("english") or "").strip() else 0.0
        score = max(0.0, (base + src_bonus + en_bonus) - 0.15 * length_penalty)

        it2 = dict(it)
        it2["score"] = float(score)
        ranked.append(it2)

    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked


def format_top_passages(items: List[Dict[str, Any]], limit: int = 3):
    out = []
    for i, it in enumerate(items[:limit], 1):
        src = (it.get("source") or "").strip()
        sa = (it.get("sanskrit") or "").strip()
        en = (it.get("english") or "").strip()

        lines = []
        lines.append(f"Passage {i} — {src}" if src else f"Passage {i}")
        if sa:
            lines.append("Sanskrit: " + sa[:600])
        if en:
            lines.append("English: " + en[:700])
        out.append("\n".join(lines).strip())
    return "\n\n".join(out).strip()


def _words(text: str):
    import re

    return re.findall(r"[a-z]{3,}", (text or "").lower())


def validate_consistency(items: List[Dict[str, Any]], min_sim: float = 0.55, max_keep: int = 30):
    # Keep a consistent cluster of passages to reduce conflicts.
    # Primary: embedding cosine similarity (if available). Fallback: word-set Jaccard.
    if not items:
        return [], {"method": "none", "kept": 0, "dropped": 0}

    base = items[: max_keep * 2]

    try:
        emb = _get_embed_model()
        device = _get_device()

        texts = []
        for it in base:
            t = ((it.get("english") or "") + " " + (it.get("source") or "")).strip()
            texts.append(t[:800])

        vecs = emb.encode(texts, device=device, show_progress_bar=False)

        def cos(a, b):
            import math

            dot = float((a * b).sum())
            na = math.sqrt(float((a * a).sum())) + 1e-9
            nb = math.sqrt(float((b * b).sum())) + 1e-9
            return dot / (na * nb)

        # pick the center: highest average similarity
        sims = []
        for i in range(len(base)):
            total = 0.0
            for j in range(len(base)):
                if i == j:
                    continue
                total += cos(vecs[i], vecs[j])
            sims.append(total / max(1, len(base) - 1))

        center = max(range(len(sims)), key=lambda i: sims[i])
        keep = []
        for i in range(len(base)):
            s = cos(vecs[center], vecs[i])
            if s >= min_sim:
                keep.append(base[i])

        if len(keep) < 3:
            keep = base[: min(6, len(base))]

        return keep, {"method": "embedding", "kept": len(keep), "dropped": len(items) - len(keep)}
    except Exception:
        # fallback: jaccard on words
        def jacc(a, b):
            sa = set(_words(a))
            sb = set(_words(b))
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / (len(sa | sb) + 1e-9)

        texts = [((it.get("english") or "") + " " + (it.get("source") or "")).lower() for it in base]
        avg = []
        for i in range(len(base)):
            total = 0.0
            for j in range(len(base)):
                if i == j:
                    continue
                total += jacc(texts[i], texts[j])
            avg.append(total / max(1, len(base) - 1))

        center = max(range(len(avg)), key=lambda i: avg[i])
        keep = []
        for i in range(len(base)):
            s = jacc(texts[center], texts[i])
            if s >= 0.12:
                keep.append(base[i])

        if len(keep) < 3:
            keep = base[: min(6, len(base))]

        return keep, {"method": "jaccard", "kept": len(keep), "dropped": len(items) - len(keep)}


def summarize_for_llm(query: str, items: List[Dict[str, Any]], max_chars: int = 6000):
    # Simple sentence selection: pick sentences with best query word overlap.
    import re

    q_words = set(_words(query))
    if not q_words or not items:
        return "", []

    sent_rows = []
    for it in items:
        src = (it.get("source") or "").strip()
        en = (it.get("english") or "").strip()
        if not en:
            continue
        sents = re.split(r"(?<=[.!?])\s+|\n+", en)
        for s in sents:
            s = s.strip()
            if len(s) < 40:
                continue
            sw = set(_words(s))
            if not sw:
                continue
            overlap = len(sw & q_words)
            if overlap <= 0:
                continue
            score = overlap / max(3, len(q_words))
            sent_rows.append((score, src, s))

    sent_rows.sort(key=lambda x: x[0], reverse=True)

    out_lines = []
    used_sources = []
    total = 0
    for score, src, s in sent_rows:
        line = f"- ({src}) {s}".strip()
        if total + len(line) + 1 > max_chars:
            break
        out_lines.append(line)
        total += len(line) + 1
        if src and src not in used_sources:
            used_sources.append(src)
        if len(out_lines) >= 18:
            break

    return "\n".join(out_lines).strip(), used_sources


def context_only_answer(question: str, context: str):
    question = (question or "").strip()
    if not context:
        return "I could not find any relevant scripture passages in the local index yet. Please run `python scripts\\build_index.py` after building the corpus."

    lines = []
    lines.append("I will answer using only the passages found in the local scripture corpus.")
    lines.append("")
    lines.append("Relevant passages:")
    lines.append(context[:12000])
    lines.append("")
    lines.append("Practical steps (safe, general):")
    lines.append("1. Pick one small right action you can do today and do it with steadiness.")
    lines.append("2. Avoid harm in speech and action; choose truth and self-control.")
    lines.append("3. Re-read one cited verse slowly and write one lesson you can apply.")
    return "\n".join(lines).strip()

