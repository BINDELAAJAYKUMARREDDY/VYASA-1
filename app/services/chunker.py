import re
from typing import List, Dict, Any


def chunk_text(text: str, max_words: int = 800):
    # This one is for long user inputs (10k+ words). Keep it simple and safe.
    text = (text or "").strip()
    if not text:
        return []

    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def chunk_verses(verses: List[Dict[str, Any]], min_verses: int = 3, max_verses: int = 5, overlap: int = 1):
    # Contextual chunking: group verses together so retrieval feels less "single-line".
    # Verses here are dicts like: {"source": "...", "sanskrit": "...", "english": "..."}
    if not verses:
        return []

    if overlap < 0:
        overlap = 0
    if max_verses < 1:
        max_verses = 1
    if min_verses > max_verses:
        min_verses = max_verses

    blocks = []
    step = max(1, max_verses - overlap)
    i = 0

    while i < len(verses):
        block_items = verses[i : i + max_verses]
        if len(block_items) < min_verses and blocks:
            break

        parts = []
        for v in block_items:
            src = (v.get("source") or "").strip()
            sa = (v.get("sanskrit") or "").strip()
            en = (v.get("english") or "").strip()

            line = []
            if src:
                line.append(src)
            if sa:
                line.append("Sanskrit: " + sa[:600])
            if en:
                line.append("English: " + en[:700])
            parts.append("\n".join(line).strip())

        blocks.append("\n\n".join([p for p in parts if p]).strip())
        i += step

    return [b for b in blocks if b]

