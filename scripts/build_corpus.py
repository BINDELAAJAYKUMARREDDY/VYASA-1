import argparse
import shutil
from pathlib import Path
import json
import re

import sys
from pathlib import Path as _Path


# Make `import app...` work when running from scripts/
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-each", type=int, default=0)
    parser.add_argument("--qa-count", type=int, default=500)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the proven script you already have (updated to strict real sources)
    # and place its outputs into data/processed/.
    import scripts.phase1_build_corpus as p1
    p1.main_args(limit_each=args.limit_each, qa_count=args.qa_count)

    # Move files into the strict structure
    out_corpus = PROCESSED_DIR / "vyasa_corpus.jsonl"
    out_qa = PROCESSED_DIR / "vyasa_qa.jsonl"

    shutil.copy2(p1.CORPUS_FILE, out_corpus)
    shutil.copy2(p1.QA_FILE, out_qa)

    # build enriched verse-level corpus with metadata + contextual chunks
    out_verses = PROCESSED_DIR / "vyasa_corpus_verses.jsonl"
    out_chunks = PROCESSED_DIR / "vyasa_corpus_chunks.jsonl"
    verses = _enrich_verse_metadata(out_corpus)
    _write_jsonl(out_verses, verses)
    chunks = _chunk_verses_for_index(verses, min_verses=3, max_verses=5, overlap=1)
    _write_jsonl(out_chunks, chunks)

    print("Corpus:", out_corpus)
    print("QA:", out_qa)
    print("Verse corpus:", out_verses)
    print("Chunk corpus:", out_chunks)


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _enrich_verse_metadata(corpus_path: Path):
    rows = []
    seen = set()
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            src = _normalize_source_name((e.get("source") or "").strip())
            en = _clean_text((e.get("english") or "").strip())
            sa = _clean_text((e.get("sanskrit") or "").strip())
            if not en and not sa:
                continue

            canto, chapter, verse = _parse_ref(src)
            e["source"] = src
            e["english"] = en
            e["sanskrit"] = sa
            e["canto"] = canto
            e["chapter"] = chapter
            e["verse"] = verse

            dedup_key = (src.lower(), en[:280].lower(), sa[:280].lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(e)
    return rows


def _parse_ref(src: str):
    # simple parser: extracts up to 3 numbers from source string
    nums = re.findall(r"\d+", src or "")
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    if len(nums) == 2:
        return "", nums[0], nums[1]
    if len(nums) == 1:
        return "", "", nums[0]
    return "", "", ""


def _chunk_verses_for_index(verses, min_verses: int = 3, max_verses: int = 5, overlap: int = 1):
    out = []
    step = max(1, max_verses - overlap)
    i = 0
    idx = 0
    while i < len(verses):
        block = verses[i : i + max_verses]
        if len(block) < min_verses and out:
            break

        text_parts = []
        src_first = (block[0].get("source") or "") if block else ""
        src_last = (block[-1].get("source") or "") if block else ""
        for v in block:
            s = (v.get("sanskrit") or "").strip()
            en = (v.get("english") or "").strip()
            src = (v.get("source") or "").strip()
            part = f"{src}\n"
            if s:
                part += f"Sanskrit: {s[:400]}\n"
            if en:
                part += f"English: {en[:500]}"
            text_parts.append(part.strip())

        text = "\n\n".join([p for p in text_parts if p]).strip()
        out.append(
            {
                "id": f"chunk_{idx}",
                "text": text,
                "source": f"{src_first} ... {src_last}".strip(),
                "canto": block[0].get("canto", "") if block else "",
                "chapter": block[0].get("chapter", "") if block else "",
                "verse": block[0].get("verse", "") if block else "",
                "size_verses": len(block),
            }
        )
        idx += 1
        i += step
    return out


def _clean_text(text: str):
    t = (text or "").replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # remove obvious noise lines
    t = t.replace("Click here", "").replace("Read more", "")
    return t


def _normalize_source_name(source: str):
    s = (source or "").strip()
    low = s.lower()
    if "atharvaveda" in low:
        return "Atharva Veda"
    if "rigveda" in low:
        return "Rig Veda"
    if "yajur" in low:
        return "Yajur Veda"
    if "samaveda" in low:
        return "Sama Veda"
    if "bhagavad" in low and "gita" in low:
        return "Bhagavad Gita"
    if "bhagavat" in low:
        return "Srimad Bhagavatam"
    if "mahabharat" in low:
        return "Mahabharata"
    if "ramayan" in low:
        return "Ramayana"
    if "upanishad" in low:
        return "Upanishads"
    return s


if __name__ == "__main__":
    main()

