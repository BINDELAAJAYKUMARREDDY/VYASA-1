import os


_map = {
    "rigveda": "Rigveda",
    "yajurveda": "Yajurveda",
    "samaveda": "Samaveda",
    "atharvaveda": "Atharvaveda",
    "bhagavad gita": "Bhagavad Gita",
    "gita": "Bhagavad Gita",
    "ramayana": "Ramayana",
    "valmiki": "Valmiki Ramayana",
    "mahabharata": "Mahabharata",
    "bhagavatam": "Srimad Bhagavatam",
    "upanishad": "Upanishads",
    "puran": "Puranas",
    "gretil": "GRETIL Corpus",
    "itihasa": "Itihasa",
    "dharmicdata": "DharmicData",
    "vedanta": "Vedanta",
}


def get_clean_source_name(src: str) -> str:
    s = (src or "").strip()
    if not s:
        return "Scripture"

    low = s.lower()
    for k, v in _map.items():
        if k in low:
            return v

    # last fallback: filename-ish cleanup
    base = os.path.basename(s)
    base = base.replace("_", " ").replace("-", " ").strip()
    return base[:80] if base else s[:80]

