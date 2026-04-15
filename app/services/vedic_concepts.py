from typing import Dict, Any, Optional


MAHAVAKYAS: Dict[str, Dict[str, str]] = {
    "prajnanam brahma": {
        "source": "Aitareya Upanishad",
        "reference": "3.3",
        "meaning": "Consciousness is Brahman",
    },
    "tat tvam asi": {
        "source": "Chandogya Upanishad",
        "reference": "6.8.7",
        "meaning": "You are That",
    },
    "aham brahmasmi": {
        "source": "Brihadaranyaka Upanishad",
        "reference": "1.4.10",
        "meaning": "I am Brahman",
    },
    "ayam atma brahma": {
        "source": "Mandukya Upanishad",
        "reference": "2",
        "meaning": "This Self is Brahman",
    },
}


def detect_canonical(query: str) -> Optional[Dict[str, Any]]:
    t = (query or "").lower().strip()
    if not t:
        return None

    for key, row in MAHAVAKYAS.items():
        if key in t:
            return {
                "matched": key,
                "source": row["source"],
                "reference": row["reference"],
                "meaning": row["meaning"],
            }

    # small synonym support
    if "consciousness is brahman" in t:
        row = MAHAVAKYAS["prajnanam brahma"]
        return {
            "matched": "prajnanam brahma",
            "source": row["source"],
            "reference": row["reference"],
            "meaning": row["meaning"],
        }

    return None
