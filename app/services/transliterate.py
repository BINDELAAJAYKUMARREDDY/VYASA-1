def _looks_sanskrit(text: str) -> bool:
    # quick check for Devanagari block
    for ch in text or "":
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return True
    return False


def to_iast(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if not _looks_sanskrit(t):
        return t

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        return transliterate(t, sanscript.DEVANAGARI, sanscript.IAST)
    except Exception:
        # fallback: if library not installed or fails, just return original
        return t

