def detect_intent(text: str) -> str:
    t = (text or "").lower()

    # order matters: pick the most specific first
    if any(w in t for w in ["what is", "what are", "define", "meaning of", "explain", "how does", "why"]):
        return "explanation"

    if any(w in t for w in ["specific line", "which verse", "exact reference", "verse number", "canto", "chapter"]):
        return "technical"

    if any(w in t for w in ["should i", "what should", "help me", "advice", "what do i do", "how to", "steps", "plan"]):
        return "advice"

    if any(w in t for w in ["i feel", "i am feeling", "scared", "anxious", "sad", "depressed", "hopeless", "lonely", "grief"]):
        return "emotional_support"

    if any(w in t for w in ["dharma", "karma", "moksha", "atman", "brahman", "meaning of life", "who am i", "soul"]):
        return "philosophical_question"

    return "advice"

